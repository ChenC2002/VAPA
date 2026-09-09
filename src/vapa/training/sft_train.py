"""Executable, resume-safe action-token supervised fine-tuning.

The module is dependency-neutral at import time.  A real run lazily constructs the
Hugging Face/PEFT adapters, while validation and tests can use the built-in byte
tokenizer or inject protocol-compatible factories.
"""

from __future__ import annotations

import argparse
import math
import random
import tomllib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from vapa import __version__
from vapa.actions import Action, ActionParseError, parse_action
from vapa.artifacts import (
    ArtifactContentKind,
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    guard_artifact_write_path,
    strict_json_loads,
    strict_jsonl_loads,
)
from vapa.model.protocols import (
    ActorModelAdapter,
    OptimizerAdapter,
    TokenizedAction,
    TokenizerAdapter,
)
from vapa.provenance import package_code_fingerprint
from vapa.schemas import ActionKind, Domain, MemoryItem, MemoryStatus, TimeWindow, normalize_field
from vapa.training.checkpoint import (
    CheckpointContract,
    RuntimeState,
    StateStore,
    TorchStateStore,
    fingerprint_payload,
    read_manifest,
    resume_checkpoint,
    save_checkpoint,
)
from vapa.training.runtime import (
    DistributedContext,
    SFTExample,
    WarmupCosineScheduler,
    build_adamw,
    resolve_device,
    seed_everything,
    train_sft_step,
)

SFT_DATA_SCHEMA_VERSION = "vapa-sft-jsonl-v1"
SFT_RUN_SCHEMA_VERSION = "vapa-sft-run-v1"
SFT_METRIC_SCHEMA_VERSION = "vapa-sft-metric-v1"


class SFTDataError(ValueError):
    """Raised when an action demonstration violates the JSONL contract."""


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class SFTTrainConfig:
    """Complete SFT lifecycle configuration, excluding the dry-run switch."""

    data_path: Path = Path("examples/tiny_sft.jsonl")
    output_dir: Path = Path("runs/sft")
    content_kind: str = "credentialed"
    run_id: str = "vapa-sft"
    model_name: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "main"
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    model_kind: str = "auto"
    use_processor: bool | None = None
    enable_thinking: bool = False
    reasoning_effort: str | None = None
    dequantize_mxfp4: bool = False
    dtype: str = "bfloat16"
    device: str = "auto"
    context_tokens: int = 32_768
    scaffold: str = ""
    epochs: int = 1
    action_token_floor: int = 131_072
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    warmup_fraction: float = 0.03
    final_lr_fraction: float = 0.1
    seed: int = 0
    shuffle: bool = True
    deterministic: bool = False
    checkpoint_every: int = 1
    resume_from: Path | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] | None = None
    generation_temperature: float = 1.0
    generation_top_p: float = 1.0
    generation_top_k: int = 0
    generation_max_tokens: int = 512
    code_version: str = f"vapa-ehr-{__version__}"

    def validate(self) -> None:
        if not isinstance(self.data_path, Path) or not isinstance(self.output_dir, Path):
            raise TypeError("data_path and output_dir must be Paths")
        if self.resume_from is not None and not isinstance(self.resume_from, Path):
            raise TypeError("resume_from must be a Path or None")
        try:
            ArtifactContentKind(self.content_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("content_kind must be a valid artifact disclosure class") from error
        for name in (
            "run_id",
            "model_name",
            "model_revision",
            "model_kind",
            "dtype",
            "device",
            "code_version",
        ):
            _nonempty_text(getattr(self, name), name)
        if not isinstance(self.scaffold, str):
            raise TypeError("scaffold must be a string")
        for name in ("tokenizer_name", "tokenizer_revision"):
            value = getattr(self, name)
            if value is not None:
                _nonempty_text(value, name)
        if self.model_kind not in {"auto", "causal", "multimodal"}:
            raise ValueError("model_kind must be 'auto', 'causal', or 'multimodal'")
        if self.use_processor is not None and not isinstance(self.use_processor, bool):
            raise TypeError("use_processor must be a boolean or None")
        if self.reasoning_effort is not None and self.reasoning_effort not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("reasoning_effort must be 'low', 'medium', 'high', or None")
        if "gpt-oss" in self.model_name.casefold():
            if self.reasoning_effort is None:
                raise ValueError("gpt-oss requires an explicit reasoning_effort")
            if self.enable_thinking:
                raise ValueError("gpt-oss uses reasoning_effort instead of enable_thinking")
        elif self.reasoning_effort is not None:
            raise ValueError("reasoning_effort is supported only for gpt-oss models")
        if self.dequantize_mxfp4 and "gpt-oss" not in self.model_name.casefold():
            raise ValueError("dequantize_mxfp4 is supported only for gpt-oss models")
        for name in (
            "shuffle",
            "deterministic",
            "enable_thinking",
            "dequantize_mxfp4",
            "local_files_only",
            "trust_remote_code",
            "use_lora",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in (
            "epochs",
            "action_token_floor",
            "checkpoint_every",
            "lora_rank",
            "lora_alpha",
            "generation_max_tokens",
            "context_tokens",
        ):
            _positive_integer(getattr(self, name), name)
        _nonnegative_integer(self.seed, "seed")
        _nonnegative_integer(self.generation_top_k, "generation_top_k")
        for name in (
            "learning_rate",
            "weight_decay",
            "epsilon",
            "gradient_clip",
            "warmup_fraction",
            "final_lr_fraction",
            "lora_dropout",
            "generation_temperature",
            "generation_top_p",
        ):
            _finite_number(getattr(self, name), name)
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate, epsilon, and gradient_clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 < self.final_lr_fraction <= 1:
            raise ValueError("final_lr_fraction must be in (0, 1]")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.generation_temperature < 0:
            raise ValueError("generation_temperature must be nonnegative")
        if self.generation_max_tokens > self.context_tokens:
            raise ValueError("generation_max_tokens cannot exceed context_tokens")
        if not 0 < self.generation_top_p <= 1:
            raise ValueError("generation_top_p must be in (0, 1]")
        if len(self.betas) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value < 1
            for value in self.betas
        ):
            raise ValueError("betas must contain two finite values in [0, 1)")
        if self.lora_target_modules is not None and (
            not self.lora_target_modules
            or any(
                not isinstance(item, str) or not item.strip() for item in self.lora_target_modules
            )
        ):
            raise ValueError("lora_target_modules must contain non-empty names")

    @property
    def effective_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.model_name

    @property
    def effective_tokenizer_revision(self) -> str:
        return self.tokenizer_revision or self.model_revision

    def fingerprint_payload(self) -> dict[str, object]:
        """Return the resume contract, intentionally excluding machine-local paths."""

        ignored = {"data_path", "output_dir", "resume_from"}
        payload = {key: value for key, value in asdict(self).items() if key not in ignored}
        payload["betas"] = list(self.betas)
        if self.lora_target_modules is not None:
            payload["lora_target_modules"] = list(self.lora_target_modules)
        return payload


def _coerce_config_values(raw: Mapping[str, object], *, base_dir: Path) -> dict[str, object]:
    known = {item.name for item in fields(SFTTrainConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown SFT configuration keys: {unknown}")
    result = dict(raw)
    for name in ("data_path", "output_dir", "resume_from"):
        value = result.get(name)
        if value is None:
            continue
        if not isinstance(value, str | Path):
            raise TypeError(f"{name} must be a path string")
        candidate = Path(value).expanduser()
        result[name] = candidate if candidate.is_absolute() else base_dir / candidate
    if "betas" in result:
        value = result["betas"]
        if not isinstance(value, list | tuple):
            raise TypeError("betas must be a two-item array")
        result["betas"] = tuple(value)
    if "lora_target_modules" in result and result["lora_target_modules"] is not None:
        value = result["lora_target_modules"]
        if not isinstance(value, list | tuple):
            raise TypeError("lora_target_modules must be an array")
        result["lora_target_modules"] = tuple(value)
    return result


def load_sft_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> SFTTrainConfig:
    """Load a strict ``[sft]`` TOML section and apply explicit CLI-style overrides."""

    config = SFTTrainConfig()
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"sft"} or not isinstance(document["sft"], Mapping):
            raise ValueError("an SFT config must contain exactly one [sft] table")
        values = _coerce_config_values(document["sft"], base_dir=config_path.parent)
        config = replace(config, **values)
    if overrides:
        values = _coerce_config_values(overrides, base_dir=Path.cwd())
        config = replace(config, **values)
    config.validate()
    return config


def load_sft_demonstrations(path: str | Path) -> tuple[SFTExample, ...]:
    """Read strict JSONL ``{messages, action, group}`` demonstrations.

    Duplicate object keys, non-finite numbers, blank records, unknown fields, and
    malformed VAPA actions are rejected with file-and-line diagnostics.
    """

    source = Path(path)
    try:
        content = source.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise SFTDataError(f"{source}: demonstration data must be UTF-8") from error
    return parse_sft_demonstrations(content, source=str(source))


def parse_sft_demonstrations(
    content: str, *, source: str = "<demonstrations>"
) -> tuple[SFTExample, ...]:
    """Validate demonstration bytes before a generator publishes them."""

    try:
        records = strict_jsonl_loads(content, source=source)
    except ValueError as error:
        raise SFTDataError(str(error)) from error
    examples: list[SFTExample] = []
    allowed_roles = {"system", "developer", "user", "assistant", "tool"}
    for line_number, raw in enumerate(records, start=1):
        location = f"{source}:{line_number}"
        if not isinstance(raw, Mapping):
            raise SFTDataError(f"{location}: each record must be a JSON object")
        expected = {"messages", "action", "group"}
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            unknown = sorted(set(raw) - expected)
            raise SFTDataError(f"{location}: schema mismatch; missing={missing}, unknown={unknown}")
        messages_raw = raw["messages"]
        if not isinstance(messages_raw, list) or not messages_raw:
            raise SFTDataError(f"{location}: messages must be a non-empty array")
        messages: list[dict[str, str]] = []
        for message_index, message in enumerate(messages_raw):
            message_location = f"{location}:messages[{message_index}]"
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise SFTDataError(f"{message_location}: messages require exactly role and content")
            role = message["role"]
            content_value = message["content"]
            if not isinstance(role, str) or role not in allowed_roles:
                raise SFTDataError(
                    f"{message_location}: role must be one of {sorted(allowed_roles)}"
                )
            if not isinstance(content_value, str) or not content_value:
                raise SFTDataError(f"{message_location}: content must be non-empty text")
            messages.append({"role": role, "content": content_value})
        action = raw["action"]
        group = raw["group"]
        if not isinstance(action, str) or not action.strip():
            raise SFTDataError(f"{location}: action must be non-empty text")
        if not isinstance(group, str) or not group.strip():
            raise SFTDataError(f"{location}: group must be a non-empty string")
        try:
            parsed_action = parse_action(action)
            _validate_demonstration_action(parsed_action)
        except (ActionParseError, KeyError, TypeError, ValueError) as error:
            raise SFTDataError(f"{location}: invalid VAPA action: {error}") from error
        examples.append(SFTExample(tuple(messages), action, group))
    return tuple(examples)


def _validate_demonstration_action(action: Action) -> None:
    """Reject payloads that the environment would reject without consulting state."""

    arguments = action.arguments
    if action.kind is ActionKind.RETRIEVE:
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieve query must be non-empty text")
        TimeWindow.parse(str(arguments["window"]))
        Domain(str(arguments["domain"]).strip().lower())
    elif action.kind is ActionKind.QUERY_FIELD:
        normalize_field(str(arguments["field"]))
        TimeWindow.parse(str(arguments["window"]))
    elif action.kind is ActionKind.CALCULATE:
        if not str(arguments["calculator"]).strip():
            raise ValueError("Calculate calculator must be non-empty")
        if not isinstance(arguments["bindings"], Mapping):
            raise ValueError("Calculate bindings must be an object")
    elif action.kind is ActionKind.UPDATE_MEMORY:
        if not isinstance(arguments["item"], Mapping):
            raise ValueError("UpdateMemory item must be an object")
        MemoryItem.from_dict(arguments["item"])
    elif action.kind is ActionKind.MARK_STATUS:
        if not str(arguments["item_id"]).strip():
            raise ValueError("MarkStatus item_id must be non-empty")
        MemoryStatus(str(arguments["status"]).lower())
        if not str(arguments["scope"]).strip() or not str(arguments["why"]).strip():
            raise ValueError("MarkStatus scope and why must be non-empty")
    elif action.kind is ActionKind.COMPRESS:
        raw_ids = arguments["item_ids"]
        if isinstance(raw_ids, str):
            item_ids = [item.strip() for item in raw_ids.split("|") if item.strip()]
        elif isinstance(raw_ids, list):
            item_ids = [item for item in raw_ids if isinstance(item, str) and item.strip()]
            if len(item_ids) != len(raw_ids):
                raise ValueError("Compress item IDs must be non-empty strings")
        else:
            raise ValueError("Compress item_ids must be text or an array")
        if len(set(item_ids)) < 2:
            raise ValueError("Compress requires at least two distinct item IDs")
    elif action.kind is ActionKind.DISCARD:
        if not str(arguments["item_id"]).strip():
            raise ValueError("Discard item_id must be non-empty")
    elif action.kind is ActionKind.ANSWER:
        prediction = arguments["prediction"]
        if prediction is None or (isinstance(prediction, str) and not prediction.strip()):
            raise ValueError("Answer prediction must be non-empty")
        evidence = arguments["evidence"]
        if isinstance(evidence, str):
            evidence_items = [
                item.strip() for item in evidence.strip("[]").split(",") if item.strip()
            ]
        elif isinstance(evidence, list):
            evidence_items = evidence
        else:
            raise ValueError("Answer evidence must be text or an array")
        if any(not isinstance(item, str) or not item.strip() for item in evidence_items):
            raise ValueError("Answer evidence must contain non-empty pointer strings")


@dataclass(frozen=True, slots=True)
class SFTBatch:
    epoch: int
    batch_in_epoch: int
    update_index: int
    examples: tuple[SFTExample, ...]
    group_ids: tuple[str, ...]
    action_tokens: int


def _epoch_seed(seed: int, epoch: int) -> int:
    digest = artifact_fingerprint({"seed": seed, "epoch": epoch, "purpose": "sft-order-v1"})
    return int(digest[:16], 16)


def build_sft_schedule(
    examples: Sequence[SFTExample],
    tokenizer: TokenizerAdapter,
    *,
    epochs: int,
    action_token_floor: int,
    seed: int,
    shuffle: bool = True,
    context_tokens: int | None = None,
) -> tuple[SFTBatch, ...]:
    """Create deterministic floor-sized batches without splitting declared groups."""

    if not examples:
        raise ValueError("the SFT schedule requires demonstrations")
    _positive_integer(epochs, "epochs")
    _positive_integer(action_token_floor, "action_token_floor")
    _nonnegative_integer(seed, "seed")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean")
    if context_tokens is not None:
        _positive_integer(context_tokens, "context_tokens")
    groups: OrderedDict[str, list[tuple[SFTExample, int]]] = OrderedDict()
    for example in examples:
        encoded = tokenizer.encode_action(example.messages, example.action_text)
        if context_tokens is not None and (
            len(encoded.prompt_ids) + len(encoded.action_ids) > context_tokens
        ):
            raise SFTDataError(f"SFT example group {example.group_id!r} exceeds context_tokens")
        groups.setdefault(example.group_id, []).append((example, encoded.token_count))
    group_rows = tuple((key, tuple(rows)) for key, rows in groups.items())
    batches: list[SFTBatch] = []
    update_index = 0
    for epoch in range(epochs):
        ordered = list(group_rows)
        if shuffle:
            random.Random(_epoch_seed(seed, epoch)).shuffle(ordered)
        pending_examples: list[SFTExample] = []
        pending_groups: list[str] = []
        pending_tokens = 0
        batch_in_epoch = 0
        for group_id, rows in ordered:
            pending_groups.append(group_id)
            pending_examples.extend(row[0] for row in rows)
            pending_tokens += sum(row[1] for row in rows)
            if pending_tokens >= action_token_floor:
                batches.append(
                    SFTBatch(
                        epoch,
                        batch_in_epoch,
                        update_index,
                        tuple(pending_examples),
                        tuple(pending_groups),
                        pending_tokens,
                    )
                )
                update_index += 1
                batch_in_epoch += 1
                pending_examples = []
                pending_groups = []
                pending_tokens = 0
        if pending_examples:
            batches.append(
                SFTBatch(
                    epoch,
                    batch_in_epoch,
                    update_index,
                    tuple(pending_examples),
                    tuple(pending_groups),
                    pending_tokens,
                )
            )
            update_index += 1
    return tuple(batches)


class _ByteTokenizer:
    """Dependency-free estimator used only by ``--dry-run``."""

    fingerprint = fingerprint_payload({"adapter": "sft-byte-estimator-v1"})

    @staticmethod
    def _encode(text: str) -> tuple[int, ...]:
        return tuple(byte + 1 for byte in text.encode("utf-8")) or (1,)

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[int, ...]:
        text = canonical_json_dumps([dict(message) for message in messages])
        if add_generation_prompt:
            text += "\nassistant:"
        return self._encode(text)

    def encode_text(self, text: str) -> tuple[int, ...]:
        return self._encode(text)

    def encode_action(
        self,
        messages: Sequence[Mapping[str, str]],
        action_text: str,
    ) -> TokenizedAction:
        return TokenizedAction(self.encode_messages(messages), self.encode_text(action_text))

    def save_pretrained(self, path: str) -> None:
        raise RuntimeError("the validation-only byte tokenizer cannot be saved")


def _schedule_fingerprint(schedule: Sequence[SFTBatch]) -> str:
    return artifact_fingerprint(
        [
            {
                "epoch": batch.epoch,
                "batch_in_epoch": batch.batch_in_epoch,
                "update_index": batch.update_index,
                "group_ids": list(batch.group_ids),
                "action_tokens": batch.action_tokens,
            }
            for batch in schedule
        ]
    )


@dataclass(frozen=True, slots=True)
class SFTDryRunReport:
    examples: int
    groups: int
    epochs: int
    planned_updates: int
    estimated_action_tokens: int
    largest_group_action_tokens: int
    schedule_fingerprint: str
    training_ready: bool
    readiness_issues: tuple[str, ...]
    scope: str = "data-and-schedule-only"
    token_count_basis: str = "utf8-byte-estimate"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_sft_run(config: SFTTrainConfig) -> SFTDryRunReport:
    """Validate data/scheduling and report model-config readiness without ML imports."""

    config.validate()
    examples = load_sft_demonstrations(config.data_path)
    tokenizer = _ByteTokenizer()
    schedule = build_sft_schedule(
        examples,
        tokenizer,
        epochs=config.epochs,
        action_token_floor=config.action_token_floor,
        seed=config.seed,
        shuffle=config.shuffle,
        context_tokens=config.context_tokens,
    )
    group_tokens: OrderedDict[str, int] = OrderedDict()
    for example in examples:
        group_tokens.setdefault(example.group_id, 0)
        group_tokens[example.group_id] += tokenizer.encode_action(
            example.messages, example.action_text
        ).token_count
    readiness_issues = _training_readiness_issues(config)
    return SFTDryRunReport(
        examples=len(examples),
        groups=len(group_tokens),
        epochs=config.epochs,
        planned_updates=len(schedule),
        estimated_action_tokens=sum(batch.action_tokens for batch in schedule),
        largest_group_action_tokens=max(group_tokens.values()),
        schedule_fingerprint=_schedule_fingerprint(schedule),
        training_ready=not readiness_issues,
        readiness_issues=readiness_issues,
    )


TokenizerFactory = Callable[[SFTTrainConfig], TokenizerAdapter]
ActorFactory = Callable[[SFTTrainConfig, str], ActorModelAdapter]
OptimizerFactory = Callable[[ActorModelAdapter, SFTTrainConfig], OptimizerAdapter]


def _default_tokenizer_factory(config: SFTTrainConfig) -> TokenizerAdapter:
    from vapa.model.transformers import TransformersTokenizerAdapter

    return TransformersTokenizerAdapter.from_pretrained(
        config.effective_tokenizer_name,
        revision=config.effective_tokenizer_revision,
        use_processor=config.use_processor,
        chat_template_kwargs=(
            {"reasoning_effort": config.reasoning_effort}
            if "gpt-oss" in config.model_name.casefold()
            else {"enable_thinking": config.enable_thinking}
        ),
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )


def _default_actor_factory(config: SFTTrainConfig, device: str) -> ActorModelAdapter:
    from vapa.model.transformers import TransformersActorAdapter, apply_lora

    actor = TransformersActorAdapter.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        device=device,
        dtype=config.dtype,
        model_kind=config.model_kind,
        dequantize_mxfp4=config.dequantize_mxfp4,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
        use_safetensors=True,
    )
    if config.use_lora:
        apply_lora(
            actor,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
    return actor


def _default_optimizer_factory(
    actor: ActorModelAdapter,
    config: SFTTrainConfig,
) -> OptimizerAdapter:
    return build_adamw(
        actor,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        epsilon=config.epsilon,
    )


def _validate_reconstructible_model(config: SFTTrainConfig) -> None:
    """Require immutable, explicit loader metadata before writing a checkpoint."""

    if config.trust_remote_code:
        raise ValueError(
            "trust_remote_code must be false so checkpoint inference uses audited model code"
        )
    _inference_spec(config)


def _training_readiness_issues(config: SFTTrainConfig) -> tuple[str, ...]:
    try:
        _validate_reconstructible_model(config)
    except (TypeError, ValueError) as error:
        return (str(error),)
    return ()


def _qualified_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    raw_records = strict_jsonl_loads(
        path.read_bytes().decode("utf-8"), source=str(path), allow_empty=True
    )
    for line_number, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number}: metrics record must be an object")
        records.append(raw)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    content = "".join(canonical_json_dumps(record) + "\n" for record in records)
    atomic_write_text(path, content)


def _inference_spec(config: SFTTrainConfig) -> dict[str, object]:
    from vapa.model.inference import TransformersInferenceSpec

    return TransformersInferenceSpec(
        model_name=config.model_name,
        model_revision=config.model_revision,
        tokenizer_name=config.effective_tokenizer_name,
        tokenizer_revision=config.effective_tokenizer_revision,
        model_kind=config.model_kind,
        use_processor=config.use_processor,  # type: ignore[arg-type]
        dtype=config.dtype,
        device=config.device,
        lora_enabled=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        enable_thinking=config.enable_thinking,
        reasoning_effort=config.reasoning_effort,
        dequantize_mxfp4=config.dequantize_mxfp4,
        context_tokens=config.context_tokens,
        scaffold=config.scaffold,
        temperature=config.generation_temperature,
        top_p=config.generation_top_p,
        top_k=config.generation_top_k,
        max_tokens=config.generation_max_tokens,
    ).to_dict()


def _manifest_payload(
    config: SFTTrainConfig,
    *,
    data_sha256: str,
    data_size: int,
    config_fingerprint: str,
    model_fingerprint: str,
    tokenizer_fingerprint: str,
    schedule_fingerprint: str,
    total_updates: int,
    factory_identities: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": SFT_RUN_SCHEMA_VERSION,
        "run_id": config.run_id,
        "content_kind": config.content_kind,
        "implementation_sha256": package_code_fingerprint(),
        "data": {
            "schema_version": SFT_DATA_SCHEMA_VERSION,
            "sha256": data_sha256,
            "size_bytes": data_size,
        },
        "config_fingerprint": config_fingerprint,
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "schedule_fingerprint": schedule_fingerprint,
        "total_updates": total_updates,
        "factories": dict(factory_identities),
        "code_version": config.code_version,
    }


def _prepare_output(
    config: SFTTrainConfig,
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    encoded_manifest = canonical_json_dumps(manifest) + "\n"
    if config.resume_from is None:
        if manifest_path.exists() or metrics_path.exists() or (output / "checkpoints").exists():
            raise FileExistsError(
                f"output already contains an SFT run: {output}; use --resume-from"
            )
        atomic_write_text(manifest_path, encoded_manifest)
        return []
    checkpoint = config.resume_from.resolve()
    if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent != output:
        raise ValueError("resume checkpoint must belong to output_dir/checkpoints")
    if not manifest_path.is_file():
        raise FileNotFoundError("resume output is missing run_manifest.json")
    stored = strict_json_loads(manifest_path.read_bytes())
    if stored != manifest:
        raise ValueError("stored SFT run manifest does not match the requested run")
    return _read_jsonl(metrics_path)


def _checkpoint_step(path: Path) -> int | None:
    prefix = "step-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _reconcile_resume_metrics(
    records: list[dict[str, object]],
    *,
    completed_steps: int,
    metrics_path: Path,
    checkpoints_path: Path,
) -> list[dict[str, object]]:
    if len(records) < completed_steps:
        raise ValueError("metrics log ends before the resume checkpoint")
    for index, record in enumerate(records, start=1):
        if record.get("step") != index:
            raise ValueError("metrics log step sequence is corrupt")
    future = (
        sorted(
            step
            for path in checkpoints_path.iterdir()
            if (step := _checkpoint_step(path)) is not None and step > completed_steps
        )
        if checkpoints_path.exists()
        else []
    )
    if future:
        raise ValueError(
            "resume checkpoint is not the latest checkpoint in output_dir; "
            f"later steps exist: {future}"
        )
    records = records[:completed_steps]
    _write_jsonl(metrics_path, records)
    return records


@dataclass(frozen=True, slots=True)
class SFTTrainResult:
    run_id: str
    status: str
    completed_updates: int
    total_updates: int
    action_tokens: int
    output_dir: str
    last_checkpoint: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def train_sft(
    config: SFTTrainConfig,
    *,
    tokenizer_factory: TokenizerFactory | None = None,
    actor_factory: ActorFactory | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    state_store: StateStore | None = None,
    stop_after_updates: int | None = None,
) -> SFTTrainResult:
    """Execute or resume deterministic, group-preserving action-token SFT."""

    config.validate()
    destination = guard_artifact_write_path(
        config.output_dir,
        content_kind=config.content_kind,
    )
    resume_from = None
    if config.resume_from is not None:
        if config.resume_from.is_symlink():
            raise ValueError("resume checkpoint cannot be a symbolic link")
        resume_from = config.resume_from.expanduser().resolve()
    config = replace(config, output_dir=destination, resume_from=resume_from)
    _validate_reconstructible_model(config)
    if stop_after_updates is not None:
        _positive_integer(stop_after_updates, "stop_after_updates")
    distributed = DistributedContext.from_environment()
    if distributed.world_size != 1:
        raise RuntimeError("train_sft currently requires a single process")
    data_fingerprint = fingerprint_file(config.data_path)
    examples = load_sft_demonstrations(config.data_path)
    if fingerprint_file(config.data_path) != data_fingerprint:
        raise RuntimeError("SFT demonstration data changed while it was being loaded")
    seed_everything(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    from vapa.inference import factory_identity

    selected_tokenizer_factory = tokenizer_factory or _default_tokenizer_factory
    selected_actor_factory = actor_factory or _default_actor_factory
    selected_optimizer_factory = optimizer_factory or _default_optimizer_factory
    factory_identities = {
        "tokenizer": factory_identity(selected_tokenizer_factory),
        "actor": factory_identity(selected_actor_factory),
        "optimizer": factory_identity(selected_optimizer_factory),
    }
    tokenizer = selected_tokenizer_factory(config)
    actor = selected_actor_factory(config, device)
    schedule = build_sft_schedule(
        examples,
        tokenizer,
        epochs=config.epochs,
        action_token_floor=config.action_token_floor,
        seed=config.seed,
        shuffle=config.shuffle,
        context_tokens=config.context_tokens,
    )
    if not schedule:
        raise RuntimeError("SFT planning produced no updates")
    optimizer = selected_optimizer_factory(actor, config)
    warmup_steps = int(len(schedule) * config.warmup_fraction)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=len(schedule),
        warmup_steps=warmup_steps,
        final_lr_fraction=config.final_lr_fraction,
    )
    selected_store = state_store or TorchStateStore()
    if fingerprint_file(config.data_path) != data_fingerprint:
        raise RuntimeError("SFT demonstration data changed during run initialization")
    config_fingerprint = fingerprint_payload(config.fingerprint_payload())
    schedule_fingerprint = _schedule_fingerprint(schedule)
    manifest = _manifest_payload(
        config,
        data_sha256=data_fingerprint.sha256,
        data_size=data_fingerprint.size_bytes,
        config_fingerprint=config_fingerprint,
        model_fingerprint=actor.fingerprint,
        tokenizer_fingerprint=tokenizer.fingerprint,
        schedule_fingerprint=schedule_fingerprint,
        total_updates=len(schedule),
        factory_identities=factory_identities,
    )
    run_manifest_fingerprint = artifact_fingerprint(manifest)
    contract = CheckpointContract(
        run_id=config.run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        config_fingerprint=config_fingerprint,
        model_fingerprint=actor.fingerprint,
        reference_model_fingerprint=actor.fingerprint,
        tokenizer_fingerprint=tokenizer.fingerprint,
        optimizer_name=_qualified_name(optimizer),
        scheduler_name="warmup-cosine-v1",
        state_format=selected_store.format_name,
    )
    metric_records = _prepare_output(config, manifest)
    output = config.output_dir.resolve()
    metrics_path = output / "metrics.jsonl"
    checkpoints_path = output / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    completed_steps = 0
    cumulative_tokens = 0
    last_checkpoint: Path | None = None
    if config.resume_from is not None:
        runtime = read_manifest(config.resume_from).runtime
        extra = runtime.extra or {}
        required_extra = {
            "objective",
            "dataset_fingerprint",
            "schedule_fingerprint",
            "next_update_index",
            "total_updates",
            "inference_spec",
        }
        if set(extra) != required_extra:
            raise ValueError("SFT checkpoint runtime metadata has an incompatible schema")
        if (
            extra["objective"] != "sft"
            or extra["dataset_fingerprint"] != data_fingerprint.sha256
            or extra["schedule_fingerprint"] != schedule_fingerprint
            or extra["total_updates"] != len(schedule)
            or extra["inference_spec"] != _inference_spec(config)
        ):
            raise ValueError("SFT checkpoint runtime metadata does not match this run")
        completed_steps = runtime.global_step
        cumulative_tokens = runtime.trainable_tokens
        if runtime.sampled_tokens != cumulative_tokens:
            raise ValueError("SFT checkpoint token counters disagree")
        if extra["next_update_index"] != completed_steps or completed_steps > len(schedule):
            raise ValueError("SFT checkpoint update cursor is invalid")
        expected_tokens = sum(batch.action_tokens for batch in schedule[:completed_steps])
        if cumulative_tokens != expected_tokens:
            raise ValueError("SFT checkpoint token cursor does not match the schedule")
        loaded_runtime = resume_checkpoint(
            config.resume_from,
            expected=contract,
            model=actor,
            reference=actor,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            store=selected_store,
        )
        if loaded_runtime != runtime:
            raise ValueError("SFT checkpoint runtime changed during resume validation")
        metric_records = _reconcile_resume_metrics(
            metric_records,
            completed_steps=completed_steps,
            metrics_path=metrics_path,
            checkpoints_path=checkpoints_path,
        )
        last_checkpoint = config.resume_from.resolve()

    actor.train()
    updates_this_invocation = 0
    last_saved_step = completed_steps if config.resume_from is not None else -1
    for batch in schedule[completed_steps:]:
        if stop_after_updates is not None and updates_this_invocation >= stop_after_updates:
            break
        seed_everything(
            _epoch_seed(config.seed, batch.update_index),
            deterministic=config.deterministic,
        )
        report = train_sft_step(
            batch.examples,
            tokenizer=tokenizer,
            actor=actor,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip=config.gradient_clip,
        )
        if report.action_tokens != batch.action_tokens:
            raise RuntimeError("SFT tokenizer token counts changed between planning and training")
        completed_steps += 1
        updates_this_invocation += 1
        cumulative_tokens += report.action_tokens
        metric_records.append(
            {
                "schema_version": SFT_METRIC_SCHEMA_VERSION,
                "run_id": config.run_id,
                "objective": report.objective,
                "step": completed_steps,
                "epoch": batch.epoch,
                "batch_in_epoch": batch.batch_in_epoch,
                "loss": report.loss,
                "policy_loss": report.policy_loss,
                "kl": report.kl,
                "action_tokens": report.action_tokens,
                "cumulative_action_tokens": cumulative_tokens,
                "microbatches": report.microbatches,
                "gradient_norm": report.gradient_norm,
                "learning_rates": list(report.learning_rates),
            }
        )
        _write_jsonl(metrics_path, metric_records)
        should_save = completed_steps % config.checkpoint_every == 0
        final_update = completed_steps == len(schedule)
        stopping = stop_after_updates is not None and updates_this_invocation == stop_after_updates
        if should_save or final_update or stopping:
            runtime = RuntimeState(
                global_step=completed_steps,
                sampled_tokens=cumulative_tokens,
                trainable_tokens=cumulative_tokens,
                seed=config.seed,
                extra={
                    "objective": "sft",
                    "dataset_fingerprint": data_fingerprint.sha256,
                    "schedule_fingerprint": schedule_fingerprint,
                    "next_update_index": completed_steps,
                    "total_updates": len(schedule),
                    "inference_spec": _inference_spec(config),
                },
            )
            last_checkpoint = checkpoints_path / f"step-{completed_steps:08d}"
            save_checkpoint(
                last_checkpoint,
                contract=contract,
                runtime=runtime,
                model=actor,
                reference=actor,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                store=selected_store,
            )
            last_saved_step = completed_steps

    if completed_steps and last_saved_step != completed_steps:
        runtime = RuntimeState(
            global_step=completed_steps,
            sampled_tokens=cumulative_tokens,
            trainable_tokens=cumulative_tokens,
            seed=config.seed,
            extra={
                "objective": "sft",
                "dataset_fingerprint": data_fingerprint.sha256,
                "schedule_fingerprint": schedule_fingerprint,
                "next_update_index": completed_steps,
                "total_updates": len(schedule),
                "inference_spec": _inference_spec(config),
            },
        )
        last_checkpoint = checkpoints_path / f"step-{completed_steps:08d}"
        save_checkpoint(
            last_checkpoint,
            contract=contract,
            runtime=runtime,
            model=actor,
            reference=actor,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            store=selected_store,
        )
    return SFTTrainResult(
        run_id=config.run_id,
        status="complete" if completed_steps == len(schedule) else "stopped",
        completed_updates=completed_steps,
        total_updates=len(schedule),
        action_tokens=cumulative_tokens,
        output_dir=str(output),
        last_checkpoint=None if last_checkpoint is None else str(last_checkpoint),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML file containing exactly [sft]")
    parser.add_argument("--data", dest="data_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--content-kind",
        choices=tuple(item.value for item in ArtifactContentKind),
        help="artifact disclosure class (credentialed by default)",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--model", dest="model_name")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer", dest="tokenizer_name")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--model-kind", choices=("auto", "causal", "multimodal"))
    parser.add_argument("--dtype")
    parser.add_argument("--device")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--scaffold")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--action-token-floor", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--betas", nargs=2, type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--warmup-fraction", type=float)
    parser.add_argument("--final-lr-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-module", dest="lora_target_modules", action="append")
    parser.add_argument("--generation-temperature", type=float)
    parser.add_argument("--generation-top-p", type=float)
    parser.add_argument("--generation-top-k", type=int)
    parser.add_argument("--generation-max-tokens", type=int)
    parser.add_argument("--code-version")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-processor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument(
        "--dequantize-mxfp4",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    return parser


def config_from_namespace(arguments: argparse.Namespace) -> SFTTrainConfig:
    raw = vars(arguments).copy()
    config_path = raw.pop("config")
    raw.pop("dry_run")
    raw.pop("stop_after_updates")
    overrides = {key: value for key, value in raw.items() if value is not None}
    return load_sft_config(config_path, overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = config_from_namespace(arguments)
    result: SFTDryRunReport | SFTTrainResult
    if arguments.dry_run:
        result = validate_sft_run(config)
    else:
        result = train_sft(config, stop_after_updates=arguments.stop_after_updates)
    print(canonical_json_dumps(result.to_dict()))
    return 0


__all__ = [
    "SFTBatch",
    "SFTDataError",
    "SFTDryRunReport",
    "SFTTrainConfig",
    "SFTTrainResult",
    "build_parser",
    "build_sft_schedule",
    "config_from_namespace",
    "load_sft_config",
    "load_sft_demonstrations",
    "main",
    "train_sft",
    "validate_sft_run",
]
