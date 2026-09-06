"""Verified Transformers checkpoint reconstruction for the generic inference runtime.

The generic :mod:`vapa.inference` layer intentionally knows nothing about model
frameworks.  This module supplies the two factories it expects while keeping model
downloads, PEFT imports, and tensor deserialization behind explicit dependency
boundaries.

A training checkpoint used by the default factories must store
``runtime.extra["inference_spec"]``.  The runtime file is covered by the checkpoint
manifest, so the exact model, tokenizer, LoRA topology, and decoding parameters travel
with the actor weights.  Revisions are immutable commit identifiers; mutable Hub
branches such as ``main`` are deliberately rejected.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vapa.artifacts import (
    artifact_fingerprint,
    canonical_json_dumps,
    fingerprint_file,
    strict_json_loads,
)
from vapa.environment.calculators import CalculatorRegistry
from vapa.environment.state_manager import StateManager
from vapa.model.protocols import ActorModelAdapter, TokenizerAdapter
from vapa.model.transformers import (
    TransformersActorAdapter,
    TransformersGenerationBackend,
    TransformersTokenizerAdapter,
    apply_lora,
)
from vapa.policies.text import GenerationBackend, TextPolicy
from vapa.provenance import package_code_fingerprint
from vapa.rollouts import ManagerFactory
from vapa.schemas import Episode
from vapa.training.checkpoint import (
    CheckpointContract,
    CheckpointManifest,
    JsonStateStore,
    RuntimeState,
    StateStore,
    TorchStateStore,
    read_manifest,
)
from vapa.training.runtime import resolve_device

INFERENCE_SPEC_FORMAT_VERSION = 1
ENVIRONMENT_SPEC_FORMAT_VERSION = 1
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_MODEL_KINDS = frozenset({"auto", "multimodal", "causal"})


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _revision(value: object, name: str) -> str:
    result = _nonempty_text(value, name)
    if _IMMUTABLE_REVISION.fullmatch(result) is None:
        raise ValueError(f"{name} must be an immutable 40- or 64-character lowercase commit digest")
    return result


@dataclass(frozen=True, slots=True)
class TransformersInferenceSpec:
    """Complete, immutable information needed to reconstruct one trained actor."""

    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    model_kind: str
    use_processor: bool
    dtype: str
    device: str | None
    lora_enabled: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...] | None
    enable_thinking: bool = False
    reasoning_effort: str | None = None
    dequantize_mxfp4: bool = False
    context_tokens: int = 32_768
    scaffold: str = ""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 512
    format_version: int = INFERENCE_SPEC_FORMAT_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != INFERENCE_SPEC_FORMAT_VERSION
        ):
            raise ValueError(f"unsupported inference spec format_version: {self.format_version!r}")
        _nonempty_text(self.model_name, "model_name")
        _nonempty_text(self.tokenizer_name, "tokenizer_name")
        _revision(self.model_revision, "model_revision")
        _revision(self.tokenizer_revision, "tokenizer_revision")
        if self.model_kind not in _MODEL_KINDS:
            raise ValueError("model_kind must be 'auto', 'multimodal', or 'causal'")
        if not isinstance(self.use_processor, bool):
            raise TypeError("use_processor must be a boolean")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of: {', '.join(sorted(_DTYPES))}")
        if self.device is not None:
            _nonempty_text(self.device, "device")
        if not isinstance(self.lora_enabled, bool):
            raise TypeError("lora_enabled must be a boolean")
        _positive_integer(self.lora_rank, "lora_rank")
        _positive_integer(self.lora_alpha, "lora_alpha")
        dropout = _finite_number(self.lora_dropout, "lora_dropout")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.lora_target_modules is not None:
            modules = tuple(self.lora_target_modules)
            if not modules or any(
                not isinstance(module, str) or not module or module != module.strip()
                for module in modules
            ):
                raise ValueError("lora_target_modules must contain non-empty, trimmed names")
            if len(set(modules)) != len(modules):
                raise ValueError("lora_target_modules cannot contain duplicates")
            object.__setattr__(self, "lora_target_modules", modules)
        if self.lora_enabled and self.lora_target_modules is None:
            raise ValueError("LoRA checkpoints require explicit lora_target_modules")
        if not self.lora_enabled and self.lora_target_modules is not None:
            raise ValueError("lora_target_modules must be null when LoRA is disabled")
        if not isinstance(self.enable_thinking, bool):
            raise TypeError("enable_thinking must be a boolean")
        if self.reasoning_effort is not None and self.reasoning_effort not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("reasoning_effort must be 'low', 'medium', 'high', or null")
        if not isinstance(self.dequantize_mxfp4, bool):
            raise TypeError("dequantize_mxfp4 must be a boolean")
        _positive_integer(self.context_tokens, "context_tokens")
        if not isinstance(self.scaffold, str):
            raise TypeError("scaffold must be a string")
        temperature = _finite_number(self.temperature, "temperature")
        top_p = _finite_number(self.top_p, "top_p")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        _nonnegative_integer(self.top_k, "top_k")
        _positive_integer(self.max_tokens, "max_tokens")
        if self.max_tokens > self.context_tokens:
            raise ValueError("max_tokens cannot exceed context_tokens")

        qwen35 = "qwen3.5" in self.model_name.casefold()
        gpt_oss = "gpt-oss" in self.model_name.casefold()
        if qwen35 and not self.use_processor:
            raise ValueError("Qwen3.5 checkpoints require use_processor=true")
        if qwen35 and self.model_kind == "causal":
            raise ValueError("Qwen3.5 checkpoints require the multimodal model loader")
        if not qwen35 and self.model_kind == "multimodal" and not self.use_processor:
            raise ValueError("multimodal checkpoints require use_processor=true")
        if gpt_oss:
            if self.reasoning_effort is None:
                raise ValueError("gpt-oss checkpoints require an explicit reasoning_effort")
            if self.enable_thinking:
                raise ValueError("gpt-oss uses reasoning_effort instead of enable_thinking")
        elif self.reasoning_effort is not None:
            raise ValueError("reasoning_effort is supported only for gpt-oss checkpoints")
        if self.dequantize_mxfp4 and not gpt_oss:
            raise ValueError("dequantize_mxfp4 is supported only for gpt-oss checkpoints")

    @property
    def chat_template_kwargs(self) -> dict[str, object]:
        if "gpt-oss" in self.model_name.casefold():
            assert self.reasoning_effort is not None
            return {"reasoning_effort": self.reasoning_effort}
        return {"enable_thinking": self.enable_thinking}

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON shape embedded in ``RuntimeState.extra``."""

        return {
            "format_version": self.format_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "model_kind": self.model_kind,
            "use_processor": self.use_processor,
            "dtype": self.dtype,
            "device": self.device,
            "lora_enabled": self.lora_enabled,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": (
                None if self.lora_target_modules is None else list(self.lora_target_modules)
            ),
            "enable_thinking": self.enable_thinking,
            "reasoning_effort": self.reasoning_effort,
            "dequantize_mxfp4": self.dequantize_mxfp4,
            "context_tokens": self.context_tokens,
            "scaffold": self.scaffold,
            "generation": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_tokens": self.max_tokens,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TransformersInferenceSpec:
        if not isinstance(value, Mapping):
            raise TypeError("inference_spec must be an object")
        expected = {
            "format_version",
            "model_name",
            "model_revision",
            "tokenizer_name",
            "tokenizer_revision",
            "model_kind",
            "use_processor",
            "dtype",
            "device",
            "lora_enabled",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "enable_thinking",
            "reasoning_effort",
            "dequantize_mxfp4",
            "context_tokens",
            "scaffold",
            "generation",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            unexpected = sorted(set(value) - expected)
            details = []
            if missing:
                details.append("missing fields: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected fields: " + ", ".join(unexpected))
            raise ValueError("invalid inference_spec: " + "; ".join(details))
        generation = value["generation"]
        if not isinstance(generation, Mapping):
            raise TypeError("inference_spec.generation must be an object")
        generation_fields = {"temperature", "top_p", "top_k", "max_tokens"}
        if set(generation) != generation_fields:
            raise ValueError("inference_spec.generation has an incompatible schema")
        raw_modules = value["lora_target_modules"]
        if raw_modules is not None and (
            not isinstance(raw_modules, Sequence) or isinstance(raw_modules, str | bytes)
        ):
            raise TypeError("lora_target_modules must be a list or null")
        return cls(
            format_version=value["format_version"],  # type: ignore[arg-type]
            model_name=value["model_name"],  # type: ignore[arg-type]
            model_revision=value["model_revision"],  # type: ignore[arg-type]
            tokenizer_name=value["tokenizer_name"],  # type: ignore[arg-type]
            tokenizer_revision=value["tokenizer_revision"],  # type: ignore[arg-type]
            model_kind=value["model_kind"],  # type: ignore[arg-type]
            use_processor=value["use_processor"],  # type: ignore[arg-type]
            dtype=value["dtype"],  # type: ignore[arg-type]
            device=value["device"],  # type: ignore[arg-type]
            lora_enabled=value["lora_enabled"],  # type: ignore[arg-type]
            lora_rank=value["lora_rank"],  # type: ignore[arg-type]
            lora_alpha=value["lora_alpha"],  # type: ignore[arg-type]
            lora_dropout=value["lora_dropout"],  # type: ignore[arg-type]
            lora_target_modules=(
                None if raw_modules is None else tuple(raw_modules)  # type: ignore[arg-type]
            ),
            enable_thinking=value["enable_thinking"],  # type: ignore[arg-type]
            reasoning_effort=value["reasoning_effort"],  # type: ignore[arg-type]
            dequantize_mxfp4=value["dequantize_mxfp4"],  # type: ignore[arg-type]
            context_tokens=value["context_tokens"],  # type: ignore[arg-type]
            scaffold=value["scaffold"],  # type: ignore[arg-type]
            temperature=generation["temperature"],  # type: ignore[arg-type]
            top_p=generation["top_p"],  # type: ignore[arg-type]
            top_k=generation["top_k"],  # type: ignore[arg-type]
            max_tokens=generation["max_tokens"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CheckpointEnvironmentSpec:
    """Complete state-machine and calculator configuration used during RL."""

    memory_capacity: int
    action_budget: int
    turn_cap: int
    retrieval_limit: int
    implementation_sha256: str
    calculator_manifest: Mapping[str, Any]
    calculator_manifest_sha256: str
    schema_version: int = ENVIRONMENT_SPEC_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENVIRONMENT_SPEC_FORMAT_VERSION:
            raise ValueError(
                f"unsupported environment_spec schema_version: {self.schema_version!r}"
            )
        for name in (
            "memory_capacity",
            "action_budget",
            "turn_cap",
            "retrieval_limit",
        ):
            _positive_integer(getattr(self, name), name)
        if self.turn_cap < self.action_budget + 1:
            raise ValueError("environment_spec turn_cap must admit all actions plus Answer")
        if not isinstance(self.calculator_manifest, Mapping):
            raise TypeError("environment_spec.calculator_manifest must be an object")
        normalized = strict_json_loads(canonical_json_dumps(self.calculator_manifest))
        if not isinstance(normalized, dict):  # pragma: no cover - guarded above
            raise TypeError("environment_spec.calculator_manifest must be an object")
        object.__setattr__(self, "calculator_manifest", normalized)
        if (
            not isinstance(self.calculator_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.calculator_manifest_sha256) is None
        ):
            raise ValueError(
                "environment_spec.calculator_manifest_sha256 must be a lowercase SHA-256"
            )
        if artifact_fingerprint(normalized) != self.calculator_manifest_sha256:
            raise ValueError("environment_spec calculator manifest fingerprint mismatch")
        if (
            not isinstance(self.implementation_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.implementation_sha256) is None
        ):
            raise ValueError("environment_spec implementation_sha256 must be a SHA-256")
        if package_code_fingerprint() != self.implementation_sha256:
            raise ValueError(
                "checkpoint environment implementation does not match installed VAPA code"
            )
        CalculatorRegistry.from_manifest(normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "memory_capacity": self.memory_capacity,
            "action_budget": self.action_budget,
            "turn_cap": self.turn_cap,
            "retrieval_limit": self.retrieval_limit,
            "implementation_sha256": self.implementation_sha256,
            "calculator_manifest": dict(self.calculator_manifest),
            "calculator_manifest_sha256": self.calculator_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CheckpointEnvironmentSpec:
        if not isinstance(value, Mapping):
            raise TypeError("environment_spec must be an object")
        expected = {
            "schema_version",
            "memory_capacity",
            "action_budget",
            "turn_cap",
            "retrieval_limit",
            "implementation_sha256",
            "calculator_manifest",
            "calculator_manifest_sha256",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            unexpected = sorted(set(value) - expected)
            details = []
            if missing:
                details.append("missing fields: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected fields: " + ", ".join(unexpected))
            raise ValueError("invalid environment_spec: " + "; ".join(details))
        manifest = value["calculator_manifest"]
        if not isinstance(manifest, Mapping):
            raise TypeError("environment_spec.calculator_manifest must be an object")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            memory_capacity=value["memory_capacity"],  # type: ignore[arg-type]
            action_budget=value["action_budget"],  # type: ignore[arg-type]
            turn_cap=value["turn_cap"],  # type: ignore[arg-type]
            retrieval_limit=value["retrieval_limit"],  # type: ignore[arg-type]
            implementation_sha256=value["implementation_sha256"],  # type: ignore[arg-type]
            calculator_manifest=manifest,
            calculator_manifest_sha256=value["calculator_manifest_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class LoadedTransformerCheckpoint:
    """Integrity-checked inference inputs, with no framework objects loaded yet."""

    path: Path
    manifest: CheckpointManifest
    spec: TransformersInferenceSpec
    actor_state: Mapping[str, Any]


class ActorLoader(Protocol):
    def __call__(self, name: str, **kwargs: Any) -> ActorModelAdapter: ...


class TokenizerLoader(Protocol):
    def __call__(self, name: str, **kwargs: Any) -> TokenizerAdapter: ...


class LoraLoader(Protocol):
    def __call__(
        self,
        actor: ActorModelAdapter,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: Sequence[str] | None,
    ) -> ActorModelAdapter: ...


class BackendFactory(Protocol):
    def __call__(
        self,
        actor: ActorModelAdapter,
        tokenizer: TokenizerAdapter,
        *,
        context_tokens: int,
    ) -> GenerationBackend: ...


StoreFactory = Callable[[str], StateStore]


@dataclass(frozen=True, slots=True)
class InferenceDependencies:
    """Injectable framework boundary used by tests and alternate runtimes."""

    actor_loader: ActorLoader = TransformersActorAdapter.from_pretrained
    tokenizer_loader: TokenizerLoader = TransformersTokenizerAdapter.from_pretrained
    lora_loader: LoraLoader = apply_lora
    backend_factory: BackendFactory = TransformersGenerationBackend
    store_factory: StoreFactory | None = None
    device_resolver: Callable[[str], str] = resolve_device


def _default_store_factory(format_name: str) -> StateStore:
    if format_name == JsonStateStore.format_name:
        return JsonStateStore()
    if format_name == TorchStateStore.format_name:
        return TorchStateStore()
    raise ValueError(f"unsupported checkpoint state_format: {format_name!r}")


def _verify_checkpoint_files(path: Path, manifest: CheckpointManifest) -> None:
    """Verify every listed file before deserializing any actor state."""

    for logical_name, record in manifest.files.items():
        candidate = path / record.path
        if candidate.is_symlink():
            raise ValueError(f"checkpoint {logical_name} file cannot be a symbolic link")
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint {logical_name} file is missing")
        fingerprint = fingerprint_file(candidate)
        if fingerprint.size_bytes != record.size or fingerprint.sha256 != record.sha256:
            raise ValueError(f"checkpoint {logical_name} file failed integrity verification")


def _verified_runtime(path: Path, manifest: CheckpointManifest) -> RuntimeState:
    runtime_record = manifest.files.get("runtime")
    if runtime_record is None:
        raise ValueError("checkpoint is missing runtime state")
    raw = strict_json_loads((path / runtime_record.path).read_bytes())
    if not isinstance(raw, Mapping):
        raise ValueError("runtime state must be an object")
    runtime = RuntimeState.from_dict(raw)
    if runtime != manifest.runtime:
        raise ValueError("manifest and runtime state disagree")
    return runtime


def _embedded_spec(runtime: RuntimeState) -> TransformersInferenceSpec:
    extra = runtime.extra
    raw = None if extra is None else extra.get("inference_spec")
    if not isinstance(raw, Mapping):
        raise ValueError('checkpoint runtime.extra must contain an "inference_spec" object')
    return TransformersInferenceSpec.from_dict(raw)


def load_transformers_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_contract: CheckpointContract | None = None,
    expected_spec: TransformersInferenceSpec | None = None,
    store_factory: StoreFactory | None = None,
) -> LoadedTransformerCheckpoint:
    """Validate one training checkpoint and deserialize only its actor state.

    Contract/spec mismatches and all checksum failures occur before the model state is
    deserialized or any external model loader is called.
    """

    unresolved = Path(checkpoint_path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("checkpoint path cannot be a symbolic link")
    path = unresolved.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    manifest = read_manifest(path)
    if expected_contract is not None:
        if not isinstance(expected_contract, CheckpointContract):
            raise TypeError("expected_contract must be a CheckpointContract or None")
        if manifest.contract != expected_contract:
            raise ValueError("checkpoint contract does not match the expected contract")
    _verify_checkpoint_files(path, manifest)
    runtime = _verified_runtime(path, manifest)
    embedded = _embedded_spec(runtime)
    if expected_spec is not None:
        if not isinstance(expected_spec, TransformersInferenceSpec):
            raise TypeError("expected_spec must be a TransformersInferenceSpec or None")
        if embedded != expected_spec:
            raise ValueError("checkpoint inference_spec does not match the expected spec")
    selected_store = (store_factory or _default_store_factory)(manifest.contract.state_format)
    if selected_store.format_name != manifest.contract.state_format:
        raise ValueError("state store format does not match the checkpoint contract")
    actor_record = manifest.files["model"]
    actor_state = selected_store.load(path / actor_record.path)
    if not isinstance(actor_state, Mapping):
        raise ValueError("checkpoint actor state must be a mapping")
    _verify_checkpoint_files(path, manifest)
    return LoadedTransformerCheckpoint(path, manifest, embedded, actor_state)


def build_transformers_policy(
    checkpoint: LoadedTransformerCheckpoint,
    *,
    dependencies: InferenceDependencies | None = None,
) -> TextPolicy:
    """Reconstruct a tokenizer and actor, restore weights, then expose ``TextPolicy``."""

    if not isinstance(checkpoint, LoadedTransformerCheckpoint):
        raise TypeError("policy_factory requires a LoadedTransformerCheckpoint")
    selected = dependencies or InferenceDependencies()
    spec = checkpoint.spec
    device = None if spec.device is None else selected.device_resolver(spec.device)
    tokenizer = selected.tokenizer_loader(
        spec.tokenizer_name,
        revision=spec.tokenizer_revision,
        use_processor=spec.use_processor,
        chat_template_kwargs=spec.chat_template_kwargs,
        trust_remote_code=False,
    )
    contract = checkpoint.manifest.contract
    if tokenizer.fingerprint != contract.tokenizer_fingerprint:
        raise ValueError("loaded tokenizer does not match the checkpoint contract")
    actor = selected.actor_loader(
        spec.model_name,
        revision=spec.model_revision,
        device=device,
        dtype=spec.dtype,
        model_kind=spec.model_kind,
        dequantize_mxfp4=spec.dequantize_mxfp4,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if spec.lora_enabled:
        actor = selected.lora_loader(
            actor,
            rank=spec.lora_rank,
            alpha=spec.lora_alpha,
            dropout=spec.lora_dropout,
            target_modules=spec.lora_target_modules,
        )
    if actor.fingerprint != contract.model_fingerprint:
        raise ValueError("loaded actor architecture does not match the checkpoint contract")
    try:
        actor.load_state_dict(checkpoint.actor_state)
    except Exception as error:
        raise ValueError("actor state is incompatible with the reconstructed model") from error
    actor.eval()
    backend = selected.backend_factory(
        actor,
        tokenizer,
        context_tokens=spec.context_tokens,
    )
    return TextPolicy(
        backend,
        temperature=spec.temperature,
        top_p=spec.top_p,
        top_k=spec.top_k,
        max_tokens=spec.max_tokens,
        scaffold=spec.scaffold,
    )


def checkpoint_factory(checkpoint_path: Path | None) -> LoadedTransformerCheckpoint:
    """Default ``scripts/evaluate.py`` checkpoint factory.

    Use with ``--checkpoint-factory vapa.model.inference:checkpoint_factory``.
    """

    if checkpoint_path is None:
        raise ValueError("the Transformers checkpoint factory requires --checkpoint")
    return load_transformers_checkpoint(checkpoint_path)


def policy_factory(checkpoint: object | None) -> TextPolicy:
    """Default ``scripts/evaluate.py`` policy factory.

    Use with ``--policy-factory vapa.model.inference:policy_factory``.
    """

    if not isinstance(checkpoint, LoadedTransformerCheckpoint):
        raise TypeError("the Transformers policy factory requires its checkpoint factory")
    return build_transformers_policy(checkpoint)


def checkpoint_manager_factory(checkpoint: object | None) -> ManagerFactory:
    """Reconstruct the exact RL state machine carried by a checkpoint.

    Use with ``--checkpoint-manager-factory
    vapa.model.inference:checkpoint_manager_factory``.  SFT-only checkpoints do
    not contain an environment specification and are intentionally rejected.
    """

    if not isinstance(checkpoint, LoadedTransformerCheckpoint):
        raise TypeError("the checkpoint manager factory requires its checkpoint factory")
    extra = checkpoint.manifest.runtime.extra
    raw = None if extra is None else extra.get("environment_spec")
    if not isinstance(raw, Mapping):
        raise ValueError('checkpoint runtime.extra must contain an "environment_spec" object')
    spec = CheckpointEnvironmentSpec.from_dict(raw)
    calculators = CalculatorRegistry.from_manifest(spec.calculator_manifest)

    def build(episode: Episode) -> StateManager:
        return StateManager(
            episode,
            memory_capacity=spec.memory_capacity,
            action_budget=spec.action_budget,
            turn_cap=spec.turn_cap,
            retrieval_limit=spec.retrieval_limit,
            calculators=calculators,
        )

    return build


@dataclass(frozen=True, slots=True)
class InferenceFactories:
    """A matching checkpoint/policy factory pair for :func:`build_inference_engine`."""

    checkpoint_factory: Callable[[Path | None], LoadedTransformerCheckpoint]
    policy_factory: Callable[[object | None], TextPolicy]

    def __iter__(self):
        yield self.checkpoint_factory
        yield self.policy_factory


def make_inference_factories(
    *,
    expected_contract: CheckpointContract | None = None,
    expected_spec: TransformersInferenceSpec | None = None,
    dependencies: InferenceDependencies | None = None,
) -> InferenceFactories:
    """Bind trust inputs and optional dependencies into generic-runtime factories."""

    selected = dependencies or InferenceDependencies()

    def load(path: Path | None) -> LoadedTransformerCheckpoint:
        if path is None:
            raise ValueError("the Transformers checkpoint factory requires a checkpoint path")
        return load_transformers_checkpoint(
            path,
            expected_contract=expected_contract,
            expected_spec=expected_spec,
            store_factory=selected.store_factory,
        )

    def build(checkpoint: object | None) -> TextPolicy:
        if not isinstance(checkpoint, LoadedTransformerCheckpoint):
            raise TypeError("the bound policy factory received an incompatible checkpoint")
        return build_transformers_policy(checkpoint, dependencies=selected)

    return InferenceFactories(load, build)


__all__ = [
    "ENVIRONMENT_SPEC_FORMAT_VERSION",
    "INFERENCE_SPEC_FORMAT_VERSION",
    "CheckpointEnvironmentSpec",
    "InferenceDependencies",
    "InferenceFactories",
    "LoadedTransformerCheckpoint",
    "TransformersInferenceSpec",
    "build_transformers_policy",
    "checkpoint_factory",
    "checkpoint_manager_factory",
    "load_transformers_checkpoint",
    "make_inference_factories",
    "policy_factory",
]
