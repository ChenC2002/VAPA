"""Executable, resume-safe VAPA reinforcement-learning lifecycle.

The method core is intentionally backend neutral.  This module connects that core to
episode streaming, exact actor-token accounting, optimizer updates, metrics, and strict
checkpoints.  Hugging Face is the default optional backend, while all orchestration can
be tested with dependency-light injected components.

The paper does not publish the complete verifier predicates or calculator
operationalizations.  A non-demo run therefore has to identify both artifacts
explicitly; this module never silently substitutes the synthetic catalog.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import math
import os
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from vapa import __version__
from vapa.artifacts import (
    ArtifactContentKind,
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    guard_artifact_write_path,
    strict_json_loads,
)
from vapa.config import ExperimentConfig, load_config
from vapa.data.episodes import episode_to_record, load_episode_objects
from vapa.data.io import DataValidationError, load_jsonl, sha256_file, sha256_json
from vapa.environment.calculators import CalculatorRegistry
from vapa.environment.state_manager import StateManager
from vapa.inference import checkpoint_identity, factory_identity, import_factory
from vapa.model.protocols import (
    ActorModelAdapter,
    OptimizerAdapter,
    SchedulerAdapter,
    TokenizerAdapter,
)
from vapa.policies.base import Policy
from vapa.prompts import render_chat
from vapa.provenance import package_code_fingerprint
from vapa.rollouts import OutcomeScorer, RolloutRunner, exact_outcome_scorer
from vapa.schemas import Episode
from vapa.training.advantages import assign_step_advantages, assign_trajectory_advantages
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
from vapa.training.curriculum import CurriculumStage, stage_at
from vapa.training.groups import assign_step_groups
from vapa.training.ledger import GroupCharge, ReplayQuotaLedger, TokenLedger
from vapa.training.runtime import (
    DistributedContext,
    TrainStepReport,
    build_adamw,
    resolve_device,
    seed_everything,
    train_vapa_update,
)
from vapa.training.trainer import InstanceBatch, InstanceBatchBuilder, UpdateBatch
from vapa.verifiers import VerifierCatalog

RL_RUN_SCHEMA_VERSION = 1
REPLAY_QUOTA_SCHEMA_VERSION = 1


class TrainingLifecycleError(RuntimeError):
    """Raised when a run cannot preserve its reproducibility contract."""


@dataclass(frozen=True, slots=True)
class _CapturedInput:
    """Immutable pre-parse bytes used to close input TOCTOU gaps."""

    label: str
    path: Path
    payload: bytes = field(repr=False)
    sha256: str

    @classmethod
    def capture(cls, label: str, path: Path) -> _CapturedInput:
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise FileNotFoundError(f"{label} input is missing or symbolic: {expanded}")
        candidate = expanded.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"{label} input is missing or symbolic: {candidate}")
        payload = candidate.read_bytes()
        return cls(label, candidate, payload, hashlib.sha256(payload).hexdigest())

    def verify(self, *, phase: str) -> None:
        if self.path.is_symlink() or not self.path.is_file():
            raise TrainingLifecycleError(f"{self.label} input changed during {phase}")
        payload = self.path.read_bytes()
        if len(payload) != len(self.payload) or hashlib.sha256(payload).hexdigest() != self.sha256:
            raise TrainingLifecycleError(f"{self.label} input changed during {phase}")

    def manifest_record(self) -> dict[str, object]:
        return {"sha256": self.sha256, "size_bytes": len(self.payload)}


def _config_input_chain(leaf: _CapturedInput) -> tuple[_CapturedInput, ...]:
    """Capture every inherited TOML file before the experiment loader reads it."""

    result = [leaf]
    seen = {leaf.path}
    current = leaf
    while True:
        document = tomllib.loads(current.payload.decode("utf-8"))
        parent = document.get("extends")
        if parent is None:
            return tuple(result)
        if not isinstance(parent, str) or not parent.strip():
            # ``load_config`` emits the public validation error; no valid parent can
            # be discovered from this malformed declaration.
            return tuple(result)
        parent_path = (current.path.parent / parent).resolve()
        if parent_path in seen:
            return tuple(result)
        captured = _CapturedInput.capture(f"config_parent_{len(result)}", parent_path)
        result.append(captured)
        seen.add(parent_path)
        current = captured


def _capture_training_inputs(
    *,
    config_path: Path,
    episode_path: Path,
    verifier_manifest: Path,
    calculator_manifest: Path,
    replay_quota_path: Path | None,
) -> tuple[_CapturedInput, ...]:
    """Capture all behavior/task/replay artifacts before calling their parsers."""

    config_leaf = _CapturedInput.capture("config", config_path)
    captures = [
        config_leaf,
        _CapturedInput.capture("episodes", episode_path),
        _CapturedInput.capture("verifier_manifest", verifier_manifest),
        _CapturedInput.capture("calculator_manifest", calculator_manifest),
    ]
    captures.extend(_config_input_chain(config_leaf)[1:])
    if replay_quota_path is None:
        return tuple(captures)

    quota = _CapturedInput.capture("replay_quota", replay_quota_path)
    source_manifest = _CapturedInput.capture(
        "replay_source_run_manifest", quota.path.parent / "run_manifest.json"
    )
    source_result = _CapturedInput.capture(
        "replay_source_result", quota.path.parent / "result.json"
    )
    captures.extend((quota, source_manifest, source_result))
    raw_result = strict_json_loads(source_result.payload)
    if not isinstance(raw_result, Mapping):
        return tuple(captures)
    checkpoint_value = raw_result.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        return tuple(captures)
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint.is_relative_to(quota.path.parent):
        raise TrainingLifecycleError(
            "replay quota donor checkpoint must remain inside the source run directory"
        )
    checkpoint_manifest = _CapturedInput.capture(
        "replay_source_checkpoint_manifest", checkpoint / "manifest.json"
    )
    captures.append(checkpoint_manifest)
    raw_checkpoint = strict_json_loads(checkpoint_manifest.payload)
    if not isinstance(raw_checkpoint, Mapping):
        return tuple(captures)
    files = raw_checkpoint.get("files")
    if not isinstance(files, Mapping):
        return tuple(captures)
    runtime = files.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("path"), str):
        return tuple(captures)
    runtime_name = str(runtime["path"])
    if Path(runtime_name).name != runtime_name:
        raise TrainingLifecycleError("replay quota donor runtime path is unsafe")
    captures.append(
        _CapturedInput.capture(
            "replay_source_checkpoint_runtime",
            checkpoint / runtime_name,
        )
    )
    return tuple(captures)


def _verify_captured_inputs(inputs: Sequence[_CapturedInput], *, phase: str) -> None:
    for captured in inputs:
        captured.verify(phase=phase)


def _input_manifest(inputs: Sequence[_CapturedInput]) -> dict[str, object]:
    return {item.label: item.manifest_record() for item in inputs}


def _captured_input(inputs: Sequence[_CapturedInput], label: str) -> _CapturedInput:
    matches = [item for item in inputs if item.label == label]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one captured {label!r} input")
    return matches[0]


@dataclass(frozen=True, slots=True)
class TransformersLoadOptions:
    """Immutable model-loading choices omitted from the paper configuration."""

    model_revision: str
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    model_kind: str = "auto"
    use_processor: bool | None = None
    device: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = False
    dequantize_mxfp4: bool | None = None
    lora_enabled: bool = True
    lora_target_modules: tuple[str, ...] = ()
    sft_checkpoint: Path | None = None
    allow_base_initialization: bool = False
    scaffold: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision must be an immutable non-empty revision")
        if self.tokenizer_name is not None and not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name cannot be empty")
        if self.tokenizer_revision is not None and not self.tokenizer_revision.strip():
            raise ValueError("tokenizer_revision cannot be empty")
        if self.model_kind not in {"auto", "multimodal", "causal"}:
            raise ValueError("model_kind must be auto, multimodal, or causal")
        if self.use_processor is not None and not isinstance(self.use_processor, bool):
            raise TypeError("use_processor must be boolean or None")
        if not isinstance(self.local_files_only, bool) or not isinstance(
            self.trust_remote_code, bool
        ):
            raise TypeError("model loading switches must be booleans")
        if self.dequantize_mxfp4 is not None and not isinstance(self.dequantize_mxfp4, bool):
            raise TypeError("dequantize_mxfp4 must be boolean or None")
        if self.trust_remote_code:
            raise ValueError(
                "trust_remote_code is forbidden: checkpoint reconstruction requires "
                "audited, package-owned model code"
            )
        if not isinstance(self.lora_enabled, bool):
            raise TypeError("lora_enabled must be boolean")
        if self.sft_checkpoint is not None and not isinstance(self.sft_checkpoint, Path):
            raise TypeError("sft_checkpoint must be a Path or None")
        if self.sft_checkpoint is not None:
            checkpoint = self.sft_checkpoint.expanduser().resolve()
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"SFT checkpoint directory does not exist: {checkpoint}")
            object.__setattr__(self, "sft_checkpoint", checkpoint)
        if not isinstance(self.allow_base_initialization, bool):
            raise TypeError("allow_base_initialization must be boolean")
        if not isinstance(self.scaffold, str):
            raise TypeError("scaffold must be a string")
        if self.sft_checkpoint is None and not self.allow_base_initialization:
            raise ValueError(
                "built-in RL requires the verified SFT checkpoint that initializes both "
                "actor and frozen reference; set allow_base_initialization only for an "
                "explicit ablation"
            )
        if self.lora_enabled and not self.lora_target_modules:
            raise ValueError(
                "LoRA target modules are not specified by the paper and must be explicit"
            )
        if any(not isinstance(item, str) or not item.strip() for item in self.lora_target_modules):
            raise ValueError("LoRA target modules must be non-empty strings")


@dataclass(slots=True)
class TrainingComponents:
    """All backend-owned objects needed by the orchestration loop."""

    policy: Policy
    tokenizer: TokenizerAdapter
    actor: ActorModelAdapter
    reference: ActorModelAdapter
    optimizer: OptimizerAdapter
    scheduler: SchedulerAdapter | None
    store: StateStore
    component_id: str
    optimizer_name: str
    scheduler_name: str
    inference_spec: Mapping[str, object] | None = None
    initialization_identity: Mapping[str, object] | None = None
    policy_scaffold: str = ""
    supports_distributed: bool = False
    initial_state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("component_id", "optimizer_name", "scheduler_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (self.scheduler_name == "none") != (self.scheduler is None):
            raise ValueError("scheduler_name must be 'none' exactly when scheduler is absent")
        if self.actor is self.reference:
            raise ValueError("actor and frozen reference must be distinct objects")
        if not isinstance(self.supports_distributed, bool):
            raise TypeError("supports_distributed must be boolean")
        if self.inference_spec is not None:
            # Validate now, rather than discovering an unserializable checkpoint extra.
            canonical_json_dumps(self.inference_spec)
        if self.initialization_identity is None:
            raise ValueError(
                "component factories must provide a content-addressed initialization_identity"
            )
        canonical_json_dumps(self.initialization_identity)
        initialization_sha = self.initialization_identity.get("sha256")
        if (
            not isinstance(initialization_sha, str)
            or len(initialization_sha) != 64
            or any(character not in "0123456789abcdef" for character in initialization_sha)
        ):
            raise ValueError("initialization_identity.sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.policy_scaffold, str):
            raise TypeError("policy_scaffold must be a string")
        actor_state = _serialized_model_state_identity(self.actor, self.store)
        reference_state = _serialized_model_state_identity(self.reference, self.store)
        if actor_state != reference_state:
            raise ValueError(
                "actor and frozen reference must start from identical Algorithm 1 weights"
            )
        self.initial_state_sha256 = actor_state["sha256"]


def _serialized_model_state_identity(
    model: ActorModelAdapter,
    store: StateStore,
) -> dict[str, object]:
    """Content-address model state through the backend's checkpoint serializer."""

    with tempfile.TemporaryDirectory(prefix="vapa-initial-state-") as directory:
        path = Path(directory) / f"state{store.extension}"
        store.save(model.state_dict(), path)
        return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


class ComponentFactory(Protocol):
    def __call__(
        self, config: ExperimentConfig, distributed: DistributedContext
    ) -> TrainingComponents: ...


class ContextLimitedPolicy:
    """Enforce the configured full-context cap before and after generation."""

    def __init__(
        self,
        policy: Policy,
        tokenizer: TokenizerAdapter,
        *,
        scaffold: str,
        context_tokens: int,
        reserved_generation_tokens: int,
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.scaffold = scaffold
        self.context_tokens = context_tokens
        self.reserved_generation_tokens = reserved_generation_tokens

    def sample(self, observation, *, rng, n: int = 1, greedy: bool = False):
        prompt_tokens = len(
            self.tokenizer.encode_messages(
                render_chat(observation, self.scaffold),
                add_generation_prompt=True,
            )
        )
        if prompt_tokens + self.reserved_generation_tokens > self.context_tokens:
            raise TrainingLifecycleError(
                "rendered policy prompt plus max_turn_tokens exceeds context_tokens"
            )
        decisions = self.policy.sample(observation, rng=rng, n=n, greedy=greedy)
        for decision in decisions:
            if prompt_tokens + decision.token_count > self.context_tokens:
                raise TrainingLifecycleError(
                    "sampled action exceeds the configured full-context token cap"
                )
        return decisions


class SampledTokenCosineScheduler:
    """Warmup/cosine schedule indexed by sampled actor tokens, not guessed steps."""

    def __init__(
        self,
        optimizer: OptimizerAdapter,
        *,
        target_tokens: int,
        warmup_fraction: float,
        final_lr_fraction: float,
    ) -> None:
        if target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 < final_lr_fraction <= 1:
            raise ValueError("final_lr_fraction must be in (0, 1]")
        if not optimizer.param_groups:
            raise ValueError("optimizer must contain parameter groups")
        self.optimizer = optimizer
        self.target_tokens = target_tokens
        self.warmup_fraction = warmup_fraction
        self.final_lr_fraction = final_lr_fraction
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        self.sampled_tokens = 0
        self.completed_updates = 0
        self._apply()

    def _factor(self) -> float:
        progress = min(1.0, self.sampled_tokens / self.target_tokens)
        if self.warmup_fraction and progress < self.warmup_fraction:
            return progress / self.warmup_fraction
        decay = (progress - self.warmup_fraction) / (1.0 - self.warmup_fraction)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay))
        return self.final_lr_fraction + (1.0 - self.final_lr_fraction) * cosine

    def _apply(self) -> None:
        factor = self._factor()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def prepare(self, sampled_tokens: int) -> None:
        if (
            isinstance(sampled_tokens, bool)
            or not isinstance(sampled_tokens, int)
            or sampled_tokens < self.sampled_tokens
        ):
            raise ValueError("scheduler sampled-token cursor must be monotonic")
        self.sampled_tokens = min(sampled_tokens, self.target_tokens)
        self._apply()

    def step(self) -> None:
        self.completed_updates += 1

    def state_dict(self) -> Mapping[str, object]:
        return {
            "format_version": 1,
            "target_tokens": self.target_tokens,
            "warmup_fraction": self.warmup_fraction,
            "final_lr_fraction": self.final_lr_fraction,
            "base_lrs": list(self.base_lrs),
            "sampled_tokens": self.sampled_tokens,
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "format_version",
            "target_tokens",
            "warmup_fraction",
            "final_lr_fraction",
            "base_lrs",
            "sampled_tokens",
            "completed_updates",
        }
        if set(state) != expected or state["format_version"] != 1:
            raise ValueError("sampled-token scheduler state has an incompatible schema")
        if (
            state["target_tokens"] != self.target_tokens
            or state["warmup_fraction"] != self.warmup_fraction
            or state["final_lr_fraction"] != self.final_lr_fraction
            or tuple(state["base_lrs"]) != self.base_lrs  # type: ignore[arg-type]
        ):
            raise ValueError("sampled-token scheduler state does not match configuration")
        sampled_tokens = state["sampled_tokens"]
        completed_updates = state["completed_updates"]
        if (
            isinstance(sampled_tokens, bool)
            or not isinstance(sampled_tokens, int)
            or not 0 <= sampled_tokens <= self.target_tokens
            or isinstance(completed_updates, bool)
            or not isinstance(completed_updates, int)
            or completed_updates < 0
        ):
            raise ValueError("sampled-token scheduler cursor is malformed")
        self.sampled_tokens = sampled_tokens
        self.completed_updates = completed_updates
        self._apply()


class VerifierFactory(Protocol):
    def __call__(self, config: ExperimentConfig, manifest_path: Path) -> VerifierCatalog: ...


@dataclass(frozen=True, slots=True)
class RLRunSettings:
    run_id: str = "vapa"
    checkpoint_every: int = 10
    max_optimizer_steps: int | None = None
    resume_from: Path | None = None
    replay_quota_path: Path | None = None
    scaffold: str = ""
    ratio_clip: float | None = None
    kl_mode: str = "forward"
    deterministic: bool = True
    dry_run: bool = False
    demo_catalogs: bool = False
    allow_non_paper_exact: bool = False
    content_kind: str = "credentialed"

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if (
            isinstance(self.checkpoint_every, bool)
            or not isinstance(self.checkpoint_every, int)
            or self.checkpoint_every < 1
        ):
            raise ValueError("checkpoint_every must be a positive integer")
        if self.max_optimizer_steps is not None and (
            isinstance(self.max_optimizer_steps, bool)
            or not isinstance(self.max_optimizer_steps, int)
            or self.max_optimizer_steps < 1
        ):
            raise ValueError("max_optimizer_steps must be positive or None")
        if self.ratio_clip is not None:
            raise ValueError("Eq. 9 uses a log-policy objective without ratio clipping")
        if self.kl_mode != "forward":
            raise ValueError("Eq. 9 requires full-vocabulary forward KL")
        for name in ("deterministic", "dry_run", "demo_catalogs", "allow_non_paper_exact"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        try:
            ArtifactContentKind(self.content_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("content_kind must be a valid artifact disclosure class") from error


@dataclass(frozen=True, slots=True)
class ReplayQuotaRecord:
    sequence: int
    curriculum_stage: str
    allocation_stratum: str
    sampled_tokens_before: int
    branch_sampled_tokens: int
    occurrence_id: str
    episode_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("replay quota sequence must be nonnegative")
        if not isinstance(self.curriculum_stage, str) or not self.curriculum_stage:
            raise ValueError("replay quota curriculum_stage must be non-empty")
        if not isinstance(self.allocation_stratum, str) or not self.allocation_stratum:
            raise ValueError("replay quota allocation_stratum must be non-empty")
        if (
            isinstance(self.sampled_tokens_before, bool)
            or not isinstance(self.sampled_tokens_before, int)
            or self.sampled_tokens_before < 0
        ):
            raise ValueError("sampled_tokens_before must be nonnegative")
        if (
            isinstance(self.branch_sampled_tokens, bool)
            or not isinstance(self.branch_sampled_tokens, int)
            or self.branch_sampled_tokens < 0
        ):
            raise ValueError("branch_sampled_tokens must be nonnegative")
        if not isinstance(self.occurrence_id, str) or not self.occurrence_id:
            raise ValueError("replay quota occurrence_id must be non-empty")
        if (
            not isinstance(self.episode_sha256, str)
            or len(self.episode_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.episode_sha256)
        ):
            raise ValueError("replay quota episode_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPLAY_QUOTA_SCHEMA_VERSION,
            "sequence": self.sequence,
            "curriculum_stage": self.curriculum_stage,
            "allocation_stratum": self.allocation_stratum,
            "sampled_tokens_before": self.sampled_tokens_before,
            "branch_sampled_tokens": self.branch_sampled_tokens,
            "occurrence_id": self.occurrence_id,
            "episode_sha256": self.episode_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ReplayQuotaRecord:
        expected = {
            "schema_version",
            "sequence",
            "curriculum_stage",
            "allocation_stratum",
            "sampled_tokens_before",
            "branch_sampled_tokens",
            "occurrence_id",
            "episode_sha256",
        }
        if set(raw) != expected or raw.get("schema_version") != REPLAY_QUOTA_SCHEMA_VERSION:
            raise DataValidationError("replay quota record has an incompatible schema")
        return cls(
            sequence=raw["sequence"],  # type: ignore[arg-type]
            curriculum_stage=raw["curriculum_stage"],  # type: ignore[arg-type]
            allocation_stratum=raw["allocation_stratum"],  # type: ignore[arg-type]
            sampled_tokens_before=raw["sampled_tokens_before"],  # type: ignore[arg-type]
            branch_sampled_tokens=raw["branch_sampled_tokens"],  # type: ignore[arg-type]
            occurrence_id=raw["occurrence_id"],  # type: ignore[arg-type]
            episode_sha256=raw["episode_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    status: str
    output_directory: Path
    global_step: int
    sampled_tokens: int
    trainable_tokens: int
    metric_records: int
    checkpoint_path: Path | None
    episode_count: int
    rank: int
    world_size: int
    training_ready: bool = True
    readiness_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    replay_quota_attestation: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "output_directory": str(self.output_directory),
            "global_step": self.global_step,
            "sampled_tokens": self.sampled_tokens,
            "trainable_tokens": self.trainable_tokens,
            "metric_records": self.metric_records,
            "checkpoint_path": (
                None if self.checkpoint_path is None else str(self.checkpoint_path)
            ),
            "episode_count": self.episode_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "training_ready": self.training_ready,
            "readiness_issues": list(self.readiness_issues),
            "warnings": list(self.warnings),
            "replay_quota_attestation": (
                None
                if self.replay_quota_attestation is None
                else dict(self.replay_quota_attestation)
            ),
        }


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json_dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reconcile_journal(path: Path, *, expected: int, kind: str) -> None:
    """Trim an uncheckpointed journal tail while validating the committed prefix."""

    if expected < 0:
        raise ValueError("expected journal length cannot be negative")
    if not path.exists():
        if expected:
            raise TrainingLifecycleError(f"{kind} journal ends before the resume checkpoint")
        return
    raw_bytes = path.read_bytes()
    # Every committed append is newline-terminated.  Ignore an arbitrary torn tail,
    # including an incomplete multibyte UTF-8 sequence, beyond the checkpoint cursor.
    terminated = raw_bytes.split(b"\n")[:-1]
    if len(terminated) < expected:
        raise TrainingLifecycleError(f"{kind} journal ends before the resume checkpoint")
    committed: list[Mapping[str, object]] = []
    for index, encoded in enumerate(terminated[:expected]):
        try:
            line = encoded.decode("utf-8")
            raw = strict_json_loads(line)
        except (TypeError, UnicodeDecodeError, ValueError) as error:
            raise TrainingLifecycleError(
                f"{kind} journal committed record {index} is corrupt"
            ) from error
        if not isinstance(raw, Mapping):
            raise TrainingLifecycleError(f"{kind} journal record {index} is not an object")
        if kind == "metrics" and raw.get("metric_index") != index:
            raise TrainingLifecycleError("metrics journal index sequence is corrupt")
        if kind == "replay_quota":
            record = ReplayQuotaRecord.from_dict(raw)
            if record.sequence != index:
                raise TrainingLifecycleError("replay quota journal sequence is corrupt")
        committed.append(raw)
    payload = "".join(canonical_json_dumps(record) + "\n" for record in committed)
    atomic_write_text(path, payload)


def _require_latest_resume_checkpoint(
    output: Path,
    selected_path: Path,
    selected_runtime: RuntimeState,
) -> None:
    """Reject rewinding model state when a later complete checkpoint is present."""

    checkpoints = output / "checkpoints"
    if not checkpoints.exists():
        return
    selected = selected_path.resolve()
    later: list[str] = []
    for candidate in sorted(path for path in checkpoints.iterdir() if path.is_dir()):
        if candidate.resolve() == selected or candidate.name.startswith("."):
            continue
        try:
            manifest = read_manifest(candidate)
        except (FileNotFoundError, TypeError, ValueError):
            continue
        cursor = (manifest.runtime.sampled_tokens, manifest.runtime.global_step)
        selected_cursor = (selected_runtime.sampled_tokens, selected_runtime.global_step)
        if cursor > selected_cursor:
            later.append(candidate.name)
    if later:
        raise TrainingLifecycleError(
            "resume checkpoint is not the latest checkpoint in the output directory; "
            f"later checkpoints exist: {later}"
        )


def shard_episodes(episodes: Iterable[Episode], context: DistributedContext) -> tuple[Episode, ...]:
    """Return a stable rank shard independent of source-file ordering."""

    ordered = sorted(episodes, key=lambda item: item.task.instance_id)
    if not ordered:
        raise ValueError("training requires at least one episode")
    seen: set[str] = set()
    for episode in ordered:
        identifier = episode.task.instance_id
        if identifier in seen:
            raise ValueError(f"duplicate training instance_id: {identifier!r}")
        seen.add(identifier)
    shard = tuple(
        episode
        for index, episode in enumerate(ordered)
        if index % context.world_size == context.rank
    )
    if not shard:
        raise ValueError(f"rank {context.rank} received an empty episode shard")
    return shard


def episode_history_quartile(episode: Episode) -> int:
    value = episode.task.metadata.get("history_quartile")
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
        raise DataValidationError(
            f"episode {episode.task.instance_id!r} requires integer metadata.history_quartile "
            "in [1, 4] for the paper curriculum"
        )
    return value


def episode_sampling_labels(episode: Episode) -> tuple[str, str]:
    """Return the explicit A.7 suite and task-type stratum labels."""

    suite = episode.task.metadata.get("suite")
    task_type = episode.task.metadata.get("task_type")
    if suite not in {"calculation", "retrieval"}:
        raise DataValidationError(
            f"episode {episode.task.instance_id!r} requires metadata.suite equal to "
            "'calculation' or 'retrieval' for hierarchical sampling"
        )
    if not isinstance(task_type, str) or not task_type.strip():
        raise DataValidationError(
            f"episode {episode.task.instance_id!r} requires non-empty "
            "metadata.task_type for hierarchical sampling"
        )
    return suite, task_type


def _keyed_order(
    values: Iterable[str],
    *,
    seed: int,
    rank: int,
    namespace: str,
    cycle: int,
) -> tuple[str, ...]:
    def key(value: str) -> bytes:
        payload = canonical_json_dumps(
            {
                "namespace": namespace,
                "seed": seed,
                "rank": rank,
                "cycle": cycle,
                "value": value,
            }
        ).encode("utf-8")
        return hashlib.sha256(payload).digest()

    return tuple(sorted(values, key=key))


class DeterministicEpisodeStream:
    """A.7 hierarchy: 50/50 suites, uniform task types, keyed instances."""

    def __init__(
        self,
        episodes: Sequence[Episode],
        *,
        seed: int,
        rank: int,
        cycle: int = 0,
        position: int = 0,
    ) -> None:
        if not episodes:
            raise ValueError("episode stream cannot be empty")
        if min(seed, rank, cycle, position) < 0:
            raise ValueError("stream seed, rank, cycle, and position must be nonnegative")
        self.episodes = tuple(episodes)
        self.seed = seed
        self.rank = rank
        self.cycle = cycle
        self.position = position
        if position > len(self.episodes):
            raise ValueError("stream position is outside its cycle")
        if self.position == len(self.episodes):
            self.cycle += 1
            self.position = 0

    @property
    def draw_index(self) -> int:
        return self.cycle * len(self.episodes) + self.position

    def _advance(self) -> None:
        self.position += 1
        if self.position >= len(self.episodes):
            self.cycle += 1
            self.position = 0

    def next(self, *, max_history_quartile: int) -> Episode:
        if not 1 <= max_history_quartile <= 4:
            raise ValueError("max_history_quartile must be in [1, 4]")
        strata: dict[str, dict[str, list[Episode]]] = {
            "calculation": {},
            "retrieval": {},
        }
        for episode in self.episodes:
            suite, task_type = episode_sampling_labels(episode)
            if episode_history_quartile(episode) <= max_history_quartile:
                strata[suite].setdefault(task_type, []).append(episode)
        missing = [suite for suite, task_types in strata.items() if not task_types]
        if missing:
            raise TrainingLifecycleError(
                f"rank {self.rank} lacks curriculum-eligible A.7 suites: {missing}"
            )

        draw = self.draw_index
        suite_cycle, suite_offset = divmod(draw, 2)
        suites = _keyed_order(
            ("calculation", "retrieval"),
            seed=self.seed,
            rank=self.rank,
            namespace="vapa-rl-suite-order-v1",
            cycle=suite_cycle,
        )
        suite = suites[suite_offset]
        # Each complete two-draw cycle visits each suite exactly once, so this is
        # also the number of prior selections from the chosen suite.
        suite_draw = suite_cycle
        task_types = tuple(strata[suite])
        task_cycle, task_offset = divmod(suite_draw, len(task_types))
        ordered_types = _keyed_order(
            task_types,
            seed=self.seed,
            rank=self.rank,
            namespace=f"vapa-rl-task-type-order-v1:{suite}",
            cycle=task_cycle,
        )
        task_type = ordered_types[task_offset]
        candidates = strata[suite][task_type]
        episode_cycle, episode_offset = divmod(task_cycle, len(candidates))
        episode_ids = _keyed_order(
            (episode.task.instance_id for episode in candidates),
            seed=self.seed,
            rank=self.rank,
            namespace=f"vapa-rl-instance-order-v1:{suite}:{task_type}",
            cycle=episode_cycle,
        )
        selected_id = episode_ids[episode_offset]
        selected = next(
            episode for episode in candidates if episode.task.instance_id == selected_id
        )
        self._advance()
        return selected

    def to_dict(self) -> dict[str, int]:
        return {"cycle": self.cycle, "position": self.position}


def _sample_seed(seed: int, rank: int, sample_index: int, instance_id: str) -> int:
    payload = canonical_json_dumps(
        {
            "namespace": "vapa-rl-sample-seed-v1",
            "seed": seed,
            "rank": rank,
            "sample_index": sample_index,
            "instance_id": instance_id,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _optimizer_seed(seed: int, rank: int, global_step: int, sampled_tokens: int) -> int:
    """Key framework RNG to a checkpointed update cursor for exact resume."""

    payload = canonical_json_dumps(
        {
            "namespace": "vapa-rl-optimizer-seed-v1",
            "seed": seed,
            "rank": rank,
            "global_step": global_step,
            "sampled_tokens": sampled_tokens,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_replay_quota_records(path: str | Path) -> tuple[ReplayQuotaRecord, ...]:
    values = load_jsonl(path)
    records: list[ReplayQuotaRecord] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise DataValidationError(f"replay quota record {index} must be an object")
        record = ReplayQuotaRecord.from_dict(raw)
        if record.sequence != index:
            raise DataValidationError("replay quota sequences must be contiguous from zero")
        records.append(record)
    return tuple(records)


def _replay_quota_attestation(
    path: Path,
    *,
    expected_records: int | None = None,
) -> dict[str, object]:
    records = load_replay_quota_records(path)
    if expected_records is not None and len(records) != expected_records:
        raise TrainingLifecycleError(
            "replay quota record count does not match the lifecycle sample cursor"
        )
    return {
        "schema_version": REPLAY_QUOTA_SCHEMA_VERSION,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "record_count": len(records),
    }


def _manifest_flag(path: Path, artifact_name: str) -> bool:
    raw = strict_json_loads(path.read_bytes())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("paper_exact"), bool):
        raise DataValidationError(f"{artifact_name} manifest requires a paper_exact boolean")
    return bool(raw["paper_exact"])


def build_environment_spec(
    config: ExperimentConfig,
    calculator_manifest: str | Path,
) -> dict[str, object]:
    """Serialize the complete behavior-defining environment for checkpoint inference."""

    raw = strict_json_loads(Path(calculator_manifest).read_bytes())
    if not isinstance(raw, Mapping):
        raise DataValidationError("calculator manifest must be a JSON object")
    manifest = dict(raw)
    return {
        "schema_version": 1,
        "memory_capacity": config.environment.memory_capacity,
        "action_budget": config.environment.action_budget,
        "turn_cap": config.environment.turn_cap,
        "retrieval_limit": 5,
        "implementation_sha256": package_code_fingerprint(),
        "calculator_manifest": manifest,
        "calculator_manifest_sha256": artifact_fingerprint(manifest),
    }


def verifier_catalog_fingerprint(catalog: VerifierCatalog) -> str:
    """Bind predicate structure and Python implementation source into one digest."""

    predicates: list[dict[str, object]] = []
    for predicate in catalog.predicates:
        try:
            source = inspect.getsource(predicate.function)
        except (OSError, TypeError) as error:
            raise TrainingLifecycleError(
                f"cannot fingerprint verifier implementation {predicate.name!r}; "
                "use source-backed predicate callables"
            ) from error
        predicates.append(
            {
                "name": predicate.name,
                "family": predicate.family.value,
                "weight": predicate.weight,
                "actions": sorted(action.value for action in predicate.actions),
                "reliability": predicate.reliability,
                "function": factory_identity(predicate.function),
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            }
        )
    return artifact_fingerprint(
        {
            "schema_version": 1,
            "catalog_id": catalog.catalog_id,
            "paper_exact": catalog.paper_exact,
            "predicates": predicates,
        }
    )


def _validate_catalogs(
    config: ExperimentConfig,
    *,
    verifier_manifest: Path,
    calculator_manifest: Path,
    verifier_factory: VerifierFactory | None,
    settings: RLRunSettings,
) -> tuple[VerifierCatalog | None, CalculatorRegistry]:
    if not verifier_manifest.is_file():
        raise FileNotFoundError(f"verifier manifest does not exist: {verifier_manifest}")
    if not calculator_manifest.is_file():
        raise FileNotFoundError(f"calculator manifest does not exist: {calculator_manifest}")
    verifier_exact = _manifest_flag(verifier_manifest, "verifier")
    calculator_exact = _manifest_flag(calculator_manifest, "calculator")
    if not settings.demo_catalogs and not settings.allow_non_paper_exact:
        if not verifier_exact or not calculator_exact:
            raise TrainingLifecycleError(
                "non-demo training requires paper-exact verifier and calculator manifests; "
                "the public examples are explicitly non-paper-exact"
            )
    calculators = CalculatorRegistry.from_json(calculator_manifest)
    if not config.credit.use_process_rewards:
        return None, calculators
    if settings.demo_catalogs:
        catalog = VerifierCatalog.demo_default()
    else:
        if verifier_factory is None:
            raise TrainingLifecycleError(
                "process rewards require --verifier-factory plus its frozen manifest"
            )
        catalog = verifier_factory(config, verifier_manifest)
        if not isinstance(catalog, VerifierCatalog):
            raise TypeError("verifier factory did not return a VerifierCatalog")
    if catalog.paper_exact != verifier_exact and not settings.allow_non_paper_exact:
        raise TrainingLifecycleError("verifier factory and manifest disagree on paper_exact")
    if not settings.demo_catalogs:
        raw_manifest = strict_json_loads(verifier_manifest.read_bytes())
        assert isinstance(raw_manifest, Mapping)
        if raw_manifest.get("catalog_id") != catalog.catalog_id:
            raise TrainingLifecycleError("verifier manifest catalog_id does not match factory")
        declared_fingerprint = raw_manifest.get("catalog_fingerprint")
        if declared_fingerprint != verifier_catalog_fingerprint(catalog):
            raise TrainingLifecycleError(
                "verifier manifest catalog_fingerprint does not match predicate behavior"
            )
    return catalog, calculators


def initialize_from_sft_checkpoint(
    checkpoint_path: str | Path,
    *,
    actor: ActorModelAdapter,
    reference: ActorModelAdapter,
    tokenizer: TokenizerAdapter,
    expected_spec: object,
) -> Mapping[str, object]:
    """Load one integrity-checked SFT actor snapshot into actor and KL reference."""

    from vapa.model.inference import (
        TransformersInferenceSpec,
        load_transformers_checkpoint,
    )

    if not isinstance(expected_spec, TransformersInferenceSpec):
        raise TypeError("expected_spec must be a TransformersInferenceSpec")
    loaded = load_transformers_checkpoint(
        checkpoint_path,
        expected_spec=expected_spec,
    )
    extra = loaded.manifest.runtime.extra
    if not isinstance(extra, Mapping) or extra.get("objective") != "sft":
        raise TrainingLifecycleError("actor initialization checkpoint is not an SFT checkpoint")
    contract = loaded.manifest.contract
    if contract.model_fingerprint != actor.fingerprint:
        raise TrainingLifecycleError("SFT checkpoint architecture does not match the RL actor")
    if reference.fingerprint != actor.fingerprint:
        raise TrainingLifecycleError(
            "RL actor and frozen reference must have identical initialized topology"
        )
    if contract.tokenizer_fingerprint != tokenizer.fingerprint:
        raise TrainingLifecycleError("SFT checkpoint tokenizer does not match the RL tokenizer")
    actor.load_state_dict(loaded.actor_state)
    reference.load_state_dict(loaded.actor_state)
    return checkpoint_identity(checkpoint_path)


def transformers_inference_spec(
    config: ExperimentConfig,
    options: TransformersLoadOptions,
):
    """Construct the shared strict reconstruction spec without loading ML libraries."""

    from vapa.model.inference import TransformersInferenceSpec

    use_processor = (
        "qwen3.5" in config.model.name.casefold()
        if options.use_processor is None
        else options.use_processor
    )
    dequantize_mxfp4 = (
        config.model.dequantize_mxfp4
        if options.dequantize_mxfp4 is None
        else options.dequantize_mxfp4
    )
    if dequantize_mxfp4 != config.model.dequantize_mxfp4:
        raise ValueError(
            "dequantize_mxfp4 load option must match the immutable experiment configuration"
        )
    return TransformersInferenceSpec(
        model_name=config.model.name,
        model_revision=options.model_revision,
        tokenizer_name=options.tokenizer_name or config.model.name,
        tokenizer_revision=options.tokenizer_revision or options.model_revision,
        model_kind=options.model_kind,
        use_processor=use_processor,
        dtype=config.model.dtype,
        device=options.device,
        lora_enabled=options.lora_enabled,
        lora_rank=config.optimization.lora_rank,
        lora_alpha=config.optimization.lora_alpha,
        lora_dropout=config.optimization.lora_dropout,
        lora_target_modules=(options.lora_target_modules if options.lora_enabled else None),
        enable_thinking=config.model.thinking,
        reasoning_effort=config.model.reasoning_effort,
        dequantize_mxfp4=dequantize_mxfp4,
        context_tokens=config.environment.context_tokens,
        scaffold=options.scaffold,
        temperature=config.model.temperature,
        top_p=config.model.top_p,
        top_k=config.model.top_k,
        max_tokens=config.environment.max_turn_tokens,
    )


def validate_transformers_load_plan(
    config: ExperimentConfig,
    options: TransformersLoadOptions,
) -> None:
    """Validate immutable metadata and an SFT checkpoint without torch/Transformers."""

    spec = transformers_inference_spec(config, options)
    if options.sft_checkpoint is None:
        return
    checkpoint = options.sft_checkpoint
    manifest = read_manifest(checkpoint)
    for logical_name, record in manifest.files.items():
        candidate = checkpoint / record.path
        if candidate.is_symlink() or not candidate.is_file():
            raise TrainingLifecycleError(
                f"SFT checkpoint {logical_name} file is missing or symbolic"
            )
        if candidate.stat().st_size != record.size or sha256_file(candidate) != record.sha256:
            raise TrainingLifecycleError(
                f"SFT checkpoint {logical_name} file failed integrity validation"
            )
    runtime_record = manifest.files.get("runtime")
    if runtime_record is None:
        raise TrainingLifecycleError("SFT checkpoint is missing runtime state")
    runtime_raw = strict_json_loads((checkpoint / runtime_record.path).read_bytes())
    if not isinstance(runtime_raw, Mapping):
        raise TrainingLifecycleError("SFT checkpoint runtime is malformed")
    runtime = RuntimeState.from_dict(runtime_raw)
    if runtime != manifest.runtime:
        raise TrainingLifecycleError("SFT checkpoint runtime and manifest disagree")
    extra = runtime.extra
    if (
        not isinstance(extra, Mapping)
        or extra.get("objective") != "sft"
        or extra.get("inference_spec") != spec.to_dict()
    ):
        raise TrainingLifecycleError("SFT checkpoint does not match the requested RL load plan")


def build_transformers_components(
    config: ExperimentConfig,
    distributed: DistributedContext,
    *,
    options: TransformersLoadOptions,
) -> TrainingComponents:
    """Build trainable actor, frozen reference, generation backend, and optimizer."""

    if distributed.world_size != 1:
        raise TrainingLifecycleError(
            "the built-in Transformers stack is single-process; use an injected synchronized "
            "component factory for multi-process training"
        )
    from vapa.model.transformers import (
        TransformersActorAdapter,
        TransformersGenerationBackend,
        TransformersTokenizerAdapter,
        apply_shared_lora,
    )
    from vapa.policies.text import TextPolicy

    inference_spec = transformers_inference_spec(config, options)
    use_processor = inference_spec.use_processor
    tokenizer_name = inference_spec.tokenizer_name
    tokenizer_revision = inference_spec.tokenizer_revision
    device = resolve_device(options.device, local_rank=distributed.local_rank)
    common = {
        "revision": options.model_revision,
        "local_files_only": options.local_files_only,
        "trust_remote_code": options.trust_remote_code,
        "dequantize_mxfp4": inference_spec.dequantize_mxfp4,
        "use_safetensors": True,
    }
    actor = TransformersActorAdapter.from_pretrained(
        config.model.name,
        device=device,
        dtype=config.model.dtype,
        model_kind=options.model_kind,
        **common,
    )
    if options.lora_enabled:
        actor, reference = apply_shared_lora(
            actor,
            rank=config.optimization.lora_rank,
            alpha=config.optimization.lora_alpha,
            dropout=config.optimization.lora_dropout,
            target_modules=options.lora_target_modules,
        )
    else:
        # Full-parameter/non-LoRA training needs a physically distinct immutable
        # reference; shared named snapshots are specific to adapter-only updates.
        reference = TransformersActorAdapter.from_pretrained(
            config.model.name,
            device=device,
            dtype=config.model.dtype,
            model_kind=options.model_kind,
            **common,
        )
    tokenizer = TransformersTokenizerAdapter.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        use_processor=use_processor,
        local_files_only=options.local_files_only,
        trust_remote_code=options.trust_remote_code,
        chat_template_kwargs=inference_spec.chat_template_kwargs,
    )
    if options.sft_checkpoint is not None:
        initialization_identity = initialize_from_sft_checkpoint(
            options.sft_checkpoint,
            actor=actor,
            reference=reference,
            tokenizer=tokenizer,
            expected_spec=inference_spec,
        )
    else:
        # This path is an explicit ablation only.  Copying guarantees that the
        # reference is the exact initial actor snapshot, including LoRA weights.
        reference.load_state_dict(actor.state_dict())
        initialization_payload = {
            "kind": "base-model-ablation",
            "model_name": config.model.name,
            "model_revision": options.model_revision,
        }
        initialization_identity = {
            **initialization_payload,
            "sha256": artifact_fingerprint(initialization_payload),
        }
    for parameter in reference.parameters():
        requires_grad = getattr(parameter, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad(False)
    reference.eval()
    backend = TransformersGenerationBackend(
        actor,
        tokenizer,
        context_tokens=config.environment.context_tokens,
    )
    policy = TextPolicy(
        backend,
        temperature=config.model.temperature,
        top_p=config.model.top_p,
        top_k=config.model.top_k,
        max_tokens=config.environment.max_turn_tokens,
        scaffold=options.scaffold,
    )
    optimizer = build_adamw(
        actor,
        learning_rate=config.optimization.rl_learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = SampledTokenCosineScheduler(
        optimizer,
        target_tokens=config.optimization.sampled_token_budget,
        warmup_fraction=config.optimization.warmup_fraction,
        final_lr_fraction=config.optimization.final_lr_fraction,
    )
    return TrainingComponents(
        policy=policy,
        tokenizer=tokenizer,
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        scheduler=scheduler,
        store=TorchStateStore(),
        component_id="vapa:transformers-v1",
        optimizer_name="torch-adamw",
        scheduler_name="sampled-token-warmup-cosine-v1",
        inference_spec=inference_spec.to_dict(),
        initialization_identity=initialization_identity,
        policy_scaffold=options.scaffold,
    )


def _combine_and_normalize(
    instances: Sequence[InstanceBatch],
    stages: Sequence[CurriculumStage],
    config: ExperimentConfig,
) -> UpdateBatch:
    bases = tuple(rollout for instance in instances for rollout in instance.base_rollouts)
    branches = tuple(rollout for instance in instances for rollout in instance.branch_rollouts)
    groups = tuple(group for instance in instances for group in instance.groups)
    credit = config.credit
    if credit.use_step_credit:
        summary = assign_step_advantages(
            bases,
            branches,
            groups,
            process_weight=credit.process_weight if credit.use_process_rewards else 0.0,
            cost_weight=credit.cost_weight,
            gamma=credit.process_discount,
            beta=credit.local_weight,
            epsilon=credit.epsilon,
        )
    else:
        assign_trajectory_advantages(
            bases,
            process_weight=credit.process_weight if credit.use_process_rewards else 0.0,
            cost_weight=credit.cost_weight,
            epsilon=credit.epsilon,
        )
        summary = None
    normalized = tuple(replace(instance, advantage_summary=summary) for instance in instances)
    return UpdateBatch(normalized, summary, tuple(stages))


def _mask_comparison_only(instance: InstanceBatch) -> None:
    for rollout in instance.base_rollouts + instance.branch_rollouts:
        for turn in rollout.turns:
            turn.loss_mask = False


def _runtime_extra(
    *,
    settings: RLRunSettings,
    stream: DeterministicEpisodeStream,
    sample_index: int,
    replay_quota: ReplayQuotaLedger,
    replay_quota_cursor: int,
    metric_records: int,
    builder_counter: int,
    inference_spec: Mapping[str, object] | None,
    environment_spec: Mapping[str, object],
    replay_quota_attestation: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": RL_RUN_SCHEMA_VERSION,
        "run_id": settings.run_id,
        "stream": stream.to_dict(),
        "sample_index": sample_index,
        "builder_counter": builder_counter,
        "metric_records": metric_records,
        "replay_quota_cursor": replay_quota_cursor,
        "replay_quota": {
            "balance_tokens": replay_quota.balance_tokens,
            "replay_tokens": replay_quota.replay_tokens,
            "extra_base_tokens": replay_quota.extra_base_tokens,
            "settled_overshoot_tokens": replay_quota.settled_overshoot_tokens,
            "next_instance": replay_quota.next_instance,
            "curriculum_stage": replay_quota.curriculum_stage,
        },
        "inference_spec": None if inference_spec is None else dict(inference_spec),
        "environment_spec": dict(environment_spec),
        "replay_quota_attestation": (
            None if replay_quota_attestation is None else dict(replay_quota_attestation)
        ),
    }


def _restore_extra(
    runtime: RuntimeState,
    *,
    settings: RLRunSettings,
) -> tuple[dict[str, int], int, ReplayQuotaLedger, int, int, int]:
    extra = runtime.extra
    if not isinstance(extra, Mapping) or extra.get("schema_version") != RL_RUN_SCHEMA_VERSION:
        raise TrainingLifecycleError("checkpoint lacks compatible RL lifecycle state")
    if extra.get("run_id") != settings.run_id:
        raise TrainingLifecycleError("checkpoint run_id does not match this run")
    stream_raw = extra.get("stream")
    quota_raw = extra.get("replay_quota")
    if not isinstance(stream_raw, Mapping) or set(stream_raw) != {"cycle", "position"}:
        raise TrainingLifecycleError("checkpoint episode-stream cursor is malformed")
    if not isinstance(quota_raw, Mapping):
        raise TrainingLifecycleError("checkpoint replay quota is malformed")

    def integer(raw: Mapping[str, object], name: str, *, signed: bool = False) -> int:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or (not signed and value < 0):
            raise TrainingLifecycleError(f"checkpoint field {name!r} is malformed")
        return value

    stream_state = {
        "cycle": integer(stream_raw, "cycle"),
        "position": integer(stream_raw, "position"),
    }
    stage = quota_raw.get("curriculum_stage")
    if stage is not None and (not isinstance(stage, str) or not stage):
        raise TrainingLifecycleError("checkpoint replay quota stage is malformed")
    replay = ReplayQuotaLedger(
        balance_tokens=integer(quota_raw, "balance_tokens", signed=True),
        replay_tokens=integer(quota_raw, "replay_tokens"),
        extra_base_tokens=integer(quota_raw, "extra_base_tokens"),
        settled_overshoot_tokens=integer(quota_raw, "settled_overshoot_tokens"),
        next_instance=integer(quota_raw, "next_instance"),
        curriculum_stage=stage,
    )
    return (
        stream_state,
        integer(extra, "sample_index"),
        replay,
        integer(extra, "replay_quota_cursor"),
        integer(extra, "metric_records"),
        integer(extra, "builder_counter"),
    )


def _checkpoint_path(output: Path, runtime: RuntimeState, *, label: str) -> Path:
    return (
        output
        / "checkpoints"
        / f"{label}-step-{runtime.global_step:08d}-tokens-{runtime.sampled_tokens:012d}"
    )


def _save_runtime_checkpoint(
    output: Path,
    *,
    label: str,
    contract: CheckpointContract,
    runtime: RuntimeState,
    components: TrainingComponents,
) -> Path:
    path = _checkpoint_path(output, runtime, label=label)
    save_checkpoint(
        path,
        contract=contract,
        runtime=runtime,
        model=components.actor,
        reference=components.reference,
        tokenizer=components.tokenizer,
        optimizer=components.optimizer,
        scheduler=components.scheduler,
        store=components.store,
    )
    return path


def _replay_quota_identity(
    path: Path | None,
    *,
    episode_path: Path,
    config: ExperimentConfig,
    target_pairing_contract: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    if path is None:
        return {"kind": "none"}
    quota_path = path.resolve()
    source_manifest_path = quota_path.parent / "run_manifest.json"
    source_result_path = quota_path.parent / "result.json"
    if not source_manifest_path.is_file() or not source_result_path.is_file():
        raise TrainingLifecycleError(
            "replay quota must remain beside its completed source run manifest and result"
        )
    source_manifest = strict_json_loads(source_manifest_path.read_bytes())
    source_result = strict_json_loads(source_result_path.read_bytes())
    if not isinstance(source_manifest, Mapping) or not isinstance(source_result, Mapping):
        raise TrainingLifecycleError("replay quota source artifacts are malformed")
    if (
        source_manifest.get("episodes_sha256") != sha256_file(episode_path)
        or source_manifest.get("sampled_token_budget") != config.optimization.sampled_token_budget
        or source_manifest.get("fork_enabled") is not True
    ):
        raise TrainingLifecycleError(
            "replay quota source does not match episodes, token budget, or fork treatment"
        )
    if (
        target_pairing_contract is not None
        and source_manifest.get("control_pairing_contract") != target_pairing_contract
    ):
        raise TrainingLifecycleError(
            "replay quota donor does not match the target seed and behavior-defining "
            "control pairing contract"
        )
    source_tokens = source_result.get("sampled_tokens")
    if (
        source_result.get("status") != "complete"
        or isinstance(source_tokens, bool)
        or not isinstance(source_tokens, int)
        or source_tokens < config.optimization.sampled_token_budget
    ):
        raise TrainingLifecycleError("replay quota source run is not complete")
    actual_attestation = _replay_quota_attestation(quota_path)
    if source_result.get("replay_quota_attestation") != actual_attestation:
        raise TrainingLifecycleError(
            "replay quota does not match the completed donor result attestation"
        )
    checkpoint_value = source_result.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise TrainingLifecycleError("replay quota donor result lacks a final checkpoint")
    source_checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not source_checkpoint.is_relative_to(quota_path.parent):
        raise TrainingLifecycleError(
            "replay quota donor checkpoint must remain inside the source run directory"
        )
    source_checkpoint_manifest = read_manifest(source_checkpoint)
    runtime_record = source_checkpoint_manifest.files.get("runtime")
    if runtime_record is None:
        raise TrainingLifecycleError("replay quota donor checkpoint lacks runtime state")
    runtime_path = source_checkpoint / runtime_record.path
    if (
        runtime_path.is_symlink()
        or not runtime_path.is_file()
        or runtime_path.stat().st_size != runtime_record.size
        or sha256_file(runtime_path) != runtime_record.sha256
    ):
        raise TrainingLifecycleError("replay quota donor checkpoint runtime failed integrity")
    runtime_raw = strict_json_loads(runtime_path.read_bytes())
    if not isinstance(runtime_raw, Mapping):
        raise TrainingLifecycleError("replay quota donor checkpoint runtime is malformed")
    source_runtime = RuntimeState.from_dict(runtime_raw)
    if source_runtime != source_checkpoint_manifest.runtime:
        raise TrainingLifecycleError("replay quota donor checkpoint runtime and manifest disagree")
    runtime_extra = source_runtime.extra
    if (
        source_runtime.sampled_tokens != source_tokens
        or not isinstance(runtime_extra, Mapping)
        or runtime_extra.get("replay_quota_attestation") != actual_attestation
        or runtime_extra.get("sample_index") != actual_attestation["record_count"]
    ):
        raise TrainingLifecycleError(
            "replay quota does not match the completed donor checkpoint attestation"
        )
    return {
        "kind": "file",
        **actual_attestation,
        "source_run_manifest_sha256": sha256_file(source_manifest_path),
        "source_result_sha256": sha256_file(source_result_path),
        "source_checkpoint_manifest_sha256": sha256_file(source_checkpoint / "manifest.json"),
    }


def _run_manifest(
    config: ExperimentConfig,
    episode_path: Path,
    shard: Sequence[Episode],
    *,
    captured_inputs: Sequence[_CapturedInput],
    environment_spec: Mapping[str, object],
    catalog: VerifierCatalog | None,
    component_factory: ComponentFactory,
    verifier_factory: VerifierFactory | None,
    outcome_scorer: OutcomeScorer,
    components: TrainingComponents,
    context: DistributedContext,
    settings: RLRunSettings,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": RL_RUN_SCHEMA_VERSION,
        "run_id": settings.run_id,
        "content_kind": settings.content_kind,
        "package_version": __version__,
        "input_artifacts": _input_manifest(captured_inputs),
        "config_sha256": _captured_input(captured_inputs, "config").sha256,
        "config_fingerprint": fingerprint_payload(config),
        "episodes_sha256": _captured_input(captured_inputs, "episodes").sha256,
        "shard_sha256": sha256_json(
            [episode_to_record(episode) for episode in shard]  # type: ignore[arg-type]
        ),
        "verifier_manifest_sha256": _captured_input(captured_inputs, "verifier_manifest").sha256,
        "calculator_manifest_sha256": _captured_input(
            captured_inputs, "calculator_manifest"
        ).sha256,
        "environment_spec": dict(environment_spec),
        "verifier_catalog_id": None if catalog is None else catalog.catalog_id,
        "verifier_paper_exact": None if catalog is None else catalog.paper_exact,
        "verifier_catalog_fingerprint": (
            None if catalog is None else verifier_catalog_fingerprint(catalog)
        ),
        "component_factory": factory_identity(component_factory),
        "verifier_factory": (
            "vapa.verifiers:VerifierCatalog.demo_default"
            if settings.demo_catalogs
            else None
            if verifier_factory is None
            else factory_identity(verifier_factory)
        ),
        "outcome_scorer": factory_identity(outcome_scorer),
        "component_id": components.component_id,
        "initialization_identity": (
            None
            if components.initialization_identity is None
            else dict(components.initialization_identity)
        ),
        "initial_state_sha256": components.initial_state_sha256,
        "inference_spec": (
            None if components.inference_spec is None else dict(components.inference_spec)
        ),
        "actor_fingerprint": components.actor.fingerprint,
        "reference_fingerprint": components.reference.fingerprint,
        "tokenizer_fingerprint": components.tokenizer.fingerprint,
        "rank": context.rank,
        "world_size": context.world_size,
        "seed": config.seed,
        "ratio_clip": settings.ratio_clip,
        "kl_mode": settings.kl_mode,
        "scaffold_sha256": hashlib.sha256(settings.scaffold.encode("utf-8")).hexdigest(),
        "scaffold_size_bytes": len(settings.scaffold.encode("utf-8")),
        "deterministic": settings.deterministic,
        "demo_catalogs": settings.demo_catalogs,
        "allow_non_paper_exact": settings.allow_non_paper_exact,
        "checkpoint_every": settings.checkpoint_every,
        "sampled_token_budget": config.optimization.sampled_token_budget,
        "fork_enabled": config.replay.fork_enabled,
    }
    control_pairing_contract: dict[str, object] = {
        "schema_version": 1,
        "package_version": manifest["package_version"],
        "episodes_sha256": manifest["episodes_sha256"],
        "shard_sha256": manifest["shard_sha256"],
        "environment_config": fingerprint_payload(config.environment),
        "credit_config": fingerprint_payload(config.credit),
        "optimization_config": fingerprint_payload(config.optimization),
        "model_config": fingerprint_payload(config.model),
        "replay_allocation": {
            "base_group_size": config.replay.base_group_size,
            "max_fork_states": config.replay.max_fork_states,
            "siblings_per_fork": config.replay.siblings_per_fork,
            "memory_pressure_fraction": config.replay.memory_pressure_fraction,
        },
        "verifier_manifest_sha256": manifest["verifier_manifest_sha256"],
        "calculator_manifest_sha256": manifest["calculator_manifest_sha256"],
        "environment_spec": manifest["environment_spec"],
        "verifier_catalog_id": manifest["verifier_catalog_id"],
        "verifier_catalog_fingerprint": manifest["verifier_catalog_fingerprint"],
        "component_factory": manifest["component_factory"],
        "verifier_factory": manifest["verifier_factory"],
        "outcome_scorer": manifest["outcome_scorer"],
        "component_id": manifest["component_id"],
        "initialization_identity": manifest["initialization_identity"],
        "initial_state_sha256": manifest["initial_state_sha256"],
        "inference_spec": manifest["inference_spec"],
        "actor_fingerprint": manifest["actor_fingerprint"],
        "reference_fingerprint": manifest["reference_fingerprint"],
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
        "rank": context.rank,
        "world_size": context.world_size,
        "seed": config.seed,
        "ratio_clip": settings.ratio_clip,
        "kl_mode": settings.kl_mode,
        "scaffold_sha256": manifest["scaffold_sha256"],
        "deterministic": settings.deterministic,
        "demo_catalogs": settings.demo_catalogs,
        "allow_non_paper_exact": settings.allow_non_paper_exact,
    }
    manifest["control_pairing_contract"] = control_pairing_contract
    manifest["replay_quota"] = _replay_quota_identity(
        settings.replay_quota_path,
        episode_path=episode_path,
        config=config,
        target_pairing_contract=control_pairing_contract,
    )
    return manifest


def _metric_payload(
    *,
    metric_index: int,
    global_step: int,
    ledger: TokenLedger,
    update: UpdateBatch,
    report: TrainStepReport | None,
    comparison_only: int,
    replay_quota: ReplayQuotaLedger,
) -> dict[str, object]:
    bases = [rollout for instance in update.instances for rollout in instance.base_rollouts]
    branches = [rollout for instance in update.instances for rollout in instance.branch_rollouts]
    return {
        "schema_version": RL_RUN_SCHEMA_VERSION,
        "metric_index": metric_index,
        "global_step": global_step,
        "sampled_tokens": ledger.spent_tokens,
        "target_tokens": ledger.target_tokens,
        "token_overshoot": ledger.overshoot,
        "batch_sampled_tokens": update.sampled_tokens,
        "batch_trainable_tokens": update.trainable_tokens,
        "instances": len(update.instances),
        "comparison_only_instances": comparison_only,
        "base_rollouts": len(bases),
        "base_rollouts_per_instance": [
            len(instance.base_rollouts) for instance in update.instances
        ],
        "branch_rollouts": len(branches),
        "success_rate": sum(item.outcome_reward > 0 for item in bases) / len(bases),
        "curriculum_stages": [stage.name for stage in update.curriculum_stages],
        "branch_sampled_tokens": update.branch_sampled_tokens,
        "replay_quota_balance": replay_quota.balance_tokens,
        "replay_quota_settled_overshoot": replay_quota.settled_overshoot_tokens,
        "loss": None if report is None else report.loss,
        "policy_loss": None if report is None else report.policy_loss,
        "kl": None if report is None else report.kl,
        "action_tokens": 0 if report is None else report.action_tokens,
        "microbatches": 0 if report is None else report.microbatches,
        "gradient_norm": None if report is None else report.gradient_norm,
        "learning_rates": [] if report is None else list(report.learning_rates),
    }


def _sample_matched_allocation_stratum(
    *,
    builder: InstanceBatchBuilder,
    policy: Policy,
    stream: DeterministicEpisodeStream,
    config: ExperimentConfig,
    context: DistributedContext,
    quota_records: Sequence[ReplayQuotaRecord],
    quota_cursor: int,
    sample_index: int,
    builder_counter: int,
    replay_quota: ReplayQuotaLedger,
) -> tuple[list[tuple[InstanceBatch, CurriculumStage]], int, int, int]:
    """Sample base groups, then fairly reallocate one donor quota stratum."""

    if replay_quota.balance_tokens != 0 or replay_quota.curriculum_stage is not None:
        raise TrainingLifecycleError("matched replay quota was not settled between strata")
    first = quota_records[quota_cursor]
    records: list[ReplayQuotaRecord] = []
    for record in quota_records[quota_cursor:]:
        if record.allocation_stratum != first.allocation_stratum:
            break
        records.append(record)
    prepared: list[tuple[Episode, int, InstanceBatch, CurriculumStage]] = []
    for record in records:
        if record.sequence != sample_index or builder_counter != sample_index:
            raise TrainingLifecycleError(
                "realized replay quota does not match the checkpointed sample cursor"
            )
        stage = stage_at(
            record.sampled_tokens_before,
            config.optimization.sampled_token_budget,
        )
        if record.curriculum_stage != stage.name:
            raise TrainingLifecycleError(
                "realized replay quota has an invalid donor curriculum cursor"
            )
        episode = stream.next(max_history_quartile=stage.max_history_quartile)
        source_id = episode.task.instance_id
        expected_occurrence = f"update{builder_counter}:item0:id{len(source_id)}:{source_id}"
        if record.occurrence_id != expected_occurrence:
            raise TrainingLifecycleError(
                "realized replay quota occurrence does not match deterministic episode order"
            )
        if record.episode_sha256 != sha256_json(episode_to_record(episode)):  # type: ignore[arg-type]
            raise TrainingLifecycleError(
                "realized replay quota was produced from different episode bytes"
            )
        seed = _sample_seed(config.seed, context.rank, sample_index, source_id)
        empty_quota = ReplayQuotaLedger(curriculum_stage=stage.name)
        sampled = builder.build_update(
            (episode,),
            policy,
            seed=seed,
            sampled_tokens_so_far=record.sampled_tokens_before,
            history_quartiles={source_id: episode_history_quartile(episode)},
            replay_quota=empty_quota,
        )
        if sampled.curriculum_stage is None or sampled.curriculum_stage.name != stage.name:
            raise AssertionError("matched base sampling changed the donor curriculum stage")
        prepared.append((episode, seed, sampled.instances[0], stage))
        replay_quota.add_replay_tokens(
            record.branch_sampled_tokens,
            curriculum_stage=record.curriculum_stage,
        )
        quota_cursor += 1
        sample_index += 1
        builder_counter += 1

    # A single fair cursor serves every base instance in the stratum.  Therefore
    # rollout counts differ by at most one even when token lengths vary.
    while replay_quota.balance_tokens > 0:
        item_index = replay_quota.next_instance % len(prepared)
        episode, seed, instance, stage = prepared[item_index]
        base_index = len(instance.base_rollouts)
        rollout = builder.runner.run(
            episode,
            policy,
            rollout_id=f"{instance.occurrence_id}:base:{base_index}",
            seed=(seed * 1_000_003) * 1_000_003 + base_index,
            instance_id=instance.occurrence_id,
            allowed_actions=stage.actions,
        )
        if config.credit.use_process_rewards:
            if builder.verifiers is None:
                raise AssertionError("process rewards require the validated verifier catalog")
            builder.verifiers.score_rollout(episode, rollout)
        replay_quota.charge_extra_base(rollout.sampled_tokens)
        replay_quota.next_instance = (item_index + 1) % len(prepared)
        prepared[item_index] = (
            episode,
            seed,
            replace(instance, base_rollouts=instance.base_rollouts + (rollout,), groups=()),
            stage,
        )
    replay_quota.settle_allocation_stratum()

    result: list[tuple[InstanceBatch, CurriculumStage]] = []
    for _, _, instance, stage in prepared:
        groups = (
            assign_step_groups(
                instance.base_rollouts,
                (),
                group_prefix=f"{instance.occurrence_id}:",
            )
            if config.credit.use_step_credit
            else ()
        )
        result.append((replace(instance, groups=groups, advantage_summary=None), stage))
    return result, quota_cursor, sample_index, builder_counter


def _rank_output(output: Path, context: DistributedContext) -> Path:
    return output if context.world_size == 1 else output / f"rank-{context.rank:05d}"


def run_vapa_training(
    config_path: str | Path,
    episode_path: str | Path,
    output_directory: str | Path,
    *,
    component_factory: ComponentFactory | None,
    verifier_manifest: str | Path,
    calculator_manifest: str | Path,
    verifier_factory: VerifierFactory | None = None,
    outcome_scorer: OutcomeScorer | None = None,
    settings: RLRunSettings | None = None,
    distributed: DistributedContext | None = None,
) -> TrainingResult:
    """Run sampling through checkpointing, or validate the exact plan in dry-run mode."""

    settings = settings or RLRunSettings()
    config_file = Path(config_path).resolve()
    episodes_file = Path(episode_path).resolve()
    verifier_file = Path(verifier_manifest).resolve()
    calculator_file = Path(calculator_manifest).resolve()
    replay_quota_file = (
        None if settings.replay_quota_path is None else Path(settings.replay_quota_path).resolve()
    )
    captured_inputs = _capture_training_inputs(
        config_path=config_file,
        episode_path=episodes_file,
        verifier_manifest=verifier_file,
        calculator_manifest=calculator_file,
        replay_quota_path=replay_quota_file,
    )
    config = load_config(config_file)
    episodes = tuple(load_episode_objects(episodes_file))
    context = distributed or DistributedContext.from_environment()
    shard = shard_episodes(episodes, context)
    warnings: list[str] = []
    readiness_issues: list[str] = []
    missing_quartiles: list[str] = []
    missing_sampling_labels: list[str] = []
    for episode in shard:
        try:
            episode_history_quartile(episode)
        except DataValidationError:
            missing_quartiles.append(episode.task.instance_id)
        try:
            episode_sampling_labels(episode)
        except DataValidationError:
            missing_sampling_labels.append(episode.task.instance_id)
    if missing_quartiles:
        readiness_issues.append(
            f"{len(missing_quartiles)} episode(s) lack valid history_quartile metadata"
        )
    if missing_sampling_labels:
        readiness_issues.append(
            f"{len(missing_sampling_labels)} episode(s) lack valid A.7 suite/task_type metadata"
        )
    if not missing_quartiles and not missing_sampling_labels:
        initial_max_quartile = stage_at(
            0,
            config.optimization.sampled_token_budget,
        ).max_history_quartile
        first_stage_suites = {
            episode_sampling_labels(episode)[0]
            for episode in shard
            if episode_history_quartile(episode) <= initial_max_quartile
        }
        if first_stage_suites != {"calculation", "retrieval"}:
            readiness_issues.append(
                "the first curriculum gate must contain both A.7 sampling suites"
            )

    if outcome_scorer is None:
        if settings.demo_catalogs:
            outcome_scorer = exact_outcome_scorer
        else:
            readiness_issues.append("non-demo training requires an explicit author outcome_scorer")
    elif outcome_scorer is exact_outcome_scorer and not settings.demo_catalogs:
        readiness_issues.append(
            "the exact-string outcome scorer is reserved for explicit demo runs"
        )

    catalog, calculators = _validate_catalogs(
        config,
        verifier_manifest=verifier_file,
        calculator_manifest=calculator_file,
        verifier_factory=verifier_factory,
        settings=settings,
    )
    environment_spec = build_environment_spec(config, calculator_file)
    output = _rank_output(Path(output_directory).resolve(), context)
    if component_factory is None:
        readiness_issues.append(
            "neural training requires an immutable model/SFT load plan or component factory"
        )
    if settings.replay_quota_path is not None and not config.replay.reallocate_disabled_forks:
        raise TrainingLifecycleError(
            "replay_quota_path is valid only for a matched no-fork configuration"
        )
    quota_records = (
        ()
        if settings.replay_quota_path is None
        else load_replay_quota_records(settings.replay_quota_path)
    )
    if config.replay.reallocate_disabled_forks and not quota_records:
        readiness_issues.append(
            "matched no-fork training requires replay_quota_path from a completed fork run"
        )
    if settings.replay_quota_path is not None:
        _replay_quota_identity(
            replay_quota_file,
            episode_path=episodes_file,
            config=config,
        )
    _verify_captured_inputs(captured_inputs, phase="input parsing")
    if settings.dry_run:
        return TrainingResult(
            status="dry_run",
            output_directory=output,
            global_step=0,
            sampled_tokens=0,
            trainable_tokens=0,
            metric_records=0,
            checkpoint_path=None,
            episode_count=len(shard),
            rank=context.rank,
            world_size=context.world_size,
            training_ready=not readiness_issues,
            readiness_issues=tuple(readiness_issues),
            warnings=tuple(warnings),
        )
    if readiness_issues:
        raise TrainingLifecycleError("; ".join(readiness_issues))
    output = guard_artifact_write_path(
        output,
        content_kind=settings.content_kind,
    )
    assert component_factory is not None
    assert outcome_scorer is not None
    local_seed = seed_everything(
        config.seed,
        rank=context.rank,
        deterministic=settings.deterministic,
    )
    components = component_factory(config, context)
    if not isinstance(components, TrainingComponents):
        raise TypeError("component factory did not return TrainingComponents")
    if context.world_size > 1 and not components.supports_distributed:
        raise TrainingLifecycleError(
            "the selected component factory does not provide synchronized distributed training"
        )
    if components.policy_scaffold != settings.scaffold:
        raise TrainingLifecycleError(
            "sampling and training scaffolds differ; component_factory must bind the same "
            "scaffold as RLRunSettings"
        )
    _verify_captured_inputs(captured_inputs, phase="component construction")
    manifest = _run_manifest(
        config,
        episodes_file,
        shard,
        captured_inputs=captured_inputs,
        environment_spec=environment_spec,
        catalog=catalog,
        component_factory=component_factory,
        verifier_factory=verifier_factory,
        outcome_scorer=outcome_scorer,
        components=components,
        context=context,
        settings=settings,
    )
    manifest_fingerprint = artifact_fingerprint(manifest)
    contract = CheckpointContract(
        run_id=settings.run_id,
        run_manifest_fingerprint=manifest_fingerprint,
        config_fingerprint=fingerprint_payload(config),
        model_fingerprint=components.actor.fingerprint,
        reference_model_fingerprint=components.reference.fingerprint,
        tokenizer_fingerprint=components.tokenizer.fingerprint,
        optimizer_name=components.optimizer_name,
        scheduler_name=components.scheduler_name,
        state_format=components.store.format_name,
        world_size=context.world_size,
    )
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    quota_output = output / "replay_quota.jsonl"
    if settings.resume_from is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                "training output already contains run artifacts; pass resume_from or use a new path"
            )
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_text(manifest_path, canonical_json_dumps(manifest) + "\n")
    else:
        if not manifest_path.is_file():
            raise TrainingLifecycleError("resume output is missing run_manifest.json")
        existing = strict_json_loads(manifest_path.read_bytes())
        if existing != manifest:
            raise TrainingLifecycleError("resume run manifest does not match the checkpoint run")

    def manager_factory(episode: Episode) -> StateManager:
        environment = config.environment
        return StateManager(
            episode,
            memory_capacity=environment.memory_capacity,
            action_budget=environment.action_budget,
            turn_cap=environment.turn_cap,
            calculators=calculators,
        )

    runner = RolloutRunner(manager_factory, outcome_scorer=outcome_scorer)
    builder = InstanceBatchBuilder(config, runner, catalog)
    rollout_policy = ContextLimitedPolicy(
        components.policy,
        components.tokenizer,
        scaffold=settings.scaffold,
        context_tokens=config.environment.context_tokens,
        reserved_generation_tokens=config.environment.max_turn_tokens,
    )
    ledger = TokenLedger(
        config.optimization.sampled_token_budget,
        config.optimization.update_token_floor,
        permit_post_target=config.replay.reallocate_disabled_forks,
    )
    replay_quota = ReplayQuotaLedger()
    stream_state = {"cycle": 0, "position": 0}
    sample_index = replay_quota_cursor = metric_records = builder_counter = 0
    global_step = trainable_tokens = 0
    last_checkpoint: Path | None = None
    last_checkpoint_state: tuple[int, int] | None = None
    if settings.resume_from is not None:
        runtime = read_manifest(settings.resume_from).runtime
        if runtime.seed != local_seed:
            raise TrainingLifecycleError("checkpoint rank-local seed does not match this run")
        (
            stream_state,
            sample_index,
            replay_quota,
            replay_quota_cursor,
            metric_records,
            builder_counter,
        ) = _restore_extra(runtime, settings=settings)
        if runtime.extra is None or runtime.extra.get("environment_spec") != environment_spec:
            raise TrainingLifecycleError(
                "checkpoint environment_spec does not match the configured training MDP"
            )
        last_checkpoint = Path(settings.resume_from).resolve()
        _require_latest_resume_checkpoint(output, last_checkpoint, runtime)
        loaded_runtime = resume_checkpoint(
            settings.resume_from,
            expected=contract,
            model=components.actor,
            reference=components.reference,
            tokenizer=components.tokenizer,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
            store=components.store,
        )
        if loaded_runtime != runtime:
            raise TrainingLifecycleError("checkpoint runtime changed during resume validation")
        ledger.spent_tokens = runtime.sampled_tokens
        global_step = runtime.global_step
        trainable_tokens = runtime.trainable_tokens
        last_checkpoint_state = (global_step, ledger.spent_tokens)
        _reconcile_journal(metrics_path, expected=metric_records, kind="metrics")
        if config.replay.fork_enabled:
            _reconcile_journal(
                quota_output,
                expected=sample_index,
                kind="replay_quota",
            )
            resumed_attestation = _replay_quota_attestation(
                quota_output,
                expected_records=sample_index,
            )
            if runtime.extra.get("replay_quota_attestation") != resumed_attestation:
                raise TrainingLifecycleError(
                    "checkpoint replay quota attestation does not match its committed journal"
                )
    stream = DeterministicEpisodeStream(
        shard,
        seed=config.seed,
        rank=context.rank,
        cycle=stream_state["cycle"],
        position=stream_state["position"],
    )
    # Occurrence IDs are part of comparison-group identity and must survive resume.
    builder._update_counter = builder_counter

    pending_instances: list[InstanceBatch] = []
    pending_stages: list[CurriculumStage] = []
    matched_ready: list[tuple[InstanceBatch, CurriculumStage]] = []
    status = "complete"
    while not ledger.complete or (
        config.replay.reallocate_disabled_forks
        and (replay_quota_cursor < len(quota_records) or matched_ready)
    ):
        if (
            not matched_ready
            and settings.max_optimizer_steps is not None
            and global_step >= settings.max_optimizer_steps
        ):
            status = "max_optimizer_steps"
            break
        if config.replay.reallocate_disabled_forks:
            if not matched_ready:
                if replay_quota_cursor >= len(quota_records):
                    raise TrainingLifecycleError(
                        "realized replay quota was exhausted before the sampled-token target"
                    )
                (
                    matched_ready,
                    replay_quota_cursor,
                    sample_index,
                    builder_counter,
                ) = _sample_matched_allocation_stratum(
                    builder=builder,
                    policy=rollout_policy,
                    stream=stream,
                    config=config,
                    context=context,
                    quota_records=quota_records,
                    quota_cursor=replay_quota_cursor,
                    sample_index=sample_index,
                    builder_counter=builder_counter,
                    replay_quota=replay_quota,
                )
            instance, stage = matched_ready.pop(0)
        else:
            stage = stage_at(ledger.spent_tokens, ledger.target_tokens)
            episode = stream.next(max_history_quartile=stage.max_history_quartile)
            seed = _sample_seed(
                config.seed,
                context.rank,
                sample_index,
                episode.task.instance_id,
            )
            sampled_update = builder.build_update(
                (episode,),
                rollout_policy,
                seed=seed,
                sampled_tokens_so_far=ledger.spent_tokens,
                history_quartiles={episode.task.instance_id: episode_history_quartile(episode)},
            )
            instance = sampled_update.instances[0]
            if config.replay.fork_enabled:
                realized = ReplayQuotaRecord(
                    sequence=sample_index,
                    curriculum_stage=stage.name,
                    allocation_stratum=f"{stage.name}:update{len(ledger.updates)}",
                    sampled_tokens_before=ledger.spent_tokens,
                    branch_sampled_tokens=sampled_update.branch_sampled_tokens,
                    occurrence_id=instance.occurrence_id,
                    episode_sha256=sha256_json(
                        episode_to_record(episode)  # type: ignore[arg-type]
                    ),
                )
                _append_jsonl(quota_output, realized.to_dict())
            sample_index += 1
            builder_counter += 1
        if instance.sampled_tokens <= 0:
            raise TrainingLifecycleError("a sampled instance produced no actor tokens")
        pending_instances.append(instance)
        pending_stages.append(stage)
        closed = ledger.add(
            GroupCharge(
                instance.occurrence_id,
                instance.sampled_tokens,
                instance.trainable_tokens,
            )
        )
        if not closed:
            continue
        charges = ledger.updates[-1]
        if len(charges) != len(pending_instances):
            raise AssertionError("token ledger and pending instance groups diverged")
        comparison_only = 0
        loss_instances: list[InstanceBatch] = []
        loss_stages: list[CurriculumStage] = []
        for charge, member, member_stage in zip(
            charges,
            pending_instances,
            pending_stages,
            strict=True,
        ):
            if not charge.include_in_loss:
                comparison_only += 1
                _mask_comparison_only(member)
            else:
                loss_instances.append(member)
                loss_stages.append(member_stage)
        loss_update = (
            None
            if not loss_instances
            else _combine_and_normalize(loss_instances, loss_stages, config)
        )
        update = UpdateBatch(
            tuple(pending_instances),
            None if loss_update is None else loss_update.advantage_summary,
            tuple(pending_stages),
        )
        report: TrainStepReport | None = None
        if loss_update is not None and loss_update.trainable_tokens:
            if isinstance(components.scheduler, SampledTokenCosineScheduler):
                components.scheduler.prepare(ledger.spent_tokens)
            seed_everything(
                _optimizer_seed(
                    config.seed,
                    context.rank,
                    global_step,
                    ledger.spent_tokens,
                ),
                deterministic=settings.deterministic,
            )
            report = train_vapa_update(
                loss_update,
                tokenizer=components.tokenizer,
                actor=components.actor,
                reference=components.reference,
                optimizer=components.optimizer,
                scheduler=components.scheduler,
                kl_weight=config.optimization.kl_weight,
                ratio_clip=settings.ratio_clip,
                kl_mode=settings.kl_mode,
                gradient_clip=config.optimization.gradient_clip,
                scaffold=settings.scaffold,
            )
            global_step += 1
            trainable_tokens += report.action_tokens
        metric = _metric_payload(
            metric_index=metric_records,
            global_step=global_step,
            ledger=ledger,
            update=update,
            report=report,
            comparison_only=comparison_only,
            replay_quota=replay_quota,
        )
        _append_jsonl(metrics_path, metric)
        metric_records += 1
        pending_instances.clear()
        pending_stages.clear()
        quota_attestation = (
            _replay_quota_attestation(quota_output, expected_records=sample_index)
            if config.replay.fork_enabled
            else None
        )
        extra = _runtime_extra(
            settings=settings,
            stream=stream,
            sample_index=sample_index,
            replay_quota=replay_quota,
            replay_quota_cursor=replay_quota_cursor,
            metric_records=metric_records,
            builder_counter=builder_counter,
            inference_spec=components.inference_spec,
            environment_spec=environment_spec,
            replay_quota_attestation=quota_attestation,
        )
        runtime = RuntimeState(
            global_step,
            ledger.spent_tokens,
            trainable_tokens,
            local_seed,
            extra,
        )
        if (
            report is not None
            and not matched_ready
            and global_step % settings.checkpoint_every == 0
        ):
            last_checkpoint = _save_runtime_checkpoint(
                output,
                label="periodic",
                contract=contract,
                runtime=runtime,
                components=components,
            )
            last_checkpoint_state = (global_step, ledger.spent_tokens)

    if pending_instances:
        raise AssertionError("training stopped with a partial token-ledger update")
    if matched_ready:
        raise AssertionError("training stopped with an uncheckpointed allocation stratum")
    if status == "complete" and config.replay.reallocate_disabled_forks:
        if replay_quota_cursor != len(quota_records):
            raise TrainingLifecycleError(
                "matched no-fork run completed without consuming the full replay quota"
            )
        if replay_quota.balance_tokens != 0 or replay_quota.curriculum_stage is not None:
            raise TrainingLifecycleError("matched no-fork run left an unsettled replay quota")
    final_quota_attestation = (
        _replay_quota_attestation(quota_output, expected_records=sample_index)
        if config.replay.fork_enabled
        else None
    )
    extra = _runtime_extra(
        settings=settings,
        stream=stream,
        sample_index=sample_index,
        replay_quota=replay_quota,
        replay_quota_cursor=replay_quota_cursor,
        metric_records=metric_records,
        builder_counter=builder_counter,
        inference_spec=components.inference_spec,
        environment_spec=environment_spec,
        replay_quota_attestation=final_quota_attestation,
    )
    final_runtime = RuntimeState(
        global_step,
        ledger.spent_tokens,
        trainable_tokens,
        local_seed,
        extra,
    )
    state_key = (global_step, ledger.spent_tokens)
    if last_checkpoint_state != state_key:
        last_checkpoint = _save_runtime_checkpoint(
            output,
            label="final" if status == "complete" else "stopped",
            contract=contract,
            runtime=final_runtime,
            components=components,
        )
    result = TrainingResult(
        status=status,
        output_directory=output,
        global_step=global_step,
        sampled_tokens=ledger.spent_tokens,
        trainable_tokens=trainable_tokens,
        metric_records=metric_records,
        checkpoint_path=last_checkpoint,
        episode_count=len(shard),
        rank=context.rank,
        world_size=context.world_size,
        training_ready=True,
        warnings=tuple(warnings),
        replay_quota_attestation=final_quota_attestation,
    )
    atomic_write_text(
        output / "result.json",
        canonical_json_dumps(result.to_dict()) + "\n",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the installed ``vapa-train`` command-line surface."""

    parser = argparse.ArgumentParser(description="Run the VAPA rollout-to-checkpoint RL lifecycle.")
    parser.add_argument("config", type=Path, help="experiment TOML")
    parser.add_argument("episodes", type=Path, help="prepared training JSON or JSONL")
    parser.add_argument("output", type=Path, help="run output directory")
    parser.add_argument("--run-id", default="vapa")
    parser.add_argument(
        "--content-kind",
        choices=tuple(item.value for item in ArtifactContentKind),
        default=ArtifactContentKind.CREDENTIALED.value,
        help="artifact disclosure class used to guard the output path",
    )
    parser.add_argument("--component-factory", help="custom module:callable backend factory")
    parser.add_argument("--verifier-factory", help="author verifier module:callable factory")
    parser.add_argument("--verifier-manifest", type=Path)
    parser.add_argument("--calculator-manifest", type=Path)
    parser.add_argument("--outcome-scorer", help="optional module:callable outcome scorer")
    parser.add_argument("--resume", type=Path, help="strict checkpoint directory")
    parser.add_argument("--replay-quota", type=Path, help="fork-run replay_quota.jsonl")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--scaffold", default="")
    parser.add_argument(
        "--ratio-clip", type=float, help="legacy option; rejected by the revised Eq. 9 objective"
    )
    parser.add_argument("--kl-mode", choices=("forward",), default="forward")
    parser.add_argument("--dry-run", action="store_true", help="validate without ML imports")
    parser.add_argument(
        "--demo-catalogs",
        action="store_true",
        help="use explicitly non-paper-exact synthetic verifier/calculator artifacts",
    )
    parser.add_argument(
        "--allow-non-paper-exact",
        action="store_true",
        help="permit explicitly supplied non-paper-exact research catalogs",
    )
    parser.add_argument("--nondeterministic", action="store_true")
    model = parser.add_argument_group("built-in Transformers backend")
    model.add_argument("--model-revision", help="immutable commit revision")
    model.add_argument("--tokenizer-name")
    model.add_argument("--tokenizer-revision")
    model.add_argument("--model-kind", choices=("auto", "multimodal", "causal"), default="auto")
    model.add_argument("--use-processor", choices=("yes", "no"))
    model.add_argument("--device", default="auto")
    model.add_argument(
        "--sft-checkpoint",
        type=Path,
        help="verified SFT checkpoint used for actor and frozen KL reference",
    )
    model.add_argument(
        "--allow-base-model-init",
        action="store_true",
        help="explicit ablation: initialize RL from the base model instead of SFT",
    )
    model.add_argument("--local-files-only", action="store_true")
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument(
        "--dequantize-mxfp4",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="request GPT-OSS MXFP4 dequantization; must match the experiment config",
    )
    model.add_argument("--no-lora", action="store_true")
    model.add_argument(
        "--lora-target-module",
        action="append",
        default=[],
        help="repeat for each author-selected module name",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the installed CLI without importing optional ML dependencies for dry-run."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.demo_catalogs:

        def demo_input(explicit: Path | None, name: str, flag: str) -> Path:
            if explicit is not None:
                return explicit
            for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
                candidate = root / "examples" / name
                if candidate.is_file():
                    return candidate
            parser.error(f"demo input {name!r} not found; supply {flag} explicitly")

        verifier_manifest = demo_input(
            arguments.verifier_manifest, "demo_verifier_catalog.json", "--verifier-manifest"
        )
        calculator_manifest = demo_input(
            arguments.calculator_manifest, "tiny_calculators.json", "--calculator-manifest"
        )
    else:
        if arguments.verifier_manifest is None or arguments.calculator_manifest is None:
            parser.error("non-demo runs require --verifier-manifest and --calculator-manifest")
        verifier_manifest = arguments.verifier_manifest
        calculator_manifest = arguments.calculator_manifest
    verifier_factory = (
        None if arguments.verifier_factory is None else import_factory(arguments.verifier_factory)
    )
    outcome_scorer = (
        None if arguments.outcome_scorer is None else import_factory(arguments.outcome_scorer)
    )
    if arguments.component_factory is not None:
        component_factory = import_factory(arguments.component_factory)
    elif arguments.dry_run and arguments.model_revision is None:
        component_factory = None
    else:
        if arguments.model_revision is None:
            parser.error("the built-in backend requires --model-revision")
        if arguments.use_processor is None:
            parser.error("the built-in backend requires explicit --use-processor yes|no")
        use_processor = {"yes": True, "no": False}[arguments.use_processor]
        options = TransformersLoadOptions(
            model_revision=arguments.model_revision,
            tokenizer_name=arguments.tokenizer_name,
            tokenizer_revision=arguments.tokenizer_revision,
            model_kind=arguments.model_kind,
            use_processor=use_processor,
            device=arguments.device,
            local_files_only=arguments.local_files_only,
            trust_remote_code=arguments.trust_remote_code,
            dequantize_mxfp4=arguments.dequantize_mxfp4,
            lora_enabled=not arguments.no_lora,
            lora_target_modules=tuple(arguments.lora_target_module),
            sft_checkpoint=arguments.sft_checkpoint,
            allow_base_initialization=arguments.allow_base_model_init,
            scaffold=arguments.scaffold,
        )
        if arguments.dry_run:
            validate_transformers_load_plan(load_config(arguments.config), options)
        component_factory = functools.partial(build_transformers_components, options=options)
    settings = RLRunSettings(
        run_id=arguments.run_id,
        checkpoint_every=arguments.checkpoint_every,
        max_optimizer_steps=arguments.max_optimizer_steps,
        resume_from=arguments.resume,
        replay_quota_path=arguments.replay_quota,
        scaffold=arguments.scaffold,
        ratio_clip=arguments.ratio_clip,
        kl_mode=arguments.kl_mode,
        deterministic=not arguments.nondeterministic,
        dry_run=arguments.dry_run,
        demo_catalogs=arguments.demo_catalogs,
        allow_non_paper_exact=arguments.allow_non_paper_exact,
        content_kind=arguments.content_kind,
    )
    result = run_vapa_training(
        arguments.config,
        arguments.episodes,
        arguments.output,
        component_factory=component_factory,
        verifier_manifest=verifier_manifest,
        calculator_manifest=calculator_manifest,
        verifier_factory=verifier_factory,
        outcome_scorer=outcome_scorer,
        settings=settings,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
