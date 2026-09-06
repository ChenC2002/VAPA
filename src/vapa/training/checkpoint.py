"""Atomic, checksummed checkpoint save/resume with an explicit compatibility contract."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from vapa.artifacts import (
    artifact_fingerprint,
    canonical_json_dumps,
    fingerprint_file,
    strict_json_loads,
)
from vapa.model.protocols import (
    ActorModelAdapter,
    OptimizerAdapter,
    OptionalDependencyError,
    SchedulerAdapter,
    TokenizerAdapter,
)

CHECKPOINT_FORMAT_VERSION = 1


def fingerprint_payload(payload: object) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible configuration data."""

    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    try:
        return artifact_fingerprint(payload)
    except (TypeError, ValueError) as error:
        raise TypeError("fingerprint payload must be finite JSON-compatible data") from error


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class CheckpointContract:
    """Fields that must match before any saved state is applied."""

    run_id: str
    run_manifest_fingerprint: str
    config_fingerprint: str
    model_fingerprint: str
    reference_model_fingerprint: str
    tokenizer_fingerprint: str
    optimizer_name: str
    scheduler_name: str
    state_format: str
    world_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "run_manifest_fingerprint",
            "config_fingerprint",
            "model_fingerprint",
            "reference_model_fingerprint",
            "tokenizer_fingerprint",
            "optimizer_name",
            "scheduler_name",
            "state_format",
        ):
            _nonempty(getattr(self, name), name)
        for name in (
            "run_manifest_fingerprint",
            "config_fingerprint",
            "model_fingerprint",
            "reference_model_fingerprint",
            "tokenizer_fingerprint",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TypeError("world_size must be an integer")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")


@dataclass(frozen=True)
class RuntimeState:
    global_step: int
    sampled_tokens: int
    trainable_tokens: int
    seed: int
    extra: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for name in ("global_step", "sampled_tokens", "trainable_tokens", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.extra is not None:
            try:
                canonical_json_dumps(self.extra)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "runtime extra state must be finite JSON-compatible data"
                ) from error

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "sampled_tokens": self.sampled_tokens,
            "trainable_tokens": self.trainable_tokens,
            "seed": self.seed,
            "extra": dict(self.extra or {}),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RuntimeState:
        expected = {"global_step", "sampled_tokens", "trainable_tokens", "seed", "extra"}
        if set(raw) != expected:
            raise ValueError("runtime state has an incompatible schema")
        extra = raw["extra"]
        if not isinstance(extra, Mapping):
            raise ValueError("runtime extra state must be an object")
        integers = [
            raw[name] for name in ("global_step", "sampled_tokens", "trainable_tokens", "seed")
        ]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("runtime counters must be integers")
        return cls(
            global_step=integers[0],
            sampled_tokens=integers[1],
            trainable_tokens=integers[2],
            seed=integers[3],
            extra=dict(extra),
        )


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        candidate = Path(self.path)
        if candidate.name != self.path or candidate.is_absolute():
            raise ValueError("checkpoint file records must use a safe basename")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("checkpoint file checksum must be lowercase SHA-256")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("checkpoint file size must be nonnegative")


@dataclass(frozen=True)
class CheckpointManifest:
    format_version: int
    created_at: str
    contract: CheckpointContract
    runtime: RuntimeState
    files: Mapping[str, FileRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "created_at": self.created_at,
            "contract": asdict(self.contract),
            "runtime": self.runtime.to_dict(),
            "files": {name: asdict(record) for name, record in sorted(self.files.items())},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CheckpointManifest:
        if set(raw) != {"format_version", "created_at", "contract", "runtime", "files"}:
            raise ValueError("checkpoint manifest has an incompatible schema")
        if raw["format_version"] != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format version")
        created_at = raw["created_at"]
        if not isinstance(created_at, str):
            raise ValueError("checkpoint created_at must be a string")
        try:
            parsed_created_at = datetime.fromisoformat(created_at)
        except ValueError as error:
            raise ValueError("checkpoint created_at is not ISO-8601") from error
        if parsed_created_at.tzinfo is None:
            raise ValueError("checkpoint created_at must include a timezone")
        contract_raw = raw["contract"]
        runtime_raw = raw["runtime"]
        files_raw = raw["files"]
        if not all(isinstance(value, Mapping) for value in (contract_raw, runtime_raw, files_raw)):
            raise ValueError("checkpoint manifest objects are malformed")
        contract_keys = {
            "run_id",
            "run_manifest_fingerprint",
            "config_fingerprint",
            "model_fingerprint",
            "reference_model_fingerprint",
            "tokenizer_fingerprint",
            "optimizer_name",
            "scheduler_name",
            "state_format",
            "world_size",
        }
        if set(contract_raw) != contract_keys:
            raise ValueError("checkpoint contract has an incompatible schema")
        contract = CheckpointContract(**dict(contract_raw))
        runtime = RuntimeState.from_dict(runtime_raw)
        files: dict[str, FileRecord] = {}
        for name, record in files_raw.items():
            if not isinstance(name, str) or not isinstance(record, Mapping):
                raise ValueError("checkpoint file table is malformed")
            if set(record) != {"path", "sha256", "size"}:
                raise ValueError("checkpoint file record has an incompatible schema")
            files[name] = FileRecord(**dict(record))
        if "model" not in files or "optimizer" not in files:
            raise ValueError("checkpoint is missing required model or optimizer state")
        return cls(CHECKPOINT_FORMAT_VERSION, created_at, contract, runtime, files)


class StateStore(Protocol):
    @property
    def format_name(self) -> str: ...

    @property
    def extension(self) -> str: ...

    def save(self, state: Mapping[str, Any], path: Path) -> None: ...

    def load(self, path: Path) -> Mapping[str, Any]: ...


class JsonStateStore:
    """Safe dependency-free store for synthetic and JSON-native adapters."""

    format_name = "json-v1"
    extension = ".json"

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        try:
            encoded = canonical_json_dumps(state)
        except (TypeError, ValueError) as error:
            raise TypeError("JSON state store received non-JSON state") from error
        path.write_text(encoded + "\n", encoding="utf-8")

    def load(self, path: Path) -> Mapping[str, Any]:
        raw = strict_json_loads(path.read_bytes())
        if not isinstance(raw, Mapping):
            raise ValueError("stored state must be an object")
        return raw


class TorchStateStore:
    """Tensor-capable store using restricted ``torch.load(weights_only=True)``."""

    format_name = "torch-weights-only-v1"
    extension = ".pt"

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as error:
            raise OptionalDependencyError(
                "torch checkpoint storage requires the 'train' extra"
            ) from error
        return torch

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        self._torch().save(dict(state), path)

    def load(self, path: Path) -> Mapping[str, Any]:
        raw = self._torch().load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, Mapping):
            raise ValueError("stored torch state must be a mapping")
        return raw


def _sha256(path: Path) -> str:
    return fingerprint_file(path).sha256


def _record(path: Path) -> FileRecord:
    return FileRecord(path.name, _sha256(path), path.stat().st_size)


def save_checkpoint(
    path: str | Path,
    *,
    contract: CheckpointContract,
    runtime: RuntimeState,
    model: ActorModelAdapter,
    reference: ActorModelAdapter,
    tokenizer: TokenizerAdapter,
    optimizer: OptimizerAdapter,
    scheduler: SchedulerAdapter | None = None,
    store: StateStore | None = None,
) -> CheckpointManifest:
    """Write a new atomic checkpoint directory; existing checkpoints are never overwritten."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    selected = store or TorchStateStore()
    if contract.state_format != selected.format_name:
        raise ValueError("checkpoint contract state_format does not match the state store")
    if contract.model_fingerprint != model.fingerprint:
        raise ValueError("checkpoint contract does not match the actor model")
    if contract.reference_model_fingerprint != reference.fingerprint:
        raise ValueError("checkpoint contract does not match the reference model")
    if contract.tokenizer_fingerprint != tokenizer.fingerprint:
        raise ValueError("checkpoint contract does not match the tokenizer")
    if (contract.scheduler_name == "none") != (scheduler is None):
        raise ValueError("scheduler_name must be 'none' exactly when no scheduler is supplied")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        files: dict[str, FileRecord] = {}
        for name, state in (
            ("model", model.state_dict()),
            ("optimizer", optimizer.state_dict()),
        ):
            state_path = temporary / f"{name}{selected.extension}"
            selected.save(state, state_path)
            files[name] = _record(state_path)
        if scheduler is not None:
            scheduler_path = temporary / f"scheduler{selected.extension}"
            selected.save(scheduler.state_dict(), scheduler_path)
            files["scheduler"] = _record(scheduler_path)
        runtime_path = temporary / "runtime.json"
        runtime_path.write_text(
            canonical_json_dumps(runtime.to_dict()) + "\n",
            encoding="utf-8",
        )
        files["runtime"] = _record(runtime_path)
        manifest = CheckpointManifest(
            format_version=CHECKPOINT_FORMAT_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            contract=contract,
            runtime=runtime,
            files=files,
        )
        (temporary / "manifest.json").write_text(
            canonical_json_dumps(manifest.to_dict()) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_manifest(path: str | Path) -> CheckpointManifest:
    checkpoint = Path(path)
    manifest_path = checkpoint / "manifest.json"
    if checkpoint.is_symlink() or manifest_path.is_symlink():
        raise ValueError("checkpoint and manifest paths cannot be symbolic links")
    raw = strict_json_loads(manifest_path.read_bytes())
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint manifest must be an object")
    return CheckpointManifest.from_dict(raw)


def _verify_files(checkpoint: Path, manifest: CheckpointManifest) -> None:
    for name, record in manifest.files.items():
        state_path = checkpoint / record.path
        if state_path.is_symlink():
            raise ValueError(f"checkpoint {name} file cannot be a symbolic link")
        if not state_path.is_file():
            raise FileNotFoundError(f"checkpoint {name} file is missing")
        if state_path.stat().st_size != record.size or _sha256(state_path) != record.sha256:
            raise ValueError(f"checkpoint {name} file failed integrity verification")


def resume_checkpoint(
    path: str | Path,
    *,
    expected: CheckpointContract,
    model: ActorModelAdapter,
    reference: ActorModelAdapter,
    tokenizer: TokenizerAdapter,
    optimizer: OptimizerAdapter,
    scheduler: SchedulerAdapter | None = None,
    store: StateStore | None = None,
) -> RuntimeState:
    """Verify the contract/checksums and deserialize every state before applying it.

    Callers must preflight objective-specific runtime cursor semantics first.  A backend
    ``load_state_dict`` exception is fatal to the current process; retry resume with
    freshly constructed runtime objects.
    """

    checkpoint = Path(path)
    manifest = read_manifest(checkpoint)
    if manifest.contract != expected:
        raise ValueError("checkpoint contract does not match the requested run")
    if expected.model_fingerprint != model.fingerprint:
        raise ValueError("checkpoint contract does not match the actor model")
    if expected.reference_model_fingerprint != reference.fingerprint:
        raise ValueError("checkpoint contract does not match the reference model")
    if expected.tokenizer_fingerprint != tokenizer.fingerprint:
        raise ValueError("checkpoint contract does not match the tokenizer")
    selected = store or TorchStateStore()
    if selected.format_name != expected.state_format:
        raise ValueError("checkpoint state_format does not match the selected state store")
    if (expected.scheduler_name == "none") != (scheduler is None):
        raise ValueError("resume scheduler does not match the checkpoint contract")
    if ("scheduler" in manifest.files) != (scheduler is not None):
        raise ValueError("checkpoint scheduler state does not match the runtime")
    _verify_files(checkpoint, manifest)
    runtime_record = manifest.files.get("runtime")
    if runtime_record is None:
        raise ValueError("checkpoint is missing runtime state")
    runtime_raw = strict_json_loads((checkpoint / runtime_record.path).read_bytes())
    if not isinstance(runtime_raw, Mapping):
        raise ValueError("runtime state must be an object")
    runtime = RuntimeState.from_dict(runtime_raw)
    if runtime != manifest.runtime:
        raise ValueError("manifest and runtime state disagree")

    # All reads happen before the first load_state_dict call, avoiding mutation on
    # corrupt or missing files. Backend application itself is not generically atomic.
    model_state = selected.load(checkpoint / manifest.files["model"].path)
    optimizer_state = selected.load(checkpoint / manifest.files["optimizer"].path)
    scheduler_state = (
        selected.load(checkpoint / manifest.files["scheduler"].path)
        if scheduler is not None
        else None
    )
    _verify_files(checkpoint, manifest)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        assert scheduler_state is not None
        scheduler.load_state_dict(scheduler_state)
    return runtime
