"""Dependency-free artifact integrity and reproducibility contracts.

The helpers in this module deliberately operate on small, explicit metadata
objects.  They do not inspect patient records and they never infer that a
credentialed artifact is safe to publish from its contents.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn

RUN_MANIFEST_SCHEMA_VERSION = "1.0"
CHECKPOINT_CONTRACT_SCHEMA_VERSION = "1.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_PRIVATE_DIRECTORIES = frozenset(
    {"artifacts", "checkpoints", "data", "results", "runs", "wandb"}
)
_PUBLIC_RESULT_SNAPSHOTS = frozenset(
    {
        "results/demo_results.json",
        "logs/demo_results.jsonl",
        "results/paper_results.json",
        "logs/paper_results.jsonl",
        "results/training_results.json",
        "logs/training_results.jsonl",
    }
)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_json_constant(value)
    return parsed


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys and every non-finite number."""

    return json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
    )


def strict_jsonl_loads(
    payload: str, *, source: str = "<jsonl>", allow_empty: bool = False
) -> list[Any]:
    """Parse LF/CRLF-delimited JSON without splitting Unicode inside strings.

    Preserve physical line numbers and reject blank records. A single final
    newline is optional; an empty journal is allowed only when requested.
    """

    if not payload and not allow_empty:
        raise ValueError(f"{source}: empty JSONL input")
    records = []
    for line_number, line in enumerate(io.StringIO(payload), start=1):
        location = f"{source}:{line_number}"
        if not line.strip():
            raise ValueError(f"{location}: blank JSONL line")
        try:
            records.append(strict_json_loads(line))
        except ValueError as error:
            raise ValueError(f"{location}: invalid strict JSON: {error}") from error
    return records


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains NaN or Infinity")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains non-string object key {key!r}")
            _validate_json_value(item, f"{location}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    raise TypeError(f"{location} contains non-JSON value of type {type(value).__name__}")


def canonical_json_dumps(value: Any) -> str:
    """Encode a JSON value deterministically as compact, key-sorted UTF-8 text."""

    _validate_json_value(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation as UTF-8 bytes, without a newline."""

    return canonical_json_dumps(value).encode("utf-8")


def artifact_fingerprint(value: Any) -> str:
    """Return a SHA-256 fingerprint of an in-memory canonical JSON artifact."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_sha256(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ArtifactFingerprint:
    """Content identity for a file without embedding a machine-specific path."""

    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_sha256(self.sha256, "sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactFingerprint:
        expected = {"sha256", "size_bytes"}
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {', '.join(unexpected)}")
            raise ValueError("invalid artifact fingerprint: " + "; ".join(details))
        return cls(sha256=value["sha256"], size_bytes=value["size_bytes"])


def fingerprint_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> ArtifactFingerprint:
    """Hash a file in bounded-memory chunks and record its exact byte length."""

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    size_bytes = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return ArtifactFingerprint(sha256=digest.hexdigest(), size_bytes=size_bytes)


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    return fingerprint_file(path).sha256


def atomic_write_text(path: str | Path, content: str, *, overwrite: bool = True) -> Path:
    """Publish complete UTF-8 bytes; preserve the existing file if writing fails.

    A no-overwrite publication uses a hard link so another writer cannot race the
    existence check. Callers publishing a set of files write their manifest last.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_output_paths(
    outputs: Sequence[Path], *, inputs: Sequence[Path] = (), overwrite: bool = False
) -> None:
    """Reject input/output aliases and accidental replacement before publishing files."""

    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")
    for output in resolved:
        for source in inputs:
            if output == source.resolve() or (
                output.exists() and source.exists() and output.samefile(source)
            ):
                raise ValueError("an output path aliases an input file")
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}; use overwrite explicitly")
        if output.exists() and not output.is_file():
            raise ValueError(f"output is not a regular file: {output}")


def repository_root_for_output(path: str | Path) -> Path | None:
    """Find the destination's repository, without treating an arbitrary cwd as one."""
    destination = Path(path).expanduser().resolve(strict=False)
    for candidate in (destination, *destination.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
            return candidate
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunManifest:
    """Immutable identity of the inputs and implementation used for one run."""

    config_sha256: str
    data_sha256: str
    verifier_sha256: str
    model_revision: str
    seed: int
    code_version: str
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported run manifest schema_version "
                f"{self.schema_version!r}; expected {RUN_MANIFEST_SCHEMA_VERSION!r}"
            )
        _validate_sha256(self.config_sha256, "config_sha256")
        _validate_sha256(self.data_sha256, "data_sha256")
        _validate_sha256(self.verifier_sha256, "verifier_sha256")
        for name, value in (
            ("model_revision", self.model_revision),
            ("code_version", self.code_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256,
            "verifier_sha256": self.verifier_sha256,
            "model_revision": self.model_revision,
            "seed": self.seed,
            "code_version": self.code_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        if not isinstance(value, Mapping):
            raise TypeError("run manifest must be a JSON object")
        expected = {
            "schema_version",
            "config_sha256",
            "data_sha256",
            "verifier_sha256",
            "model_revision",
            "seed",
            "code_version",
        }
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {', '.join(unexpected)}")
            raise ValueError("invalid run manifest: " + "; ".join(details))
        return cls(
            schema_version=value["schema_version"],
            config_sha256=value["config_sha256"],
            data_sha256=value["data_sha256"],
            verifier_sha256=value["verifier_sha256"],
            model_revision=value["model_revision"],
            seed=value["seed"],
            code_version=value["code_version"],
        )


def save_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Atomically save a run manifest as canonical JSON plus one final newline."""

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    content = canonical_json_dumps(manifest.to_dict()) + "\n"
    return atomic_write_text(path, content)


def load_manifest(path: str | Path) -> RunManifest:
    """Load and validate a strictly encoded run manifest."""

    value = strict_json_loads(Path(path).read_bytes())
    if not isinstance(value, Mapping):
        raise TypeError("run manifest must be a JSON object")
    return RunManifest.from_dict(value)


def build_checkpoint_contract(
    manifest: RunManifest,
    *,
    artifacts: Mapping[str, str | Path | ArtifactFingerprint] | None = None,
) -> dict[str, Any]:
    """Build JSON-safe checkpoint identity metadata from a run and its files.

    ``artifacts`` maps stable logical names to either file paths or precomputed
    :class:`ArtifactFingerprint` values.  Absolute paths are intentionally not
    stored in the contract.
    """

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    if artifacts is not None and not isinstance(artifacts, Mapping):
        raise TypeError("artifacts must map logical names to paths or fingerprints")
    artifact_items = list((artifacts or {}).items())
    for name, _ in artifact_items:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("artifact names must be non-empty strings")
    artifact_items.sort(key=lambda item: item[0])

    artifact_values: dict[str, dict[str, Any]] = {}
    for name, source in artifact_items:
        fingerprint = (
            source if isinstance(source, ArtifactFingerprint) else fingerprint_file(source)
        )
        artifact_values[name] = fingerprint.to_dict()
    return {
        "schema_version": CHECKPOINT_CONTRACT_SCHEMA_VERSION,
        "run_manifest": manifest.to_dict(),
        "artifacts": artifact_values,
    }


class CheckpointContractMismatchError(ValueError):
    """Raised when stored checkpoint identity differs from the current run."""

    def __init__(self, mismatches: Sequence[str]):
        self.mismatches = tuple(mismatches)
        message = "checkpoint contract mismatch:\n" + "\n".join(
            f"- {mismatch}" for mismatch in self.mismatches
        )
        super().__init__(message)


def _contract_differences(actual: Any, expected: Any, location: str) -> list[str]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{location}: expected object, got {type(actual).__name__} {actual!r}"]
        differences: list[str] = []
        actual_keys = set(actual)
        expected_keys = set(expected)
        for key in sorted(expected_keys - actual_keys):
            child = f"{location}.{key}" if location else str(key)
            differences.append(f"{child}: missing field; expected {expected[key]!r}")
        for key in sorted(actual_keys - expected_keys):
            child = f"{location}.{key}" if location else str(key)
            differences.append(f"{child}: unexpected field with value {actual[key]!r}")
        for key in sorted(actual_keys & expected_keys):
            child = f"{location}.{key}" if location else str(key)
            differences.extend(_contract_differences(actual[key], expected[key], child))
        return differences
    if type(actual) is not type(expected):
        return [
            f"{location}: expected {type(expected).__name__} {expected!r}, "
            f"got {type(actual).__name__} {actual!r}"
        ]
    if actual != expected:
        return [f"{location}: expected {expected!r}, got {actual!r}"]
    return []


def validate_checkpoint_contract(
    stored_contract: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
    """Raise field-level diagnostics for any checkpoint contract mismatch."""

    differences = _contract_differences(stored_contract, expected_contract, "")
    if differences:
        raise CheckpointContractMismatchError(differences)


class ArtifactContentKind(StrEnum):
    """Explicit disclosure class used by :func:`guard_artifact_write_path`."""

    PUBLIC = "public"
    DERIVED = "derived"
    CREDENTIALED = "credentialed"
    RAW_RECORDS = "raw_records"


def guard_artifact_write_path(
    path: str | Path,
    *,
    content_kind: ArtifactContentKind | str,
    repository_root: str | Path | None = None,
) -> Path:
    """Reject sensitive records targeting tracked public repository folders.

    The guard is classification-based by design: callers must say whether the
    content is credentialed or contains raw records.  Derived/public artifacts
    are allowed, and sensitive artifacts remain allowed in dedicated locations
    such as ``data/raw`` or an external run directory.
    """

    try:
        kind = ArtifactContentKind(content_kind)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ArtifactContentKind)
        raise ValueError(
            f"unknown content_kind {content_kind!r}; expected one of: {allowed}"
        ) from exc

    destination = Path(path).expanduser().resolve(strict=False)
    selected_root = (
        repository_root if repository_root is not None else repository_root_for_output(destination)
    )
    if selected_root is None:
        return destination
    root = Path(selected_root).expanduser().resolve()
    try:
        relative = destination.relative_to(root)
    except ValueError:
        return destination

    if kind not in {ArtifactContentKind.CREDENTIALED, ArtifactContentKind.RAW_RECORDS}:
        return destination

    private_root = relative.parts[0].casefold() if relative.parts else None
    if (
        private_root in _SENSITIVE_PRIVATE_DIRECTORIES
        and relative.as_posix().casefold() not in _PUBLIC_RESULT_SNAPSHOTS
    ):
        return destination
    raise ValueError(
        f"refusing to write {kind.value} content to tracked public location ({relative}); "
        "use data/, a gitignored run directory, or a location outside the repository"
    )


__all__ = [
    "ArtifactContentKind",
    "ArtifactFingerprint",
    "CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "CheckpointContractMismatchError",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "artifact_fingerprint",
    "atomic_write_text",
    "build_checkpoint_contract",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "file_sha256",
    "fingerprint_file",
    "guard_artifact_write_path",
    "load_manifest",
    "repository_root_for_output",
    "save_manifest",
    "strict_json_loads",
    "strict_jsonl_loads",
    "validate_checkpoint_contract",
    "validate_output_paths",
]
