"""Strict, dependency-free data loading and integrity manifests.

The paper does not prescribe one released EHR file layout.  These helpers keep
the on-disk boundary deliberately small: an episode or event file is either a
JSON array (optionally wrapped by an ``episodes``/``events`` key) or JSONL with
one object per line.  Schema-specific adapters can build dataclasses after this
boundary without weakening its validation guarantees.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

from vapa.artifacts import artifact_fingerprint, strict_json_loads

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
Record: TypeAlias = dict[str, Any]
IdField: TypeAlias = str | Sequence[str]


class DataValidationError(ValueError):
    """Raised when an input is valid JSON but violates the data contract."""


class IntegrityError(DataValidationError):
    """Raised when a file does not match its SHA-256 manifest."""


def _validate_finite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DataValidationError(f"non-finite number at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{location}[{index}]")


def _loads_json(text: str, source: str) -> JSONValue:
    try:
        return strict_json_loads(text)
    except json.JSONDecodeError as error:
        raise DataValidationError(
            f"{source}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}"
        ) from error
    except ValueError as error:
        raise DataValidationError(f"{source}: {error}") from error


def load_json(path: str | Path) -> JSONValue:
    """Load strict UTF-8 JSON, rejecting duplicate keys and non-finite numbers."""

    input_path = Path(path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DataValidationError(f"{input_path}: input is not valid UTF-8") from error
    if not text.strip():
        raise DataValidationError(f"{input_path}: empty JSON input")
    return _loads_json(text, str(input_path))


def load_jsonl(path: str | Path) -> list[JSONValue]:
    """Load strict JSONL.

    Every physical line must contain one JSON value.  Blank lines are rejected
    instead of being silently skipped so line numbers and hashes remain stable.
    """

    input_path = Path(path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DataValidationError(f"{input_path}: input is not valid UTF-8") from error
    if not text:
        raise DataValidationError(f"{input_path}: empty JSONL input")

    values: list[JSONValue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise DataValidationError(f"{input_path}:{line_number}: blank JSONL line")
        values.append(_loads_json(line, f"{input_path}:{line_number}"))
    if not values:
        raise DataValidationError(f"{input_path}: empty JSONL input")
    return values


def _id_fields(id_field: IdField) -> tuple[str, ...]:
    fields = (id_field,) if isinstance(id_field, str) else tuple(id_field)
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("id_field must contain one or more non-empty field names")
    return fields


def _record_identifier(
    record: Mapping[str, Any], fields: tuple[str, ...], index: int
) -> tuple[Any, ...]:
    identifier: list[Any] = []
    for field in fields:
        if field not in record:
            raise DataValidationError(f"record {index} is missing required ID field {field!r}")
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, str | int):
            raise DataValidationError(
                f"record {index} field {field!r} must be a string or integer ID"
            )
        if isinstance(value, str) and not value.strip():
            raise DataValidationError(f"record {index} field {field!r} cannot be empty")
        # Include the JSON type so integer 1 and string "1" cannot alias.
        identifier.append((type(value).__name__, value))
    return tuple(identifier)


def _extract_records(payload: JSONValue, collection_key: str | None) -> list[Record]:
    if isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, dict) and collection_key is not None:
        if collection_key not in payload:
            raise DataValidationError(f"JSON object must contain a {collection_key!r} array")
        raw_records = payload[collection_key]
        if not isinstance(raw_records, list):
            raise DataValidationError(f"{collection_key!r} must be a JSON array")
    else:
        expected = "a JSON array"
        if collection_key is not None:
            expected += f" or an object containing {collection_key!r}"
        raise DataValidationError(f"expected {expected}")

    records: list[Record] = []
    for index, value in enumerate(raw_records):
        if not isinstance(value, dict):
            raise DataValidationError(f"record {index} must be a JSON object")
        records.append(dict(value))
    return records


def load_records(
    path: str | Path,
    *,
    id_field: IdField,
    collection_key: str | None = None,
) -> list[Record]:
    """Load records and reject missing or duplicate IDs.

    ``id_field`` may be a sequence to enforce a composite identifier, for
    example ``("patient_id", "task_id", "target", "cutoff")``.
    """

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records = _extract_records(load_jsonl(input_path), None)
    elif suffix == ".json":
        records = _extract_records(load_json(input_path), collection_key)
    else:
        raise DataValidationError(
            f"{input_path}: unsupported format {suffix!r}; expected .json, .jsonl, or .ndjson"
        )

    fields = _id_fields(id_field)
    seen: dict[tuple[Any, ...], int] = {}
    for index, record in enumerate(records):
        identifier = _record_identifier(record, fields, index)
        if identifier in seen:
            rendered = {field: record[field] for field in fields}
            raise DataValidationError(
                f"duplicate record ID {rendered!r} at indices {seen[identifier]} and {index}"
            )
        seen[identifier] = index
    return records


def load_episodes(path: str | Path, *, id_field: IdField = "episode_id") -> list[Record]:
    """Load strict episode records from JSON or JSONL."""

    return load_records(path, id_field=id_field, collection_key="episodes")


def load_events(path: str | Path, *, id_field: IdField = "event_id") -> list[Record]:
    """Load strict event records from JSON or JSONL.

    Datasets that use the paper's evidence pointer as the identifier can pass
    ``id_field="pointer"``.
    """

    return load_records(path, id_field=id_field, collection_key="events")


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it wholly into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: JSONValue) -> str:
    """Hash canonical compact JSON with sorted object keys."""

    _validate_finite(value)
    return artifact_fingerprint(value)


def build_sha256_manifest(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest for regular files.

    All paths must resolve beneath ``root`` and are stored as portable POSIX
    names.  ``root`` defaults to the current working directory.
    """

    resolved_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_path in paths:
        file_path = Path(raw_path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"manifest input is not a regular file: {file_path}")
        try:
            name = file_path.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError(f"manifest path escapes root {resolved_root}: {file_path}") from error
        if name in seen_names:
            raise DataValidationError(f"duplicate manifest path: {name}")
        seen_names.add(name)
        entries.append(
            {
                "path": name,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    entries.sort(key=lambda entry: entry["path"])
    return {"manifest_version": 1, "algorithm": "sha256", "files": entries}


def write_sha256_manifest(
    manifest_path: str | Path,
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write a stable, human-readable SHA-256 manifest."""

    output_path = Path(manifest_path)
    manifest_root = root if root is not None else output_path.parent
    manifest = build_sha256_manifest(paths, root=manifest_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validated_manifest(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if manifest.get("manifest_version") != 1:
        raise DataValidationError("unsupported manifest_version")
    if manifest.get("algorithm") != "sha256":
        raise DataValidationError("manifest algorithm must be 'sha256'")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise DataValidationError("manifest 'files' must be an array")

    seen: set[str] = set()
    output: list[Mapping[str, Any]] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise DataValidationError(f"manifest file entry {index} must be an object")
        path = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise DataValidationError(f"manifest file entry {index} has an unsafe path")
        if path in seen:
            raise DataValidationError(f"duplicate manifest path: {path}")
        seen.add(path)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DataValidationError(f"manifest file entry {index} has invalid size_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DataValidationError(f"manifest file entry {index} has invalid SHA-256")
        output.append(entry)
    return output


def verify_sha256_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    root: str | Path | None = None,
) -> bool:
    """Verify every file in a manifest, raising :class:`IntegrityError` on mismatch."""

    if isinstance(manifest, str | Path):
        manifest_path = Path(manifest)
        loaded = load_json(manifest_path)
        if not isinstance(loaded, dict):
            raise DataValidationError("manifest root must be a JSON object")
        manifest_mapping: Mapping[str, Any] = loaded
        resolved_root = Path(root).resolve() if root is not None else manifest_path.parent.resolve()
    else:
        manifest_mapping = manifest
        resolved_root = Path(root).resolve() if root is not None else Path.cwd().resolve()

    for entry in _validated_manifest(manifest_mapping):
        file_path = (resolved_root / str(entry["path"])).resolve()
        try:
            file_path.relative_to(resolved_root)
        except ValueError as error:
            raise IntegrityError(
                f"manifest path escapes verification root: {entry['path']}"
            ) from error
        if not file_path.is_file():
            raise IntegrityError(f"manifest file is missing: {entry['path']}")
        actual_size = file_path.stat().st_size
        if actual_size != entry["size_bytes"]:
            expected_size = entry["size_bytes"]
            raise IntegrityError(
                f"size mismatch for {entry['path']}: expected {expected_size}, got {actual_size}"
            )
        actual_digest = sha256_file(file_path)
        if actual_digest != entry["sha256"]:
            raise IntegrityError(f"SHA-256 mismatch for {entry['path']}")
    return True


# Short aliases retained for script-level readability.
build_manifest = build_sha256_manifest
verify_manifest = verify_sha256_manifest
