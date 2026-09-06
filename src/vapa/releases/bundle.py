"""Load and verify the immutable ``public_core_v1`` package-data release.

The implementation intentionally uses only the Python standard library.  JSON is
decoded strictly, every declared byte is authenticated, and release paths are treated
as untrusted input even though the bundled manifest is normally package-owned.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, NoReturn

RELEASE_ID = "public_core_v1"
RELEASE_VERSION = "1.0.0"
RELEASE_SCHEMA_VERSION = "1.0"
RELEASE_STATUS = "public_reference"
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
NON_PAPER_EXACT_WARNING = (
    "This public core release is a non-paper-exact interoperability bundle; "
    "it is not the unreleased author artifact."
)
PROMPT_MARKER = "[VAPA PUBLIC CORE V1 - NON-PAPER-EXACT TEMPLATE]"
INTEGRITY_SCOPE = (
    "Every release asset listed in files is covered by its SHA-256 digest and byte "
    "size; manifest.json is excluded to avoid a self-referential digest."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_FILES = {
    "README.md": "release_documentation",
    "prompts/system_prompt.txt": "non_paper_exact_system_prompt",
    "prompts/user_prompt.txt": "non_paper_exact_user_prompt",
    "release.json": "release_descriptor",
    "schemas/calculator_manifest.schema.json": "calculator_manifest_schema",
    "schemas/task_manifest.schema.json": "task_manifest_schema",
    "schemas/verifier_manifest.schema.json": "verifier_manifest_schema",
}
_PROMPT_PATHS = {
    "system": "prompts/system_prompt.txt",
    "user": "prompts/user_prompt.txt",
}
_SCHEMA_PATHS = {
    "calculator": "schemas/calculator_manifest.schema.json",
    "task": "schemas/task_manifest.schema.json",
    "verifier": "schemas/verifier_manifest.schema.json",
}
_SCHEMA_IDS = {
    "calculator": "urn:vapa:public-core-v1:schema:calculator-manifest",
    "task": "urn:vapa:public-core-v1:schema:task-manifest",
    "verifier": "urn:vapa:public-core-v1:schema:verifier-manifest",
}
_PROMPT_PLACEHOLDERS = {
    "system": frozenset(
        {
            "{{memory_capacity}}",
            "{{action_budget}}",
            "{{turn_cap}}",
            "{{action_grammar}}",
        }
    ),
    "user": frozenset(
        {
            "{{instruction}}",
            "{{cutoff}}",
            "{{tool_return}}",
            "{{memory}}",
            "{{history}}",
            "{{remaining_budget}}",
            "{{legal_actions}}",
        }
    ),
}


class ReleaseValidationError(ValueError):
    """Raised when a release cannot be trusted as ``public_core_v1``."""


@dataclass(frozen=True, slots=True)
class ReleaseValidationReport:
    """Dependency-free validation result suitable for a CLI or release check."""

    valid: bool
    release_id: str | None
    version: str | None
    files_checked: int
    issues: tuple[str, ...] = ()

    def ok(self) -> bool:
        return self.valid

    def require_valid(self) -> None:
        if not self.valid:
            raise ReleaseValidationError("; ".join(self.issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "release_id": self.release_id,
            "version": self.version,
            "files_checked": self.files_checked,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class PublicCoreRelease:
    """Validated public assets; all mappings are read-only at their top level."""

    manifest: Mapping[str, Any]
    descriptor: Mapping[str, Any]
    prompts: Mapping[str, str]
    schemas: Mapping[str, Mapping[str, Any]]
    files_checked: int

    @property
    def release_id(self) -> str:
        return str(self.descriptor["release_id"])

    @property
    def version(self) -> str:
        return str(self.descriptor["version"])

    @property
    def paper_exact(self) -> bool:
        return bool(self.descriptor["paper_exact"])

    def prompt(self, name: str) -> str:
        try:
            return self.prompts[name]
        except KeyError as exc:
            raise KeyError(f"unknown public-core prompt {name!r}") from exc

    def schema(self, name: str) -> Mapping[str, Any]:
        try:
            return self.schemas[name]
        except KeyError as exc:
            raise KeyError(f"unknown public-core schema {name!r}") from exc


def _reject_json_constant(value: str) -> NoReturn:
    raise ReleaseValidationError(f"non-finite JSON number {value!r} is not allowed")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_json_constant(value)
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseValidationError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def _strict_json_loads(payload: bytes, location: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(f"{location} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except ReleaseValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise ReleaseValidationError(
            f"{location} is invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ReleaseValidationError(f"manifest is not finite canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ReleaseValidationError(f"{location} fields drifted: {'; '.join(details)}")


def _safe_parts(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ReleaseValidationError(f"{location} must be a non-empty relative path")
    if "\x00" in value or "\\" in value:
        raise ReleaseValidationError(f"{location} is an unsafe release path: {value!r}")
    raw_parts = value.split("/")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ReleaseValidationError(f"{location} is an unsafe release path: {value!r}")
    return tuple(posix.parts)


def _release_root(root: str | os.PathLike[str] | None) -> Path | Traversable:
    if root is not None:
        return Path(root)
    return resources.files("vapa.releases.public_core_v1")


def _read_release_file(root: Path | Traversable, relative: str, location: str) -> bytes:
    parts = _safe_parts(relative, location)
    if isinstance(root, Path):
        try:
            resolved_root = root.resolve(strict=True)
            candidate = root.joinpath(*parts)
            resolved_candidate = candidate.resolve(strict=True)
            resolved_candidate.relative_to(resolved_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ReleaseValidationError(
                f"{location} does not resolve to a file inside the release: {relative!r}"
            ) from exc
        if not resolved_candidate.is_file():
            raise ReleaseValidationError(f"{location} is not a regular file: {relative!r}")
        try:
            return resolved_candidate.read_bytes()
        except OSError as exc:
            raise ReleaseValidationError(f"cannot read {location}: {relative!r}") from exc

    candidate: Traversable = root
    for part in parts:
        candidate = candidate.joinpath(part)
    if not candidate.is_file():
        raise ReleaseValidationError(f"{location} is not a packaged file: {relative!r}")
    try:
        return candidate.read_bytes()
    except OSError as exc:
        raise ReleaseValidationError(f"cannot read packaged {location}: {relative!r}") from exc


def _validate_manifest(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Mapping):
        raise ReleaseValidationError("manifest.json root must be an object")
    _exact_keys(
        value,
        {
            "files",
            "integrity_scope",
            "paper_exact",
            "release_id",
            "schema_version",
            "status",
            "version",
        },
        "manifest.json",
    )
    expected_scalars = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "version": RELEASE_VERSION,
        "status": RELEASE_STATUS,
        "paper_exact": False,
        "integrity_scope": INTEGRITY_SCOPE,
    }
    for field, expected in expected_scalars.items():
        if type(value[field]) is not type(expected) or value[field] != expected:
            raise ReleaseValidationError(
                f"manifest.json {field} drifted: expected {expected!r}, got {value[field]!r}"
            )

    files = value["files"]
    if not isinstance(files, list):
        raise ReleaseValidationError("manifest.json files must be an array")
    records: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for index, record in enumerate(files):
        location = f"manifest.json files[{index}]"
        if not isinstance(record, Mapping):
            raise ReleaseValidationError(f"{location} must be an object")
        _exact_keys(record, {"path", "required", "role", "sha256", "size_bytes"}, location)
        path = record["path"]
        _safe_parts(path, f"{location}.path")
        if not isinstance(record["role"], str) or not record["role"]:
            raise ReleaseValidationError(f"{location}.role must be a non-empty string")
        if record["required"] is not True:
            raise ReleaseValidationError(f"{location}.required must be true")
        if not isinstance(record["sha256"], str) or _SHA256.fullmatch(record["sha256"]) is None:
            raise ReleaseValidationError(f"{location}.sha256 is not a lowercase SHA-256 digest")
        size = record["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseValidationError(f"{location}.size_bytes must be a non-negative integer")
        paths.append(path)
        records.append(record)

    if len(paths) != len(set(paths)):
        raise ReleaseValidationError("manifest.json contains duplicate file paths")
    if paths != sorted(paths):
        raise ReleaseValidationError("manifest.json file records must be sorted by path")
    if set(paths) != set(_EXPECTED_FILES):
        missing = sorted(set(_EXPECTED_FILES) - set(paths))
        unexpected = sorted(set(paths) - set(_EXPECTED_FILES))
        raise ReleaseValidationError(
            f"manifest.json file inventory drifted: missing={missing}, unexpected={unexpected}"
        )
    for record in records:
        expected_role = _EXPECTED_FILES[record["path"]]
        if record["role"] != expected_role:
            raise ReleaseValidationError(
                f"manifest.json role drifted for {record['path']!r}: "
                f"expected {expected_role!r}, got {record['role']!r}"
            )
    return tuple(records)


def _validate_descriptor(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseValidationError("release.json root must be an object")
    _exact_keys(
        value,
        {
            "paper_exact",
            "prompts",
            "release_id",
            "schemas",
            "schema_version",
            "status",
            "version",
            "warning",
        },
        "release.json",
    )
    expected_scalars = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "version": RELEASE_VERSION,
        "status": RELEASE_STATUS,
        "paper_exact": False,
        "warning": NON_PAPER_EXACT_WARNING,
    }
    for field, expected in expected_scalars.items():
        if type(value[field]) is not type(expected) or value[field] != expected:
            raise ReleaseValidationError(
                f"release.json {field} drifted: expected {expected!r}, got {value[field]!r}"
            )
    for field, expected in (("prompts", _PROMPT_PATHS), ("schemas", _SCHEMA_PATHS)):
        actual = value[field]
        if not isinstance(actual, Mapping) or dict(actual) != expected:
            raise ReleaseValidationError(
                f"release.json {field} inventory drifted: expected {expected!r}, got {actual!r}"
            )
        for name, path in actual.items():
            _safe_parts(path, f"release.json {field}.{name}")
    return value


def _validate_prompt(name: str, payload: bytes) -> str:
    try:
        prompt = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(f"{name} prompt is not valid UTF-8") from exc
    if not prompt.endswith("\n") or "\x00" in prompt:
        raise ReleaseValidationError(f"{name} prompt must be NUL-free UTF-8 with a final newline")
    if not prompt.startswith(PROMPT_MARKER + "\n"):
        raise ReleaseValidationError(f"{name} prompt is missing the non-paper-exact marker")
    missing = sorted(token for token in _PROMPT_PLACEHOLDERS[name] if token not in prompt)
    if missing:
        raise ReleaseValidationError(f"{name} prompt is missing placeholders: {missing}")
    return prompt


def _validate_schema(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseValidationError(f"{name} schema root must be an object")
    if value.get("$schema") != JSON_SCHEMA_DRAFT:
        raise ReleaseValidationError(f"{name} schema draft drifted")
    if value.get("$id") != _SCHEMA_IDS[name]:
        raise ReleaseValidationError(f"{name} schema identifier drifted")
    if value.get("type") != "object" or value.get("additionalProperties") is not False:
        raise ReleaseValidationError(f"{name} schema must be a closed object schema")
    required = value.get("required")
    properties = value.get("properties")
    if not isinstance(required, list) or not isinstance(properties, Mapping):
        raise ReleaseValidationError(f"{name} schema requires properties and required arrays")
    common = {"schema_version", "release_id", "catalog_id", "paper_exact", "warning"}
    collection = {"calculator": "calculators", "task": "tasks", "verifier": "predicates"}[name]
    if set(required) != common | {collection}:
        raise ReleaseValidationError(f"{name} schema required fields drifted")
    if not common | {collection} <= set(properties):
        raise ReleaseValidationError(f"{name} schema properties drifted")
    expected_constants = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "paper_exact": False,
        "warning": NON_PAPER_EXACT_WARNING,
    }
    for field, expected in expected_constants.items():
        field_schema = properties.get(field)
        actual = field_schema.get("const") if isinstance(field_schema, Mapping) else None
        if (
            not isinstance(field_schema, Mapping)
            or type(actual) is not type(expected)
            or actual != expected
        ):
            raise ReleaseValidationError(f"{name} schema {field} contract drifted")
    collection_schema = properties[collection]
    if not isinstance(collection_schema, Mapping) or collection_schema.get("type") != "array":
        raise ReleaseValidationError(f"{name} schema {collection} must be an array")
    definitions = value.get("$defs")
    item = definitions.get("item") if isinstance(definitions, Mapping) else None
    if not isinstance(item, Mapping) or item.get("additionalProperties") is not False:
        raise ReleaseValidationError(f"{name} schema item definition must be closed")
    return value


def _load_validated(root_value: str | os.PathLike[str] | None) -> PublicCoreRelease:
    root = _release_root(root_value)
    manifest_bytes = _read_release_file(root, "manifest.json", "manifest.json")
    manifest = _strict_json_loads(manifest_bytes, "manifest.json")
    if manifest_bytes != _canonical_json_bytes(manifest):
        raise ReleaseValidationError("manifest.json is not canonical JSON")
    records = _validate_manifest(manifest)

    payloads: dict[str, bytes] = {}
    for record in records:
        path = record["path"]
        payload = _read_release_file(root, path, f"release asset {path}")
        if len(payload) != record["size_bytes"]:
            raise ReleaseValidationError(
                f"size drift for {path!r}: expected {record['size_bytes']}, got {len(payload)}"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise ReleaseValidationError(
                f"SHA-256 drift for {path!r}: expected {record['sha256']}, got {digest}"
            )
        payloads[path] = payload

    descriptor = _validate_descriptor(_strict_json_loads(payloads["release.json"], "release.json"))
    prompts = {name: _validate_prompt(name, payloads[path]) for name, path in _PROMPT_PATHS.items()}
    schemas = {
        name: _validate_schema(name, _strict_json_loads(payloads[path], path))
        for name, path in _SCHEMA_PATHS.items()
    }
    return PublicCoreRelease(
        manifest=MappingProxyType(dict(manifest)),
        descriptor=MappingProxyType(dict(descriptor)),
        prompts=MappingProxyType(prompts),
        schemas=MappingProxyType(
            {name: MappingProxyType(dict(schema)) for name, schema in schemas.items()}
        ),
        files_checked=len(records),
    )


def load_public_core_release(
    root: str | os.PathLike[str] | None = None,
) -> PublicCoreRelease:
    """Load ``public_core_v1`` after authenticating and validating every asset."""

    return _load_validated(root)


def validate_public_core_release(
    root: str | os.PathLike[str] | None = None,
) -> ReleaseValidationReport:
    """Return a non-throwing validation report for ``public_core_v1``."""

    try:
        release = _load_validated(root)
    except ReleaseValidationError as exc:
        return ReleaseValidationReport(False, None, None, 0, (str(exc),))
    return ReleaseValidationReport(
        True,
        release.release_id,
        release.version,
        release.files_checked,
    )


__all__ = [
    "NON_PAPER_EXACT_WARNING",
    "PROMPT_MARKER",
    "PublicCoreRelease",
    "RELEASE_ID",
    "RELEASE_SCHEMA_VERSION",
    "RELEASE_VERSION",
    "ReleaseValidationError",
    "ReleaseValidationReport",
    "load_public_core_release",
    "validate_public_core_release",
]
