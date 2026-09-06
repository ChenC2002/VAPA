"""Manifest-driven preparation of cutoff-safe, patient-disjoint episodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vapa.artifacts import (
    ArtifactContentKind,
    atomic_write_text,
    canonical_json_dumps,
    guard_artifact_write_path,
)
from vapa.data.adapters import DatasetAdapter, builtin_adapters
from vapa.data.episodes import episode_to_record
from vapa.data.io import DataValidationError, load_json, sha256_file
from vapa.data.splits import SplitFractions, patient_disjoint_hash_split
from vapa.provenance import package_code_fingerprint
from vapa.schemas import Episode

PREPARATION_SCHEMA_VERSION = 1
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "content_kind",
        "adapter",
        "events",
        "tasks",
        "source_files",
        "adapter_config",
        "splits",
    }
)
_SPLIT_KEYS = frozenset({"train", "validation", "test", "seed", "namespace"})
_OUTPUT_NAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


@dataclass(frozen=True, slots=True)
class PreparationResult:
    output_directory: Path
    split_paths: Mapping[str, Path]
    episode_counts: Mapping[str, int]
    patient_counts: Mapping[str, int]
    manifest_path: Path

    @property
    def total_episodes(self) -> int:
        return sum(self.episode_counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_directory": str(self.output_directory),
            "manifest_path": str(self.manifest_path),
            "total_episodes": self.total_episodes,
            "episode_counts": dict(self.episode_counts),
            "patient_counts": dict(self.patient_counts),
            "split_paths": {name: str(path) for name, path in self.split_paths.items()},
        }


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{location} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DataValidationError(f"{location} keys must be strings")
    return value


def _parse_splits(raw: Any, *, seed_override: int | None) -> tuple[SplitFractions, int, str]:
    value = _mapping(raw, "splits")
    unknown = set(value) - _SPLIT_KEYS
    if unknown:
        raise DataValidationError(f"splits has unknown fields: {sorted(unknown)}")
    try:
        fractions = SplitFractions(
            train=value.get("train", 0.8),
            validation=value.get("validation", 0.1),
            test=value.get("test", 0.1),
        )
        fractions.decimals()
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"invalid split fractions: {error}") from error

    seed = value.get("seed", 0) if seed_override is None else seed_override
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataValidationError("split seed must be a non-negative integer")
    namespace = value.get("namespace", "vapa-prepared-split-v1")
    if not isinstance(namespace, str) or not namespace.strip():
        raise DataValidationError("split namespace must be a non-empty string")
    return fractions, seed, namespace.strip()


def _load_manifest(path: Path) -> Mapping[str, Any]:
    raw = load_json(path)
    manifest = _mapping(raw, "dataset manifest")
    unknown = set(manifest) - _MANIFEST_KEYS
    if unknown:
        raise DataValidationError(f"dataset manifest has unknown fields: {sorted(unknown)}")
    if manifest.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise DataValidationError(
            f"dataset manifest schema_version must be {PREPARATION_SCHEMA_VERSION}"
        )
    adapter_id = manifest.get("adapter")
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise DataValidationError("dataset manifest adapter must be a non-empty string")
    if "splits" not in manifest:
        raise DataValidationError("dataset manifest is missing splits")
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise DataValidationError("dataset manifest source_files must be a non-empty array")
    if any(not isinstance(item, str) or not item.strip() for item in sources):
        raise DataValidationError("dataset manifest source_files must contain path strings")
    if len(set(sources)) != len(sources):
        raise DataValidationError("dataset manifest source_files cannot contain duplicates")
    try:
        ArtifactContentKind(manifest.get("content_kind"))
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in ArtifactContentKind)
        raise DataValidationError(
            f"dataset manifest content_kind must be one of: {choices}"
        ) from error
    return manifest


def _source_file_records(
    manifest: Mapping[str, Any],
    *,
    manifest_directory: Path,
) -> list[dict[str, Any]]:
    root = manifest_directory.resolve()
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(manifest["source_files"]):
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in {"", "."}:
            raise DataValidationError(
                f"dataset manifest source_files[{index}] must be a safe relative path"
            )
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise DataValidationError(
                f"dataset manifest source_files[{index}] escapes the manifest directory"
            ) from error
        if not candidate.is_file():
            raise DataValidationError(
                f"dataset manifest source_files[{index}] is not a regular file"
            )
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    return records


def _validate_consumed_sources_declared(
    manifest: Mapping[str, Any],
    source_file_records: list[dict[str, Any]],
) -> None:
    """Bind the generic adapter's consumed event/task paths to hashed sources."""

    adapter_id = manifest.get("adapter")
    if adapter_id not in {"generic_events", "latest_field_events"}:
        return
    declared = {str(record["path"]) for record in source_file_records}
    consumed: set[str] = set()
    sections = ("events", "tasks") if adapter_id == "generic_events" else ("events",)
    for section_name in sections:
        section = _mapping(manifest.get(section_name), section_name)
        raw_path = section.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise DataValidationError(f"{section_name}.path must be a non-empty string")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DataValidationError(f"{section_name}.path must be a safe relative path")
        consumed.add(relative.as_posix())
    missing = consumed - declared
    if missing:
        raise DataValidationError(
            "adapter inputs must be listed in source_files: " + ", ".join(sorted(missing))
        )


def _adapter_registry(
    adapters: Mapping[str, DatasetAdapter] | None,
) -> dict[str, DatasetAdapter]:
    output = builtin_adapters()
    if adapters is None:
        return output
    for name, adapter in adapters.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("adapter registry names must be non-empty strings")
        if not isinstance(adapter, DatasetAdapter):
            raise TypeError(f"adapter {name!r} does not implement DatasetAdapter")
        if adapter.adapter_id != name:
            raise ValueError(
                f"adapter registry key {name!r} does not match adapter_id {adapter.adapter_id!r}"
            )
        output[name] = adapter
    return output


def _validate_episodes(episodes: list[Episode]) -> None:
    seen: set[str] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Episode):
            raise DataValidationError(f"adapter result {index} is not an Episode")
        if episode.task.instance_id in seen:
            raise DataValidationError(
                f"adapter returned duplicate instance_id {episode.task.instance_id!r}"
            )
        seen.add(episode.task.instance_id)
        if len(episode.events) != len(episode.admissible_events):
            raise DataValidationError(
                f"adapter returned post-cutoff events for {episode.task.instance_id!r}"
            )


def _episode_jsonl(records: list[Mapping[str, Any]]) -> str:
    return "".join(canonical_json_dumps(record) + "\n" for record in records)


def prepare_dataset(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int | None = None,
    overwrite: bool = False,
    adapters: Mapping[str, DatasetAdapter] | None = None,
    repository_root: str | Path | None = None,
) -> PreparationResult:
    """Prepare deterministic JSONL splits from an explicit dataset manifest.

    Only known output files are replaced when ``overwrite`` is true.  This
    function never removes an output directory or unrelated files.
    """

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    source_path = Path(manifest_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {source_path}")
    source_manifest_sha256 = sha256_file(source_path)
    pipeline_implementation_sha256 = package_code_fingerprint()
    manifest = _load_manifest(source_path)
    if sha256_file(source_path) != source_manifest_sha256:
        raise DataValidationError("the dataset manifest changed while it was being read")
    source_file_records = _source_file_records(
        manifest,
        manifest_directory=source_path.parent,
    )
    _validate_consumed_sources_declared(manifest, source_file_records)
    registry = _adapter_registry(adapters)
    adapter_id = str(manifest["adapter"]).strip()
    try:
        adapter = registry[adapter_id]
    except KeyError as error:
        raise DataValidationError(
            f"unknown dataset adapter {adapter_id!r}; available={sorted(registry)}"
        ) from error
    # Import lazily to avoid a package-initialization cycle: ``vapa.inference``
    # itself imports the episode schema.  The identity binds implementation
    # source, closures, defaults, and adapter receiver state.
    from vapa.inference import factory_identity

    adapter_implementation = factory_identity(adapter.build_episodes)

    fractions, split_seed, namespace = _parse_splits(manifest["splits"], seed_override=seed)
    episodes = list(
        adapter.build_episodes(manifest, manifest_directory=source_path.parent.resolve())
    )
    source_file_records_after = _source_file_records(
        manifest,
        manifest_directory=source_path.parent,
    )
    if source_file_records_after != source_file_records:
        raise DataValidationError("a source file changed while dataset preparation was running")
    if sha256_file(source_path) != source_manifest_sha256:
        raise DataValidationError("the dataset manifest changed during dataset preparation")
    if package_code_fingerprint() != pipeline_implementation_sha256:
        raise DataValidationError("the VAPA implementation changed during dataset preparation")
    _validate_episodes(episodes)
    records = [episode_to_record(episode) for episode in episodes]
    split_records = patient_disjoint_hash_split(
        records,
        fractions=fractions,
        seed=split_seed,
        namespace=namespace,
    )
    for values in split_records.values():
        values.sort(key=lambda record: str(record["instance_id"]))

    destination = guard_artifact_write_path(
        output_directory,
        content_kind=ArtifactContentKind(manifest["content_kind"]),
        repository_root=repository_root,
    )
    split_paths = {name: destination / filename for name, filename in _OUTPUT_NAMES.items()}
    prepared_manifest_path = destination / "prepared_manifest.json"
    targets = [*split_paths.values(), prepared_manifest_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(path.name for path in existing)
        raise FileExistsError(f"prepared output already exists ({rendered}); pass overwrite=True")
    if any(path.exists() and not path.is_file() for path in targets):
        raise FileExistsError("a prepared output target exists but is not a regular file")

    for name, path in split_paths.items():
        atomic_write_text(path, _episode_jsonl(split_records[name]))

    episode_counts = {name: len(values) for name, values in split_records.items()}
    patient_counts = {
        name: len({str(record["patient_id"]) for record in values})
        for name, values in split_records.items()
    }
    prepared = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "adapter": adapter_id,
        "adapter_implementation": adapter_implementation,
        "pipeline_implementation_sha256": pipeline_implementation_sha256,
        "content_kind": manifest["content_kind"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_files": source_file_records,
        "split": {
            "seed": split_seed,
            "namespace": namespace,
            "fractions": {
                "train": fractions.train,
                "validation": fractions.validation,
                "test": fractions.test,
            },
        },
        "episode_counts": episode_counts,
        "patient_counts": patient_counts,
        "files": {
            name: {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in split_paths.items()
        },
    }
    atomic_write_text(prepared_manifest_path, canonical_json_dumps(prepared) + "\n")
    return PreparationResult(
        output_directory=destination,
        split_paths=split_paths,
        episode_counts=episode_counts,
        patient_counts=patient_counts,
        manifest_path=prepared_manifest_path,
    )
