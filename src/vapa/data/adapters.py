"""Dataset adapter contracts and the schema-neutral tabular adapter.

The public adapter intentionally requires an explicit column map for both an
event source and a task source.  It does not guess the layouts of credentialed
benchmarks.  External integrations can implement :class:`DatasetAdapter` and
register them with the data pipeline without weakening the shared episode
contract.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vapa.artifacts import strict_json_loads
from vapa.data.io import DataValidationError, load_json, load_jsonl
from vapa.schemas import (
    Episode,
    RecordEvent,
    TaskSpec,
    TimeWindow,
    normalize_field,
    parse_timestamp,
)

GENERIC_ADAPTER_ID = "generic_events"
LATEST_FIELD_ADAPTER_ID = "latest_field_events"

_EVENT_FIELDS = frozenset(
    {
        "pointer",
        "patient_id",
        "timestamp",
        "domain",
        "field",
        "value",
        "unit",
        "text",
        "source",
        "code",
    }
)
_REQUIRED_EVENT_FIELDS = frozenset(
    {"pointer", "patient_id", "timestamp", "domain", "field", "value"}
)
_TASK_FIELDS = frozenset(
    {
        "instance_id",
        "patient_id",
        "instruction",
        "cutoff",
        "family",
        "gold_answer",
        "requested_fields",
        "requested_window",
        "answer_type",
        "metadata",
        "reference_evidence",
    }
)
_REQUIRED_TASK_FIELDS = frozenset(
    {"instance_id", "patient_id", "instruction", "cutoff", "family", "gold_answer"}
)
_SOURCE_KEYS = frozenset({"path", "format", "columns", "json_fields", "collection_key"})


@runtime_checkable
class DatasetAdapter(Protocol):
    """Convert one adapter-specific manifest section to strict episodes."""

    adapter_id: str

    def build_episodes(
        self,
        manifest: Mapping[str, Any],
        *,
        manifest_directory: Path,
    ) -> Sequence[Episode]:
        """Return deterministic, cutoff-safe episodes."""


class CredentialedAdapterUnavailable(DataValidationError):
    """Raised when a benchmark adapter has not been supplied by its data owner."""


@dataclass(frozen=True, slots=True)
class UnavailableCredentialedAdapter:
    """Named placeholder that deliberately contains no invented benchmark schema."""

    adapter_id: str
    installation_hint: str

    def build_episodes(
        self,
        manifest: Mapping[str, Any],
        *,
        manifest_directory: Path,
    ) -> Sequence[Episode]:
        del manifest, manifest_directory
        raise CredentialedAdapterUnavailable(
            f"adapter {self.adapter_id!r} is not bundled because its frozen source schema "
            f"is not public; {self.installation_hint}"
        )


def credentialed_adapter_placeholders() -> dict[str, DatasetAdapter]:
    """Return honest extension points for datasets named by the manuscript."""

    hint = "install and register an authorized adapter that implements DatasetAdapter"
    return {
        name: UnavailableCredentialedAdapter(name, hint)
        for name in (
            "mimic_iv",
            "medcalc_bench",
            "medagentbench_query",
            "ehrshot",
        )
    }


@dataclass(frozen=True, slots=True)
class TabularSource:
    path: Path
    format: str
    columns: Mapping[str, str]
    json_fields: frozenset[str]
    collection_key: str | None


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    unknown = set(value) - expected
    if unknown:
        raise DataValidationError(f"{location} has unknown fields: {sorted(unknown)}")


def _nonempty_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{location} must be a non-empty string")
    return value.strip()


def _source_path(raw: Any, *, manifest_directory: Path, location: str) -> Path:
    relative = Path(_nonempty_text(raw, f"{location}.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise DataValidationError(f"{location}.path must be relative and cannot contain '..'")
    resolved_root = manifest_directory.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:  # Also protects against symlinks escaping the manifest root.
        raise DataValidationError(f"{location}.path escapes the manifest directory") from error
    if not resolved.is_file():
        raise DataValidationError(f"{location}.path is not a regular file: {relative.as_posix()}")
    return resolved


def _parse_source(
    raw: Any,
    *,
    manifest_directory: Path,
    location: str,
    permitted_fields: frozenset[str],
    required_fields: frozenset[str],
) -> TabularSource:
    if not isinstance(raw, Mapping):
        raise DataValidationError(f"{location} must be an object")
    _strict_keys(raw, _SOURCE_KEYS, location)
    path = _source_path(raw.get("path"), manifest_directory=manifest_directory, location=location)

    raw_format = raw.get("format", path.suffix.lower().lstrip("."))
    source_format = _nonempty_text(raw_format, f"{location}.format").lower()
    if source_format == "ndjson":
        source_format = "jsonl"
    if source_format not in {"csv", "json", "jsonl"}:
        raise DataValidationError(f"{location}.format must be csv, json, or jsonl")

    raw_columns = raw.get("columns")
    if not isinstance(raw_columns, Mapping):
        raise DataValidationError(f"{location}.columns must be an object")
    canonical_fields = set(raw_columns)
    unknown = canonical_fields - permitted_fields
    missing = required_fields - canonical_fields
    if unknown or missing:
        raise DataValidationError(
            f"{location}.columns mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    columns: dict[str, str] = {}
    for canonical, source_name in raw_columns.items():
        columns[str(canonical)] = _nonempty_text(source_name, f"{location}.columns.{canonical}")
    duplicate_sources = sorted(
        source_name
        for source_name in set(columns.values())
        if list(columns.values()).count(source_name) > 1
    )
    if duplicate_sources:
        raise DataValidationError(
            f"{location}.columns maps multiple fields to: {duplicate_sources}"
        )

    raw_json_fields = raw.get("json_fields", [])
    if not isinstance(raw_json_fields, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_json_fields
    ):
        raise DataValidationError(f"{location}.json_fields must be an array of field names")
    json_fields = frozenset(raw_json_fields)
    if not json_fields.issubset(canonical_fields):
        raise DataValidationError(
            f"{location}.json_fields contains unmapped fields: "
            f"{sorted(json_fields - canonical_fields)}"
        )

    collection_key = raw.get("collection_key")
    if collection_key is not None:
        collection_key = _nonempty_text(collection_key, f"{location}.collection_key")
    if source_format == "json" and collection_key is None:
        collection_key = "events" if location == "events" else "tasks"
    if source_format != "json" and collection_key is not None:
        raise DataValidationError(f"{location}.collection_key is only valid for JSON sources")
    return TabularSource(path, source_format, columns, json_fields, collection_key)


def _load_csv(path: Path) -> list[dict[str, Any]]:
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except UnicodeDecodeError as error:
        raise DataValidationError(f"{path}: input is not valid UTF-8") from error
    with stream:
        try:
            reader = csv.reader(stream, strict=True)
            rows = list(reader)
        except (csv.Error, UnicodeDecodeError) as error:
            raise DataValidationError(f"{path}: invalid UTF-8 CSV: {error}") from error
    if not rows:
        raise DataValidationError(f"{path}: empty CSV input")
    header = rows[0]
    if not header or any(not name.strip() for name in header):
        raise DataValidationError(f"{path}: CSV header names must be non-empty")
    if len(set(header)) != len(header):
        duplicates = sorted({name for name in header if header.count(name) > 1})
        raise DataValidationError(f"{path}: duplicate CSV header names: {duplicates}")
    output: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise DataValidationError(
                f"{path}:{line_number}: expected {len(header)} columns, got {len(row)}"
            )
        output.append(dict(zip(header, row, strict=True)))
    if not output:
        raise DataValidationError(f"{path}: CSV contains no records")
    return output


def _json_records(source: TabularSource) -> list[dict[str, Any]]:
    if source.format == "jsonl":
        raw: Any = load_jsonl(source.path)
    else:
        raw = load_json(source.path)
        if isinstance(raw, Mapping):
            if source.collection_key is None:
                raise DataValidationError(f"{source.path}: JSON object requires collection_key")
            raw = raw.get(source.collection_key)
    if not isinstance(raw, list):
        raise DataValidationError(f"{source.path}: expected an array of records")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise DataValidationError(f"{source.path}: record {index} must be an object")
        output.append(dict(item))
    if not output:
        raise DataValidationError(f"{source.path}: input contains no records")
    return output


def _decode_json_field(value: Any, *, path: Path, index: int, field: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return strict_json_loads(value)
    except ValueError as error:
        raise DataValidationError(
            f"{path}: record {index} field {field!r} is not valid JSON: {error}"
        ) from error


def _mapped_records(source: TabularSource) -> list[dict[str, Any]]:
    rows = _load_csv(source.path) if source.format == "csv" else _json_records(source)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = sorted(set(source.columns.values()) - set(row))
        if missing:
            raise DataValidationError(
                f"{source.path}: record {index} is missing mapped columns: {missing}"
            )
        mapped = {canonical: row[column] for canonical, column in source.columns.items()}
        for field in source.json_fields:
            mapped[field] = _decode_json_field(
                mapped[field], path=source.path, index=index, field=field
            )
        output.append(mapped)
    return output


def _optional_text(value: Any, location: str) -> str | None:
    if value is None or value == "":
        return None
    return _nonempty_text(value, location)


def _identifier(value: Any, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise DataValidationError(f"{location} must be a string or integer ID")
    rendered = str(value).strip()
    if not rendered:
        raise DataValidationError(f"{location} cannot be empty")
    return rendered


def _string_sequence(value: Any, location: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if not isinstance(value, list | tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DataValidationError(f"{location} must be an array of non-empty strings")
    return tuple(item.strip() for item in value)


def _json_scalar(value: Any, location: str) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # RecordEvent/TaskSpec perform the finite check; keep the error contextual here.
        if value != value or value in {float("inf"), float("-inf")}:
            raise DataValidationError(f"{location} must be finite")
        return value
    raise DataValidationError(f"{location} must be a JSON scalar")


def _metadata(value: Any, location: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DataValidationError(f"{location} must be a JSON object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class GenericEventAdapter:
    """Join explicitly mapped task rows to generic CSV/JSON event rows."""

    adapter_id: str = GENERIC_ADAPTER_ID

    def build_episodes(
        self,
        manifest: Mapping[str, Any],
        *,
        manifest_directory: Path,
    ) -> Sequence[Episode]:
        adapter_config = manifest.get("adapter_config")
        if adapter_config not in (None, {}):
            raise DataValidationError("generic_events does not accept adapter_config")
        event_source = _parse_source(
            manifest.get("events"),
            manifest_directory=manifest_directory,
            location="events",
            permitted_fields=_EVENT_FIELDS,
            required_fields=_REQUIRED_EVENT_FIELDS,
        )
        task_source = _parse_source(
            manifest.get("tasks"),
            manifest_directory=manifest_directory,
            location="tasks",
            permitted_fields=_TASK_FIELDS,
            required_fields=_REQUIRED_TASK_FIELDS,
        )
        events = self._events(_mapped_records(event_source))
        return self._episodes(_mapped_records(task_source), events)

    @staticmethod
    def _events(rows: Iterable[Mapping[str, Any]]) -> tuple[RecordEvent, ...]:
        events: list[RecordEvent] = []
        pointers: dict[str, int] = {}
        for index, row in enumerate(rows):
            pointer = _identifier(row["pointer"], f"events[{index}].pointer")
            if pointer in pointers:
                raise DataValidationError(
                    f"duplicate event pointer {pointer!r} at rows {pointers[pointer]} and {index}"
                )
            pointers[pointer] = index
            try:
                event = RecordEvent.from_dict(
                    {
                        "pointer": pointer,
                        "patient_id": _identifier(row["patient_id"], f"events[{index}].patient_id"),
                        "timestamp": row["timestamp"],
                        "domain": _nonempty_text(row["domain"], f"events[{index}].domain"),
                        "field": _nonempty_text(row["field"], f"events[{index}].field"),
                        "value": _json_scalar(row["value"], f"events[{index}].value"),
                        "unit": _optional_text(row.get("unit"), f"events[{index}].unit"),
                        "text": _optional_text(row.get("text"), f"events[{index}].text"),
                        "source": _optional_text(row.get("source"), f"events[{index}].source"),
                        "code": _optional_text(row.get("code"), f"events[{index}].code"),
                    }
                )
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise DataValidationError(f"invalid events[{index}]: {error}") from error
            events.append(event)
        return tuple(events)

    @staticmethod
    def _episodes(
        rows: Iterable[Mapping[str, Any]], events: Sequence[RecordEvent]
    ) -> tuple[Episode, ...]:
        by_patient: dict[str, list[RecordEvent]] = {}
        for event in events:
            by_patient.setdefault(event.patient_id, []).append(event)
        for patient_events in by_patient.values():
            patient_events.sort(key=lambda event: (event.timestamp, event.pointer))

        episodes: list[Episode] = []
        seen_instances: dict[str, int] = {}
        for index, row in enumerate(rows):
            instance_id = _identifier(row["instance_id"], f"tasks[{index}].instance_id")
            if instance_id in seen_instances:
                raise DataValidationError(
                    f"duplicate task instance_id {instance_id!r} at rows "
                    f"{seen_instances[instance_id]} and {index}"
                )
            seen_instances[instance_id] = index
            patient_id = _identifier(row["patient_id"], f"tasks[{index}].patient_id")
            try:
                cutoff = parse_timestamp(row["cutoff"])
                task = TaskSpec(
                    instance_id=instance_id,
                    patient_id=patient_id,
                    instruction=_nonempty_text(row["instruction"], f"tasks[{index}].instruction"),
                    cutoff=cutoff,
                    family=_nonempty_text(row["family"], f"tasks[{index}].family"),
                    requested_fields=_string_sequence(
                        row.get("requested_fields"), f"tasks[{index}].requested_fields"
                    ),
                    requested_window=TimeWindow.parse(row.get("requested_window", "all")),
                    answer_type=_nonempty_text(
                        row.get("answer_type", "text"), f"tasks[{index}].answer_type"
                    ),
                    metadata=_metadata(row.get("metadata"), f"tasks[{index}].metadata"),
                )
                # The prepared artifact intentionally omits every post-cutoff event.
                admissible = tuple(
                    event for event in by_patient.get(patient_id, ()) if event.timestamp <= cutoff
                )
                reference = _string_sequence(
                    row.get("reference_evidence"), f"tasks[{index}].reference_evidence"
                )
                episode = Episode(
                    task=task,
                    events=admissible,
                    gold_answer=_json_scalar(row["gold_answer"], f"tasks[{index}].gold_answer"),
                    reference_evidence=reference,
                )
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise DataValidationError(f"invalid tasks[{index}]: {error}") from error
            episodes.append(episode)
        episodes.sort(key=lambda episode: episode.task.instance_id)
        return tuple(episodes)


@dataclass(frozen=True, slots=True)
class LatestFieldEventAdapter:
    """Construct deterministic latest-field tasks directly from mapped events.

    This public task builder is intentionally narrow and auditable. It is useful for
    smoke runs and source integration, but it is not a substitute for the manuscript's
    unpublished benchmark-specific task constructors.
    """

    adapter_id: str = LATEST_FIELD_ADAPTER_ID

    def build_episodes(
        self,
        manifest: Mapping[str, Any],
        *,
        manifest_directory: Path,
    ) -> Sequence[Episode]:
        if manifest.get("tasks") is not None:
            raise DataValidationError(
                "latest_field_events constructs tasks and rejects tasks input"
            )
        event_source = _parse_source(
            manifest.get("events"),
            manifest_directory=manifest_directory,
            location="events",
            permitted_fields=_EVENT_FIELDS,
            required_fields=_REQUIRED_EVENT_FIELDS,
        )
        events = GenericEventAdapter._events(_mapped_records(event_source))
        config = self._parse_config(manifest.get("adapter_config"))
        return self._episodes(events, **config)

    @staticmethod
    def _parse_config(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise DataValidationError("latest_field_events requires adapter_config")
        expected = frozenset(
            {
                "fields",
                "requested_window",
                "family",
                "task_type",
                "instruction_template",
                "answer_type",
            }
        )
        _strict_keys(raw, expected, "adapter_config")
        fields = _string_sequence(raw.get("fields"), "adapter_config.fields")
        if not fields:
            raise DataValidationError("adapter_config.fields cannot be empty")
        normalized_fields = tuple(normalize_field(field) for field in fields)
        if len(set(normalized_fields)) != len(normalized_fields):
            raise DataValidationError("adapter_config.fields collide after normalization")
        try:
            window = TimeWindow.parse(raw.get("requested_window", "all"))
        except (AttributeError, TypeError, ValueError) as error:
            raise DataValidationError(
                f"invalid adapter_config.requested_window: {error}"
            ) from error
        family = _nonempty_text(raw.get("family", "latest_field"), "adapter_config.family")
        task_type = _nonempty_text(raw.get("task_type", family), "adapter_config.task_type")
        template = _nonempty_text(
            raw.get("instruction_template", "What is the latest {field} in {window}?"),
            "adapter_config.instruction_template",
        )
        remainder = template.replace("{field}", "").replace("{window}", "")
        if "{" in remainder or "}" in remainder:
            raise DataValidationError(
                "adapter_config.instruction_template supports only {field} and {window}"
            )
        answer_type = _nonempty_text(raw.get("answer_type", "auto"), "adapter_config.answer_type")
        return {
            "fields": normalized_fields,
            "window": window,
            "family": family,
            "task_type": task_type,
            "template": template,
            "answer_type": answer_type,
        }

    @staticmethod
    def _episodes(
        events: Sequence[RecordEvent],
        *,
        fields: Sequence[str],
        window: TimeWindow,
        family: str,
        task_type: str,
        template: str,
        answer_type: str,
    ) -> tuple[Episode, ...]:
        by_patient: dict[str, list[RecordEvent]] = {}
        for event in events:
            by_patient.setdefault(event.patient_id, []).append(event)
        candidates: list[tuple[Episode, int]] = []
        for patient_id in sorted(by_patient):
            patient_events = sorted(
                by_patient[patient_id], key=lambda event: (event.timestamp, event.pointer)
            )
            cutoff = patient_events[-1].timestamp
            for field in fields:
                matching = [
                    event
                    for event in patient_events
                    if event.field == field and window.contains(event.timestamp, cutoff)
                ]
                if not matching:
                    continue
                latest_time = matching[-1].timestamp
                tied = [event for event in matching if event.timestamp == latest_time]
                if any(
                    (event.value, event.unit) != (tied[0].value, tied[0].unit)
                    or type(event.value) is not type(tied[0].value)
                    for event in tied[1:]
                ):
                    raise DataValidationError(
                        f"ambiguous latest value for patient {patient_id!r}, field {field!r}"
                    )
                latest = tied[0]  # Same pointer tie-break as StateManager.QueryField.
                if latest.value is None:
                    raise DataValidationError(f"latest value for {field!r} is null")
                resolved_answer_type = answer_type
                if resolved_answer_type == "auto":
                    resolved_answer_type = (
                        "number"
                        if isinstance(latest.value, int | float)
                        and not isinstance(latest.value, bool)
                        else "text"
                    )
                instance_id = f"latest-field:{patient_id}:{field}:{cutoff.isoformat()}"
                task = TaskSpec(
                    instance_id=instance_id,
                    patient_id=patient_id,
                    instruction=template.replace("{field}", field).replace(
                        "{window}", window.label
                    ),
                    cutoff=cutoff,
                    family=family,
                    requested_fields=(field,),
                    requested_window=window,
                    answer_type=resolved_answer_type,
                    metadata={
                        "history_quartile": 1,
                        "suite": "retrieval",
                        "task_type": task_type,
                        "task_builder": LATEST_FIELD_ADAPTER_ID,
                    },
                )
                candidates.append(
                    (
                        Episode(
                            task=task,
                            events=tuple(patient_events),
                            gold_answer=latest.value,
                            reference_evidence=(latest.pointer,),
                        ),
                        len(patient_events),
                    )
                )
        if not candidates:
            raise DataValidationError("latest_field_events produced no answerable episodes")
        counts = sorted(count for _, count in candidates)
        total = len(counts)
        episodes: list[Episode] = []
        for episode, count in candidates:
            quartile = min(4, 1 + (bisect_left(counts, count) * 4) // total)
            task = replace(
                episode.task,
                metadata={**episode.task.metadata, "history_quartile": quartile},
            )
            episodes.append(
                Episode(task, episode.events, episode.gold_answer, episode.reference_evidence)
            )
        return tuple(sorted(episodes, key=lambda episode: episode.task.instance_id))


def builtin_adapters() -> dict[str, DatasetAdapter]:
    """Return fresh built-in and unavailable-placeholder adapter mappings."""

    adapters: dict[str, DatasetAdapter] = {
        GENERIC_ADAPTER_ID: GenericEventAdapter(),
        LATEST_FIELD_ADAPTER_ID: LatestFieldEventAdapter(),
    }
    adapters.update(credentialed_adapter_placeholders())
    return adapters
