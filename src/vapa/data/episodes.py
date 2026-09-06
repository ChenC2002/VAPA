"""Conversion between strict JSON records and the in-memory episode contract."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vapa.artifacts import ArtifactContentKind, atomic_write_text, guard_artifact_write_path
from vapa.data.io import DataValidationError, load_json, load_jsonl
from vapa.schemas import Episode, RecordEvent, TaskSpec, TimeWindow, parse_timestamp

_EPISODE_KEYS = {
    "schema_version",
    "instance_id",
    "episode_id",
    "patient_id",
    "instruction",
    "family",
    "cutoff",
    "requested_fields",
    "requested_window",
    "answer_type",
    "metadata",
    "gold_answer",
    "reference_evidence",
    "events",
}


def _text(value: object, name: str, *, identifier: bool = False) -> str:
    if identifier and isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} must be non-empty text")
    return value


def _text_array(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DataValidationError(f"{name} must be an array")
    return tuple(_text(item, name) for item in value)


def episode_from_record(record: Mapping[str, Any]) -> Episode:
    unknown = set(record) - _EPISODE_KEYS
    if unknown:
        raise DataValidationError(f"unknown episode fields: {sorted(unknown)}")
    version = record.get("schema_version", 1)
    if type(version) is not int or version != 1:
        raise DataValidationError(f"unsupported episode schema_version: {version}")
    instance_id = record.get("instance_id", record.get("episode_id"))
    if instance_id is None:
        raise DataValidationError("episode is missing instance_id")
    if "instance_id" in record and "episode_id" in record:
        if _text(record["episode_id"], "episode_id", identifier=True) != _text(
            instance_id, "instance_id", identifier=True
        ):
            raise DataValidationError("instance_id and episode_id disagree")
    raw_events = record.get("events")
    if not isinstance(raw_events, list) or any(not isinstance(item, dict) for item in raw_events):
        raise DataValidationError("episode events must be an array of objects")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DataValidationError("episode metadata must be an object")
    try:
        task = TaskSpec(
            instance_id=_text(instance_id, "instance_id", identifier=True),
            patient_id=_text(record["patient_id"], "patient_id", identifier=True),
            instruction=_text(record["instruction"], "instruction"),
            cutoff=parse_timestamp(record["cutoff"]),
            family=_text(record["family"], "family"),
            requested_fields=_text_array(record.get("requested_fields", []), "requested_fields"),
            requested_window=TimeWindow.parse(
                _text(record.get("requested_window", "all"), "requested_window")
            ),
            answer_type=_text(record.get("answer_type", "text"), "answer_type"),
            metadata=metadata,
        )
        events = tuple(RecordEvent.from_dict(item) for item in raw_events)
        reference = _text_array(record.get("reference_evidence", []), "reference_evidence")
        return Episode(task, events, record.get("gold_answer"), reference)
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"invalid episode {instance_id!r}: {exc}") from exc


def episode_to_record(episode: Episode) -> dict[str, Any]:
    task = episode.task
    return {
        "schema_version": 1,
        "instance_id": task.instance_id,
        "patient_id": task.patient_id,
        "instruction": task.instruction,
        "family": task.family,
        "cutoff": task.cutoff.isoformat(),
        "requested_fields": list(task.requested_fields),
        "requested_window": task.requested_window.label,
        "answer_type": task.answer_type,
        "metadata": dict(task.metadata),
        "gold_answer": episode.gold_answer,
        "reference_evidence": list(episode.reference_evidence),
        "events": [event.to_dict() for event in episode.events],
    }


def load_episode_objects(path: str | Path) -> list[Episode]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = load_json(source)
        if isinstance(payload, dict) and ({"instance_id", "episode_id"} & set(payload)):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            records = payload["episodes"]
        else:
            raise DataValidationError("episode JSON must be a record, array, or episodes array")
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        records = load_jsonl(source)
    else:
        raise DataValidationError(f"unsupported episode format: {source.suffix}")
    if any(not isinstance(record, Mapping) for record in records):
        raise DataValidationError("every episode must be a JSON object")
    episodes = [episode_from_record(record) for record in records]
    seen: set[str] = set()
    for episode in episodes:
        instance_id = episode.task.instance_id
        if instance_id in seen:
            raise DataValidationError(f"duplicate episode ID: {instance_id!r}")
        seen.add(instance_id)
    return episodes


def write_episode_jsonl(
    episodes: Iterable[Episode],
    path: str | Path,
    *,
    repository_root: str | Path,
    content_kind: ArtifactContentKind | str = ArtifactContentKind.CREDENTIALED,
) -> Path:
    """Write protected episode JSONL only after the release-path guard passes."""

    destination = guard_artifact_write_path(
        path, content_kind=content_kind, repository_root=repository_root
    )
    payload = "".join(
        json.dumps(
            episode_to_record(episode), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
        for episode in episodes
    )
    return atomic_write_text(destination, payload)
