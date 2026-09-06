"""Typed contracts shared by data, environment, verification, and training.

The contracts deliberately keep the terminal gold answer outside the model-visible
``Observation``.  This makes the paper's information boundary inspectable in code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

JsonScalar = str | int | float | bool | None


class Domain(str, Enum):
    LAB = "lab"
    MED = "med"
    DX = "dx"
    PROC = "proc"
    NOTE = "note"


class MemoryStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class ReturnCode(str, Enum):
    FOUND = "FOUND"
    NOT_RECORDED = "NOTRECORDED"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_WINDOW = "OUTOFWINDOW"
    VALUE = "VALUE"
    UNBOUND = "UNBOUND"
    UNRESOLVED = "UNRESOLVED"
    OK = "OK"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    ANSWERED = "ANSWERED"


class ActionKind(str, Enum):
    RETRIEVE = "RETRIEVE"
    QUERY_FIELD = "QUERYFIELD"
    CALCULATE = "CALCULATE"
    UPDATE_MEMORY = "UPDATEMEMORY"
    MARK_STATUS = "MARKSTATUS"
    COMPRESS = "COMPRESS"
    DISCARD = "DISCARD"
    ANSWER = "ANSWER"
    MALFORMED = "MALFORMED"  # Internal history marker; never a legal model action.


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    else:
        raise TypeError("timestamp must be ISO-8601 text or a datetime")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_field(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("field name cannot be empty")
    return normalized


def _finite(value: Any, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


@dataclass(frozen=True)
class TimeWindow:
    """A cutoff-relative evidence window.

    ``lookback=None`` means all admissible pre-cutoff evidence.  ``lag`` can exclude
    the most recent interval, which is useful for explicit historical windows.
    """

    lookback: timedelta | None = None
    lag: timedelta = timedelta(0)
    label: str = "all"

    def __post_init__(self) -> None:
        if self.lookback is not None and self.lookback.total_seconds() < 0:
            raise ValueError("lookback cannot be negative")
        if self.lag.total_seconds() < 0:
            raise ValueError("lag cannot be negative")

    @classmethod
    def parse(cls, value: str | TimeWindow | None) -> TimeWindow:
        if isinstance(value, TimeWindow):
            return value
        if value is None or value.strip().lower() == "all":
            return cls()
        text = value.strip().lower().replace("_", " ")
        match = re.fullmatch(r"last\s+(\d+)\s*([dhmy])", text)
        if not match:
            raise ValueError(f"unsupported time window: {value!r}")
        amount = int(match.group(1))
        unit = match.group(2)
        days = amount * {"d": 1, "h": 1 / 24, "m": 30, "y": 365}[unit]
        return cls(timedelta(days=days), label=f"last {amount}{unit}")

    def contains(self, timestamp: datetime, cutoff: datetime) -> bool:
        event_time = parse_timestamp(timestamp)
        end = parse_timestamp(cutoff) - self.lag
        if event_time > end:
            return False
        return self.lookback is None or event_time >= end - self.lookback

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback_seconds": None if self.lookback is None else self.lookback.total_seconds(),
            "lag_seconds": self.lag.total_seconds(),
            "label": self.label,
        }


@dataclass(frozen=True)
class RecordEvent:
    pointer: str
    patient_id: str
    timestamp: datetime
    domain: Domain
    field: str
    value: JsonScalar
    unit: str | None = None
    text: str | None = None
    source: str | None = None
    code: str | None = None

    def __post_init__(self) -> None:
        if not self.pointer or not self.patient_id:
            raise ValueError("event pointer and patient_id are required")
        object.__setattr__(self, "timestamp", parse_timestamp(self.timestamp))
        object.__setattr__(self, "field", normalize_field(self.field))
        _finite(self.value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RecordEvent:
        required = {"pointer", "patient_id", "timestamp", "domain", "field", "value"}
        optional = {"unit", "text", "source", "code"}
        if required - set(value) or set(value) - required - optional:
            raise ValueError("event has missing or unknown fields")
        for name in ("pointer", "patient_id"):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, str | int) or not str(item).strip():
                raise ValueError(f"event {name} must be a non-empty string or integer ID")
        for name in ("domain", "field"):
            if not isinstance(value[name], str) or not value[name].strip():
                raise ValueError(f"event {name} must be non-empty text")
        for name in optional:
            if value.get(name) is not None and not isinstance(value[name], str):
                raise ValueError(f"event {name} must be text or null")
        if value["value"] is not None and not isinstance(value["value"], str | int | float | bool):
            raise ValueError("event value must be a JSON scalar")
        return cls(
            pointer=str(value["pointer"]),
            patient_id=str(value["patient_id"]),
            timestamp=parse_timestamp(value["timestamp"]),
            domain=Domain(value["domain"].lower()),
            field=value["field"],
            value=value["value"],
            unit=value.get("unit"),
            text=value.get("text"),
            source=value.get("source"),
            code=value.get("code"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "domain": self.domain.value,
            "field": self.field,
            "value": self.value,
            "unit": self.unit,
            "text": self.text,
            "source": self.source,
            "code": self.code,
        }


@dataclass(frozen=True)
class TaskSpec:
    instance_id: str
    patient_id: str
    instruction: str
    cutoff: datetime
    family: str
    requested_fields: tuple[str, ...] = ()
    requested_window: TimeWindow = TimeWindow()
    answer_type: str = "text"
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_id or not self.patient_id or not self.instruction.strip():
            raise ValueError("task identifiers and instruction are required")
        object.__setattr__(self, "cutoff", parse_timestamp(self.cutoff))
        object.__setattr__(
            self, "requested_fields", tuple(normalize_field(item) for item in self.requested_fields)
        )
        object.__setattr__(self, "requested_window", TimeWindow.parse(self.requested_window))
        _finite(self.metadata, "metadata")


@dataclass(frozen=True)
class Episode:
    """Full training/evaluation instance; ``gold_answer`` is never model-visible."""

    task: TaskSpec
    events: tuple[RecordEvent, ...]
    gold_answer: JsonScalar
    reference_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for event in self.events:
            if event.patient_id != self.task.patient_id:
                raise ValueError("episode contains an event from another patient")
            if event.pointer in seen:
                raise ValueError(f"duplicate event pointer: {event.pointer}")
            seen.add(event.pointer)
        missing = set(self.reference_evidence) - seen
        if missing:
            raise ValueError(f"reference evidence pointers not found: {sorted(missing)}")
        _finite(self.gold_answer, "gold_answer")

    @property
    def admissible_events(self) -> tuple[RecordEvent, ...]:
        return tuple(event for event in self.events if event.timestamp <= self.task.cutoff)


@dataclass(frozen=True)
class MemoryItem:
    item_id: str
    field: str
    value: JsonScalar | tuple[JsonScalar, ...]
    status: MemoryStatus
    validity_scope: str
    evidence_pointers: tuple[str, ...]
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.item_id or not self.validity_scope.strip():
            raise ValueError("memory item id and validity scope are required")
        object.__setattr__(self, "field", normalize_field(self.field))
        if not self.evidence_pointers:
            raise ValueError("memory items require at least one evidence pointer")
        pointers = tuple(dict.fromkeys(str(item) for item in self.evidence_pointers))
        object.__setattr__(self, "evidence_pointers", pointers)
        _finite(self.value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryItem:
        raw_value = value.get("value")
        if isinstance(raw_value, list):
            raw_value = tuple(raw_value)
        pointers = value.get("evidence_pointers", value.get("pointers", ()))
        if isinstance(pointers, str):
            pointers = (pointers,)
        return cls(
            item_id=str(value.get("item_id", value.get("id", ""))),
            field=str(value["field"]),
            value=raw_value,
            unit=None if value.get("unit") is None else str(value["unit"]),
            status=MemoryStatus(str(value.get("status", "uncertain")).lower()),
            validity_scope=str(value.get("validity_scope", value.get("scope", "unspecified"))),
            evidence_pointers=tuple(str(pointer) for pointer in pointers),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "field": self.field,
            "value": list(self.value) if isinstance(self.value, tuple) else self.value,
            "unit": self.unit,
            "status": self.status.value,
            "validity_scope": self.validity_scope,
            "evidence_pointers": list(self.evidence_pointers),
        }


@dataclass(frozen=True)
class ToolReturn:
    code: ReturnCode
    value: JsonScalar | tuple[JsonScalar, ...] = None
    unit: str | None = None
    evidence_pointers: tuple[str, ...] = ()
    events: tuple[RecordEvent, ...] = ()
    message: str = ""

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code.value,
            "value": list(self.value) if isinstance(self.value, tuple) else self.value,
            "unit": self.unit,
            "evidence_pointers": list(self.evidence_pointers),
            "message": self.message,
        }
        if include_events:
            result["events"] = [event.to_dict() for event in self.events]
        return result


@dataclass(frozen=True)
class HistoryEntry:
    turn: int
    action: ActionKind
    query_scope: str
    cost: int
    return_code: ReturnCode

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "action": self.action.value,
            "query_scope": self.query_scope,
            "cost": self.cost,
            "return_code": self.return_code.value,
        }


@dataclass(frozen=True)
class Observation:
    task: TaskSpec
    last_return: ToolReturn | None
    memory: tuple[MemoryItem, ...]
    history: tuple[HistoryEntry, ...]
    budget_remaining: int
    budget_cap: int
    turn: int
    turn_cap: int
    memory_capacity: int
    legal_actions: tuple[ActionKind, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.budget_remaining <= self.budget_cap:
            raise ValueError("remaining budget is outside its cap")
        if len(self.memory) > self.memory_capacity:
            raise ValueError("memory exceeds capacity")

    def canonical_payload(self) -> dict[str, Any]:
        """Appendix A.4 collision signature with normalized structured fields."""

        def normalized_value(value: Any) -> Any:
            if isinstance(value, bool) or value is None:
                return value
            if isinstance(value, int | float):
                number = Decimal(str(value))
                rendered = "0" if number == 0 else format(number.normalize(), "f")
                return {"number": rendered}
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list | tuple):
                return [normalized_value(item) for item in value]
            if isinstance(value, Mapping):
                return {
                    str(key): normalized_value(item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                }
            raise TypeError(f"unsupported canonical value: {type(value).__name__}")

        memory = [
            {
                "id": item.item_id.strip(),
                "field": item.field,
                "value": normalized_value(item.value),
                "unit": None if item.unit is None else " ".join(item.unit.split()).casefold(),
                "status": item.status.value,
                "validity_scope": " ".join(item.validity_scope.split()).casefold(),
                "evidence_pointers": sorted(pointer.strip() for pointer in item.evidence_pointers),
            }
            for item in sorted(self.memory, key=lambda value: value.item_id)
        ]
        tool_return = None
        if self.last_return is not None:
            tool_return = {
                "code": self.last_return.code.value,
                "value": normalized_value(self.last_return.value),
                "unit": None
                if self.last_return.unit is None
                else " ".join(self.last_return.unit.split()).casefold(),
                "evidence_pointers": sorted(
                    pointer.strip() for pointer in self.last_return.evidence_pointers
                ),
                "events": [
                    {
                        "pointer": event.pointer.strip(),
                        "timestamp": event.timestamp.isoformat(),
                        "domain": event.domain.value,
                        "field": event.field,
                        "value": normalized_value(event.value),
                        "unit": None
                        if event.unit is None
                        else " ".join(event.unit.split()).casefold(),
                        "code": None if event.code is None else event.code.strip().casefold(),
                    }
                    for event in sorted(self.last_return.events, key=lambda value: value.pointer)
                ],
                "message": " ".join(self.last_return.message.split()),
            }

        return {
            "serialization_version": 3,
            "task": {
                "instruction": " ".join(self.task.instruction.split()),
                "family": self.task.family.strip().casefold(),
                "requested_fields": sorted(self.task.requested_fields),
                "requested_window": self.task.requested_window.to_dict(),
                "answer_type": self.task.answer_type.strip().casefold(),
            },
            "cutoff": self.task.cutoff.isoformat(),
            "tool_return": tool_return,
            "memory": memory,
            "action_window": [entry.to_dict() for entry in self.history],
            "budget_remaining": self.budget_remaining,
            "legal_action_mask": sorted(action.value for action in self.legal_actions),
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    def exact_bytes(self) -> bytes:
        """Lossless deterministic serialization for exact replay/fork assertions."""

        task = {
            "instance_id": self.task.instance_id,
            "patient_id": self.task.patient_id,
            "instruction": self.task.instruction,
            "cutoff": self.task.cutoff.isoformat(),
            "family": self.task.family,
            "requested_fields": list(self.task.requested_fields),
            "requested_window": self.task.requested_window.to_dict(),
            "answer_type": self.task.answer_type,
            "metadata": dict(self.task.metadata),
        }
        payload = {
            "serialization_version": 1,
            "task": task,
            "last_return": None if self.last_return is None else self.last_return.to_dict(),
            "memory": [item.to_dict() for item in self.memory],
            "history": [entry.to_dict() for entry in self.history],
            "budget_remaining": self.budget_remaining,
            "budget_cap": self.budget_cap,
            "turn": self.turn,
            "turn_cap": self.turn_cap,
            "memory_capacity": self.memory_capacity,
            "legal_actions": [action.value for action in self.legal_actions],
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @property
    def state_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def event_index(events: Iterable[RecordEvent]) -> dict[str, RecordEvent]:
    result: dict[str, RecordEvent] = {}
    for event in events:
        if event.pointer in result:
            raise ValueError(f"duplicate event pointer: {event.pointer}")
        result[event.pointer] = event
    return result
