"""Model-visible surfaces for bounded-memory interaction."""

from __future__ import annotations

import json
from collections.abc import Iterable

from vapa.schemas import HistoryEntry, MemoryItem, Observation, RecordEvent, ToolReturn

SYSTEM_PROMPT = (
    "You answer one question about one patient record. "
    "Use only evidence recorded on or before\n"
    "the cutoff. Emit exactly one action per turn.\n"
    """Actions: Retrieve(query, window, domain) | QueryField(field, window) |
Calculate(calculator, bindings) | UpdateMemory(item) |
MarkStatus(id, status, scope, why) | Compress(ids) | Discard(id) |
Answer(prediction, evidence)
End your reply with the action call on one line, arguments in the order listed above and
comma-separated, and write nothing after that line.
window is a relative span such as last 90d, last 1y, or all. domain is one of lab, med,
dx, proc, note. bindings maps each required calculator input to a memory id or an
evidence pointer. evidence is a bracketed pointer list such as [e#4471,e#5210], and a
pointer is the ptr field of a tool return.
Memory holds at most {memory_capacity} items, each with a value, a status of current | stale |
contradicted | uncertain, a validity scope, and an evidence pointer. A write to full
memory is rejected. Status changes require MarkStatus.
Every non-answer action costs one unit. Answer is free, ends the episode, and must cite
the pointers it relies on. At zero budget only Answer is legal.{scaffold}"""
)


def render_system_prompt(memory_capacity: int, scaffold: str = "") -> str:
    inserted = f"\n{scaffold.strip()}" if scaffold.strip() else ""
    return SYSTEM_PROMPT.format(memory_capacity=memory_capacity, scaffold=inserted)


def _memory_payload(memory: Iterable[MemoryItem]) -> str:
    return json.dumps(
        [item.to_dict() for item in memory],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _history_payload(history: Iterable[HistoryEntry]) -> str:
    return json.dumps(
        [entry.to_dict() for entry in history],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _safe_event_payload(event: RecordEvent) -> dict[str, object]:
    """Project a record event onto the model-visible evidence contract.

    Patient identifiers, source-system metadata, and free text deliberately stay
    outside the generation surface. Structured evidence retains only the fields
    needed to resolve and cite the returned pointer.
    """

    return {
        "pointer": event.pointer,
        "timestamp": event.timestamp.isoformat(),
        "domain": event.domain.value,
        "field": event.field,
        "value": event.value,
        "unit": event.unit,
        "code": event.code,
    }


def render_tool_return(result: ToolReturn | None) -> str:
    if result is None:
        return "NONE"
    payload = {
        "code": result.code.value,
        "value": list(result.value) if isinstance(result.value, tuple) else result.value,
        "unit": result.unit,
        "evidence_pointers": list(result.evidence_pointers),
        "message": result.message,
        "events": [_safe_event_payload(event) for event in result.events],
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def render_observation(observation: Observation) -> str:
    return (
        f"Cutoff {observation.task.cutoff.isoformat()} "
        f"Budget {observation.budget_remaining}/{observation.budget_cap} "
        f"Memory {len(observation.memory)}/{observation.memory_capacity}\n"
        f"Task: {observation.task.instruction}\n"
        f"Legal actions: {[action.value for action in observation.legal_actions]}\n"
        f"Memory: {_memory_payload(observation.memory)} "
        f"Last return: {render_tool_return(observation.last_return)} "
        f"History: {_history_payload(observation.history)}"
    )


def render_chat(observation: Observation, scaffold: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": render_system_prompt(observation.memory_capacity, scaffold)},
        {"role": "user", "content": render_observation(observation)},
    ]
