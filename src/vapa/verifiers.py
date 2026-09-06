"""Deterministic, record-grounded process verifiers.

The PDF publishes verifier families but not the complete predicate catalog or its
numeric weights.  ``VerifierCatalog.demo_default`` therefore provides an auditable
reference implementation for synthetic tests; it is deliberately marked non-paper-exact.
Production reconstruction must load the authors' frozen catalog and hash it in the run
manifest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from vapa.actions import Action
from vapa.rollouts import Rollout
from vapa.schemas import (
    ActionKind,
    Episode,
    MemoryStatus,
    Observation,
    RecordEvent,
    ReturnCode,
    TaskSpec,
    ToolReturn,
    event_index,
    normalize_field,
)


class VerifierFamily(str, Enum):
    EXTRACTION = "extraction"
    GROUNDING = "grounding"
    TEMPORAL = "temporal_validity"
    MISSINGNESS = "missingness"
    STATUS_SCOPE = "status_and_scope"
    MEMORY = "compression_and_discard"
    STOPPING = "stopping"


@dataclass(frozen=True, slots=True)
class VerifierContext:
    """The complete data boundary available to a process-verifier predicate."""

    task: TaskSpec
    admissible_events: tuple[RecordEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "admissible_events", tuple(self.admissible_events))
        seen: set[str] = set()
        for event in self.admissible_events:
            if event.patient_id != self.task.patient_id:
                raise ValueError("verifier context contains an event from another patient")
            if event.timestamp > self.task.cutoff:
                raise ValueError("verifier context cannot contain post-cutoff events")
            if event.pointer in seen:
                raise ValueError(f"duplicate verifier-context event pointer: {event.pointer}")
            seen.add(event.pointer)


@dataclass(frozen=True, slots=True)
class VerifierTurn:
    """Read-only turn projection without rewards, advantages, or rollout metadata."""

    index: int
    observation: Observation
    action: Action | None
    tool_return: ToolReturn
    cost: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class VerifierTrace:
    """Completed trace available to consequence-aware predicates."""

    turns: tuple[VerifierTurn, ...]

    @classmethod
    def from_rollout(cls, context: VerifierContext, rollout: Rollout) -> VerifierTrace:
        allowed_events = event_index(context.admissible_events)
        allowed_pointers = set(allowed_events)
        projected: list[VerifierTurn] = []
        for turn in rollout.turns:
            if turn.observation.task != context.task:
                raise ValueError("verifier trace observation belongs to another task")
            for item in turn.observation.memory:
                if not set(item.evidence_pointers) <= allowed_pointers:
                    raise ValueError("verifier trace memory contains inadmissible evidence")
            visible_returns = [turn.tool_return]
            if turn.observation.last_return is not None:
                visible_returns.append(turn.observation.last_return)
            for tool_return in visible_returns:
                if not set(tool_return.evidence_pointers) <= allowed_pointers:
                    raise ValueError("verifier trace return contains inadmissible evidence")
                for event in tool_return.events:
                    if (
                        event.timestamp > context.task.cutoff
                        or event.pointer not in allowed_pointers
                        or event != allowed_events[event.pointer]
                    ):
                        raise ValueError(
                            "verifier trace contains an inadmissible tool-return event"
                        )
            projected.append(
                VerifierTurn(
                    turn.index,
                    turn.observation,
                    turn.action,
                    turn.tool_return,
                    turn.cost,
                    turn.accepted,
                )
            )
        return cls(tuple(projected))


VerifierFunction = Callable[[VerifierContext, VerifierTrace, int], float | None]


@dataclass(frozen=True)
class Predicate:
    name: str
    family: VerifierFamily
    weight: float
    actions: frozenset[ActionKind]
    function: VerifierFunction
    reliability: str = "exact"

    def __post_init__(self) -> None:
        if not self.name or self.weight <= 0 or not math.isfinite(self.weight):
            raise ValueError("predicate name and finite positive weight are required")
        if not self.actions:
            raise ValueError("predicate must apply to at least one action family")


@dataclass(frozen=True)
class PredicateResult:
    name: str
    family: VerifierFamily
    weight: float
    fired: bool
    value: float | None


@dataclass(frozen=True)
class ProcessScore:
    reward: float
    denominator: float
    results: tuple[PredicateResult, ...]

    @property
    def fired_count(self) -> int:
        return sum(result.fired for result in self.results)


class VerifierCatalog:
    def __init__(
        self, predicates: Iterable[Predicate], *, catalog_id: str, paper_exact: bool
    ) -> None:
        self.predicates = tuple(predicates)
        self.catalog_id = catalog_id
        self.paper_exact = paper_exact
        names = [predicate.name for predicate in self.predicates]
        if len(names) != len(set(names)):
            raise ValueError("verifier predicate names must be unique")

    def score_turn(self, episode: Episode, rollout: Rollout, turn_index: int) -> ProcessScore:
        context = VerifierContext(episode.task, episode.admissible_events)
        return self._score_turn(context, VerifierTrace.from_rollout(context, rollout), turn_index)

    def _score_turn(
        self, context: VerifierContext, trace: VerifierTrace, turn_index: int
    ) -> ProcessScore:
        turn = trace.turns[turn_index]
        if turn.action is None:
            return ProcessScore(0.0, 0.0, ())
        applicable = [
            predicate for predicate in self.predicates if turn.action.kind in predicate.actions
        ]
        denominator = sum(predicate.weight for predicate in applicable)
        weighted = 0.0
        results: list[PredicateResult] = []
        for predicate in applicable:
            value = predicate.function(context, trace, turn_index)
            if value is not None:
                if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                    raise ValueError(f"predicate {predicate.name} returned an invalid score")
                weighted += predicate.weight * value
            results.append(
                PredicateResult(
                    predicate.name,
                    predicate.family,
                    predicate.weight,
                    value is not None,
                    value,
                )
            )
        reward = 0.0 if denominator == 0 else weighted / denominator
        return ProcessScore(reward, denominator, tuple(results))

    def score_rollout(self, episode: Episode, rollout: Rollout) -> None:
        """Attach scores in place, preserving already copied tree-prefix scores."""

        context = VerifierContext(episode.task, episode.admissible_events)
        trace = VerifierTrace.from_rollout(context, rollout)
        for index, turn in enumerate(rollout.turns):
            if turn.copied_prefix:
                continue
            score = self._score_turn(context, trace, index)
            turn.process_reward = score.reward
            turn.verifier_scores = {result.name: result.value for result in score.results}

    @classmethod
    def demo_default(cls) -> VerifierCatalog:
        evidence_actions = frozenset(
            {
                ActionKind.QUERY_FIELD,
                ActionKind.CALCULATE,
                ActionKind.UPDATE_MEMORY,
                ActionKind.MARK_STATUS,
                ActionKind.COMPRESS,
                ActionKind.DISCARD,
                ActionKind.ANSWER,
            }
        )
        return cls(
            [
                Predicate(
                    "extracted_value_matches",
                    VerifierFamily.EXTRACTION,
                    2.0,
                    frozenset(
                        {
                            ActionKind.QUERY_FIELD,
                            ActionKind.UPDATE_MEMORY,
                            ActionKind.ANSWER,
                        }
                    ),
                    _extracted_value_matches,
                ),
                Predicate(
                    "pointers_resolve",
                    VerifierFamily.GROUNDING,
                    2.0,
                    evidence_actions,
                    _pointers_resolve,
                ),
                Predicate(
                    "window_respected",
                    VerifierFamily.TEMPORAL,
                    2.0,
                    evidence_actions,
                    _window_respected,
                ),
                Predicate(
                    "missingness_correct",
                    VerifierFamily.MISSINGNESS,
                    2.0,
                    frozenset({ActionKind.QUERY_FIELD, ActionKind.ANSWER}),
                    _missingness_correct,
                ),
                Predicate(
                    "status_scope_consistent",
                    VerifierFamily.STATUS_SCOPE,
                    1.0,
                    frozenset(
                        {ActionKind.UPDATE_MEMORY, ActionKind.MARK_STATUS, ActionKind.ANSWER}
                    ),
                    _status_scope_consistent,
                    reliability="heuristic",
                ),
                Predicate(
                    "memory_preserves_needed_evidence",
                    VerifierFamily.MEMORY,
                    2.0,
                    frozenset({ActionKind.COMPRESS, ActionKind.DISCARD}),
                    _memory_preserves_needed_evidence,
                ),
                Predicate(
                    "stopping_progress",
                    VerifierFamily.STOPPING,
                    1.0,
                    frozenset(set(ActionKind) - {ActionKind.MALFORMED}),
                    _stopping_progress,
                    reliability="heuristic",
                ),
            ],
            catalog_id="demo-v1-not-paper-exact",
            paper_exact=False,
        )


def _turn_pointers(turn: VerifierTurn) -> tuple[str, ...]:
    pointers: list[str] = list(turn.tool_return.evidence_pointers)
    if turn.action is None:
        return tuple(dict.fromkeys(pointers))

    def visit(value: object) -> None:
        if isinstance(value, str) and value.startswith("e#"):
            pointers.append(value)
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                visit(nested)

    visit(turn.action.arguments)
    return tuple(dict.fromkeys(pointers))


def _asserted_value(turn: VerifierTurn) -> object | None:
    if turn.action is None:
        return None
    if turn.action.kind is ActionKind.UPDATE_MEMORY:
        item = turn.action.arguments.get("item")
        return item.get("value") if isinstance(item, Mapping) else None
    if turn.action.kind is ActionKind.ANSWER:
        return turn.action.arguments.get("prediction")
    if turn.action.kind is ActionKind.QUERY_FIELD:
        return turn.tool_return.value
    return None


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return str(left).strip().casefold() == str(right).strip().casefold()


def _extracted_value_matches(
    context: VerifierContext, trace: VerifierTrace, index: int
) -> float | None:
    turn = trace.turns[index]
    pointers = _turn_pointers(turn)
    value = _asserted_value(turn)
    if not pointers or value is None:
        return None
    events = event_index(context.admissible_events)
    resolved = [events[pointer] for pointer in pointers if pointer in events]
    if not resolved:
        return -1.0
    return 1.0 if any(_same_value(value, event.value) for event in resolved) else -1.0


def _pointers_resolve(context: VerifierContext, trace: VerifierTrace, index: int) -> float | None:
    pointers = _turn_pointers(trace.turns[index])
    if not pointers:
        return None
    events = event_index(context.admissible_events)
    return 1.0 if all(pointer in events for pointer in pointers) else -1.0


def _window_respected(context: VerifierContext, trace: VerifierTrace, index: int) -> float | None:
    pointers = _turn_pointers(trace.turns[index])
    if not pointers:
        return None
    events = event_index(context.admissible_events)
    if any(pointer not in events for pointer in pointers):
        return -1.0
    window = context.task.requested_window
    return (
        1.0
        if all(
            window.contains(events[pointer].timestamp, context.task.cutoff) for pointer in pointers
        )
        else -1.0
    )


def _missingness_correct(
    context: VerifierContext, trace: VerifierTrace, index: int
) -> float | None:
    turn = trace.turns[index]
    if turn.action is None:
        return None
    fields = context.task.requested_fields
    if turn.action.kind is ActionKind.QUERY_FIELD:
        fields = (normalize_field(str(turn.action.arguments["field"])),)
        claims_missing = turn.tool_return.code is ReturnCode.NOT_RECORDED
    elif turn.action.kind is ActionKind.ANSWER:
        claims_missing = (
            str(turn.action.arguments["prediction"]).replace("_", "").upper() == "NOTRECORDED"
        )
        if not claims_missing:
            return None
    else:
        return None
    if not fields:
        return None
    exists = any(event.field in fields for event in context.admissible_events)
    return 1.0 if claims_missing is (not exists) else -1.0


def _status_scope_consistent(
    context: VerifierContext, trace: VerifierTrace, index: int
) -> float | None:
    turn = trace.turns[index]
    if turn.action is None:
        return None
    if turn.action.kind is ActionKind.UPDATE_MEMORY:
        raw = turn.action.arguments.get("item")
        if not isinstance(raw, Mapping) or not raw.get("evidence_pointers"):
            return None
        status = str(raw.get("status", "uncertain")).lower()
        pointers = tuple(str(item) for item in raw["evidence_pointers"])
    elif turn.action.kind is ActionKind.MARK_STATUS:
        status = str(turn.action.arguments["status"]).lower()
        item_id = str(turn.action.arguments["item_id"])
        before = {item.item_id: item for item in turn.observation.memory}
        if item_id not in before:
            return -1.0
        pointers = before[item_id].evidence_pointers
    elif turn.action.kind is ActionKind.ANSWER:
        pointers = _turn_pointers(turn)
        if not pointers:
            return None
        supporting = [
            item for item in turn.observation.memory if set(item.evidence_pointers) & set(pointers)
        ]
        if not supporting:
            return None
        return 1.0 if all(item.status is MemoryStatus.CURRENT for item in supporting) else -1.0
    else:
        return None
    events = event_index(context.admissible_events)
    if any(pointer not in events for pointer in pointers):
        return -1.0
    in_scope = all(
        context.task.requested_window.contains(events[pointer].timestamp, context.task.cutoff)
        for pointer in pointers
    )
    expected = MemoryStatus.CURRENT.value if in_scope else MemoryStatus.STALE.value
    return 1.0 if status == expected else -1.0


def _memory_preserves_needed_evidence(
    context: VerifierContext, trace: VerifierTrace, index: int
) -> float | None:
    turn = trace.turns[index]
    if turn.action is None:
        return None
    if turn.action.kind is ActionKind.COMPRESS:
        raw_ids = turn.action.arguments["item_ids"]
        ids = raw_ids if isinstance(raw_ids, list | tuple) else str(raw_ids).split("|")
        before = {item.item_id: item for item in turn.observation.memory}
        if any(str(item) not in before for item in ids):
            return -1.0
        expected = {pointer for item in ids for pointer in before[str(item)].evidence_pointers}
        return 1.0 if expected <= set(turn.tool_return.evidence_pointers) else -1.0
    if turn.action.kind is ActionKind.DISCARD:
        item_id = str(turn.action.arguments["item_id"])
        before = {item.item_id: item for item in turn.observation.memory}
        if item_id not in before:
            return -1.0
        discarded_item = before[item_id]
        discarded = set(discarded_item.evidence_pointers)
        later_pointers = {
            pointer for later in trace.turns[index + 1 :] for pointer in _turn_pointers(later)
        }
        events = event_index(context.admissible_events)
        requested_fields = set(context.task.requested_fields)
        discards_task_evidence = any(
            pointer in events
            and events[pointer].field in requested_fields
            and context.task.requested_window.contains(
                events[pointer].timestamp, context.task.cutoff
            )
            for pointer in discarded
        )
        if discarded & later_pointers or discards_task_evidence:
            return -1.0
        return 1.0
    return None


def _stopping_progress(context: VerifierContext, trace: VerifierTrace, index: int) -> float | None:
    turn = trace.turns[index]
    if turn.action is None:
        return None
    if turn.action.kind is ActionKind.ANSWER:
        pointers = set(_turn_pointers(turn))
        events = event_index(context.admissible_events)
        requested_fields = set(context.task.requested_fields)
        task_evidence = {
            event.pointer
            for event in context.admissible_events
            if (not requested_fields or event.field in requested_fields)
            and context.task.requested_window.contains(event.timestamp, context.task.cutoff)
        }
        claims_missing = (
            str(turn.action.arguments["prediction"]).replace("_", "").upper() == "NOTRECORDED"
        )
        if claims_missing:
            return 1.0 if not task_evidence else -1.0
        if not pointers or any(pointer not in events for pointer in pointers):
            return -1.0
        return 1.0 if pointers & task_evidence else -1.0
    scope = turn.observation.history[-1].query_scope if turn.observation.history else None
    current_scope = ""
    if turn.action.kind is ActionKind.QUERY_FIELD:
        current_scope = f"{turn.action.arguments['field']}:{turn.action.arguments['window']}"
    elif turn.action.kind is ActionKind.RETRIEVE:
        current_scope = f"{turn.action.arguments['domain']}:{turn.action.arguments['window']}"
    if current_scope and any(
        entry.query_scope == current_scope for entry in turn.observation.history
    ):
        return -1.0
    if turn.tool_return.code in {ReturnCode.FOUND, ReturnCode.VALUE, ReturnCode.OK}:
        return 1.0
    if scope is not None and turn.tool_return.code is ReturnCode.REJECTED:
        return -1.0
    return 0.0
