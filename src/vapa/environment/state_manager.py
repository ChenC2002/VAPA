"""Frozen state transition manager for cutoff-safe, bounded-memory episodes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from vapa.actions import Action, ActionParseError, parse_action
from vapa.environment.calculators import CalculatorRegistry
from vapa.schemas import (
    ActionKind,
    Domain,
    Episode,
    HistoryEntry,
    MemoryItem,
    MemoryStatus,
    Observation,
    RecordEvent,
    ReturnCode,
    TimeWindow,
    ToolReturn,
    event_index,
    normalize_field,
)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    tool_return: ToolReturn
    cost: int
    terminated: bool
    accepted: bool


_INVALID_TERMINAL_PREDICTION = object()


class StateManager:
    """Execute actions deterministically and never expose post-cutoff events."""

    def __init__(
        self,
        episode: Episode,
        *,
        memory_capacity: int = 8,
        action_budget: int = 12,
        turn_cap: int = 16,
        calculators: CalculatorRegistry | None = None,
        retrieval_limit: int = 5,
    ) -> None:
        if min(memory_capacity, action_budget, turn_cap, retrieval_limit) <= 0:
            raise ValueError("environment caps must be positive")
        if turn_cap < action_budget + 1:
            raise ValueError("turn cap must admit all actions plus ANSWER")
        self.episode = episode
        self.memory_capacity = memory_capacity
        self.action_budget = action_budget
        self.turn_cap = turn_cap
        self.retrieval_limit = retrieval_limit
        self.calculators = calculators or CalculatorRegistry()
        self._events = event_index(episode.admissible_events)
        self._all_pre_cutoff = tuple(
            sorted(episode.admissible_events, key=lambda item: (item.timestamp, item.pointer))
        )
        self._action_mask: frozenset[ActionKind] | None = None
        self.reset()

    def reset(self) -> Observation:
        self._memory: dict[str, MemoryItem] = {}
        self._history: list[HistoryEntry] = []
        self._actions: list[Action | None] = []
        self._last_return: ToolReturn | None = None
        self._budget = self.action_budget
        self._turn = 1
        self._terminated = False
        self.prediction: Any = None
        self.answer_evidence: tuple[str, ...] = ()
        return self.observation()

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def actions(self) -> tuple[Action | None, ...]:
        return tuple(self._actions)

    @property
    def memory(self) -> tuple[MemoryItem, ...]:
        return tuple(self._memory[key] for key in sorted(self._memory))

    def set_action_mask(self, actions: Iterable[ActionKind] | None) -> Observation:
        """Apply one curriculum action mask to sampling and transition legality."""

        if actions is None:
            self._action_mask = None
        else:
            mask = frozenset(actions)
            if not mask or any(not isinstance(action, ActionKind) for action in mask):
                raise TypeError("action mask must contain ActionKind values")
            if ActionKind.ANSWER not in mask:
                raise ValueError("action mask must always permit Answer")
            if ActionKind.MALFORMED in mask:
                raise ValueError("Malformed is internal and cannot be legal")
            self._action_mask = mask
        return self.observation()

    def legal_actions(self) -> tuple[ActionKind, ...]:
        if self._terminated:
            return ()
        if self._budget == 0 or self._turn >= self.turn_cap:
            return (ActionKind.ANSWER,)
        legal = [ActionKind.RETRIEVE, ActionKind.QUERY_FIELD]
        if len(self.calculators):
            legal.append(ActionKind.CALCULATE)
        legal.append(ActionKind.UPDATE_MEMORY)
        if self._memory:
            legal.extend([ActionKind.MARK_STATUS, ActionKind.DISCARD])
        if len(self._memory) >= 2:
            legal.append(ActionKind.COMPRESS)
        legal.append(ActionKind.ANSWER)
        if self._action_mask is not None:
            legal = [action for action in legal if action in self._action_mask]
        return tuple(legal)

    def observation(self) -> Observation:
        return Observation(
            task=self.episode.task,
            last_return=self._last_return,
            memory=self.memory,
            history=tuple(self._history),
            budget_remaining=self._budget,
            budget_cap=self.action_budget,
            turn=self._turn,
            turn_cap=self.turn_cap,
            memory_capacity=self.memory_capacity,
            legal_actions=self.legal_actions(),
        )

    def _scope(self, action: Action) -> str:
        arguments = action.arguments
        if action.kind is ActionKind.RETRIEVE:
            return f"{arguments['domain']}:{arguments['window']}"
        if action.kind is ActionKind.QUERY_FIELD:
            return f"{arguments['field']}:{arguments['window']}"
        if action.kind is ActionKind.CALCULATE:
            return str(arguments["calculator"])
        if action.kind in {ActionKind.UPDATE_MEMORY, ActionKind.MARK_STATUS, ActionKind.DISCARD}:
            return str(arguments.get("item_id", "memory"))
        return action.kind.value.lower()

    def _visible_event_pointers(self) -> set[str]:
        pointers = {pointer for item in self._memory.values() for pointer in item.evidence_pointers}
        if self._last_return is not None:
            pointers.update(self._last_return.evidence_pointers)
            pointers.update(event.pointer for event in self._last_return.events)
        return pointers & self._events.keys()

    def _mark_terminal_failure(self) -> None:
        self.prediction = _INVALID_TERMINAL_PREDICTION
        self.answer_evidence = ()

    def _record_step(self, action: Action | None, returned: ToolReturn, cost: int) -> None:
        kind = ActionKind.MALFORMED if action is None else action.kind
        scope = "malformed" if action is None else self._scope(action)
        self._history.append(HistoryEntry(self._turn, kind, scope, cost, returned.code))
        self._actions.append(action)
        self._last_return = returned
        self._turn += 1

    def _consume_rejection(
        self, action: Action | None, message: str, *, terminate: bool = False
    ) -> StepResult:
        cost = 1 if self._budget > 0 else 0
        self._budget -= cost
        returned = ToolReturn(ReturnCode.REJECTED, message=message)
        self._record_step(action, returned, cost)
        if terminate:
            self._mark_terminal_failure()
            self._terminated = True
        return StepResult(self.observation(), returned, cost, self._terminated, False)

    def step_text(self, model_output: str) -> StepResult:
        try:
            action = parse_action(model_output)
        except ActionParseError as exc:
            if self._terminated:
                raise RuntimeError("episode has already terminated") from exc
            malformed_answer = any(
                re.match(r"(?i)^answer\s*\(", line.strip()) is not None
                for line in model_output.split("\n")
            )
            terminate = malformed_answer or self._budget == 0 or self._turn >= self.turn_cap
            return self._consume_rejection(None, str(exc), terminate=terminate)
        return self.step(action)

    def step(self, action: Action) -> StepResult:
        if self._terminated:
            raise RuntimeError("episode has already terminated")
        if action.kind not in self.legal_actions():
            terminal_illegal = self._budget == 0 or self._turn >= self.turn_cap
            return self._consume_rejection(
                action, "action is not legal in the current state", terminate=terminal_illegal
            )
        if action.kind is ActionKind.ANSWER:
            returned = self._answer(action)
            self._record_step(action, returned, 0)
            self._terminated = True
            return StepResult(
                self.observation(), returned, 0, True, returned.code is ReturnCode.ANSWERED
            )
        self._budget -= 1
        handlers = {
            ActionKind.RETRIEVE: self._retrieve,
            ActionKind.QUERY_FIELD: self._query_field,
            ActionKind.CALCULATE: self._calculate,
            ActionKind.UPDATE_MEMORY: self._update_memory,
            ActionKind.MARK_STATUS: self._mark_status,
            ActionKind.COMPRESS: self._compress,
            ActionKind.DISCARD: self._discard,
        }
        try:
            returned = handlers[action.kind](action)
        except (KeyError, TypeError, ValueError) as exc:
            returned = ToolReturn(ReturnCode.REJECTED, message=str(exc))
        self._record_step(action, returned, 1)
        return StepResult(
            self.observation(),
            returned,
            1,
            False,
            returned.code not in {ReturnCode.REJECTED, ReturnCode.ERROR},
        )

    def _retrieve(self, action: Action) -> ToolReturn:
        query = str(action.arguments["query"]).strip().lower()
        window = TimeWindow.parse(str(action.arguments["window"]))
        domain = Domain(str(action.arguments["domain"]).strip().lower())
        terms = tuple(term for term in query.replace("_", " ").split() if term)
        candidates: list[tuple[int, RecordEvent]] = []
        for event in self._all_pre_cutoff:
            if event.domain is not domain or not window.contains(
                event.timestamp, self.episode.task.cutoff
            ):
                continue
            haystack = " ".join(
                item
                for item in [
                    event.field.replace("_", " "),
                    str(event.value),
                    event.text or "",
                    event.code or "",
                ]
                if item
            ).lower()
            score = sum(term in haystack for term in terms)
            if not terms or score:
                candidates.append((score, event))
        ranked = sorted(
            candidates, key=lambda item: (-item[0], -item[1].timestamp.timestamp(), item[1].pointer)
        )
        events = tuple(event for _, event in ranked[: self.retrieval_limit])
        if not events:
            return ToolReturn(ReturnCode.NOT_RECORDED, message="no qualifying event")
        return ToolReturn(
            ReturnCode.FOUND,
            evidence_pointers=tuple(event.pointer for event in events),
            events=events,
        )

    def _query_field(self, action: Action) -> ToolReturn:
        field = normalize_field(str(action.arguments["field"]))
        window = TimeWindow.parse(str(action.arguments["window"]))
        field_events = [event for event in self._all_pre_cutoff if event.field == field]
        if not field_events:
            return ToolReturn(ReturnCode.NOT_RECORDED, message=f"{field} is not recorded")
        candidates = [
            event
            for event in field_events
            if window.contains(event.timestamp, self.episode.task.cutoff)
        ]
        if not candidates:
            return ToolReturn(
                ReturnCode.OUT_OF_WINDOW, message=f"{field} exists only outside the window"
            )
        latest_time = max(event.timestamp for event in candidates)
        latest = [event for event in candidates if event.timestamp == latest_time]
        canonical = {(json.dumps(event.value, sort_keys=True), event.unit) for event in latest}
        if len(canonical) > 1:
            return ToolReturn(
                ReturnCode.AMBIGUOUS,
                evidence_pointers=tuple(event.pointer for event in latest),
                events=tuple(latest),
                message="latest tied values disagree",
            )
        chosen = sorted(latest, key=lambda item: item.pointer)[0]
        return ToolReturn(
            ReturnCode.FOUND,
            value=chosen.value,
            unit=chosen.unit,
            evidence_pointers=(chosen.pointer,),
            events=(chosen,),
        )

    def _calculate(self, action: Action) -> ToolReturn:
        bindings = action.arguments["bindings"]
        if not isinstance(bindings, Mapping):
            raise ValueError("calculator bindings must be a mapping")
        return self.calculators.calculate(
            str(action.arguments["calculator"]),
            bindings,
            events={
                pointer: self._events[pointer] for pointer in sorted(self._visible_event_pointers())
            },
            memory=self._memory,
        )

    def _update_memory(self, action: Action) -> ToolReturn:
        raw = action.arguments["item"]
        if not isinstance(raw, Mapping):
            raise ValueError("memory item must be a mapping")
        item = MemoryItem.from_dict(raw)
        if item.item_id not in self._memory and len(self._memory) >= self.memory_capacity:
            return ToolReturn(
                ReturnCode.REJECTED, message="memory is full; discard or compress first"
            )
        hidden = set(item.evidence_pointers) - self._visible_event_pointers()
        if hidden:
            return ToolReturn(
                ReturnCode.REJECTED,
                message=f"evidence pointers are not visible: {sorted(hidden)}",
            )
        previous = self._memory.get(item.item_id)
        if previous is not None and (
            item.status is not previous.status or item.validity_scope != previous.validity_scope
        ):
            return ToolReturn(
                ReturnCode.REJECTED,
                message="status and validity-scope changes require MarkStatus",
            )
        self._memory[item.item_id] = item
        return ToolReturn(
            ReturnCode.OK, evidence_pointers=item.evidence_pointers, message="memory updated"
        )

    def _mark_status(self, action: Action) -> ToolReturn:
        item_id = str(action.arguments["item_id"])
        if item_id not in self._memory:
            return ToolReturn(ReturnCode.REJECTED, message=f"unknown memory item {item_id}")
        status = MemoryStatus(str(action.arguments["status"]).lower())
        scope = str(action.arguments["scope"]).strip()
        if not scope:
            raise ValueError("validity scope cannot be empty")
        self._memory[item_id] = replace(self._memory[item_id], status=status, validity_scope=scope)
        return ToolReturn(
            ReturnCode.OK,
            evidence_pointers=self._memory[item_id].evidence_pointers,
            message="status updated",
        )

    def _compress(self, action: Action) -> ToolReturn:
        raw_ids = action.arguments["item_ids"]
        if isinstance(raw_ids, str):
            item_ids = [item.strip() for item in raw_ids.split("|") if item.strip()]
        elif isinstance(raw_ids, list | tuple):
            item_ids = [str(item) for item in raw_ids]
        else:
            raise ValueError("Compress expects a list of memory ids")
        item_ids = list(dict.fromkeys(item_ids))
        if len(item_ids) < 2 or any(item not in self._memory for item in item_ids):
            return ToolReturn(
                ReturnCode.REJECTED, message="Compress needs at least two existing items"
            )
        source = [self._memory[item] for item in item_ids]
        digest = hashlib.sha256("|".join(sorted(item_ids)).encode()).hexdigest()[:10]
        pointers = tuple(
            dict.fromkeys(pointer for item in source for pointer in item.evidence_pointers)
        )
        statuses = {item.status for item in source}
        units = {item.unit for item in source}
        compressed = MemoryItem(
            item_id=f"cmp_{digest}",
            field="compressed_summary",
            value=tuple(item.value for item in source),
            unit=units.pop() if len(units) == 1 else None,
            status=statuses.pop() if len(statuses) == 1 else MemoryStatus.UNCERTAIN,
            validity_scope="; ".join(sorted({item.validity_scope for item in source})),
            evidence_pointers=pointers,
        )
        for item_id in item_ids:
            del self._memory[item_id]
        self._memory[compressed.item_id] = compressed
        return ToolReturn(ReturnCode.OK, evidence_pointers=pointers, message=compressed.item_id)

    def _discard(self, action: Action) -> ToolReturn:
        item_id = str(action.arguments["item_id"])
        item = self._memory.pop(item_id, None)
        if item is None:
            return ToolReturn(ReturnCode.REJECTED, message=f"unknown memory item {item_id}")
        return ToolReturn(
            ReturnCode.OK, evidence_pointers=item.evidence_pointers, message="memory discarded"
        )

    def _answer(self, action: Action) -> ToolReturn:
        self._mark_terminal_failure()
        raw_evidence = action.arguments["evidence"]
        if isinstance(raw_evidence, str):
            evidence = tuple(
                item.strip() for item in raw_evidence.strip("[]").split(",") if item.strip()
            )
        elif isinstance(raw_evidence, list):
            if any(not isinstance(item, str) or not item.strip() for item in raw_evidence):
                return ToolReturn(
                    ReturnCode.REJECTED,
                    message="answer evidence must contain non-empty pointer strings",
                )
            evidence = tuple(item.strip() for item in raw_evidence)
        else:
            return ToolReturn(ReturnCode.REJECTED, message="answer evidence must be a pointer list")
        hidden = set(evidence) - self._visible_event_pointers()
        if hidden:
            return ToolReturn(
                ReturnCode.REJECTED,
                message=f"answer evidence pointers are not visible: {sorted(hidden)}",
            )
        prediction = action.arguments["prediction"]
        if prediction is None or (isinstance(prediction, str) and not prediction.strip()):
            return ToolReturn(ReturnCode.REJECTED, message="answer prediction is required")
        self.prediction = prediction
        self.answer_evidence = tuple(dict.fromkeys(evidence))
        return ToolReturn(
            ReturnCode.ANSWERED,
            evidence_pointers=self.answer_evidence,
            message="answer committed",
        )

    def replay(self, actions: Iterable[Action], *, stop_before: int | None = None) -> Observation:
        """Reset and replay a logged prefix; ``stop_before`` is a zero-based turn index."""

        self.reset()
        for index, action in enumerate(actions):
            if stop_before is not None and index >= stop_before:
                break
            self.step(action)
            if self._terminated:
                break
        return self.observation()
