"""Small deterministic/stochastic policies for tests and the public synthetic demo."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

from vapa.actions import Action, make_action
from vapa.policies.base import PolicyDecision
from vapa.schemas import ActionKind, MemoryStatus, Observation, ReturnCode


class ScriptedPolicy:
    """Call a user function for each decision; useful for auditable fixtures."""

    def __init__(self, chooser: Callable[[Observation, random.Random], Action]) -> None:
        self.chooser = chooser

    def sample(
        self,
        observation: Observation,
        *,
        rng: random.Random,
        n: int = 1,
        greedy: bool = False,
    ) -> list[PolicyDecision]:
        if n < 1:
            raise ValueError("n must be positive")
        return [PolicyDecision(self.chooser(observation, rng)) for _ in range(n)]


class HeuristicPolicy:
    """A bounded-memory baseline for latest-field synthetic episodes.

    This is not a clinical model.  It exists so environment, replay, verification, and
    advantage code can run end to end without a GPU or model download.
    """

    def sample(
        self,
        observation: Observation,
        *,
        rng: random.Random,
        n: int = 1,
        greedy: bool = False,
    ) -> list[PolicyDecision]:
        return [PolicyDecision(self._choose(observation, rng, greedy)) for _ in range(n)]

    def _choose(self, observation: Observation, rng: random.Random, greedy: bool) -> Action:
        field = (
            observation.task.requested_fields[0] if observation.task.requested_fields else "unknown"
        )
        matching = [item for item in observation.memory if item.field == field]
        latest = observation.last_return
        if observation.budget_remaining == 0 or observation.turn >= observation.turn_cap:
            return self._answer(matching, latest)
        if latest is not None and latest.code is ReturnCode.FOUND and latest.value is not None:
            if not matching:
                pointer = latest.evidence_pointers[0]
                item = {
                    "id": f"m_{field}",
                    "field": field,
                    "value": latest.value,
                    "unit": latest.unit,
                    "status": MemoryStatus.CURRENT.value,
                    "validity_scope": observation.task.requested_window.label,
                    "evidence_pointers": [pointer],
                }
                return make_action(ActionKind.UPDATE_MEMORY, item=item)
        if matching:
            if greedy or rng.random() < 0.70:
                return self._answer(matching, latest)
        return make_action(
            ActionKind.QUERY_FIELD,
            field=field,
            window=observation.task.requested_window.label,
        )

    @staticmethod
    def _answer(items: Sequence[Any], latest: Any) -> Action:
        if items:
            item = items[-1]
            return make_action(
                ActionKind.ANSWER,
                prediction=item.value,
                evidence=list(item.evidence_pointers),
            )
        if latest is not None and latest.value is not None:
            return make_action(
                ActionKind.ANSWER,
                prediction=latest.value,
                evidence=list(latest.evidence_pointers),
            )
        return make_action(ActionKind.ANSWER, prediction="NOTRECORDED", evidence=[])
