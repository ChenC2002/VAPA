"""Policy protocol used by base rollouts and frozen-policy replay."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from vapa.actions import Action, format_action
from vapa.schemas import Observation


@dataclass(frozen=True)
class PolicyDecision:
    action: Action | None
    text: str = ""
    token_count: int = 1
    token_ids: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("token IDs must be nonnegative integers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in self.behavior_log_probs
        ):
            raise ValueError("behavior log probabilities must be finite numbers")
        if self.token_ids and len(self.token_ids) != self.token_count:
            raise ValueError("token_ids length must match token_count")
        if self.behavior_log_probs and len(self.behavior_log_probs) != self.token_count:
            raise ValueError("log-probability length must match token_count")
        if self.action is None and not self.text:
            raise ValueError("a malformed decision must retain its generated text")
        if not self.text and self.action is not None:
            object.__setattr__(self, "text", format_action(self.action))


class Policy(Protocol):
    def sample(
        self,
        observation: Observation,
        *,
        rng: random.Random,
        n: int = 1,
        greedy: bool = False,
    ) -> list[PolicyDecision]:
        """Sample with replacement; duplicate decisions must be retained."""
