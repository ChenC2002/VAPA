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
    prompt_ids: tuple[int, ...] = ()
    behavior_policy_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.behavior_policy_id, str):
            raise TypeError("behavior_policy_id must be a string")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        if any(
            type(token) is not int or token < 0 for token in (*self.token_ids, *self.prompt_ids)
        ):
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


@dataclass(frozen=True)
class PolicyRequest:
    """Independent random stream and candidate count for one observed state."""

    observation: Observation
    rng: random.Random
    n: int = 1
    greedy: bool = False


def sample_many(policy: Policy, requests: list[PolicyRequest]) -> list[list[PolicyDecision]]:
    """Use an optional batching interface, retaining a serial backend fallback."""
    if not requests:
        return []
    batch = getattr(policy, "sample_many", None)
    outputs = (
        batch(requests)
        if callable(batch)
        else [policy.sample(r.observation, rng=r.rng, n=r.n, greedy=r.greedy) for r in requests]
    )
    if not isinstance(outputs, list | tuple) or len(outputs) != len(requests):
        raise ValueError("policy batch must return one candidate list per request")
    for request, decisions in zip(requests, outputs, strict=True):
        if not isinstance(decisions, list | tuple) or len(decisions) != request.n:
            raise ValueError(f"policy must return exactly {request.n} sampled decisions")
        if any(not isinstance(decision, PolicyDecision) for decision in decisions):
            raise TypeError("policy samples must be PolicyDecision objects")
    return list(outputs)


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
