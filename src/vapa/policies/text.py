"""Adapter from any text-generation backend to the VAPA policy protocol."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from vapa.actions import ActionParseError, parse_action
from vapa.policies.base import PolicyDecision
from vapa.prompts import render_chat
from vapa.schemas import ActionKind, Observation


@dataclass(frozen=True)
class GeneratedCandidate:
    text: str
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("generated candidates require at least one actor token")
        if len(self.token_ids) != len(self.log_probs):
            raise ValueError("candidate token IDs and log probabilities must align")


class GenerationBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        seed: int,
        greedy: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        allowed_actions: tuple[ActionKind, ...],
    ) -> list[GeneratedCandidate]: ...


class TextPolicy:
    """Preserve malformed generations so they can be charged and masked correctly."""

    def __init__(
        self,
        backend: GenerationBackend,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        max_tokens: int = 512,
        scaffold: str = "",
    ) -> None:
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.scaffold = scaffold

    def sample(
        self,
        observation: Observation,
        *,
        rng: random.Random,
        n: int = 1,
        greedy: bool = False,
    ) -> list[PolicyDecision]:
        candidates = self.backend.generate(
            render_chat(observation, self.scaffold),
            n=n,
            seed=rng.randrange(0, 2**63),
            greedy=greedy,
            temperature=0.0 if greedy else self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            allowed_actions=observation.legal_actions,
        )
        if len(candidates) != n:
            raise RuntimeError(
                f"generation backend returned {len(candidates)} candidates; expected {n}"
            )
        decisions: list[PolicyDecision] = []
        for candidate in candidates:
            try:
                action = parse_action(candidate.text)
            except ActionParseError:
                action = None
            if action is not None and action.kind not in observation.legal_actions:
                action = None
            decisions.append(
                PolicyDecision(
                    action=action,
                    text=candidate.text,
                    token_count=len(candidate.token_ids),
                    token_ids=candidate.token_ids,
                    behavior_log_probs=candidate.log_probs,
                )
            )
        return decisions

    def binary_answer_probability(
        self,
        observation: Observation,
        decision: PolicyDecision | None = None,
    ) -> float:
        """Return binary ``p(class=1)`` at a direct or sampled Answer position.

        Passing the terminal decision keeps same-turn reasoning/control tokens in the
        scoring context.  Omitting it retains the direct-final fixed-position API.
        """

        scorer = getattr(self.backend, "binary_answer_probability", None)
        if not callable(scorer):
            raise TypeError("this text-generation backend does not support binary scoring")
        messages = render_chat(observation, self.scaffold)
        if decision is None:
            probability = scorer(messages)
        else:
            if not isinstance(decision, PolicyDecision):
                raise TypeError("decision must be a PolicyDecision or None")
            action = decision.action
            if action is None or action.kind is not ActionKind.ANSWER:
                raise ValueError("binary readout requires a parsed terminal Answer decision")
            prediction = action.arguments.get("prediction")
            if (
                isinstance(prediction, bool)
                or not isinstance(prediction, int)
                or prediction not in {0, 1}
            ):
                raise ValueError("binary Answer prediction must be the integer 0 or 1")
            if not decision.token_ids:
                raise ValueError("binary readout requires retained terminal token IDs")
            probability = scorer(messages, action_ids=decision.token_ids)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, int | float)
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise RuntimeError("the backend returned an invalid binary probability")
        return float(probability)
