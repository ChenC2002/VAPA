"""Dependency-free reference for the log-policy objective and forward KL (Eq. 9)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenObjective:
    policy: float
    kl: float
    total: float
    token_count: int


def forward_kl(policy: Iterable[float], reference: Iterable[float]) -> float:
    """Exact KL over normalized, identically masked next-token log distributions."""

    rows = list(zip(policy, reference, strict=True))
    if not rows:
        raise ValueError("KL requires a nonempty vocabulary")
    for column in zip(*rows, strict=True):
        if any(value > 0 or (not math.isfinite(value) and value != -math.inf) for value in column):
            raise ValueError("log distributions must contain nonpositive values or -inf")
        if not math.isclose(math.fsum(math.exp(value) for value in column), 1.0, abs_tol=1e-6):
            raise ValueError("log distributions must be normalized over the legal vocabulary")
    if any((policy == -math.inf) != (reference == -math.inf) for policy, reference in rows):
        raise ValueError("actor and reference must share the same legal vocabulary")
    return math.fsum(
        math.exp(policy) * (policy - reference) for policy, reference in rows if policy != -math.inf
    )


def token_objective(
    action_log_probs: Iterable[float],
    policy_log_distributions: Iterable[Iterable[float]],
    reference_log_distributions: Iterable[Iterable[float]],
    advantages: Iterable[float],
    masks: Iterable[bool],
    *,
    kl_weight: float = 0.01,
) -> TokenObjective:
    """Compute the token-mean Eq. 9 loss with fixed advantages and eligibility masks."""

    rows = list(
        zip(
            action_log_probs,
            policy_log_distributions,
            reference_log_distributions,
            advantages,
            masks,
            strict=True,
        )
    )
    if any(not isinstance(row[-1], bool) for row in rows):
        raise TypeError("objective masks must be boolean")
    kept = [row for row in rows if row[-1]]
    if not kept:
        raise ValueError("objective has no unmasked tokens")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and nonnegative")
    policy_terms: list[float] = []
    kl_terms: list[float] = []
    for action, policy_distribution, reference_distribution, advantage, _ in kept:
        if not all(math.isfinite(value) for value in (action, advantage)):
            raise ValueError("unmasked objective inputs must be finite")
        if action > 0:
            raise ValueError("action log probabilities must be nonpositive")
        policy_terms.append(-advantage * action)
        kl_terms.append(forward_kl(policy_distribution, reference_distribution))
    policy = math.fsum(policy_terms) / len(policy_terms)
    kl = math.fsum(kl_terms) / len(kl_terms)
    return TokenObjective(policy, kl, policy + kl_weight * kl, len(kept))
