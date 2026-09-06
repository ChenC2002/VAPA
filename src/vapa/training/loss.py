"""Backend-neutral token objective with an explicit KL convention."""

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


def _k3_kl(policy_log_prob: float, reference_log_prob: float) -> float:
    log_ratio = reference_log_prob - policy_log_prob
    return math.exp(log_ratio) - log_ratio - 1.0


def token_objective(
    new_log_probs: Iterable[float],
    behavior_log_probs: Iterable[float],
    reference_log_probs: Iterable[float],
    advantages: Iterable[float],
    masks: Iterable[bool],
    *,
    kl_weight: float = 0.01,
    ratio_clip: float | None = None,
    kl_mode: str = "k3",
) -> TokenObjective:
    """Compute the token-mean actor objective.

    ``ratio_clip=None`` is the paper-compatible default because Appendix A.7 says no
    PPO ratio clipping.  The exact KL estimator is absent from the PDF; ``k3`` is an
    explicit, configurable convention rather than a reproduction claim.
    """

    rows = list(
        zip(
            new_log_probs,
            behavior_log_probs,
            reference_log_probs,
            advantages,
            masks,
            strict=True,
        )
    )
    kept = [row for row in rows if row[-1]]
    if not kept:
        raise ValueError("objective has no unmasked tokens")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and nonnegative")
    if ratio_clip is not None and (not math.isfinite(ratio_clip) or ratio_clip < 0):
        raise ValueError("ratio_clip must be finite and nonnegative")
    policy_terms: list[float] = []
    kl_terms: list[float] = []
    for new, old, reference, advantage, _ in kept:
        if not all(math.isfinite(value) for value in (new, old, reference, advantage)):
            raise ValueError("unmasked objective inputs must be finite")
        ratio = math.exp(new - old)
        surrogate = ratio * advantage
        if ratio_clip is not None:
            clipped = min(max(ratio, 1.0 - ratio_clip), 1.0 + ratio_clip)
            surrogate = min(surrogate, clipped * advantage)
        policy_terms.append(-surrogate)
        if kl_mode == "k3":
            kl_terms.append(_k3_kl(new, reference))
        elif kl_mode == "log_ratio":
            kl_terms.append(new - reference)
        else:
            raise ValueError(f"unknown KL mode: {kl_mode}")
    policy = sum(policy_terms) / len(policy_terms)
    kl = sum(kl_terms) / len(kl_terms)
    return TokenObjective(policy, kl, policy + kl_weight * kl, len(kept))
