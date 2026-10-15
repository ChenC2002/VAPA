"""Eq. 9: a scalar reference and an optional, framework-independent PyTorch kernel."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def action_token_sums(
    actor_logits: Tensor,
    reference_logits: Tensor,
    targets: Tensor,
    *,
    advantage: float,
) -> tuple[Tensor, Tensor]:
    """Return unnormalized policy and KL(actor || reference) sums for one action.

    Inputs are [tokens, vocabulary] logits with identical legal support; excluded
    entries are -inf and each row must admit its target token. Grammar enforcement
    belongs to the caller. Normalize once by the *global* eligible-token count,
    not by a mean of microbatch means. The reference never receives gradients.

    No model, trainer, rollout engine, or behavior-probability dependency is needed.
    Importing this module still works without PyTorch installed.
    """
    import torch

    if actor_logits.ndim != 2 or actor_logits.shape != reference_logits.shape:
        raise ValueError("actor and reference logits must have matching [tokens, vocabulary] shape")
    if targets.shape != actor_logits.shape[:1] or not targets.numel():
        raise ValueError("targets must contain one token ID for each nonempty logit row")
    actor_log = torch.log_softmax(actor_logits.float(), dim=-1)
    reference_log = torch.log_softmax(reference_logits.detach().float(), dim=-1)
    sampled_log = actor_log.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Avoid (-inf)-(-inf) and its NaN gradients on excluded vocabulary entries.
    excluded = torch.isneginf(actor_log)
    difference = actor_log.masked_fill(excluded, 0.0) - reference_log.masked_fill(excluded, 0.0)
    return -advantage * sampled_log.sum(), (actor_log.exp() * difference).sum()


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
