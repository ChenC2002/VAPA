"""Pre-action-only fork-anchor selection and branch construction."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from vapa.policies.base import Policy
from vapa.rollouts import Rollout, RolloutRunner
from vapa.schemas import ActionKind, Episode


@dataclass(frozen=True)
class ForkAnchor:
    rollout_id: str
    turn_index: int
    stratum: str
    weight: float


def _weighted_choice(
    candidates: list[ForkAnchor], rng: random.Random, excluded: set[tuple[str, int]]
) -> ForkAnchor | None:
    available = [
        candidate
        for candidate in candidates
        if (candidate.rollout_id, candidate.turn_index) not in excluded
    ]
    if not available:
        return None
    total = sum(candidate.weight for candidate in available)
    target = rng.random() * total
    cumulative = 0.0
    for candidate in available:
        cumulative += candidate.weight
        if target <= cumulative:
            return candidate
    return available[-1]


def select_fork_anchors(
    base_rollouts: Iterable[Rollout],
    *,
    seed: int,
    action_budget: int = 12,
    max_anchors: int = 2,
    memory_pressure_fraction: float = 0.75,
    allowed_actions: frozenset[ActionKind] | None = None,
) -> tuple[ForkAnchor, ...]:
    """Select at most one early and one late anchor, with the paper's carry rule."""

    if max_anchors < 0:
        raise ValueError("max_anchors cannot be negative")
    if not 0 <= memory_pressure_fraction <= 1:
        raise ValueError("memory pressure fraction must be in [0, 1]")
    strata: dict[str, list[ForkAnchor]] = {"early": [], "late": []}
    for rollout in sorted(base_rollouts, key=lambda item: item.rollout_id):
        if not rollout.is_base:
            raise ValueError("anchor candidates must be base rollouts")
        for turn in rollout.turns:
            observation = turn.observation
            legal_actions = set(observation.legal_actions)
            if allowed_actions is not None:
                legal_actions &= allowed_actions
            if len(legal_actions) < 2:
                continue
            spent = action_budget - observation.budget_remaining
            stratum = "early" if spent < action_budget / 2 else "late"
            pressure = len(observation.memory) / observation.memory_capacity
            weight = 2.0 if pressure >= memory_pressure_fraction else 1.0
            strata[stratum].append(ForkAnchor(rollout.rollout_id, turn.index, stratum, weight))
    rng = random.Random(seed)
    selected: list[ForkAnchor] = []
    excluded: set[tuple[str, int]] = set()
    for stratum in ("early", "late"):
        if len(selected) >= max_anchors:
            break
        candidate = _weighted_choice(strata[stratum], rng, excluded)
        if candidate is not None:
            selected.append(candidate)
            excluded.add((candidate.rollout_id, candidate.turn_index))
    # If one stratum is empty, transfer its draw to the other stratum.
    all_candidates = strata["early"] + strata["late"]
    while len(selected) < max_anchors:
        candidate = _weighted_choice(all_candidates, rng, excluded)
        if candidate is None:
            break
        selected.append(candidate)
        excluded.add((candidate.rollout_id, candidate.turn_index))
    return tuple(selected)


def build_fork_branches(
    episode: Episode,
    base_rollouts: Iterable[Rollout],
    policy: Policy,
    runner: RolloutRunner,
    *,
    anchors: Iterable[ForkAnchor],
    siblings_per_fork: int,
    seed: int,
    allowed_actions: frozenset[ActionKind] | None = None,
) -> tuple[Rollout, ...]:
    if siblings_per_fork < 1:
        return ()
    by_id: Mapping[str, Rollout] = {rollout.rollout_id: rollout for rollout in base_rollouts}
    branches: list[Rollout] = []
    for ordinal, anchor in enumerate(anchors):
        if anchor.rollout_id not in by_id:
            raise ValueError(f"fork anchor references unknown rollout {anchor.rollout_id}")
        branches.extend(
            runner.fork(
                episode,
                by_id[anchor.rollout_id],
                policy,
                anchor_turn=anchor.turn_index,
                siblings=siblings_per_fork,
                seed=seed * 10_007 + ordinal,
                allowed_actions=allowed_actions,
            )
        )
    return tuple(branches)
