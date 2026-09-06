"""Equations 4-7 and the trajectory-level factorial controls."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from vapa.rollouts import Rollout
from vapa.training.groups import StepGroup, TurnRef


@dataclass(frozen=True)
class AdvantageSummary:
    episode_scales: Mapping[str, float]
    local_scale: float
    local_nonzero: int
    masked_tokens: int
    trainable_tokens: int


def credit_return(
    rollout: Rollout,
    turn_index: int,
    *,
    process_weight: float,
    cost_weight: float,
    gamma: float,
) -> float:
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if type(turn_index) is not int or not 0 <= turn_index < len(rollout.turns):
        raise ValueError("turn_index must identify an existing turn")
    _validate_weights(process_weight=process_weight, cost_weight=cost_weight)
    suffix = rollout.turns[turn_index:]
    costs = sum(turn.cost for turn in suffix)
    process = sum((gamma**offset) * turn.process_reward for offset, turn in enumerate(suffix))
    return rollout.outcome_reward - cost_weight * costs + process_weight * process


def _population_std(values: Iterable[float]) -> float:
    collected = list(values)
    return statistics.pstdev(collected) if collected else 0.0


def _validate_weights(**weights: float) -> None:
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0 or (name == "epsilon" and value == 0):
            raise ValueError(
                f"{name} must be finite and {'positive' if name == 'epsilon' else 'nonnegative'}"
            )


def assign_step_advantages(
    base_rollouts: Iterable[Rollout],
    branch_rollouts: Iterable[Rollout],
    groups: Iterable[StepGroup],
    *,
    process_weight: float = 0.05,
    cost_weight: float = 0.02,
    gamma: float = 1.0,
    beta: float = 1.0,
    epsilon: float = 1e-8,
) -> AdvantageSummary:
    """Assign normalized two-level advantages and final token masks in place."""

    bases = list(base_rollouts)
    branches = list(branch_rollouts)
    group_list = list(groups)
    _validate_weights(
        epsilon=epsilon, beta=beta, process_weight=process_weight, cost_weight=cost_weight
    )
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not bases or any(rollout.is_base for rollout in branches):
        raise ValueError("step advantages require base rollouts and correctly labeled branches")
    if len({group.group_id for group in group_list}) != len(group_list):
        raise ValueError("local group IDs must be unique")
    all_rollouts = bases + branches
    by_id = {rollout.rollout_id: rollout for rollout in all_rollouts}
    if len(by_id) != len(all_rollouts):
        raise ValueError("rollout ids must be unique")

    # Episode term: terminal answer only, base rollouts only, mean includes self.
    instances: dict[str, list[Rollout]] = defaultdict(list)
    for rollout in bases:
        if not rollout.is_base:
            raise ValueError("base_rollouts includes a branch")
        instances[rollout.instance_id].append(rollout)
    episode_scales: dict[str, float] = {}
    normalized_episode: dict[str, float] = {}
    for instance_id, members in instances.items():
        mean_outcome = sum(member.outcome_reward for member in members) / len(members)
        raw = {member.rollout_id: member.outcome_reward - mean_outcome for member in members}
        scale = _population_std(raw.values())
        episode_scales[instance_id] = scale
        for rollout_id, value in raw.items():
            normalized_episode[rollout_id] = value / (scale + epsilon)

    # Local leave-one-out term over each complete group.
    local_raw: dict[TurnRef, float] = {}
    for group in group_list:
        returns: dict[TurnRef, float] = {}
        instances_in_group: set[str] = set()
        for reference in group.members:
            rollout = by_id.get(reference.rollout_id)
            if rollout is None or reference.turn_index >= len(rollout.turns):
                raise ValueError(f"group {group.group_id} contains an invalid turn reference")
            instances_in_group.add(rollout.instance_id)
            turn = rollout.turns[reference.turn_index]
            if turn.copied_prefix or turn.action is None:
                raise ValueError("local groups cannot include copied prefixes or malformed actions")
            returns[reference] = credit_return(
                rollout,
                reference.turn_index,
                process_weight=process_weight,
                cost_weight=cost_weight,
                gamma=gamma,
            )
        if len(instances_in_group) != 1:
            raise ValueError("local groups cannot mix episode occurrences")
        total = sum(returns.values())
        size = len(returns)
        for reference, value in returns.items():
            if reference in local_raw:
                raise ValueError(f"turn {reference} belongs to more than one local group")
            local_raw[reference] = value - (total - value) / (size - 1)
    local_scale = _population_std(local_raw.values())

    masked_tokens = 0
    trainable_tokens = 0
    for rollout in all_rollouts:
        for turn in rollout.turns:
            reference = TurnRef(rollout.rollout_id, turn.index)
            local = local_raw.get(reference, 0.0)
            normalized_local = local / (local_scale + epsilon)
            episode = normalized_episode.get(rollout.rollout_id, 0.0)
            turn.local_advantage = local
            turn.episode_advantage = episode
            if rollout.is_base:
                turn.normalized_advantage = episode + beta * normalized_local
            else:
                turn.normalized_advantage = beta * normalized_local
                if reference not in local_raw:
                    turn.loss_mask = False
            if turn.copied_prefix or turn.action is None or turn.decision is None:
                turn.loss_mask = False
            tokens = 0 if turn.decision is None else turn.decision.token_count
            if turn.copied_prefix:
                continue  # Replay prefixes are neither generated nor actor-token charged.
            if turn.loss_mask:
                trainable_tokens += tokens
            else:
                masked_tokens += tokens
    return AdvantageSummary(
        episode_scales=episode_scales,
        local_scale=local_scale,
        local_nonzero=sum(not math.isclose(value, 0.0) for value in local_raw.values()),
        masked_tokens=masked_tokens,
        trainable_tokens=trainable_tokens,
    )


def assign_trajectory_advantages(
    base_rollouts: Iterable[Rollout],
    *,
    process_weight: float,
    cost_weight: float,
    epsilon: float = 1e-8,
) -> None:
    """GRPO-style trajectory score copied to every generated base turn."""

    _validate_weights(epsilon=epsilon, process_weight=process_weight, cost_weight=cost_weight)
    instances: dict[str, list[Rollout]] = defaultdict(list)
    seen: set[str] = set()
    for rollout in base_rollouts:
        if not rollout.is_base or rollout.rollout_id in seen:
            raise ValueError("trajectory advantages require distinct base rollouts")
        seen.add(rollout.rollout_id)
        instances[rollout.instance_id].append(rollout)
    for members in instances.values():
        returns = [
            rollout.outcome_reward
            + process_weight * sum(turn.process_reward for turn in rollout.turns)
            - cost_weight * sum(turn.cost for turn in rollout.turns)
            for rollout in members
        ]
        mean_return = sum(returns) / len(returns)
        advantages = [value - mean_return for value in returns]
        scale = _population_std(advantages)
        for rollout, advantage in zip(members, advantages, strict=True):
            normalized = advantage / (scale + epsilon)
            for turn in rollout.turns:
                turn.episode_advantage = normalized
                turn.local_advantage = 0.0
                turn.normalized_advantage = normalized
                turn.loss_mask = turn.action is not None and turn.decision is not None
