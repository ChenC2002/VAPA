"""Priority-ordered exact-fork, collision, and turn-index groups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from vapa.rollouts import Rollout, Turn


class GroupTier(str, Enum):
    EXACT_FORK = "exact_fork"
    COLLISION = "canonical_collision"
    TURN_INDEX = "turn_index"


@dataclass(frozen=True, order=True)
class TurnRef:
    rollout_id: str
    turn_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, str) or not self.rollout_id.strip():
            raise ValueError("rollout_id must be non-empty text")
        if type(self.turn_index) is not int or self.turn_index < 0:
            raise ValueError("turn_index must be a nonnegative integer")


@dataclass(frozen=True)
class StepGroup:
    group_id: str
    tier: GroupTier
    members: tuple[TurnRef, ...]

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("step groups require at least two members")
        if len(set(self.members)) != len(self.members):
            raise ValueError("step group contains duplicate members")


def _eligible_turns(rollouts: Iterable[Rollout]) -> Iterable[tuple[Rollout, Turn]]:
    for rollout in rollouts:
        for turn in rollout.turns:
            if turn.copied_prefix or turn.action is None:
                continue
            yield rollout, turn


def assign_step_groups(
    base_rollouts: Iterable[Rollout],
    branch_rollouts: Iterable[Rollout],
    *,
    group_prefix: str = "",
) -> tuple[StepGroup, ...]:
    """Assign each turn to its strongest available non-overlapping local group."""

    bases = list(base_rollouts)
    branches = list(branch_rollouts)
    if not bases:
        raise ValueError("at least one base rollout is required")
    instance_ids = {rollout.instance_id for rollout in bases + branches}
    if len(instance_ids) != 1:
        raise ValueError("groups cannot mix instances")
    by_id = {rollout.rollout_id: rollout for rollout in bases + branches}
    if len(by_id) != len(bases) + len(branches):
        raise ValueError("rollout ids must be unique")
    assigned: set[TurnRef] = set()
    groups: list[StepGroup] = []

    # Tier 1: exact state manufactured by deterministic replay.
    fork_buckets: dict[tuple[str, int], list[Rollout]] = defaultdict(list)
    for branch in branches:
        if branch.parent_rollout_id is None or branch.fork_turn is None:
            raise ValueError("branch rollout is missing fork metadata")
        TurnRef(branch.parent_rollout_id, branch.fork_turn)
        fork_buckets[(branch.parent_rollout_id, branch.fork_turn)].append(branch)
    for ordinal, ((base_id, turn_index), siblings) in enumerate(sorted(fork_buckets.items())):
        base = by_id.get(base_id)
        if base is None or not base.is_base or turn_index >= len(base.turns):
            raise ValueError("branch points to an invalid base anchor")
        members = []
        if base.turns[turn_index].action is not None:
            members.append(TurnRef(base_id, turn_index))
        for sibling in sorted(siblings, key=lambda item: item.rollout_id):
            if turn_index >= len(sibling.turns):
                raise ValueError("branch does not contain its fork turn")
            if (
                sibling.turns[turn_index].observation.exact_bytes()
                != base.turns[turn_index].observation.exact_bytes()
            ):
                raise ValueError("fork siblings do not share the base pre-action observation")
            sibling_turn = sibling.turns[turn_index]
            if sibling_turn.action is not None:
                members.append(TurnRef(sibling.rollout_id, turn_index))
        if len(members) >= 2:
            group = StepGroup(f"{group_prefix}fork:{ordinal}", GroupTier.EXACT_FORK, tuple(members))
            groups.append(group)
            assigned.update(members)

    # Tier 2: exact canonical structured-state collisions. Branch continuations may join.
    collision_buckets: dict[str, list[TurnRef]] = defaultdict(list)
    for rollout, turn in _eligible_turns(bases + branches):
        reference = TurnRef(rollout.rollout_id, turn.index)
        if reference in assigned:
            continue
        collision_buckets[turn.observation.state_hash].append(reference)
    collision_ordinal = 0
    for state_hash, members in sorted(collision_buckets.items()):
        unique = tuple(sorted(set(members)))
        if len(unique) < 2:
            continue
        group = StepGroup(
            f"{group_prefix}collision:{collision_ordinal}:{state_hash[:12]}",
            GroupTier.COLLISION,
            unique,
        )
        groups.append(group)
        assigned.update(unique)
        collision_ordinal += 1
        for reference in unique:
            rollout = by_id[reference.rollout_id]
            if not rollout.is_base:
                rollout.turns[reference.turn_index].loss_mask = True

    # Tier 3: weak same-instance, same-turn-index baseline for unmatched base turns only.
    turn_buckets: dict[int, list[TurnRef]] = defaultdict(list)
    for rollout, turn in _eligible_turns(bases):
        reference = TurnRef(rollout.rollout_id, turn.index)
        if reference not in assigned:
            turn_buckets[turn.index].append(reference)
    for turn_index, members in sorted(turn_buckets.items()):
        unique = tuple(sorted(set(members)))
        if len(unique) < 2:
            continue
        group = StepGroup(f"{group_prefix}turn:{turn_index}", GroupTier.TURN_INDEX, unique)
        groups.append(group)
        assigned.update(unique)

    for group in groups:
        for reference in group.members:
            turn = by_id[reference.rollout_id].turns[reference.turn_index]
            turn.group_id = group.group_id
            turn.group_tier = group.tier.value
    return tuple(groups)
