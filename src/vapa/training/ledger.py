"""Sampled-actor-token accounting with intact-group update boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GroupCharge:
    group_id: str
    sampled_tokens: int
    trainable_tokens: int
    include_in_loss: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("a group charge requires a non-empty group ID")
        if type(self.sampled_tokens) is not int or self.sampled_tokens <= 0:
            raise ValueError("sampled tokens must be a positive integer")
        if (
            type(self.trainable_tokens) is not int
            or not 0 <= self.trainable_tokens <= self.sampled_tokens
        ):
            raise ValueError("trainable tokens must be an integer within sampled tokens")
        if not isinstance(self.include_in_loss, bool):
            raise TypeError("include_in_loss must be a boolean")


@dataclass
class ReplayQuotaLedger:
    """Cumulative replay-token quota carried into a paired no-fork control."""

    balance_tokens: int = 0
    replay_tokens: int = 0
    extra_base_tokens: int = 0
    settled_overshoot_tokens: int = 0
    next_instance: int = 0
    curriculum_stage: str | None = None

    def add_replay_tokens(self, tokens: int, *, curriculum_stage: str | None = None) -> None:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("replay token quota must be a nonnegative integer")
        if curriculum_stage is not None:
            if not isinstance(curriculum_stage, str) or not curriculum_stage.strip():
                raise ValueError("curriculum stage must be a non-empty string")
            if self.curriculum_stage != curriculum_stage and self.balance_tokens > 0:
                raise ValueError("unserved replay quota cannot cross a curriculum stage")
            self.curriculum_stage = curriculum_stage
        self.balance_tokens += tokens
        self.replay_tokens += tokens

    def charge_extra_base(self, tokens: int) -> None:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("extra base rollout must contain actor tokens")
        self.balance_tokens -= tokens
        self.extra_base_tokens += tokens

    def settle_allocation_stratum(self) -> int:
        """Close one matched allocation stratum without carrying atomic overshoot.

        Extra base rollouts are indivisible, so serving a positive quota may leave a
        negative balance.  That overshoot belongs to the stratum that caused it and
        must not cancel replay quota from a later update or curriculum stage.
        """

        if self.balance_tokens > 0:
            raise ValueError("cannot settle an allocation stratum with unserved replay quota")
        overshoot = -self.balance_tokens
        self.settled_overshoot_tokens += overshoot
        self.balance_tokens = 0
        self.next_instance = 0
        self.curriculum_stage = None
        return overshoot


@dataclass
class TokenLedger:
    target_tokens: int
    update_floor: int
    permit_post_target: bool = False
    spent_tokens: int = 0
    updates: list[list[GroupCharge]] = field(default_factory=list)
    _current: list[GroupCharge] = field(default_factory=list)
    _current_tokens: int = 0

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (self.target_tokens, self.update_floor)
        ):
            raise ValueError("ledger limits must be positive integers")
        if not isinstance(self.permit_post_target, bool):
            raise TypeError("permit_post_target must be a boolean")

    @property
    def complete(self) -> bool:
        return self.spent_tokens >= self.target_tokens

    @property
    def overshoot(self) -> int:
        return max(0, self.spent_tokens - self.target_tokens)

    @property
    def loss_updates(self) -> list[list[GroupCharge]]:
        """Optimizer view, including the intact group that crosses the run quota."""

        return [[charge for charge in update if charge.include_in_loss] for update in self.updates]

    def add(self, charge: GroupCharge) -> bool:
        """Add one whole group and return whether an optimizer batch just closed."""

        if self.complete and not self.permit_post_target:
            raise RuntimeError("token target is already complete")
        if self.complete:
            # Only extra matched-control bookkeeping after target completion is
            # comparison-only. The group that first reaches/crosses it is trainable.
            charge = replace(charge, include_in_loss=False)
        self._current.append(charge)
        self._current_tokens += charge.sampled_tokens
        self.spent_tokens += charge.sampled_tokens
        if self._current_tokens >= self.update_floor or self.complete:
            self.updates.append(self._current)
            self._current = []
            self._current_tokens = 0
            return True
        return False
