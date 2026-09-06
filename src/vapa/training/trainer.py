"""Backend-neutral construction of one complete VAPA instance group.

This module performs the method-specific work through advantage assignment.  A model
backend only needs to supply sampled actions/log probabilities and consume the resulting
turn masks and advantages; no critic or learned reward model is instantiated here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from vapa.config import ExperimentConfig
from vapa.policies.base import Policy
from vapa.rollouts import Rollout, RolloutRunner
from vapa.schemas import ActionKind, Episode
from vapa.training.advantages import (
    AdvantageSummary,
    assign_step_advantages,
    assign_trajectory_advantages,
)
from vapa.training.curriculum import CurriculumStage, stage_at
from vapa.training.groups import StepGroup, assign_step_groups
from vapa.training.ledger import ReplayQuotaLedger
from vapa.training.replay import build_fork_branches, select_fork_anchors
from vapa.verifiers import VerifierCatalog


@dataclass(frozen=True)
class InstanceBatch:
    episode_id: str
    occurrence_id: str
    base_rollouts: tuple[Rollout, ...]
    branch_rollouts: tuple[Rollout, ...]
    groups: tuple[StepGroup, ...]
    advantage_summary: AdvantageSummary | None

    @property
    def sampled_tokens(self) -> int:
        return sum(rollout.sampled_tokens for rollout in self.base_rollouts + self.branch_rollouts)

    @property
    def trainable_tokens(self) -> int:
        return sum(
            turn.decision.token_count
            for rollout in self.base_rollouts + self.branch_rollouts
            for turn in rollout.turns
            if turn.loss_mask and turn.decision is not None
        )


@dataclass(frozen=True)
class UpdateBatch:
    """One intact optimizer batch with a single batch-wide local scale."""

    instances: tuple[InstanceBatch, ...]
    advantage_summary: AdvantageSummary | None
    curriculum_stages: tuple[CurriculumStage, ...] = ()

    @property
    def curriculum_stage(self) -> CurriculumStage | None:
        """Return the common stage, or ``None`` when an update crosses a gate."""

        unique = {stage.name for stage in self.curriculum_stages}
        return self.curriculum_stages[0] if len(unique) == 1 else None

    @property
    def sampled_tokens(self) -> int:
        return sum(instance.sampled_tokens for instance in self.instances)

    @property
    def trainable_tokens(self) -> int:
        return sum(instance.trainable_tokens for instance in self.instances)

    @property
    def branch_sampled_tokens(self) -> int:
        return sum(
            rollout.sampled_tokens
            for instance in self.instances
            for rollout in instance.branch_rollouts
        )


class InstanceBatchBuilder:
    def __init__(
        self,
        config: ExperimentConfig,
        runner: RolloutRunner,
        verifiers: VerifierCatalog | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.runner = runner
        self.verifiers = verifiers
        self._update_counter = 0

    def build(self, episode: Episode, policy: Policy, *, seed: int) -> InstanceBatch:
        """Build a one-instance update (the synthetic/demo convenience path)."""

        return self.build_update((episode,), policy, seed=seed).instances[0]

    def build_update(
        self,
        episodes: Iterable[Episode],
        policy: Policy,
        *,
        seed: int,
        sampled_tokens_so_far: int | None = None,
        history_quartiles: Mapping[str, int] | None = None,
        replay_quota: ReplayQuotaLedger | None = None,
    ) -> UpdateBatch:
        """Sample complete instances, then normalize local credit once across the update."""

        episode_list = list(episodes)
        if not episode_list:
            raise ValueError("an update requires at least one episode")
        update_index = self._update_counter
        self._update_counter += 1
        partial: list[
            tuple[Episode, tuple[Rollout, ...], tuple[Rollout, ...], tuple[StepGroup, ...]]
        ] = []
        curriculum_stages: list[CurriculumStage] = []
        tokens_within_update = 0
        for ordinal, episode in enumerate(episode_list):
            curriculum_stage: CurriculumStage | None = None
            if sampled_tokens_so_far is not None:
                curriculum_stage = stage_at(
                    sampled_tokens_so_far + tokens_within_update,
                    self.config.optimization.sampled_token_budget,
                )
                raw_quartile = (
                    history_quartiles.get(episode.task.instance_id)
                    if history_quartiles is not None
                    else episode.task.metadata.get("history_quartile")
                )
                if isinstance(raw_quartile, bool) or not isinstance(raw_quartile, int):
                    raise ValueError(
                        "curriculum updates require an integer history quartile per episode"
                    )
                if not 1 <= raw_quartile <= curriculum_stage.max_history_quartile:
                    raise ValueError(
                        f"episode {episode.task.instance_id!r} is outside the "
                        f"{curriculum_stage.name} history curriculum"
                    )
                curriculum_stages.append(curriculum_stage)
            source_id = episode.task.instance_id
            occurrence_id = f"update{update_index}:item{ordinal}:id{len(source_id)}:{source_id}"
            sampled = self._sample_instance(
                episode,
                policy,
                seed=seed * 1_000_003 + ordinal,
                occurrence_id=occurrence_id,
                allowed_actions=None if curriculum_stage is None else curriculum_stage.actions,
                defer_groups=(
                    not self.config.replay.fork_enabled
                    and self.config.replay.reallocate_disabled_forks
                ),
            )
            partial.append(sampled)
            tokens_within_update += sum(
                rollout.sampled_tokens for rollout in sampled[1] + sampled[2]
            )

        replay = self.config.replay
        if not replay.fork_enabled and replay.reallocate_disabled_forks:
            if replay_quota is None:
                raise ValueError("matched no-fork construction requires a ReplayQuotaLedger")
            matched_stage: CurriculumStage | None = None
            if sampled_tokens_so_far is not None:
                unique_stages = {stage.name for stage in curriculum_stages}
                if len(unique_stages) != 1:
                    raise ValueError(
                        "matched no-fork updates must remain within one curriculum stage"
                    )
                matched_stage = curriculum_stages[0]
                if replay_quota.curriculum_stage != matched_stage.name:
                    raise ValueError(
                        "replay quota stage does not match the no-fork curriculum stage"
                    )
            while replay_quota.balance_tokens > 0:
                item_index = replay_quota.next_instance % len(partial)
                episode, member_bases, member_branches, _ = partial[item_index]
                occurrence_id = member_bases[0].instance_id
                base_index = len(member_bases)
                instance_seed = seed * 1_000_003 + item_index
                rollout = self.runner.run(
                    episode,
                    policy,
                    rollout_id=f"{occurrence_id}:base:{base_index}",
                    seed=instance_seed * 1_000_003 + base_index,
                    instance_id=occurrence_id,
                    allowed_actions=None if matched_stage is None else matched_stage.actions,
                )
                if self.config.credit.use_process_rewards:
                    assert self.verifiers is not None
                    self.verifiers.score_rollout(episode, rollout)
                replay_quota.charge_extra_base(rollout.sampled_tokens)
                replay_quota.next_instance = (item_index + 1) % len(partial)
                partial[item_index] = (
                    episode,
                    member_bases + (rollout,),
                    member_branches,
                    (),
                )
            if self.config.credit.use_step_credit:
                partial = [
                    (
                        episode,
                        member_bases,
                        member_branches,
                        assign_step_groups(
                            member_bases,
                            member_branches,
                            group_prefix=f"{member_bases[0].instance_id}:",
                        ),
                    )
                    for episode, member_bases, member_branches, _ in partial
                ]

        bases = tuple(rollout for _, members, _, _ in partial for rollout in members)
        branches = tuple(rollout for _, _, members, _ in partial for rollout in members)
        groups = tuple(group for _, _, _, members in partial for group in members)
        credit = self.config.credit
        summary: AdvantageSummary | None = None
        if credit.use_step_credit:
            summary = assign_step_advantages(
                bases,
                branches,
                groups,
                process_weight=credit.process_weight if credit.use_process_rewards else 0.0,
                cost_weight=credit.cost_weight,
                gamma=credit.process_discount,
                beta=credit.local_weight,
                epsilon=credit.epsilon,
            )
        else:
            assign_trajectory_advantages(
                bases,
                process_weight=credit.process_weight if credit.use_process_rewards else 0.0,
                cost_weight=credit.cost_weight,
                epsilon=credit.epsilon,
            )

        instances = tuple(
            InstanceBatch(
                episode.task.instance_id,
                member_bases[0].instance_id,
                member_bases,
                member_branches,
                member_groups,
                summary,
            )
            for episode, member_bases, member_branches, member_groups in partial
        )
        return UpdateBatch(instances, summary, tuple(curriculum_stages))

    def _sample_instance(
        self,
        episode: Episode,
        policy: Policy,
        *,
        seed: int,
        occurrence_id: str,
        allowed_actions: frozenset[ActionKind] | None,
        defer_groups: bool = False,
    ) -> tuple[Episode, tuple[Rollout, ...], tuple[Rollout, ...], tuple[StepGroup, ...]]:
        replay = self.config.replay
        credit = self.config.credit
        bases = tuple(
            self.runner.run(
                episode,
                policy,
                rollout_id=f"{occurrence_id}:base:{index}",
                seed=seed * 1_000_003 + index,
                instance_id=occurrence_id,
                allowed_actions=allowed_actions,
            )
            for index in range(replay.base_group_size)
        )
        if credit.use_process_rewards:
            if self.verifiers is None:
                raise ValueError("process rewards require a verifier catalog")
            for rollout in bases:
                self.verifiers.score_rollout(episode, rollout)

        branches: tuple[Rollout, ...] = ()
        groups: tuple[StepGroup, ...] = ()
        if credit.use_step_credit:
            if replay.fork_enabled:
                anchors = select_fork_anchors(
                    bases,
                    seed=seed + 17,
                    action_budget=self.config.environment.action_budget,
                    max_anchors=replay.max_fork_states,
                    memory_pressure_fraction=replay.memory_pressure_fraction,
                    allowed_actions=allowed_actions,
                )
                branches = build_fork_branches(
                    episode,
                    bases,
                    policy,
                    self.runner,
                    anchors=anchors,
                    siblings_per_fork=replay.siblings_per_fork,
                    seed=seed + 31,
                    allowed_actions=allowed_actions,
                )
                if credit.use_process_rewards:
                    assert self.verifiers is not None
                    for rollout in branches:
                        self.verifiers.score_rollout(episode, rollout)
            if not defer_groups:
                groups = assign_step_groups(
                    bases,
                    branches,
                    group_prefix=f"{occurrence_id}:",
                )
        return episode, bases, branches, groups
