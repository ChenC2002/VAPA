"""Trajectory generation and deterministic fork continuation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from vapa.actions import Action
from vapa.environment.state_manager import StateManager, StepResult
from vapa.policies.base import Policy, PolicyDecision
from vapa.schemas import ActionKind, Episode, Observation, ReturnCode, ToolReturn


@dataclass
class Turn:
    index: int
    observation: Observation
    decision: PolicyDecision | None
    action: Action | None
    tool_return: ToolReturn
    cost: int
    accepted: bool
    process_reward: float = 0.0
    verifier_scores: dict[str, float | None] = field(default_factory=dict)
    group_id: str | None = None
    group_tier: str | None = None
    local_advantage: float = 0.0
    episode_advantage: float = 0.0
    normalized_advantage: float = 0.0
    loss_mask: bool = True
    copied_prefix: bool = False


@dataclass
class Rollout:
    rollout_id: str
    instance_id: str
    turns: list[Turn]
    is_base: bool = True
    parent_rollout_id: str | None = None
    fork_turn: int | None = None
    outcome_reward: float = -1.0
    prediction: object = None
    answer_evidence: tuple[str, ...] = ()
    sampled_tokens: int = 0
    did_terminate: bool = False

    @property
    def terminated(self) -> bool:
        return self.did_terminate or bool(
            self.turns and self.turns[-1].tool_return.code is ReturnCode.ANSWERED
        )

    @property
    def action_cost(self) -> int:
        return sum(turn.cost for turn in self.turns if not turn.copied_prefix)

    def actions(self) -> list[Action]:
        return [turn.action for turn in self.turns if turn.action is not None]


OutcomeScorer = Callable[[Episode, object, tuple[str, ...]], float]
ManagerFactory = Callable[[Episode], StateManager]


def exact_outcome_scorer(episode: Episode, prediction: object, evidence: tuple[str, ...]) -> float:
    del evidence
    if prediction is None:
        return -1.0
    return (
        1.0 if str(prediction).strip().lower() == str(episode.gold_answer).strip().lower() else -1.0
    )


class RolloutRunner:
    def __init__(
        self,
        manager_factory: ManagerFactory,
        *,
        outcome_scorer: OutcomeScorer = exact_outcome_scorer,
    ) -> None:
        self.manager_factory = manager_factory
        self.outcome_scorer = outcome_scorer

    def run(
        self,
        episode: Episode,
        policy: Policy,
        *,
        rollout_id: str,
        seed: int,
        greedy: bool = False,
        instance_id: str | None = None,
        allowed_actions: Iterable[ActionKind] | None = None,
    ) -> Rollout:
        manager = self.manager_factory(episode)
        action_mask = None if allowed_actions is None else frozenset(allowed_actions)
        manager.set_action_mask(action_mask)
        rng = random.Random(seed)
        turns: list[Turn] = []
        sampled_tokens = 0
        while not manager.terminated:
            observation = manager.observation()
            decision = self._sample(policy, observation, rng=rng, n=1, greedy=greedy)[0]
            result = (
                manager.step(decision.action)
                if decision.action is not None
                else manager.step_text(decision.text)
            )
            sampled_tokens += decision.token_count
            turns.append(self._turn(len(turns), observation, decision, result))
        reward = self._score_outcome(episode, manager, turns)
        return Rollout(
            rollout_id=rollout_id,
            instance_id=instance_id or episode.task.instance_id,
            turns=turns,
            outcome_reward=reward,
            prediction=manager.prediction,
            answer_evidence=manager.answer_evidence,
            sampled_tokens=sampled_tokens,
            did_terminate=manager.terminated,
        )

    def fork(
        self,
        episode: Episode,
        base: Rollout,
        policy: Policy,
        *,
        anchor_turn: int,
        siblings: int,
        seed: int,
        allowed_actions: Iterable[ActionKind] | None = None,
    ) -> list[Rollout]:
        if type(anchor_turn) is not int or not 0 <= anchor_turn < len(base.turns):
            raise IndexError("fork anchor is outside the base rollout")
        if type(siblings) is not int or siblings < 1:
            raise ValueError("siblings must be a positive integer")
        action_mask = None if allowed_actions is None else frozenset(allowed_actions)
        manager = self.manager_factory(episode)
        manager.set_action_mask(action_mask)
        prefix_turns = base.turns[:anchor_turn]
        self._replay_prefix(manager, prefix_turns)
        recovered = manager.observation()
        expected = base.turns[anchor_turn].observation
        if recovered.exact_bytes() != expected.exact_bytes():
            raise RuntimeError("deterministic replay did not recover the logged observation")
        rng = random.Random(seed)
        decisions = self._sample(policy, recovered, rng=rng, n=siblings, greedy=False)
        branches: list[Rollout] = []
        for branch_index, first_decision in enumerate(decisions):
            branch_manager = self.manager_factory(episode)
            branch_manager.set_action_mask(action_mask)
            self._replay_prefix(branch_manager, prefix_turns)
            copied = [self._copy_prefix(turn) for turn in base.turns[:anchor_turn]]
            result = (
                branch_manager.step(first_decision.action)
                if first_decision.action is not None
                else branch_manager.step_text(first_decision.text)
            )
            branch_turns = copied + [self._turn(anchor_turn, recovered, first_decision, result)]
            sampled_tokens = first_decision.token_count
            branch_rng = random.Random(seed * 1_000_003 + branch_index)
            while not branch_manager.terminated:
                observation = branch_manager.observation()
                decision = self._sample(policy, observation, rng=branch_rng, n=1, greedy=False)[0]
                result = (
                    branch_manager.step(decision.action)
                    if decision.action is not None
                    else branch_manager.step_text(decision.text)
                )
                sampled_tokens += decision.token_count
                continuation_turn = self._turn(len(branch_turns), observation, decision, result)
                continuation_turn.loss_mask = False
                branch_turns.append(continuation_turn)
            reward = self._score_outcome(episode, branch_manager, branch_turns)
            branches.append(
                Rollout(
                    rollout_id=f"{base.rollout_id}:fork{anchor_turn}:{branch_index}",
                    instance_id=base.instance_id,
                    turns=branch_turns,
                    is_base=False,
                    parent_rollout_id=base.rollout_id,
                    fork_turn=anchor_turn,
                    outcome_reward=reward,
                    prediction=branch_manager.prediction,
                    answer_evidence=branch_manager.answer_evidence,
                    sampled_tokens=sampled_tokens,
                    did_terminate=branch_manager.terminated,
                )
            )
        return branches

    def _score_outcome(self, episode: Episode, manager: StateManager, turns: list[Turn]) -> float:
        valid_answer = bool(
            manager.terminated
            and turns
            and turns[-1].action is not None
            and turns[-1].action.kind is ActionKind.ANSWER
            and turns[-1].accepted
            and turns[-1].tool_return.code is ReturnCode.ANSWERED
        )
        if not valid_answer:
            return -1.0
        reward = self.outcome_scorer(episode, manager.prediction, manager.answer_evidence)
        if isinstance(reward, bool) or not isinstance(reward, int | float):
            raise TypeError("outcome scorer must return a number")
        reward = float(reward)
        if not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError("outcome scorer must return a finite value in [-1, 1]")
        return reward

    @staticmethod
    def _sample(
        policy: Policy, observation: Observation, *, rng: random.Random, n: int, greedy: bool
    ) -> list[PolicyDecision]:
        decisions = policy.sample(observation, rng=rng, n=n, greedy=greedy)
        if not isinstance(decisions, list | tuple) or len(decisions) != n:
            raise ValueError(f"policy must return exactly {n} sampled decisions")
        if any(not isinstance(decision, PolicyDecision) for decision in decisions):
            raise TypeError("policy samples must be PolicyDecision objects")
        return [
            RolloutRunner._mask_illegal_decision(decision, observation) for decision in decisions
        ]

    @staticmethod
    def _mask_illegal_decision(
        decision: PolicyDecision, observation: Observation
    ) -> PolicyDecision:
        """Treat an off-mask generation as malformed while retaining its actor tokens."""

        if decision.action is None or decision.action.kind in observation.legal_actions:
            return decision
        return replace(decision, action=None)

    @staticmethod
    def _turn(
        index: int, observation: Observation, decision: PolicyDecision, result: StepResult
    ) -> Turn:
        return Turn(
            index=index,
            observation=observation,
            decision=decision,
            action=decision.action,
            tool_return=result.tool_return,
            cost=result.cost,
            accepted=result.accepted,
        )

    @staticmethod
    def _copy_prefix(turn: Turn) -> Turn:
        return Turn(
            index=turn.index,
            observation=turn.observation,
            decision=turn.decision,
            action=turn.action,
            tool_return=turn.tool_return,
            cost=turn.cost,
            accepted=turn.accepted,
            process_reward=turn.process_reward,
            verifier_scores=dict(turn.verifier_scores),
            loss_mask=False,
            copied_prefix=True,
        )

    @staticmethod
    def _replay_prefix(manager: StateManager, turns: list[Turn]) -> None:
        manager.reset()
        for turn in turns:
            if turn.action is not None:
                manager.step(turn.action)
            elif turn.decision is not None:
                manager.step_text(turn.decision.text)
            else:
                raise ValueError("logged prefix lacks both action and generated text")
