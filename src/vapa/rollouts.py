"""Trajectory generation and deterministic fork continuation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace

from vapa.actions import Action
from vapa.environment.state_manager import StateManager, StepResult
from vapa.policies.base import Policy, PolicyDecision, PolicyRequest, sample_many
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


@dataclass(frozen=True)
class RolloutSpec:
    episode: Episode
    rollout_id: str
    seed: int
    greedy: bool = False
    instance_id: str | None = None
    allowed_actions: frozenset[ActionKind] | None = None


@dataclass(frozen=True)
class ForkSpec:
    episode: Episode
    base: Rollout
    anchor_turn: int
    siblings: int
    seed: int
    allowed_actions: frozenset[ActionKind] | None = None


@dataclass
class _LiveRollout:
    episode: Episode
    manager: StateManager
    rollout: Rollout
    rng: random.Random
    greedy: bool = False


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
        batched: bool = False,
    ) -> None:
        self.manager_factory = manager_factory
        self.outcome_scorer = outcome_scorer
        self.batched = batched

    def run_many(self, specs: Iterable[RolloutSpec], policy: Policy) -> list[Rollout]:
        """Advance independent trajectories in lockstep, preserving input order.

        Batching is opt-in: the default retains even the call order of stateful
        legacy policies. Each batched request owns its original random stream.
        """
        specs = list(specs)
        if not self.batched:
            return [
                self.run(
                    s.episode,
                    policy,
                    rollout_id=s.rollout_id,
                    seed=s.seed,
                    greedy=s.greedy,
                    instance_id=s.instance_id,
                    allowed_actions=s.allowed_actions,
                )
                for s in specs
            ]
        live = []
        for spec in specs:
            manager = self.manager_factory(spec.episode)
            manager.set_action_mask(spec.allowed_actions)
            live.append(
                _LiveRollout(
                    spec.episode,
                    manager,
                    Rollout(spec.rollout_id, spec.instance_id or spec.episode.task.instance_id, []),
                    random.Random(spec.seed),
                    spec.greedy,
                )
            )
        return self._finish_many(live, policy)

    def fork_many(self, specs: Iterable[ForkSpec], policy: Policy) -> list[Rollout]:
        """Batch anchor candidate groups, then their independent continuations."""
        specs = list(specs)
        if not self.batched:
            return [
                branch
                for s in specs
                for branch in self.fork(
                    s.episode,
                    s.base,
                    policy,
                    anchor_turn=s.anchor_turn,
                    siblings=s.siblings,
                    seed=s.seed,
                    allowed_actions=s.allowed_actions,
                )
            ]
        requests = [self._fork_request(spec) for spec in specs]
        outputs = self._sample_many(policy, requests)
        live = [
            state
            for spec, request, decisions in zip(specs, requests, outputs, strict=True)
            for state in self._fork_states(spec, request.observation, decisions)
        ]
        return self._finish_many(live, policy)

    def _fork_request(self, spec: ForkSpec) -> PolicyRequest:
        if type(spec.anchor_turn) is not int or not 0 <= spec.anchor_turn < len(spec.base.turns):
            raise IndexError("fork anchor is outside the base rollout")
        if type(spec.siblings) is not int or spec.siblings < 1:
            raise ValueError("siblings must be a positive integer")
        manager = self.manager_factory(spec.episode)
        manager.set_action_mask(spec.allowed_actions)
        self._replay_prefix(manager, spec.base.turns[: spec.anchor_turn])
        observation = manager.observation()
        if observation.exact_bytes() != spec.base.turns[spec.anchor_turn].observation.exact_bytes():
            raise RuntimeError("deterministic replay did not recover the logged observation")
        return PolicyRequest(observation, random.Random(spec.seed), spec.siblings)

    def _fork_states(
        self, spec: ForkSpec, observation: Observation, decisions: Iterable[PolicyDecision]
    ) -> Iterator[_LiveRollout]:
        # Lazy construction preserves serial environment/scorer call order as well
        # as RNG order; the batched caller explicitly collects the states first.
        for index, decision in enumerate(decisions):
            manager = self.manager_factory(spec.episode)
            manager.set_action_mask(spec.allowed_actions)
            self._replay_prefix(manager, spec.base.turns[: spec.anchor_turn])
            rollout = Rollout(
                f"{spec.base.rollout_id}:fork{spec.anchor_turn}:{index}",
                spec.base.instance_id,
                [self._copy_prefix(t) for t in spec.base.turns[: spec.anchor_turn]],
                is_base=False,
                parent_rollout_id=spec.base.rollout_id,
                fork_turn=spec.anchor_turn,
            )
            state = _LiveRollout(
                spec.episode, manager, rollout, random.Random(spec.seed * 1_000_003 + index)
            )
            self._advance(state, observation, decision, loss_mask=True)
            yield state

    def _finish_many(
        self, live: list[_LiveRollout], policy: Policy, *, batched: bool = True
    ) -> list[Rollout]:
        while active := [state for state in live if not state.manager.terminated]:
            requests = [
                PolicyRequest(s.manager.observation(), s.rng, greedy=s.greedy) for s in active
            ]
            outputs = (
                self._sample_many(policy, requests)
                if batched
                else [
                    self._sample(policy, r.observation, rng=r.rng, n=r.n, greedy=r.greedy)
                    for r in requests
                ]
            )
            for state, request, decisions in zip(active, requests, outputs, strict=True):
                self._advance(
                    state, request.observation, decisions[0], loss_mask=state.rollout.is_base
                )
        for state in live:
            rollout, manager = state.rollout, state.manager
            rollout.outcome_reward = self._score_outcome(state.episode, manager, rollout.turns)
            rollout.prediction = manager.prediction
            rollout.answer_evidence = manager.answer_evidence
            rollout.did_terminate = manager.terminated
        return [s.rollout for s in live]

    @staticmethod
    def _sample_many(policy: Policy, requests: list[PolicyRequest]) -> list[list[PolicyDecision]]:
        return [
            [RolloutRunner._mask_illegal_decision(d, r.observation) for d in decisions]
            for r, decisions in zip(requests, sample_many(policy, requests), strict=True)
        ]

    @staticmethod
    def _advance(
        state: _LiveRollout, observation: Observation, decision: PolicyDecision, *, loss_mask: bool
    ) -> None:
        result = (
            state.manager.step(decision.action)
            if decision.action is not None
            else state.manager.step_text(decision.text)
        )
        turn = RolloutRunner._turn(len(state.rollout.turns), observation, decision, result)
        turn.loss_mask = loss_mask
        state.rollout.turns.append(turn)
        state.rollout.sampled_tokens += decision.token_count

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
        state = _LiveRollout(
            episode,
            manager,
            Rollout(rollout_id, instance_id or episode.task.instance_id, []),
            random.Random(seed),
            greedy,
        )
        return self._finish_many([state], policy, batched=False)[0]

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
        action_mask = None if allowed_actions is None else frozenset(allowed_actions)
        spec = ForkSpec(episode, base, anchor_turn, siblings, seed, action_mask)
        request = self._fork_request(spec)
        decisions = self._sample(
            policy, request.observation, rng=request.rng, n=siblings, greedy=False
        )
        return [
            self._finish_many([state], policy, batched=False)[0]
            for state in self._fork_states(spec, request.observation, decisions)
        ]

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
        trace_scorer = getattr(self.outcome_scorer, "score_with_trace", None)
        reward = (
            trace_scorer(episode, manager.prediction, manager.answer_evidence, tuple(turns))
            if callable(trace_scorer)
            else self.outcome_scorer(episode, manager.prediction, manager.answer_evidence)
        )
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
