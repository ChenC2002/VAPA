import random
from dataclasses import replace

import pytest

from vapa.actions import make_action
from vapa.config import ExperimentConfig
from vapa.demo import tiny_episode
from vapa.environment import StateManager
from vapa.policies.base import PolicyDecision, PolicyRequest, sample_many
from vapa.policies.heuristic import HeuristicPolicy
from vapa.policies.text import GeneratedCandidate, TextPolicy
from vapa.rollouts import ForkSpec, RolloutRunner, RolloutSpec
from vapa.schemas import ActionKind
from vapa.training.trainer import InstanceBatchBuilder
from vapa.verifiers import VerifierCatalog


class CountingPolicy(HeuristicPolicy):
    def __init__(self):
        super().__init__()
        self.widths = []

    def sample_many(self, requests):
        self.widths.append([r.n for r in requests])
        return [self.sample(r.observation, rng=r.rng, n=r.n, greedy=r.greedy) for r in requests]


def test_batched_update_preserves_all_groups_rewards_masks_and_counts():
    episode = tiny_episode()
    config = ExperimentConfig()
    serial = InstanceBatchBuilder(
        config, RolloutRunner(StateManager), VerifierCatalog.demo_default()
    )
    batch = InstanceBatchBuilder(
        config, RolloutRunner(StateManager, batched=True), VerifierCatalog.demo_default()
    )
    policy = CountingPolicy()
    expected = serial.build_update((episode, episode), HeuristicPolicy(), seed=17)
    actual = batch.build_update((episode, episode), policy, seed=17)
    assert actual == expected
    assert max(map(len, policy.widths)) >= config.replay.base_group_size
    assert any(
        width == [config.replay.siblings_per_fork] * config.replay.max_fork_states
        for width in policy.widths
    )


class MixedPolicy:
    """RNG-only policy with early termination, malformed and off-mask actions."""

    def sample(self, observation, *, rng, n=1, greedy=False):
        decisions = []
        for _ in range(n):
            choice = rng.randrange(4)
            if greedy or choice == 0:
                action = make_action(ActionKind.ANSWER, prediction="ok", evidence=[])
            elif choice == 1:
                action = make_action(ActionKind.QUERY_FIELD, field="value", window="all")
            elif choice == 2:
                action = make_action(ActionKind.CALCULATE, calculator="unknown", bindings={})
            else:
                action = None
            decisions.append(
                PolicyDecision(
                    action,
                    text="malformed" if action is None else "",
                    token_count=3,
                    token_ids=(1, 2, 3),
                    behavior_log_probs=(-1.0,) * 3,
                )
            )
        return decisions


def test_ragged_rollouts_and_forks_match_sequential_with_serial_policy_fallback():
    episode = tiny_episode()
    mask = frozenset((ActionKind.QUERY_FIELD, ActionKind.ANSWER))
    specs = [RolloutSpec(episode, f"base:{i}", i, allowed_actions=mask) for i in range(12)]
    serial, batch = RolloutRunner(StateManager), RolloutRunner(StateManager, batched=True)
    bases = serial.run_many(specs, MixedPolicy())
    assert batch.run_many(specs, MixedPolicy()) == bases
    assert len({len(r.turns) for r in bases}) > 1
    forks = [ForkSpec(episode, b, len(b.turns) - 1, 3, i, mask) for i, b in enumerate(bases)]
    expected = serial.fork_many(forks, MixedPolicy())
    actual = batch.fork_many(forks, MixedPolicy())
    assert actual == expected
    for branch in actual:
        assert all(not t.loss_mask for t in branch.turns[: branch.fork_turn])
        assert all(not t.loss_mask for t in branch.turns[branch.fork_turn + 1 :])
        assert branch.sampled_tokens == sum(
            t.decision.token_count for t in branch.turns if not t.copied_prefix
        )
    assert batch.run_many([], MixedPolicy()) == []
    assert batch.fork_many([], MixedPolicy()) == []


def test_batch_replay_verifies_exact_observation_before_sampling():
    runner = RolloutRunner(StateManager, batched=True)
    episode = tiny_episode()
    base = runner.run(episode, MixedPolicy(), rollout_id="b", seed=0)
    base.turns[0].observation = replace(base.turns[0].observation, budget_remaining=0)
    with pytest.raises(RuntimeError, match="deterministic replay"):
        runner.fork_many([ForkSpec(episode, base, 0, 2, 7)], MixedPolicy())


def test_batch_policy_validates_request_and_candidate_counts():
    class Broken:
        def sample_many(self, requests):
            return [[]] * len(requests)

    observation = StateManager(tiny_episode()).observation()
    with pytest.raises(ValueError, match="exactly 1"):
        sample_many(Broken(), [PolicyRequest(observation, random.Random(1))])


def test_text_batch_preserves_request_seeds_n_and_prompt_identity():
    class Backend:
        def generate_batch(self, requests):
            self.requests = requests
            return [
                [
                    GeneratedCandidate(
                        'Answer(prediction="ok", evidence=[])', (1, 2), (-0.5, -0.5), (8, 9)
                    )
                ]
                * r.n
                for r in requests
            ]

    observation = StateManager(tiny_episode()).observation()
    backend = Backend()
    requests = [PolicyRequest(observation, random.Random(i), i + 1) for i in range(3)]
    result = TextPolicy(backend).sample_many(requests)
    assert [len(group) for group in result] == [1, 2, 3]
    assert [r.seed for r in backend.requests] == [
        random.Random(i).randrange(2**63) for i in range(3)
    ]
    assert result[0][0].prompt_ids == (8, 9)
