from __future__ import annotations

import math
import random
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

from vapa.cli import main
from vapa.config import (
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    ReplayConfig,
    load_config,
)
from vapa.data import load_episode_objects
from vapa.demo import demo_result_log, publish_demo_results, run_demo, run_demo_suite, tiny_episode
from vapa.environment import CalculatorRegistry, StateManager
from vapa.evaluation import holm_adjust, ordinary_least_squares_slope, paired_t_interval
from vapa.policies.base import PolicyDecision
from vapa.policies.heuristic import HeuristicPolicy
from vapa.policies.text import GeneratedCandidate, TextPolicy
from vapa.rollouts import RolloutRunner
from vapa.schemas import ActionKind
from vapa.training.factorial import Arm, configure_arm, factorial_contrasts
from vapa.training.ledger import GroupCharge, ReplayQuotaLedger, TokenLedger
from vapa.training.loss import token_objective
from vapa.training.trainer import InstanceBatchBuilder
from vapa.verifiers import VerifierCatalog

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [True, False, 0.5, float("nan"), float("inf")])
def test_token_accounting_rejects_non_integer_counts(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GroupCharge("group", value, 0)
    with pytest.raises(ValueError, match="integer within"):
        GroupCharge("group", 3, value)
    with pytest.raises(ValueError, match="positive integers"):
        TokenLedger(target_tokens=value, update_floor=2)
    with pytest.raises(ValueError, match="positive integers"):
        TokenLedger(target_tokens=10, update_floor=value)


def test_installed_training_plan_does_not_depend_on_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["show-training-plan"]) == 0


@pytest.mark.parametrize(
    "section",
    [
        {"seed": "not-an-int"},
        {"environment": EnvironmentConfig(context_tokens=-1)},
        {"model": ModelConfig(thinking="yes")},
        {"model": ModelConfig(dtype=" ")},
        {"model": ModelConfig(top_k=-4)},
        {"model": ModelConfig(name="openai/gpt-oss-20b")},
        {"optimization": OptimizationConfig(weight_decay=-1)},
        {"optimization": OptimizationConfig(gradient_clip=-1)},
        {"optimization": OptimizationConfig(kl_weight=-1)},
        {"optimization": OptimizationConfig(lora_rank=0)},
        {"optimization": OptimizationConfig(lora_dropout=2)},
    ],
)
def test_config_rejects_wrong_types_and_unsafe_ranges(section: dict[str, object]) -> None:
    base = load_config(ROOT / "configs" / "vapa_qwen.toml")
    with pytest.raises((TypeError, ValueError)):
        replace(base, **section).validate()


def test_every_public_experiment_config_is_valid_and_inheritance_resolves() -> None:
    configs = {path.name: load_config(path) for path in sorted((ROOT / "configs").glob("*.toml"))}
    assert configs["vapa_qwen.toml"].optimization.sampled_token_budget == 2**25
    assert configs["a2_step_qwen.toml"].credit.use_step_credit
    assert not configs["a2_step_qwen.toml"].credit.use_process_rewards
    assert configs["a3_process_qwen.toml"].credit.use_process_rewards
    assert not configs["a3_process_qwen.toml"].credit.use_step_credit
    assert configs["vapa_gpt_oss.toml"].model.reasoning_effort == "low"
    assert configs["vapa_gpt_oss.toml"].model.dequantize_mxfp4 is True
    assert configs["vapa_qwen.toml"].model.dequantize_mxfp4 is False


def test_mxfp4_dequantization_is_rejected_for_non_gpt_oss_models() -> None:
    with pytest.raises(ValueError, match="only for gpt-oss"):
        ModelConfig(dequantize_mxfp4=True).validate()


def test_public_fixtures_load_and_demo_blocks_future_evidence() -> None:
    episodes = load_episode_objects(ROOT / "examples" / "tiny_episode.json")
    calculators = CalculatorRegistry.from_json(ROOT / "examples" / "tiny_calculators.json")
    result = run_demo(seed=7, compact=True)

    assert len(episodes) == 1
    assert len(calculators) == 1
    assert result["status"] == "ok"
    assert result["successes"] == result["base_rollouts"]
    assert not result["future_event_visible"]
    assert result["group_tiers"]["exact_fork"] == 2


def test_update_builder_uses_one_local_scale_and_unique_duplicate_instance_ids() -> None:
    episode = tiny_episode()
    duplicate = replace(episode, task=replace(episode.task, instance_id="x"))
    colliding_source = replace(episode, task=replace(episode.task, instance_id="x#0"))
    config = replace(
        ExperimentConfig(),
        replay=ReplayConfig(base_group_size=2, max_fork_states=1, siblings_per_fork=1),
    )
    runner = RolloutRunner(lambda item: StateManager(item))
    builder = InstanceBatchBuilder(config, runner, VerifierCatalog.demo_default())
    update = builder.build_update(
        (duplicate, duplicate, colliding_source), HeuristicPolicy(), seed=11
    )

    rollouts = [
        rollout
        for instance in update.instances
        for rollout in instance.base_rollouts + instance.branch_rollouts
    ]
    assert len({rollout.rollout_id for rollout in rollouts}) == len(rollouts)
    occurrence_ids = {instance.occurrence_id for instance in update.instances}
    assert len(occurrence_ids) == 3
    assert {rollout.instance_id for rollout in rollouts} == occurrence_ids
    assert update.advantage_summary is not None
    local_values = [
        turn.local_advantage
        for rollout in rollouts
        for turn in rollout.turns
        if turn.group_id is not None and not turn.copied_prefix
    ]
    assert update.advantage_summary.local_scale == pytest.approx(statistics.pstdev(local_values))
    assert (
        update.advantage_summary.masked_tokens + update.advantage_summary.trainable_tokens
        == update.sampled_tokens
    )
    assert all(
        instance.advantage_summary is update.advantage_summary for instance in update.instances
    )


def test_curriculum_validation_and_matched_no_fork_allocation() -> None:
    episode = tiny_episode()
    runner = RolloutRunner(lambda item: StateManager(item))
    base = ExperimentConfig()
    builder = InstanceBatchBuilder(base, runner, VerifierCatalog.demo_default())
    with pytest.raises(ValueError, match="history curriculum"):
        builder.build_update(
            (episode,),
            HeuristicPolicy(),
            seed=3,
            sampled_tokens_so_far=0,
            history_quartiles={episode.task.instance_id: 3},
        )
    foundation = builder.build_update(
        (episode,),
        HeuristicPolicy(),
        seed=3,
        sampled_tokens_so_far=0,
        history_quartiles={episode.task.instance_id: 1},
    )
    off_mask_queries = [
        turn
        for rollout in foundation.instances[0].base_rollouts
        for turn in rollout.turns
        if turn.action is None
        and turn.decision is not None
        and turn.decision.text.startswith("QueryField(")
    ]
    assert off_mask_queries
    assert all(not turn.accepted and not turn.loss_mask for turn in off_mask_queries)

    settled = ReplayQuotaLedger()
    settled.add_replay_tokens(5, curriculum_stage="foundation")
    settled.charge_extra_base(7)
    assert settled.settle_allocation_stratum() == 2
    assert settled.settled_overshoot_tokens == 2
    settled.add_replay_tokens(1, curriculum_stage="full")
    assert settled.balance_tokens == 1

    short_schedule = replace(
        base,
        replay=ReplayConfig(
            base_group_size=2,
            max_fork_states=0,
            siblings_per_fork=0,
            fork_enabled=False,
        ),
        optimization=replace(
            base.optimization,
            sampled_token_budget=100,
            update_token_floor=10,
        ),
    )
    crossing = InstanceBatchBuilder(
        short_schedule, runner, VerifierCatalog.demo_default()
    ).build_update(
        (episode, episode),
        HeuristicPolicy(),
        seed=3,
        sampled_tokens_so_far=0,
        history_quartiles={episode.task.instance_id: 1},
    )
    assert [stage.name for stage in crossing.curriculum_stages] == ["foundation", "status"]
    assert crossing.curriculum_stage is None

    matched_short = replace(
        short_schedule,
        replay=replace(
            short_schedule.replay,
            reallocate_disabled_forks=True,
        ),
    )
    staged_quota = ReplayQuotaLedger()
    staged_quota.add_replay_tokens(10, curriculum_stage="foundation")
    with pytest.raises(ValueError, match="within one curriculum stage"):
        InstanceBatchBuilder(matched_short, runner, VerifierCatalog.demo_default()).build_update(
            (episode, episode),
            HeuristicPolicy(),
            seed=3,
            sampled_tokens_so_far=0,
            history_quartiles={episode.task.instance_id: 1},
            replay_quota=staged_quota,
        )

    with pytest.raises(ValueError, match="cannot cross a curriculum stage"):
        staged_quota.add_replay_tokens(1, curriculum_stage="status")
    staged_quota.charge_extra_base(11)
    staged_quota.add_replay_tokens(1, curriculum_stage="status")
    assert staged_quota.curriculum_stage == "status"

    replay_config = replace(
        base,
        replay=ReplayConfig(base_group_size=2, max_fork_states=1, siblings_per_fork=2),
    )
    replay_update = InstanceBatchBuilder(
        replay_config, runner, VerifierCatalog.demo_default()
    ).build_update((episode,), HeuristicPolicy(), seed=3)
    quota = ReplayQuotaLedger()
    quota.add_replay_tokens(replay_update.branch_sampled_tokens)

    no_fork = replace(
        base,
        replay=ReplayConfig(
            base_group_size=2,
            max_fork_states=1,
            siblings_per_fork=2,
            fork_enabled=False,
            reallocate_disabled_forks=True,
        ),
    )
    matched_update = InstanceBatchBuilder(
        no_fork, runner, VerifierCatalog.demo_default()
    ).build_update((episode,), HeuristicPolicy(), seed=3, replay_quota=quota)
    matched = matched_update.instances[0]
    assert len(matched.base_rollouts) > 2
    assert matched.branch_rollouts == ()
    assert quota.replay_tokens == replay_update.branch_sampled_tokens
    assert quota.extra_base_tokens >= quota.replay_tokens
    assert quota.balance_tokens <= 0


class _MalformedBackend:
    def generate(self, messages, **kwargs):
        del messages
        return [
            GeneratedCandidate("not an action", (1, 2), (-0.1, -0.2)) for _ in range(kwargs["n"])
        ]


class _OffMaskBackend:
    def __init__(self) -> None:
        self.allowed_actions = ()

    def generate(self, messages, **kwargs):
        assert "Legal actions:" in messages[-1]["content"]
        self.allowed_actions = kwargs["allowed_actions"]
        return [GeneratedCandidate("QueryField(value, all)", (1,), (-0.1,))]


def test_text_policy_preserves_malformed_generation_for_cost_and_masking() -> None:
    manager = StateManager(tiny_episode())
    policy = TextPolicy(_MalformedBackend())
    decision = policy.sample(manager.observation(), rng=random.Random(1))[0]

    assert decision.action is None
    assert decision.text == "not an action"
    assert decision.token_count == 2
    result = manager.step_text(decision.text)
    assert result.cost == 1
    assert not result.accepted

    runner = RolloutRunner(lambda episode: StateManager(episode, action_budget=1, turn_cap=2))
    rollout = runner.run(tiny_episode(), policy, rollout_id="malformed", seed=1)
    assert rollout.terminated
    assert rollout.outcome_reward == -1.0


def test_text_policy_passes_and_enforces_the_sampling_action_mask() -> None:
    manager = StateManager(tiny_episode())
    manager.set_action_mask({ActionKind.RETRIEVE, ActionKind.ANSWER})
    backend = _OffMaskBackend()
    decision = TextPolicy(backend).sample(manager.observation(), rng=random.Random(1))[0]

    assert backend.allowed_actions == (ActionKind.RETRIEVE, ActionKind.ANSWER)
    assert decision.action is None
    result = manager.step_text(decision.text)
    assert not result.accepted


def test_factorial_configuration_and_contrasts_match_definitions() -> None:
    base = load_config(ROOT / "configs" / "vapa_qwen.toml")
    arms = {arm: configure_arm(base, arm) for arm in Arm}
    assert not arms[Arm.A1_GRPO].credit.use_process_rewards
    assert not arms[Arm.A1_GRPO].credit.use_step_credit
    assert arms[Arm.A4_VAPA].credit.use_process_rewards
    assert arms[Arm.A4_VAPA].credit.use_step_credit

    contrast = factorial_contrasts(
        {Arm.A1_GRPO: 10, Arm.A2_STEP: 12, Arm.A3_PROCESS: 13, Arm.A4_VAPA: 18}
    )
    assert contrast.process_given_step == 6
    assert contrast.step_given_process == 5
    assert contrast.interaction == 3
    assert contrast.joint == 8


def test_token_ledger_never_splits_groups_and_tracks_overshoot() -> None:
    ledger = TokenLedger(target_tokens=10, update_floor=6)
    assert not ledger.add(GroupCharge("g1", 4, 3))
    assert ledger.add(GroupCharge("g2", 3, 2))
    assert ledger.add(GroupCharge("g3", 5, 1))
    assert ledger.complete
    assert ledger.overshoot == 2
    assert [[item.group_id for item in update] for update in ledger.updates] == [
        ["g1", "g2"],
        ["g3"],
    ]
    assert [[item.group_id for item in update] for update in ledger.loss_updates] == [
        ["g1", "g2"],
        [],
    ]


def test_token_objective_masks_and_uses_explicit_no_clip_default() -> None:
    result = token_objective(
        [0.0, 10.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [2.0, 100.0],
        [True, False],
    )
    assert result.token_count == 1
    assert result.policy == -2.0
    assert result.kl == 0.0
    assert result.total == -2.0


def test_statistical_helpers_follow_paired_seed_and_holm_conventions() -> None:
    interval = paired_t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert interval.estimate == 3.0
    assert interval.lower < 3 < interval.upper
    assert ordinary_least_squares_slope([1, 2, 3], [2, 4, 6]) == 2.0
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.9})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.9})


@pytest.mark.parametrize(
    "ratio, advantage, loss", [(2, 1, -1.2), (2, -1, 2), (0.5, 1, -0.5), (0.5, -1, 0.8)]
)
def test_ratio_clipping_uses_pessimistic_surrogate(
    ratio: float, advantage: float, loss: float
) -> None:
    result = token_objective(
        [math.log(ratio)],
        [0.0],
        [math.log(ratio)],
        [advantage],
        [True],
        kl_weight=0.0,
        ratio_clip=0.2,
    )
    assert result.policy == pytest.approx(loss)


@pytest.mark.parametrize("updates", [{"ratio_clip": float("nan")}, {"kl_weight": -1.0}])
def test_token_objective_rejects_invalid_weights(updates: dict) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        token_objective([0.0], [0.0], [0.0], [1.0], [True], **updates)


def test_full_demo_retains_reproducible_results_and_publishes_synchronized_log(
    tmp_path: Path,
) -> None:
    import json

    first, second = tmp_path / "first", tmp_path / "second"
    result = run_demo_suite(ROOT / "examples", first)
    assert run_demo_suite(ROOT / "examples", second) == result
    assert len(result["records"]) == 7
    records = {row["id"]: row for row in result["records"]}
    assert records["evaluation"]["metrics"]["successful"] == 3
    assert records["sft"]["kind"] == "dry_run_only"
    assert records["sft"]["metrics"]["training_ready"] is False
    assert (first / "evaluation/predictions.jsonl").is_file()
    assert (first / "demonstrations.jsonl").is_file()
    assert (first / "events.jsonl").read_text() == demo_result_log(result)
    publish_demo_results(first, tmp_path / "repo")
    assert json.loads((tmp_path / "repo/results/demo_results.json").read_text()) == result
    assert (tmp_path / "repo/logs/demo_results.jsonl").read_text() == demo_result_log(result)
    with pytest.raises(FileExistsError):
        run_demo_suite(ROOT / "examples", first)
    assert json.loads((first / "results.json").read_text()) == result
    (first / "events.jsonl").write_text("damaged\n")
    with pytest.raises(ValueError, match="summary and event log disagree"):
        publish_demo_results(first, tmp_path / "repo")
    assert json.loads((tmp_path / "repo/results/demo_results.json").read_text()) == result


def test_demo_rejects_private_fixture_manifest_before_creating_output(tmp_path: Path) -> None:
    import json
    import shutil

    examples = tmp_path / "examples"
    shutil.copytree(ROOT / "examples", examples)
    manifest = examples / "tiny_dataset_manifest.json"
    data = json.loads(manifest.read_text())
    data["content_kind"] = "credentialed"
    manifest.write_text(json.dumps(data))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="explicitly public fixtures"):
        run_demo_suite(examples, output)
    assert not output.exists()


def test_demo_failure_cannot_publish_a_success_summary(tmp_path: Path, monkeypatch) -> None:
    import vapa.evaluation.analysis as module

    def fail(*_args, **_kwargs):
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(module, "analyze_binary_results", fail)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="analysis failed"):
        run_demo_suite(ROOT / "examples", output)
    assert not (output / "results.json").exists()
    assert not (output / "events.jsonl").exists()
    with pytest.raises(FileNotFoundError):
        publish_demo_results(output, tmp_path / "repo")
    assert not (tmp_path / "repo").exists()


@pytest.mark.parametrize("count", [0, 2])
def test_rollout_refuses_missing_or_extra_policy_candidates(count: int) -> None:
    class WrongCount:
        def sample(self, observation, *, rng, n=1, greedy=False):
            return HeuristicPolicy().sample(observation, rng=rng, n=1, greedy=greedy) * count

    runner = RolloutRunner(StateManager)
    with pytest.raises(ValueError, match="exactly 1 sampled decisions"):
        runner.run(tiny_episode(), WrongCount(), rollout_id="bad", seed=1)


def test_fork_refuses_incomplete_sibling_groups() -> None:
    class WrongCount:
        def sample(self, observation, *, rng, n=1, greedy=False):
            return HeuristicPolicy().sample(observation, rng=rng, n=1, greedy=greedy)

    episode, runner = tiny_episode(), RolloutRunner(StateManager)
    base = runner.run(episode, HeuristicPolicy(), rollout_id="base", seed=1)
    with pytest.raises(ValueError, match="exactly 3 sampled decisions"):
        runner.fork(episode, base, WrongCount(), anchor_turn=0, siblings=3, seed=1)


@pytest.mark.parametrize(
    "updates",
    [
        {"token_count": True},
        {"token_count": 1.5},
        {"token_ids": (True,)},
        {"behavior_log_probs": (float("nan"),)},
    ],
)
def test_policy_decision_rejects_invalid_token_accounting(updates: dict) -> None:
    with pytest.raises(ValueError):
        PolicyDecision(None, "malformed", **updates)
