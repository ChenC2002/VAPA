from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from vapa.actions import make_action
from vapa.config import load_config
from vapa.data.episodes import episode_to_record
from vapa.demo import tiny_episode
from vapa.model.inference import TransformersInferenceSpec
from vapa.model.protocols import LossReport, TokenizedAction, VAPAAction
from vapa.policies.base import PolicyDecision
from vapa.schemas import ActionKind, Episode
from vapa.training.checkpoint import (
    CheckpointContract,
    JsonStateStore,
    RuntimeState,
    fingerprint_payload,
    read_manifest,
    save_checkpoint,
)
from vapa.training.rl_train import (
    DeterministicEpisodeStream,
    RLRunSettings,
    TrainingComponents,
    TransformersLoadOptions,
    build_transformers_components,
    initialize_from_sft_checkpoint,
    main,
    run_vapa_training,
    shard_episodes,
    transformers_inference_spec,
)
from vapa.training.runtime import DistributedContext


class TinyTokenizer:
    fingerprint = fingerprint_payload({"kind": "rl-entrypoint-test-tokenizer"})

    @staticmethod
    def encode_messages(
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[int, ...]:
        del messages, add_generation_prompt
        return (1, 2)

    @staticmethod
    def encode_text(text: str) -> tuple[int, ...]:
        return tuple(text.encode()) or (1,)

    def encode_action(
        self,
        messages: Sequence[Mapping[str, str]],
        action_text: str,
    ) -> TokenizedAction:
        return TokenizedAction(self.encode_messages(messages), self.encode_text(action_text))

    @staticmethod
    def save_pretrained(path: str) -> None:
        Path(path, "tokenizer.json").write_text("{}\n", encoding="utf-8")


def test_transformers_plan_rejects_remote_model_code():
    with pytest.raises(ValueError, match="trust_remote_code is forbidden"):
        TransformersLoadOptions(
            model_revision="a" * 40,
            trust_remote_code=True,
            lora_enabled=False,
            allow_base_initialization=True,
        )


def test_gpt_oss_rl_mxfp4_setting_is_bound_to_the_load_plan() -> None:
    config = load_config(Path("configs/vapa_gpt_oss.toml"))
    options = TransformersLoadOptions(
        model_revision="a" * 40,
        model_kind="causal",
        use_processor=False,
        lora_enabled=False,
        allow_base_initialization=True,
    )
    spec = transformers_inference_spec(config, options)
    assert spec.dequantize_mxfp4 is True
    assert spec.to_dict()["dequantize_mxfp4"] is True

    with pytest.raises(ValueError, match="must match"):
        transformers_inference_spec(config, replace(options, dequantize_mxfp4=False))


def test_builtin_lora_rl_loads_one_backbone_and_uses_shared_named_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vapa.model.transformers as transformer_module
    import vapa.training.rl_train as rl_module

    config_path, _ = _write_inputs(tmp_path)
    config = load_config(config_path)
    actor = TinyModel("actor", 0.5)
    reference = TinyModel("actor", 0.5)
    model_loads: list[dict[str, object]] = []
    shared_calls: list[dict[str, object]] = []

    def model_loader(name: str, **kwargs: object) -> TinyModel:
        model_loads.append({"name": name, **kwargs})
        return actor

    def shared_lora(runtime_actor: TinyModel, **kwargs: object):
        assert runtime_actor is actor
        shared_calls.append(kwargs)
        return actor, reference

    monkeypatch.setattr(
        transformer_module.TransformersActorAdapter,
        "from_pretrained",
        model_loader,
    )
    monkeypatch.setattr(transformer_module, "apply_shared_lora", shared_lora)
    monkeypatch.setattr(
        transformer_module.TransformersTokenizerAdapter,
        "from_pretrained",
        lambda *args, **kwargs: TinyTokenizer(),
    )
    monkeypatch.setattr(
        transformer_module,
        "TransformersGenerationBackend",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(rl_module, "build_adamw", lambda model, **kwargs: TinyOptimizer(model))
    monkeypatch.setattr(rl_module, "TorchStateStore", JsonStateStore)

    components = build_transformers_components(
        config,
        DistributedContext(),
        options=TransformersLoadOptions(
            model_revision="a" * 40,
            model_kind="causal",
            use_processor=False,
            lora_target_modules=("q_proj", "v_proj"),
            allow_base_initialization=True,
        ),
    )

    assert len(model_loads) == 1
    assert len(shared_calls) == 1
    assert shared_calls[0]["target_modules"] == ("q_proj", "v_proj")
    assert components.actor is actor
    assert components.reference is reference


class SeededAnswerPolicy:
    def sample(self, observation, *, rng, n: int = 1, greedy: bool = False):
        del observation, greedy
        decisions = []
        for _ in range(n):
            prediction = "ok" if rng.randrange(2) else "wrong"
            action = make_action(ActionKind.ANSWER, prediction=prediction, evidence=[])
            decisions.append(
                PolicyDecision(
                    action,
                    token_count=2,
                    token_ids=(10, 11),
                    behavior_log_probs=(-0.5, -0.5),
                )
            )
        return decisions


class VariableLengthAnswerPolicy:
    def sample(self, observation, *, rng, n: int = 1, greedy: bool = False):
        del observation, greedy
        decisions = []
        for _ in range(n):
            width = 1 + rng.randrange(7)
            prediction = "ok" if rng.randrange(2) else "wrong"
            action = make_action(ActionKind.ANSWER, prediction=prediction, evidence=[])
            decisions.append(
                PolicyDecision(
                    action,
                    token_count=width,
                    token_ids=tuple(range(10, 10 + width)),
                    behavior_log_probs=tuple(-0.5 for _ in range(width)),
                )
            )
        return decisions


class TinyModel:
    def __init__(self, name: str, weight: float) -> None:
        self.name = name
        self.weight = weight
        self.grad = 0.0
        self.mode = "eval"

    @property
    def fingerprint(self) -> str:
        return fingerprint_payload({"kind": "rl-entrypoint-test-model", "name": self.name})

    def train(self) -> None:
        self.mode = "train"

    def eval(self) -> None:
        self.mode = "eval"

    def parameters(self):
        return (self,)

    def sft_loss(self, examples: Sequence[TokenizedAction]) -> LossReport:
        raise AssertionError("SFT is outside this test")

    def vapa_loss(
        self,
        examples: Sequence[VAPAAction],
        *,
        reference: TinyModel,
        kl_weight: float,
        ratio_clip: float | None,
        kl_mode: str,
    ) -> LossReport:
        assert ratio_clip is None
        assert kl_mode == "k3"
        tokens = sum(item.token_count for item in examples)
        advantage = sum(item.advantage * item.token_count for item in examples) / tokens
        policy = -self.weight * advantage
        kl = (self.weight - reference.weight) ** 2
        total = policy + kl_weight * kl
        derivative = -advantage + 2 * kl_weight * (self.weight - reference.weight)
        # Models with dropout consume framework RNG.  The lifecycle must key this to
        # the checkpointed update cursor so interrupted and uninterrupted runs match.
        derivative += random.random() * 0.01

        def backward(scale: float) -> None:
            self.grad += scale * derivative

        return LossReport(total, policy, kl, tokens, backward)

    def clip_grad_norm(self, max_norm: float) -> float:
        norm = abs(self.grad)
        if norm > max_norm:
            self.grad *= max_norm / norm
        return norm

    def state_dict(self) -> Mapping[str, Any]:
        return {"name": self.name, "weight": self.weight}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.name = str(state["name"])
        self.weight = float(state["weight"])


class TinyOptimizer:
    def __init__(self, model: TinyModel) -> None:
        self.model = model
        self.param_groups = [{"lr": 0.1}]
        self.steps = 0

    def zero_grad(self) -> None:
        self.model.grad = 0.0

    def step(self) -> None:
        self.model.weight -= self.param_groups[0]["lr"] * self.model.grad
        self.steps += 1

    def state_dict(self) -> Mapping[str, Any]:
        return {"steps": self.steps, "param_groups": self.param_groups}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.param_groups = [dict(item) for item in state["param_groups"]]


def _episode(
    identifier: str = "train-1",
    *,
    suite: str = "calculation",
    task_type: str = "type-a",
) -> Episode:
    source = tiny_episode()
    task = replace(
        source.task,
        instance_id=identifier,
        patient_id=f"patient-{identifier}",
        metadata={
            "history_quartile": 1,
            "suite": suite,
            "task_type": task_type,
        },
    )
    events = tuple(replace(event, patient_id=task.patient_id) for event in source.events)
    return Episode(task, events, gold_answer="ok")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "vapa.toml"
    config.write_text(
        """
name = "entrypoint-test"
seed = 19

[environment]
memory_capacity = 8
action_budget = 2
turn_cap = 3
context_tokens = 128
max_turn_tokens = 8

[credit]
process_weight = 0.05
cost_weight = 0.02
process_discount = 1.0
local_weight = 1.0
epsilon = 1e-8
use_process_rewards = true
use_step_credit = true

[replay]
base_group_size = 2
max_fork_states = 1
siblings_per_fork = 1
memory_pressure_fraction = 0.75
fork_enabled = true
reallocate_disabled_forks = false

[optimization]
sft_learning_rate = 0.0001
rl_learning_rate = 0.000005
sampled_token_budget = 13
update_token_floor = 6
warmup_fraction = 0.0
final_lr_fraction = 0.1
weight_decay = 0.01
gradient_clip = 1.0
kl_weight = 0.01
lora_rank = 2
lora_alpha = 4
lora_dropout = 0.0

[model]
name = "fake/model"
thinking = false
dtype = "float32"
temperature = 1.0
top_p = 1.0
top_k = 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    episodes = tmp_path / "train.jsonl"
    import json

    records = (
        _episode("train-calculation", suite="calculation", task_type="calc-a"),
        _episode("train-retrieval", suite="retrieval", task_type="retrieval-a"),
    )
    episodes.write_text(
        "".join(json.dumps(episode_to_record(item), sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    return config, episodes


def _factory_sink(
    sink: list[TrainingComponents],
    *,
    initial_weight: float = 0.5,
    policy: Any | None = None,
):
    selected_policy = SeededAnswerPolicy() if policy is None else policy

    def factory(config, distributed):
        del config
        assert distributed == DistributedContext()
        actor = TinyModel("actor", initial_weight)
        optimizer = TinyOptimizer(actor)
        components = TrainingComponents(
            policy=selected_policy,
            tokenizer=TinyTokenizer(),
            actor=actor,
            reference=TinyModel("actor", initial_weight),
            optimizer=optimizer,
            scheduler=None,
            store=JsonStateStore(),
            component_id="test:tiny-v1",
            optimizer_name="tiny-sgd",
            scheduler_name="none",
            inference_spec={"format_version": 1, "model_name": "fake/model"},
            initialization_identity={
                "kind": "test-fixture",
                "sha256": fingerprint_payload(
                    {"fixture": "tiny-initial-v1", "weight": initial_weight}
                ),
            },
        )
        sink.append(components)
        return components

    factory.__vapa_content_id__ = fingerprint_payload(
        {
            "fixture": "tiny-component-factory-v1",
            "initial_weight": initial_weight,
            "policy": f"{type(selected_policy).__module__}.{type(selected_policy).__qualname__}",
        }
    )
    return factory


def test_full_rl_lifecycle_resumes_and_excludes_final_comparison_group(tmp_path: Path):
    config, episodes = _write_inputs(tmp_path)
    output = tmp_path / "run"
    verifier = Path("examples/demo_verifier_catalog.json").resolve()
    calculators = Path("examples/tiny_calculators.json").resolve()
    stacks: list[TrainingComponents] = []
    factory = _factory_sink(stacks)

    stopped = run_vapa_training(
        config,
        episodes,
        output,
        component_factory=factory,
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(
            run_id="resume-test",
            checkpoint_every=1,
            max_optimizer_steps=1,
            demo_catalogs=True,
        ),
    )
    assert stopped.status == "max_optimizer_steps"
    assert stopped.global_step == 1
    assert stopped.sampled_tokens == 6
    assert stopped.checkpoint_path is not None
    assert stacks[-1].optimizer.steps == 1
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"uncheckpointed":')
    with (output / "replay_quota.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"uncheckpointed":')
    with (output / "metrics.jsonl").open("ab") as stream:
        stream.write(b"\xc3")
    with (output / "replay_quota.jsonl").open("ab") as stream:
        stream.write(b"\xc3")

    completed = run_vapa_training(
        config,
        episodes,
        output,
        component_factory=factory,
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(
            run_id="resume-test",
            checkpoint_every=1,
            resume_from=stopped.checkpoint_path,
            demo_catalogs=True,
        ),
    )
    assert completed.status == "complete"
    assert completed.global_step == 2
    assert completed.sampled_tokens == 18
    assert completed.trainable_tokens == 12
    assert completed.metric_records == 3
    assert stacks[-1].optimizer.steps == 2
    assert completed.checkpoint_path is not None
    manifest = read_manifest(completed.checkpoint_path)
    assert manifest.runtime.sampled_tokens == 18
    assert manifest.runtime.extra["inference_spec"]["format_version"] == 1
    environment_spec = manifest.runtime.extra["environment_spec"]
    assert set(environment_spec) == {
        "schema_version",
        "memory_capacity",
        "action_budget",
        "turn_cap",
        "retrieval_limit",
        "implementation_sha256",
        "calculator_manifest",
        "calculator_manifest_sha256",
    }
    assert environment_spec["retrieval_limit"] == 5

    import json

    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["batch_sampled_tokens"] for row in metrics] == [6, 6, 6]
    assert metrics[-1]["comparison_only_instances"] == 1
    assert metrics[-1]["action_tokens"] == 0
    assert metrics[-1]["loss"] is None
    quotas = (output / "replay_quota.jsonl").read_text().splitlines()
    assert len(quotas) == 3
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    for field in (
        "scaffold_sha256",
        "outcome_scorer",
        "deterministic",
        "replay_quota",
        "verifier_factory",
        "verifier_catalog_fingerprint",
        "initial_state_sha256",
        "control_pairing_contract",
        "input_artifacts",
    ):
        assert field in run_manifest
    assert run_manifest["input_artifacts"]["config"]["sha256"]
    assert run_manifest["input_artifacts"]["episodes"]["sha256"]

    uninterrupted_stacks: list[TrainingComponents] = []
    uninterrupted = run_vapa_training(
        config,
        episodes,
        tmp_path / "uninterrupted",
        component_factory=_factory_sink(uninterrupted_stacks),
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(
            run_id="resume-test",
            checkpoint_every=1,
            demo_catalogs=True,
        ),
    )
    assert uninterrupted.global_step == completed.global_step
    assert uninterrupted_stacks[-1].actor.weight == pytest.approx(stacks[-1].actor.weight)


def test_resume_rejects_a_different_frozen_reference_initialization(tmp_path: Path):
    config, episodes = _write_inputs(tmp_path)
    verifier = Path("examples/demo_verifier_catalog.json").resolve()
    calculators = Path("examples/tiny_calculators.json").resolve()
    first_stacks: list[TrainingComponents] = []
    stopped = run_vapa_training(
        config,
        episodes,
        tmp_path / "reference-contract",
        component_factory=_factory_sink(first_stacks, initial_weight=0.5),
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(
            run_id="reference-contract",
            max_optimizer_steps=1,
            demo_catalogs=True,
        ),
    )
    assert stopped.checkpoint_path is not None
    changed_stacks: list[TrainingComponents] = []
    with pytest.raises(Exception, match="run manifest does not match"):
        run_vapa_training(
            config,
            episodes,
            tmp_path / "reference-contract",
            component_factory=_factory_sink(changed_stacks, initial_weight=0.6),
            verifier_manifest=verifier,
            calculator_manifest=calculators,
            settings=RLRunSettings(
                run_id="reference-contract",
                resume_from=stopped.checkpoint_path,
                demo_catalogs=True,
            ),
        )


def test_dry_run_never_builds_model_components(tmp_path: Path):
    config, episodes = _write_inputs(tmp_path)
    called = False

    def forbidden_factory(config, distributed):
        nonlocal called
        called = True
        raise AssertionError((config, distributed))

    result = run_vapa_training(
        config,
        episodes,
        tmp_path / "dry-run-output",
        component_factory=forbidden_factory,
        verifier_manifest=Path("examples/demo_verifier_catalog.json"),
        calculator_manifest=Path("examples/tiny_calculators.json"),
        settings=RLRunSettings(dry_run=True, demo_catalogs=True),
    )
    assert result.status == "dry_run"
    assert result.episode_count == 2
    assert not called
    assert not result.output_directory.exists()


def test_rl_rejects_sensitive_artifacts_in_tracked_public_paths(tmp_path: Path):
    config, episodes = _write_inputs(tmp_path)
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked public location"):
        run_vapa_training(
            config,
            episodes,
            repository / "docs" / "patient-run",
            component_factory=_factory_sink([]),
            verifier_manifest=Path("examples/demo_verifier_catalog.json"),
            calculator_manifest=Path("examples/tiny_calculators.json"),
            settings=RLRunSettings(demo_catalogs=True),
        )


@pytest.mark.parametrize("installed", [False, True])
def test_installed_cli_demo_dry_run_is_dependency_free_and_scoped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch, installed: bool
):
    if installed:
        import vapa.training.rl_train as module

        monkeypatch.setattr(
            module, "__file__", str(tmp_path / "venv/lib/vapa/training/rl_train.py")
        )
    output = tmp_path / "cli-dry-run"
    exit_code = main(
        [
            "configs/vapa_qwen.toml",
            "examples/tiny_episode.json",
            str(output),
            "--dry-run",
            "--demo-catalogs",
        ]
    )
    assert exit_code == 0
    import json

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run"
    assert report["training_ready"] is False
    assert any("model/SFT load plan" in item for item in report["readiness_issues"])
    assert not any("suite/task_type" in item for item in report["readiness_issues"])
    assert not any("history_quartile" in item for item in report["readiness_issues"])
    assert not output.exists()


def test_cli_exact_model_plan_dry_run_validates_without_loading_ml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    output = tmp_path / "planned-dry-run"
    assert (
        main(
            [
                "configs/vapa_qwen.toml",
                "examples/tiny_episode.json",
                str(output),
                "--dry-run",
                "--demo-catalogs",
                "--model-revision",
                "a" * 40,
                "--use-processor",
                "yes",
                "--no-lora",
                "--allow-base-model-init",
            ]
        )
        == 0
    )
    import json

    report = json.loads(capsys.readouterr().out)
    assert not any("model/SFT load plan" in item for item in report["readiness_issues"])
    assert not output.exists()


def test_rank_sharding_and_episode_stream_are_stable():
    episodes = tuple(_episode(f"e-{index}") for index in range(5))
    left = shard_episodes(episodes[::-1], DistributedContext(rank=0, world_size=2))
    right = shard_episodes(episodes, DistributedContext(rank=1, world_size=2))
    assert [item.task.instance_id for item in left] == ["e-0", "e-2", "e-4"]
    assert [item.task.instance_id for item in right] == ["e-1", "e-3"]

    balanced = tuple(
        _episode(
            f"{suite}-{task_type}-{index}",
            suite=suite,
            task_type=task_type,
        )
        for suite in ("calculation", "retrieval")
        for task_type in ("a", "b")
        for index in range(2)
    )
    first = DeterministicEpisodeStream(balanced, seed=7, rank=0)
    second = DeterministicEpisodeStream(balanced, seed=7, rank=0)
    first_items = [first.next(max_history_quartile=2) for _ in range(16)]
    second_items = [second.next(max_history_quartile=2) for _ in range(16)]
    first_ids = [item.task.instance_id for item in first_items]
    second_ids = [item.task.instance_id for item in second_items]
    assert first_ids == second_ids
    suites = [item.task.metadata["suite"] for item in first_items]
    assert suites.count("calculation") == suites.count("retrieval") == 8
    for suite in ("calculation", "retrieval"):
        task_types = [
            item.task.metadata["task_type"]
            for item in first_items
            if item.task.metadata["suite"] == suite
        ]
        assert task_types.count("a") == task_types.count("b") == 4
    restored = DeterministicEpisodeStream(
        balanced,
        seed=7,
        rank=0,
        cycle=first.cycle,
        position=first.position,
    )
    assert (
        restored.next(max_history_quartile=2).task.instance_id
        == first.next(max_history_quartile=2).task.instance_id
    )


def test_non_demo_training_rejects_public_placeholder_catalogs(tmp_path: Path):
    config, episodes = _write_inputs(tmp_path)
    with pytest.raises(Exception, match="paper-exact"):
        run_vapa_training(
            config,
            episodes,
            tmp_path / "run",
            component_factory=None,
            verifier_manifest=Path("examples/demo_verifier_catalog.json"),
            calculator_manifest=Path("examples/tiny_calculators.json"),
            settings=RLRunSettings(dry_run=True),
        )


def test_non_demo_training_requires_an_explicit_outcome_scorer(tmp_path: Path):
    config, episodes = _write_inputs(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "use_process_rewards = true", "use_process_rewards = false"
        ),
        encoding="utf-8",
    )
    verifier = tmp_path / "verifier.json"
    calculator = tmp_path / "calculators.json"
    verifier.write_text(
        Path("examples/demo_verifier_catalog.json")
        .read_text(encoding="utf-8")
        .replace('"paper_exact": false', '"paper_exact": true'),
        encoding="utf-8",
    )
    calculator.write_text(
        Path("examples/tiny_calculators.json")
        .read_text(encoding="utf-8")
        .replace('"paper_exact": false', '"paper_exact": true'),
        encoding="utf-8",
    )
    stacks: list[TrainingComponents] = []
    with pytest.raises(Exception, match="explicit author outcome_scorer"):
        run_vapa_training(
            config,
            episodes,
            tmp_path / "missing-scorer",
            component_factory=_factory_sink(stacks),
            verifier_manifest=verifier,
            calculator_manifest=calculator,
        )


def test_verified_sft_checkpoint_initializes_actor_and_reference(tmp_path: Path):
    digest = "a" * 40
    spec = TransformersInferenceSpec(
        model_name="fake/model",
        model_revision=digest,
        tokenizer_name="fake/model",
        tokenizer_revision=digest,
        model_kind="causal",
        use_processor=False,
        dtype="float32",
        device="cpu",
        lora_enabled=False,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=None,
        enable_thinking=False,
        context_tokens=128,
        scaffold="",
        max_tokens=8,
    )
    source = TinyModel("actor", 0.875)
    tokenizer = TinyTokenizer()
    optimizer = TinyOptimizer(source)
    contract = CheckpointContract(
        run_id="sft-source",
        run_manifest_fingerprint=fingerprint_payload({"run": "sft"}),
        config_fingerprint=fingerprint_payload({"config": "sft"}),
        model_fingerprint=source.fingerprint,
        reference_model_fingerprint=source.fingerprint,
        tokenizer_fingerprint=tokenizer.fingerprint,
        optimizer_name="tiny-sgd",
        scheduler_name="none",
        state_format="json-v1",
    )
    checkpoint = tmp_path / "sft-checkpoint"
    save_checkpoint(
        checkpoint,
        contract=contract,
        runtime=RuntimeState(
            1,
            10,
            10,
            0,
            {"objective": "sft", "inference_spec": spec.to_dict()},
        ),
        model=source,
        reference=source,
        tokenizer=tokenizer,
        optimizer=optimizer,
        store=JsonStateStore(),
    )
    actor = TinyModel("actor", 0.0)
    reference = TinyModel("actor", -1.0)
    identity = initialize_from_sft_checkpoint(
        checkpoint,
        actor=actor,
        reference=reference,
        tokenizer=tokenizer,
        expected_spec=spec,
    )
    assert actor.weight == reference.weight == 0.875
    assert identity["kind"] == "directory"


def test_no_fork_control_consumes_realized_per_occurrence_quota(tmp_path: Path):
    config, episodes = _write_inputs(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8")
        .replace("sampled_token_budget = 13", "sampled_token_budget = 100")
        .replace("update_token_floor = 6", "update_token_floor = 12"),
        encoding="utf-8",
    )
    verifier = Path("examples/demo_verifier_catalog.json").resolve()
    calculators = Path("examples/tiny_calculators.json").resolve()
    donor_stacks: list[TrainingComponents] = []
    donor = run_vapa_training(
        config,
        episodes,
        tmp_path / "donor",
        component_factory=_factory_sink(donor_stacks),
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(run_id="fork-donor", demo_catalogs=True),
    )
    assert donor.status == "complete"
    donor_result = json.loads((tmp_path / "donor/result.json").read_text())
    donor_attestation = donor_result["replay_quota_attestation"]
    assert donor_attestation["record_count"] > 0
    assert (
        read_manifest(Path(donor.checkpoint_path or "")).runtime.extra["replay_quota_attestation"]
        == donor_attestation
    )

    no_fork_config = tmp_path / "no-fork.toml"
    no_fork_config.write_text(
        config.read_text(encoding="utf-8")
        .replace('name = "entrypoint-test"', 'name = "entrypoint-no-fork"')
        .replace("fork_enabled = true", "fork_enabled = false")
        .replace("reallocate_disabled_forks = false", "reallocate_disabled_forks = true"),
        encoding="utf-8",
    )
    control_stacks: list[TrainingComponents] = []
    control = run_vapa_training(
        no_fork_config,
        episodes,
        tmp_path / "control",
        component_factory=_factory_sink(control_stacks),
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(
            run_id="no-fork-control",
            replay_quota_path=tmp_path / "donor/replay_quota.jsonl",
            demo_catalogs=True,
        ),
    )
    assert control.status == "complete"
    metrics = [
        json.loads(line) for line in (tmp_path / "control/metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["branch_rollouts"] == 0 for row in metrics)
    assert all(row["replay_quota_balance"] <= 0 for row in metrics)
    assert metrics[0]["base_rollouts_per_instance"] == [3, 3]
    assert all(
        max(row["base_rollouts_per_instance"]) - min(row["base_rollouts_per_instance"]) <= 1
        for row in metrics
    )

    mismatched = tmp_path / "no-fork-wrong-seed.toml"
    mismatched.write_text(
        no_fork_config.read_text(encoding="utf-8").replace("seed = 19", "seed = 20"),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="control pairing contract"):
        run_vapa_training(
            mismatched,
            episodes,
            tmp_path / "wrong-seed-control",
            component_factory=_factory_sink([]),
            verifier_manifest=verifier,
            calculator_manifest=calculators,
            settings=RLRunSettings(
                run_id="wrong-seed-control",
                replay_quota_path=tmp_path / "donor/replay_quota.jsonl",
                demo_catalogs=True,
            ),
        )

    quota_path = tmp_path / "donor/replay_quota.jsonl"
    quota_rows = [json.loads(line) for line in quota_path.read_text().splitlines()]
    quota_rows[0]["branch_sampled_tokens"] += 1
    quota_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in quota_rows),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="donor result attestation"):
        run_vapa_training(
            no_fork_config,
            episodes,
            tmp_path / "mutated-quota-control",
            component_factory=_factory_sink([]),
            verifier_manifest=verifier,
            calculator_manifest=calculators,
            settings=RLRunSettings(
                run_id="mutated-quota-control",
                replay_quota_path=quota_path,
                demo_catalogs=True,
            ),
        )


def test_matched_control_exhausts_variable_length_donor_trace(tmp_path: Path) -> None:
    config, episodes = _write_inputs(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8")
        .replace("seed = 19", "seed = 5")
        .replace("sampled_token_budget = 13", "sampled_token_budget = 300")
        .replace("update_token_floor = 6", "update_token_floor = 100"),
        encoding="utf-8",
    )
    verifier = Path("examples/demo_verifier_catalog.json").resolve()
    calculators = Path("examples/tiny_calculators.json").resolve()
    policy = VariableLengthAnswerPolicy()
    donor = run_vapa_training(
        config,
        episodes,
        tmp_path / "variable-donor",
        component_factory=_factory_sink([], policy=policy),
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(run_id="variable-donor", demo_catalogs=True),
    )
    no_fork_config = tmp_path / "variable-no-fork.toml"
    no_fork_config.write_text(
        config.read_text(encoding="utf-8")
        .replace('name = "entrypoint-test"', 'name = "entrypoint-no-fork"')
        .replace("fork_enabled = true", "fork_enabled = false")
        .replace("reallocate_disabled_forks = false", "reallocate_disabled_forks = true"),
        encoding="utf-8",
    )
    control = run_vapa_training(
        no_fork_config,
        episodes,
        tmp_path / "variable-control",
        component_factory=_factory_sink([], policy=policy),
        verifier_manifest=verifier,
        calculator_manifest=calculators,
        settings=RLRunSettings(
            run_id="variable-control",
            replay_quota_path=tmp_path / "variable-donor/replay_quota.jsonl",
            demo_catalogs=True,
        ),
    )

    donor_records = (tmp_path / "variable-donor/replay_quota.jsonl").read_text().splitlines()
    control_runtime = read_manifest(Path(control.checkpoint_path or "")).runtime
    assert donor.status == control.status == "complete"
    assert control.sampled_tokens >= 300
    assert control_runtime.extra["replay_quota_cursor"] == len(donor_records)
    assert control_runtime.extra["replay_quota"]["balance_tokens"] == 0
    metrics = [
        json.loads(line)
        for line in (tmp_path / "variable-control/metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["replay_quota_balance"] == 0 for row in metrics)
    assert metrics[-1]["comparison_only_instances"] == 1
    assert metrics[-1]["action_tokens"] == 0


@pytest.mark.parametrize(
    ("input_name", "expected_label"),
    (
        ("config", "config"),
        ("episodes", "episodes"),
        ("verifier", "verifier_manifest"),
        ("calculators", "calculator_manifest"),
    ),
)
def test_component_factory_cannot_swap_captured_training_inputs(
    tmp_path: Path,
    input_name: str,
    expected_label: str,
) -> None:
    config, episodes = _write_inputs(tmp_path)
    verifier = tmp_path / "verifier.json"
    calculators = tmp_path / "calculators.json"
    verifier.write_bytes(Path("examples/demo_verifier_catalog.json").read_bytes())
    calculators.write_bytes(Path("examples/tiny_calculators.json").read_bytes())
    targets = {
        "config": config,
        "episodes": episodes,
        "verifier": verifier,
        "calculators": calculators,
    }
    base_factory = _factory_sink([])

    def mutating_factory(experiment, distributed):
        components = base_factory(experiment, distributed)
        target = targets[input_name]
        target.write_bytes(target.read_bytes() + b" ")
        return components

    with pytest.raises(Exception, match=f"{expected_label} input changed"):
        run_vapa_training(
            config,
            episodes,
            tmp_path / f"mutated-{input_name}",
            component_factory=mutating_factory,
            verifier_manifest=verifier,
            calculator_manifest=calculators,
            settings=RLRunSettings(run_id="input-snapshot", demo_catalogs=True),
        )
