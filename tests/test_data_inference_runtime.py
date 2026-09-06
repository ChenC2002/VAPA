from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from vapa.actions import make_action
from vapa.data.adapters import (
    CredentialedAdapterUnavailable,
    GenericEventAdapter,
    builtin_adapters,
)
from vapa.data.episodes import load_episode_objects
from vapa.data.io import DataValidationError, load_json, load_jsonl
from vapa.data.pipeline import prepare_dataset
from vapa.environment.state_manager import StateManager
from vapa.evaluation.runner import evaluate_episodes
from vapa.inference import build_inference_engine, deterministic_instance_seed, factory_identity
from vapa.policies.base import PolicyDecision
from vapa.policies.heuristic import HeuristicPolicy
from vapa.schemas import ActionKind, Episode


def _write_synthetic_sources(root: Path, *, event_format: str = "csv") -> Path:
    events = [
        {
            "event_id": "e-p1-old",
            "patient": "p1",
            "when": "2025-12-20T00:00:00Z",
            "kind": "lab",
            "name": "hba1c",
            "reading": 6.4,
            "unit": "%",
        },
        {
            "event_id": "e-p1-future",
            "patient": "p1",
            "when": "2026-01-11T00:00:00Z",
            "kind": "lab",
            "name": "hba1c",
            "reading": 9.9,
            "unit": "%",
        },
        {
            "event_id": "e-p2-old",
            "patient": "p2",
            "when": "2025-12-21T00:00:00Z",
            "kind": "lab",
            "name": "hba1c",
            "reading": 5.8,
            "unit": "%",
        },
        {
            "event_id": "e-p3-old",
            "patient": "p3",
            "when": "2025-12-22T00:00:00Z",
            "kind": "lab",
            "name": "hba1c",
            "reading": 7.1,
            "unit": "%",
        },
    ]
    if event_format == "csv":
        events_path = root / "events.csv"
        rows = ["event_id,patient,when,kind,name,reading,unit"]
        rows.extend(
            f"{event['event_id']},{event['patient']},{event['when']},"
            f"{event['kind']},{event['name']},{event['reading']},{event['unit']}"
            for event in events
        )
        events_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        event_json_fields = ["value"]
    else:
        events_path = root / "events.jsonl"
        events_path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
        event_json_fields = []

    tasks = [
        {
            "task_id": f"instance-{index}",
            "patient": f"p{index}",
            "prompt": "What is the latest pre-cutoff HbA1c in the last 90 days?",
            "cutoff_at": "2026-01-10T00:00:00Z",
            "task_family": "latest_laboratory_value",
            "answer": answer,
            "fields": ["hba1c"],
            "window": "last 90d",
            "answer_kind": "number",
            "evidence": [f"e-p{index}-old"],
            "metadata": {
                "history_quartile": 1,
                "suite": "retrieval",
                "task_type": "latest_laboratory_value",
            },
        }
        for index, answer in ((1, 6.4), (2, 5.8), (3, 7.1))
    ]
    tasks_path = root / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "content_kind": "public",
        "adapter": "generic_events",
        "source_files": [events_path.name, tasks_path.name],
        "events": {
            "path": events_path.name,
            "format": event_format,
            "columns": {
                "pointer": "event_id",
                "patient_id": "patient",
                "timestamp": "when",
                "domain": "kind",
                "field": "name",
                "value": "reading",
                "unit": "unit",
            },
            "json_fields": event_json_fields,
        },
        "tasks": {
            "path": tasks_path.name,
            "format": "jsonl",
            "columns": {
                "instance_id": "task_id",
                "patient_id": "patient",
                "instruction": "prompt",
                "cutoff": "cutoff_at",
                "family": "task_family",
                "gold_answer": "answer",
                "requested_fields": "fields",
                "requested_window": "window",
                "answer_type": "answer_kind",
                "metadata": "metadata",
                "reference_evidence": "evidence",
            },
        },
        "splits": {
            "train": 1.0,
            "validation": 0.0,
            "test": 0.0,
            "seed": 13,
            "namespace": "synthetic-runtime-test-v1",
        },
    }
    manifest_path = root / "dataset.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_factory_identities_bind_source_and_partial_arguments():
    def first(value):
        return value + 1

    def second(value):
        return value + 2

    second.__qualname__ = first.__qualname__
    assert factory_identity(first) != factory_identity(second)
    assert factory_identity(partial(first, 1)) != factory_identity(partial(first, 2))

    def make_scorer(tolerance):
        def scorer(value):
            return abs(value) <= tolerance

        return scorer

    class CallableScorer:
        def __init__(self, tolerance):
            self.tolerance = tolerance

        def __call__(self, value):
            return abs(value) <= self.tolerance

        def score(self, value):
            return self(value)

    assert factory_identity(make_scorer(0.1)) != factory_identity(make_scorer(0.2))
    assert factory_identity(CallableScorer(0.1)) != factory_identity(CallableScorer(0.2))
    assert factory_identity(CallableScorer(0.1).score) != factory_identity(
        CallableScorer(0.2).score
    )


@pytest.mark.parametrize("event_format", ["csv", "jsonl"])
def test_manifest_pipeline_is_cutoff_safe_and_patient_disjoint(tmp_path, event_format):
    manifest_path = _write_synthetic_sources(tmp_path, event_format=event_format)
    result = prepare_dataset(manifest_path, tmp_path / "prepared")

    assert result.total_episodes == 3
    assert result.episode_counts == {"train": 3, "validation": 0, "test": 0}
    episodes = load_episode_objects(result.split_paths["train"])
    assert [episode.task.instance_id for episode in episodes] == [
        "instance-1",
        "instance-2",
        "instance-3",
    ]
    p1 = episodes[0]
    assert [event.pointer for event in p1.events] == ["e-p1-old"]
    assert all(
        event.timestamp <= p1.task.cutoff for episode in episodes for event in episode.events
    )
    assert {episode.task.patient_id for episode in episodes} == {"p1", "p2", "p3"}

    prepared_manifest = load_json(result.manifest_path)
    assert prepared_manifest["source_manifest_sha256"]
    assert prepared_manifest["adapter_implementation"].startswith(
        "vapa.data.adapters:GenericEventAdapter.build_episodes@sha256:"
    )
    assert len(prepared_manifest["pipeline_implementation_sha256"]) == 64
    assert [item["path"] for item in prepared_manifest["source_files"]] == [
        f"events.{event_format}",
        "tasks.jsonl",
    ]
    assert all(item["sha256"] for item in prepared_manifest["source_files"])
    assert prepared_manifest["files"]["train"]["sha256"]
    with pytest.raises(FileExistsError):
        prepare_dataset(manifest_path, tmp_path / "prepared")


def test_credentialed_placeholders_do_not_invent_source_schemas(tmp_path):
    adapter = builtin_adapters()["mimic_iv"]
    with pytest.raises(CredentialedAdapterUnavailable, match="authorized adapter"):
        adapter.build_episodes({}, manifest_directory=tmp_path)


def test_credentialed_preparation_rejects_tracked_public_output(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    manifest_path = _write_synthetic_sources(repository)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_kind"] = "credentialed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tracked public location"):
        prepare_dataset(
            manifest_path,
            repository / "docs" / "patient-episodes",
        )


def test_generic_preparation_hashes_every_consumed_source(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    (tmp_path / "dummy.txt").write_text("not the dataset\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"] = ["dummy.txt"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataValidationError, match="must be listed in source_files"):
        prepare_dataset(manifest_path, tmp_path / "prepared")


def test_preparation_rejects_manifest_changed_by_adapter(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter"] = "mutating"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class MutatingAdapter:
        adapter_id = "mutating"

        def build_episodes(self, payload, *, manifest_directory):
            episodes = GenericEventAdapter().build_episodes(
                payload,
                manifest_directory=manifest_directory,
            )
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            return episodes

    with pytest.raises(DataValidationError, match="manifest changed"):
        prepare_dataset(
            manifest_path,
            tmp_path / "prepared",
            adapters={"mutating": MutatingAdapter()},
        )


def test_preparation_binds_custom_adapter_code_and_state(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter"] = "parameterized"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class ParameterizedAdapter:
        adapter_id = "parameterized"

        def __init__(self, suffix):
            self.suffix = suffix

        def build_episodes(self, payload, *, manifest_directory):
            episodes = GenericEventAdapter().build_episodes(
                payload,
                manifest_directory=manifest_directory,
            )
            return [
                type(episode)(
                    episode.task,
                    episode.events,
                    gold_answer=f"{episode.gold_answer}{self.suffix}",
                    reference_evidence=episode.reference_evidence,
                )
                for episode in episodes
            ]

    first = prepare_dataset(
        manifest_path,
        tmp_path / "prepared-a",
        adapters={"parameterized": ParameterizedAdapter("-a")},
    )
    second = prepare_dataset(
        manifest_path,
        tmp_path / "prepared-b",
        adapters={"parameterized": ParameterizedAdapter("-b")},
    )
    first_manifest = load_json(first.manifest_path)
    second_manifest = load_json(second.manifest_path)
    assert first_manifest["adapter_implementation"] != second_manifest["adapter_implementation"]


def test_evaluation_defaults_to_sensitive_output_guard(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    prepared = prepare_dataset(manifest_path, tmp_path / "prepared")
    episodes = load_episode_objects(prepared.split_paths["train"])
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tracked public location"):
        evaluate_episodes(
            episodes,
            repository / "docs" / "patient-evaluation",
            repository_root=repository,
        )


def test_resume_safe_deterministic_evaluation_end_to_end(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    prepared = prepare_dataset(manifest_path, tmp_path / "prepared")
    episodes = list(reversed(load_episode_objects(prepared.split_paths["train"])))
    output = tmp_path / "evaluation"
    calls = 0
    policy_loads = 0

    def counting_policy_factory(checkpoint):
        nonlocal policy_loads
        assert checkpoint is None
        policy_loads += 1
        return HeuristicPolicy()

    counting_policy_factory.__vapa_content_id__ = "1" * 64

    def interrupt_after_first(record):
        nonlocal calls
        assert record["status"] == "ok"
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated scheduler interruption")

    with pytest.raises(RuntimeError, match="scheduler interruption"):
        evaluate_episodes(
            episodes,
            output,
            seed=29,
            max_instances=2,
            policy_factory=counting_policy_factory,
            policy_id="synthetic-heuristic-v1",
            on_result=interrupt_after_first,
        )
    durable = load_jsonl(output / "predictions.jsonl")
    assert [record["instance_id"] for record in durable] == ["instance-1"]
    evaluation_manifest = load_json(output / "run_manifest.json")
    assert len(evaluation_manifest["implementation_sha256"]) == 64

    resumed = evaluate_episodes(
        episodes,
        output,
        seed=29,
        max_instances=2,
        policy_factory=counting_policy_factory,
        policy_id="synthetic-heuristic-v1",
    )
    assert resumed.resumed == 1
    assert resumed.newly_completed == 1
    assert resumed.metrics["instances"] == 2
    assert resumed.metrics["successful"] == 2
    assert resumed.metrics["task_success"] == 100.0
    assert resumed.metrics["reference_evidence_coverage"] == 100.0

    records = load_jsonl(resumed.predictions_path)
    assert [record["instance_id"] for record in records] == ["instance-1", "instance-2"]
    assert [record["prediction"] for record in records] == [6.4, 5.8]
    assert records[0]["seed"] == deterministic_instance_seed(29, "instance-1")

    completed = evaluate_episodes(
        episodes,
        output,
        seed=29,
        max_instances=2,
        policy_factory=counting_policy_factory,
        policy_id="synthetic-heuristic-v1",
    )
    assert completed.resumed == 2
    assert completed.newly_completed == 0
    assert policy_loads == 2

    with pytest.raises(DataValidationError, match="manifest mismatch"):
        evaluate_episodes(
            episodes,
            output,
            seed=30,
            max_instances=2,
            policy_factory=counting_policy_factory,
            policy_id="synthetic-heuristic-v1",
        )


def test_binary_evaluation_emits_fixed_position_scores_and_frozen_roster(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    prepared = prepare_dataset(manifest_path, tmp_path / "prepared")
    originals = load_episode_objects(prepared.split_paths["train"])
    labels = (1, 0, 1)
    episodes = [
        Episode(
            task=replace(
                episode.task,
                family="binary_transfer",
                metadata={
                    **episode.task.metadata,
                    "evaluation_task_id": "ehrshot_binary",
                },
            ),
            events=episode.events,
            gold_answer=label,
        )
        for episode, label in zip(originals, labels, strict=True)
    ]

    class BinaryPolicy:
        def sample(self, observation, *, rng, n=1, greedy=False):
            del rng, greedy
            prediction = "yes" if observation.task.patient_id == "p2" else 1
            action = make_action(ActionKind.ANSWER, prediction=prediction, evidence=[])
            return [PolicyDecision(action) for _ in range(n)]

        def binary_answer_probability(self, observation, decision=None):
            assert decision is not None
            assert decision.action is not None
            assert decision.action.arguments["prediction"] in {0, 1}
            return {"p1": 0.9, "p2": 0.2, "p3": 0.8}[observation.task.patient_id]

    def binary_policy_factory(checkpoint):
        assert checkpoint is None
        return BinaryPolicy()

    binary_policy_factory.__vapa_content_id__ = "3" * 64
    result = evaluate_episodes(
        episodes,
        tmp_path / "binary-evaluation",
        policy_factory=binary_policy_factory,
        policy_id="synthetic-binary-v1",
        binary_readout=True,
        analysis_seed=7,
        selection_seed=19,
    )

    assert result.binary_predictions_path is not None
    binary_records = load_jsonl(result.binary_predictions_path)
    assert [record["score"] for record in binary_records] == [0.9, None, 0.8]
    assert [record["label"] for record in binary_records] == list(labels)
    assert {record["task_id"] for record in binary_records} == {"ehrshot_binary"}
    assert {record["seed"] for record in binary_records} == {7}
    assert [record["format_valid"] for record in binary_records] == [True, False, True]
    run_manifest = load_json(result.run_manifest_path)
    assert run_manifest["selection"]["method"] == ("frozen-label-blind-patient-round-robin-v1")
    assert run_manifest["selection"]["task_cap"] == 2_500
    assert run_manifest["binary_readout"] is True

    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate_episodes(
            episodes,
            tmp_path / "invalid-binary-evaluation",
            max_instances=1,
            policy_factory=binary_policy_factory,
            policy_id="synthetic-binary-v1",
            binary_readout=True,
        )


def test_evaluation_macro_averages_equal_task_ids_not_family_rows(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    prepared = prepare_dataset(manifest_path, tmp_path / "prepared")
    originals = load_episode_objects(prepared.split_paths["train"])
    episodes = [
        Episode(
            task=replace(
                episode.task,
                metadata={
                    **episode.task.metadata,
                    "evaluation_task_id": "task-a" if index < 2 else "task-b",
                },
            ),
            events=episode.events,
            gold_answer=episode.gold_answer if index < 2 else 99,
            reference_evidence=episode.reference_evidence,
        )
        for index, episode in enumerate(originals)
    ]

    result = evaluate_episodes(
        episodes,
        tmp_path / "task-macro-evaluation",
        policy_id="synthetic-heuristic-v1",
    )

    assert result.metrics["task_success"] == pytest.approx(200 / 3)
    assert result.metrics["macro_task_success"] == pytest.approx(50.0)
    assert result.metrics["cost_normalized_success"] == pytest.approx(100 / 2 * (5 / 6))
    assert result.metrics["by_task"]["task-a"]["task_success"] == 100.0
    assert result.metrics["by_task"]["task-b"]["task_success"] == 0.0


def test_checkpoint_and_policy_factories_are_composable(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    prepared = prepare_dataset(manifest_path, tmp_path / "prepared")
    episode = load_episode_objects(prepared.split_paths["train"])[0]
    checkpoint_path = tmp_path / "checkpoint.txt"
    checkpoint_path.write_text("synthetic-checkpoint-v1\n", encoding="utf-8")
    calls = []

    with pytest.raises(ValueError, match="task-specific outcome_scorer"):
        evaluate_episodes(
            [episode],
            tmp_path / "checkpoint-evaluation",
            checkpoint_path=checkpoint_path,
            manager_factory=lambda item: StateManager(item),
        )

    def checkpoint_factory(path):
        calls.append(("checkpoint", path))
        return path.read_text(encoding="utf-8").strip()

    def policy_factory(checkpoint):
        calls.append(("policy", checkpoint))
        return HeuristicPolicy()

    def checkpoint_manager_factory(checkpoint):
        calls.append(("manager", checkpoint))
        return lambda item: StateManager(
            item,
            memory_capacity=4,
            action_budget=6,
            turn_cap=8,
            retrieval_limit=2,
        )

    engine = build_inference_engine(
        checkpoint_path=checkpoint_path,
        checkpoint_factory=checkpoint_factory,
        policy_factory=policy_factory,
        checkpoint_manager_factory=checkpoint_manager_factory,
    )
    result = engine.run(episode, seed=5)
    assert result.success
    assert calls == [
        ("checkpoint", checkpoint_path.resolve()),
        ("policy", "synthetic-checkpoint-v1"),
        ("manager", "synthetic-checkpoint-v1"),
    ]

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_inference_engine(
            policy_factory=lambda checkpoint: HeuristicPolicy(),
            manager_factory=lambda item: StateManager(item),
            checkpoint_manager_factory=lambda checkpoint: lambda item: StateManager(item),
        )


def test_evaluation_rejects_checkpoint_mutated_during_policy_load(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    prepared = prepare_dataset(manifest_path, tmp_path / "prepared")
    episode = load_episode_objects(prepared.split_paths["train"])[0]
    checkpoint_path = tmp_path / "checkpoint.txt"
    checkpoint_path.write_text("checkpoint-a\n", encoding="utf-8")

    def mutating_checkpoint_factory(path):
        path.write_text("checkpoint-b\n", encoding="utf-8")
        return path

    def policy_factory(checkpoint):
        assert checkpoint == checkpoint_path.resolve()
        return HeuristicPolicy()

    with pytest.raises(DataValidationError, match="checkpoint changed"):
        evaluate_episodes(
            [episode],
            tmp_path / "mutated-checkpoint-evaluation",
            checkpoint_path=checkpoint_path,
            checkpoint_factory=mutating_checkpoint_factory,
            policy_factory=policy_factory,
            manager_factory=lambda item: StateManager(item),
            policy_id="mutation-test-v1",
            allow_synthetic_outcome_scorer=True,
        )


def test_error_results_are_durable_and_resume_without_policy_reload(tmp_path):
    manifest_path = _write_synthetic_sources(tmp_path)
    prepared = prepare_dataset(manifest_path, tmp_path / "prepared")
    episodes = load_episode_objects(prepared.split_paths["train"])

    class FailingPolicy:
        def sample(self, observation, *, rng, n=1, greedy=False):
            del observation, rng, n, greedy
            raise RuntimeError("synthetic backend failure")

    policy_loads = 0

    def failing_factory(checkpoint):
        nonlocal policy_loads
        assert checkpoint is None
        policy_loads += 1
        return FailingPolicy()

    failing_factory.__vapa_content_id__ = "2" * 64

    failed = evaluate_episodes(
        episodes,
        tmp_path / "failed-evaluation",
        max_instances=1,
        policy_factory=failing_factory,
        policy_id="always-fails-v1",
        continue_on_error=True,
    )
    assert failed.metrics["errors"] == 1
    assert failed.metrics["task_success"] == 0.0
    record = load_jsonl(failed.predictions_path)[0]
    assert record["status"] == "error"
    assert record["reference_evidence_covered"] is False

    resumed = evaluate_episodes(
        episodes,
        tmp_path / "failed-evaluation",
        max_instances=1,
        policy_factory=failing_factory,
        policy_id="always-fails-v1",
        continue_on_error=True,
    )
    assert resumed.resumed == 1
    assert resumed.newly_completed == 0
    assert policy_loads == 1
