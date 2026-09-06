"""Public synthetic fixture and end-to-end method smoke run."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vapa.artifacts import (
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    strict_json_loads,
    validate_output_paths,
)
from vapa.config import ExperimentConfig, ReplayConfig
from vapa.environment.state_manager import StateManager
from vapa.policies.heuristic import HeuristicPolicy
from vapa.rollouts import RolloutRunner
from vapa.schemas import Domain, Episode, RecordEvent, TaskSpec, TimeWindow
from vapa.training.trainer import InstanceBatchBuilder
from vapa.verifiers import VerifierCatalog

DEMO_INPUTS = (
    "tiny_dataset_manifest.json",
    "tiny_latest_field_manifest.json",
    "tiny_events.csv",
    "tiny_tasks.jsonl",
    "tiny_expected_metrics.json",
    "tiny_analysis_manifest.json",
    "tiny_binary_predictions.jsonl",
    "tiny_experiment_report_manifest.json",
    "tiny_experiment_metrics.jsonl",
)
DEMO_RESULT_SCHEMA = "vapa-demo-results-v1"


def tiny_episode() -> Episode:
    """Synthetic values only; no patient data or clinical recommendation."""

    cutoff = datetime(2026, 1, 10, tzinfo=UTC)
    task = TaskSpec(
        instance_id="synthetic-latest-hba1c",
        patient_id="synthetic-patient-001",
        instruction="What is the latest pre-cutoff HbA1c in the last 90 days?",
        cutoff=cutoff,
        family="latest_laboratory_value",
        requested_fields=("hba1c",),
        requested_window=TimeWindow.parse("last 90d"),
        answer_type="number",
        metadata={
            "history_quartile": 1,
            "suite": "retrieval",
            "task_type": "latest_laboratory_value",
        },
    )
    events = (
        RecordEvent(
            "e#1001",
            task.patient_id,
            datetime(2025, 6, 10, tzinfo=UTC),
            Domain.LAB,
            "hba1c",
            7.9,
            "%",
            source="synthetic",
        ),
        RecordEvent(
            "e#1002",
            task.patient_id,
            datetime(2025, 12, 20, tzinfo=UTC),
            Domain.LAB,
            "hba1c",
            6.4,
            "%",
            source="synthetic",
        ),
        # The cutoff guard must make this tempting future event invisible.
        RecordEvent(
            "e#future",
            task.patient_id,
            datetime(2026, 1, 11, tzinfo=UTC),
            Domain.LAB,
            "hba1c",
            9.9,
            "%",
            source="synthetic",
        ),
    )
    return Episode(task, events, gold_answer=6.4, reference_evidence=("e#1002",))


def run_demo(seed: int = 7, *, compact: bool = False) -> dict[str, Any]:
    episode = tiny_episode()
    config = ExperimentConfig(seed=seed)
    if compact:
        config = replace(
            config, replay=ReplayConfig(base_group_size=4, max_fork_states=2, siblings_per_fork=2)
        )

    def manager_factory(item: Episode) -> StateManager:
        environment = config.environment
        return StateManager(
            item,
            memory_capacity=environment.memory_capacity,
            action_budget=environment.action_budget,
            turn_cap=environment.turn_cap,
        )

    runner = RolloutRunner(manager_factory)
    builder = InstanceBatchBuilder(config, runner, VerifierCatalog.demo_default())
    batch = builder.build(episode, HeuristicPolicy(), seed=seed)
    tiers = Counter(group.tier.value for group in batch.groups)
    return {
        "status": "ok",
        "fixture": "synthetic",
        "paper_exact": False,
        "verifier_catalog": "demo-v1-not-paper-exact",
        "base_rollouts": len(batch.base_rollouts),
        "branch_rollouts": len(batch.branch_rollouts),
        "successes": sum(rollout.outcome_reward > 0 for rollout in batch.base_rollouts),
        "sampled_tokens": batch.sampled_tokens,
        "trainable_tokens": batch.trainable_tokens,
        "group_tiers": dict(sorted(tiers.items())),
        "local_scale": None
        if batch.advantage_summary is None
        else batch.advantage_summary.local_scale,
        "predictions": [rollout.prediction for rollout in batch.base_rollouts],
        "future_event_visible": any(
            "e#future" in turn.tool_return.evidence_pointers
            for rollout in batch.base_rollouts + batch.branch_rollouts
            for turn in rollout.turns
        ),
    }


def demo_result_log(result: dict[str, Any]) -> str:
    """Derive a portable result-event log, not a fabricated optimizer trace."""
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != DEMO_RESULT_SCHEMA
        or result.get("kind") != "synthetic_run"
        or result.get("paper_exact") is not False
    ):
        raise ValueError("not a VAPA executable demo result")
    if set(result) != {
        "schema_version",
        "kind",
        "paper_exact",
        "scope",
        "seed",
        "compact",
        "implementation_sha256",
        "inputs",
        "records",
    }:
        raise ValueError("demo result has missing or unknown fields")
    records = result.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("demo result must contain records")
    if any(not isinstance(row, dict) or set(row) != {"id", "kind", "metrics"} for row in records):
        raise ValueError("demo records require exactly id, kind, and metrics")
    ids = [row["id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate demo result IDs")
    if ids != [
        "method",
        "preparation",
        "demonstrations",
        "sft",
        "evaluation",
        "binary_analysis",
        "experiment_report",
    ]:
        raise ValueError("demo result stages are incomplete or out of order")
    events = [
        {
            "event": "manifest",
            "sequence": 0,
            **{key: value for key, value in result.items() if key != "records"},
            "result_sha256": artifact_fingerprint(result),
            "record_count": len(records),
        }
    ]
    events.extend(
        {"event": "result", "sequence": i, "record": row} for i, row in enumerate(records, 1)
    )
    events.append({"event": "complete", "sequence": len(records) + 1, "record_count": len(records)})
    return "".join(canonical_json_dumps(event) + "\n" for event in events)


def run_demo_suite(
    examples_directory: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 7,
    compact: bool = True,
) -> dict[str, Any]:
    """Run public fixtures and retain outputs; publish a summary only after all checks pass.

    The output directory must be new. No weights are loaded and no optimizer step is
    performed. Analysis-fixture values and executable policy results are labeled separately.
    """
    from vapa.data import load_episode_objects, load_json, prepare_dataset
    from vapa.evaluation.analysis import analyze_binary_results
    from vapa.evaluation.reporting import analyze_experiments
    from vapa.evaluation.runner import evaluate_episodes
    from vapa.provenance import package_code_fingerprint
    from vapa.training.demonstrations import generate_sft_demonstrations
    from vapa.training.sft_train import SFTTrainConfig, validate_sft_run

    if type(seed) is not int or seed < 0 or not isinstance(compact, bool):
        raise ValueError("demo seed must be nonnegative and compact must be boolean")
    examples = Path(examples_directory).resolve()
    destination = Path(output_directory).resolve()
    inputs = {name: fingerprint_file(examples / name).to_dict() for name in DEMO_INPUTS}
    implementation = package_code_fingerprint()
    for name in ("tiny_dataset_manifest.json", "tiny_latest_field_manifest.json"):
        manifest = load_json(examples / name)
        if not isinstance(manifest, dict) or manifest.get("content_kind") != "public":
            raise ValueError("the demo suite accepts only explicitly public fixtures")
    destination.mkdir(parents=True, exist_ok=False)
    method = run_demo(seed, compact=compact)
    if method["future_event_visible"] or method["successes"] != method["base_rollouts"]:
        raise RuntimeError("method demo failed its success/cutoff checks")
    prepared = prepare_dataset(examples / "tiny_dataset_manifest.json", destination / "prepared")
    episodes = load_episode_objects(prepared.split_paths["train"])
    if any(
        event.timestamp > episode.task.cutoff for episode in episodes for event in episode.events
    ):
        raise RuntimeError("prepared demo episodes expose post-cutoff evidence")
    constructed = prepare_dataset(
        examples / "tiny_latest_field_manifest.json", destination / "constructed"
    )
    demonstrations = generate_sft_demonstrations(
        constructed.split_paths["train"],
        destination / "demonstrations.jsonl",
        seed=seed,
        content_kind="public",
    )
    sft = validate_sft_run(SFTTrainConfig(data_path=demonstrations.output_path, seed=seed))
    evaluation = evaluate_episodes(
        episodes,
        destination / "evaluation",
        seed=seed,
        policy_id="vapa-public-heuristic-v1",
        content_kind="public",
    )
    expected = load_json(examples / "tiny_expected_metrics.json")
    if (
        expected.get("fixture_id") != "vapa-public-tiny-v1"
        or evaluation.metrics != expected["metrics"]
    ):
        raise RuntimeError("demo evaluation differs from the frozen expected metrics")
    analysis = analyze_binary_results(
        examples / "tiny_analysis_manifest.json",
        examples / "tiny_binary_predictions.jsonl",
        destination / "analysis.json",
        content_kind="public",
    )
    systems = analysis["metrics"]["systems"]
    binary = {name: systems[name]["runs"]["fixed"]["macro_auroc"] for name in ("good", "bad")}
    if binary != {"good": 100.0, "bad": 0.0}:
        raise RuntimeError("binary analysis fixture failed")
    report = analyze_experiments(
        examples / "tiny_experiment_report_manifest.json",
        examples / "tiny_experiment_metrics.jsonl",
        destination / "experiment-report.json",
        content_kind="public",
    )
    if report["factorial"]["success"]["contrasts"]["interaction"]["estimate"] != 1.0:
        raise RuntimeError("factorial analysis fixture failed")
    if package_code_fingerprint() != implementation or inputs != {
        name: fingerprint_file(examples / name).to_dict() for name in DEMO_INPUTS
    }:
        raise RuntimeError("demo inputs or implementation changed during execution")
    result = {
        "schema_version": DEMO_RESULT_SCHEMA,
        "kind": "synthetic_run",
        "paper_exact": False,
        "scope": "Executable synthetic lifecycle; no model training or clinical benchmark results.",
        "seed": seed,
        "compact": compact,
        "implementation_sha256": implementation,
        "inputs": inputs,
        "records": [
            {"id": "method", "kind": "executed_method", "metrics": method},
            {
                "id": "preparation",
                "kind": "executed_preparation",
                "metrics": {
                    "episodes": prepared.total_episodes,
                    "patients": len({episode.task.patient_id for episode in episodes}),
                    "post_cutoff_events": 0,
                    "constructed_episodes": constructed.total_episodes,
                },
            },
            {
                "id": "demonstrations",
                "kind": "generated_and_validated",
                "metrics": {
                    "episodes": demonstrations.episodes,
                    "examples": demonstrations.examples,
                    "sha256": demonstrations.output_sha256,
                },
            },
            {"id": "sft", "kind": "dry_run_only", "metrics": sft.to_dict()},
            {
                "id": "evaluation",
                "kind": "executed_reference_policy",
                "metrics": dict(evaluation.metrics),
            },
            {
                "id": "binary_analysis",
                "kind": "synthetic_analysis_fixture",
                "metrics": {
                    "macro_auroc_percent": binary,
                },
            },
            {
                "id": "experiment_report",
                "kind": "synthetic_analysis_fixture",
                "metrics": {
                    key: report[key] for key in ("factorial", "horizon", "profile", "holm_family")
                },
            },
        ],
    }
    result = strict_json_loads(canonical_json_dumps(result))
    atomic_write_text(destination / "events.jsonl", demo_result_log(result), overwrite=False)
    atomic_write_text(
        destination / "results.json", canonical_json_dumps(result) + "\n", overwrite=False
    )
    return result


def publish_demo_results(run_directory: str | Path, repository_root: str | Path) -> None:
    """Explicitly update the two allowlisted public snapshots from a completed run."""
    source, root = Path(run_directory).resolve(), Path(repository_root).resolve()
    result = strict_json_loads((source / "results.json").read_bytes())
    log = demo_result_log(result)
    if (source / "events.jsonl").read_text(encoding="utf-8") != log:
        raise ValueError("demo summary and event log disagree")
    output, events = root / "results/demo_results.json", root / "logs/demo_results.jsonl"
    validate_output_paths(
        [output, events], inputs=[source / "results.json", source / "events.jsonl"], overwrite=True
    )
    # A summary/log mismatch is detectable if publication is interrupted between files.
    atomic_write_text(events, log)
    atomic_write_text(output, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
