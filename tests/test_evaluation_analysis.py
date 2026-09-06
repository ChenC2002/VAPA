from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vapa.artifacts import canonical_json_dumps, fingerprint_file
from vapa.evaluation.analysis import (
    ANALYSIS_WARNING,
    AnalysisInputError,
    AnalysisManifest,
    BinaryPrediction,
    CohortInstance,
    MethodContrast,
    TrajectoryCandidate,
    analyze_binary_results,
    binary_task_metrics,
    frozen_task_cap,
    joint_patient_cluster_bootstrap,
    main,
    oracle_best_of_n,
    paired_seed_contrasts,
    paired_seed_t_summary,
    self_consistency,
)


def _panel(*, seeds: tuple[int | None, ...] = (None,)) -> list[BinaryPrediction]:
    output: list[BinaryPrediction] = []
    for system_id in ("good", "bad"):
        for seed in seeds:
            for task_index, task_id in enumerate(("task-a", "task-b")):
                for patient_index in range(3):
                    for label in (0, 1):
                        score = (
                            0.85 - 0.02 * patient_index if label else 0.15 + 0.02 * patient_index
                        )
                        if system_id == "bad":
                            score = 1.0 - score
                        if seed is not None:
                            score += (seed - 1) * 0.005 * (1 if label else -1)
                        valid = not (
                            system_id == "good"
                            and seed in {None, 1}
                            and task_index == 0
                            and patient_index == 0
                            and label == 0
                        )
                        output.append(
                            BinaryPrediction(
                                system_id=system_id,
                                seed=seed,
                                task_id=task_id,
                                patient_id=f"patient-{patient_index}",
                                instance_id=(f"{task_id}-patient-{patient_index}-label-{label}"),
                                label=label,
                                score=score if valid else None,
                                format_valid=valid,
                            )
                        )
    return output


def _record(prediction: BinaryPrediction) -> dict[str, object]:
    return {
        "schema_version": prediction.schema_version,
        "system_id": prediction.system_id,
        "seed": prediction.seed,
        "task_id": prediction.task_id,
        "patient_id": prediction.patient_id,
        "instance_id": prediction.instance_id,
        "label": prediction.label,
        "score": prediction.score,
        "format_valid": prediction.format_valid,
    }


def test_frozen_cap_retains_every_patient_before_repeated_instances():
    instances = [
        CohortInstance("large-task", f"patient-{patient}", f"instance-{patient}-{repeat}")
        for patient in range(3)
        for repeat in range(3)
    ]
    instances.extend(
        CohortInstance("small-task", f"small-{index}", f"small-instance-{index}")
        for index in range(2)
    )
    selected = frozen_task_cap(instances, cap=4, seed=71)
    assert selected == frozen_task_cap(reversed(instances), cap=4, seed=71)
    large = [item for item in selected if item.task_id == "large-task"]
    small = [item for item in selected if item.task_id == "small-task"]
    assert len(large) == 4
    assert {item.patient_id for item in large} == {
        "patient-0",
        "patient-1",
        "patient-2",
    }
    assert len(small) == 2
    with pytest.raises(AnalysisInputError, match="retain every patient"):
        frozen_task_cap(instances, cap=2, seed=71)

    defaults = AnalysisManifest.from_dict(
        {
            "schema_version": 1,
            "analysis_id": "defaults",
            "paper_exact": False,
            "records_sha256": "0" * 64,
        }
    )
    assert defaults.task_cap == 2_500
    assert defaults.bootstrap_replicates == 10_000


def test_binary_metrics_use_same_system_task_median_and_equal_task_macro():
    metrics = binary_task_metrics(_panel())
    good = metrics["systems"]["good"]["runs"]["fixed"]  # type: ignore[index]
    bad = metrics["systems"]["bad"]["runs"]["fixed"]  # type: ignore[index]
    task_a = good["tasks"]["task-a"]
    assert task_a["format_failures"] == 1
    assert task_a["imputation_score"] == pytest.approx(0.81)
    assert good["macro_auprc"] == pytest.approx(
        sum(task["auprc"] for task in good["tasks"].values()) / 2
    )
    assert good["macro_auroc"] > bad["macro_auroc"]

    incomplete = _panel()[:-1]
    with pytest.raises(AnalysisInputError, match="incomplete"):
        binary_task_metrics(incomplete)


def test_joint_patient_bootstrap_is_paired_joint_and_deterministic():
    contrast = MethodContrast("good-minus-bad", "good", "bad")
    first = joint_patient_cluster_bootstrap(
        _panel(),
        contrasts=(contrast,),
        replicates=80,
        seed=17,
    )
    second = joint_patient_cluster_bootstrap(
        reversed(_panel()),
        contrasts=(contrast,),
        replicates=80,
        seed=17,
    )
    assert first == second
    assert first["patient_union_size"] == 3
    paired = first["paired_contrasts"]["good-minus-bad"]  # type: ignore[index]
    assert paired["metrics"]["macro_auroc"]["estimate"] > 0
    assert paired["metrics"]["macro_auroc"]["replicates"] == 80


def test_paired_seed_interval_uses_within_seed_differences():
    summary = paired_seed_t_summary(
        {0: 72.0, 1: 74.0, 2: 76.0},
        {0: 70.0, 1: 72.0, 2: 74.0},
    )
    assert summary["estimate"] == 2.0
    assert summary["lower"] == summary["upper"] == 2.0
    assert summary["within_seed_differences"] == [2.0, 2.0, 2.0]
    with pytest.raises(AnalysisInputError, match="same two or more"):
        paired_seed_t_summary({0: 1.0, 1: 2.0}, {0: 1.0, 2: 2.0})


def test_seeded_panel_reports_across_seed_dispersion_and_paired_contrast():
    metrics = binary_task_metrics(_panel(seeds=(0, 1, 2)))
    aggregate = metrics["systems"]["good"]["seed_aggregate"]  # type: ignore[index]
    assert aggregate["seeds"] == 3
    assert aggregate["macro_auroc_standard_deviation"] >= 0
    contrasts = paired_seed_contrasts(
        metrics,
        (MethodContrast("good-minus-bad", "good", "bad"),),
    )
    interval = contrasts["good-minus-bad"]["metrics"]["macro_auroc"]  # type: ignore[index]
    assert interval["n"] == 3
    assert interval["estimate"] > 0

    mixed_panel = [
        item
        for item in (*_panel(seeds=(0, 1, 2)), *_panel())
        if (item.system_id == "good" and item.seed is not None)
        or (item.system_id == "bad" and item.seed is None)
    ]
    mixed_metrics = binary_task_metrics(mixed_panel)
    mixed = paired_seed_contrasts(
        mixed_metrics,
        (MethodContrast("trained-minus-fixed", "good", "bad"),),
    )
    assert mixed["trained-minus-fixed"]["metrics"]["macro_auroc"]["n"] == 3


def test_self_consistency_and_oracle_controls_have_distinct_information_rules():
    candidates = [
        TrajectoryCandidate("a-1", "A", True, -1.0, 1, ("a1",)),
        TrajectoryCandidate("a-2", "A", True, -1.1, 1, ("a2",)),
        TrajectoryCandidate("a-3", "A", True, -1.2, 1, ("a3",)),
        TrajectoryCandidate("b-1", "B", True, -0.6, 2, ("b1",)),
        TrajectoryCandidate("b-2", "B", True, -0.2, 1, ("b2",)),
        TrajectoryCandidate("b-3", "B", True, -1.0, 2, ("b3",)),
        TrajectoryCandidate("c-1", "C", True, -0.1, 1, ("c1",)),
        TrajectoryCandidate("c-2", "C", True, -0.1, 1, ("c2",)),
    ]
    selected = self_consistency(candidates)
    assert selected["answer"] == "B"
    assert selected["plurality_count"] == 3
    assert selected["supporting_trajectory_id"] == "b-2"
    assert selected["evidence_chain"] == ["b2"]
    assert selected["deployable"] is True

    binary = [
        replace(candidate, binary_score=index / 10) for index, candidate in enumerate(candidates)
    ]
    averaged = self_consistency(binary, binary=True)
    assert averaged["binary_score"] == pytest.approx(0.35)

    oracle = oracle_best_of_n(
        candidates,
        {
            "accuracy": {
                candidate.trajectory_id: candidate.trajectory_id == "a-1"
                for candidate in candidates
            },
            "grounding": {
                candidate.trajectory_id: candidate.trajectory_id == "b-1"
                for candidate in candidates
            },
        },
    )
    assert oracle["deployable"] is False
    assert oracle["metrics"]["accuracy"]["trajectory_id"] == "a-1"
    assert oracle["metrics"]["grounding"]["trajectory_id"] == "b-1"


def test_cli_emits_canonical_checksummed_analysis_and_rejects_schema_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    records_path = tmp_path / "predictions.jsonl"
    records_path.write_text(
        "".join(canonical_json_dumps(_record(item)) + "\n" for item in _panel()),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "analysis.json"
    manifest = {
        "schema_version": 1,
        "analysis_id": "dependency-free-test",
        "paper_exact": False,
        "records_sha256": fingerprint_file(records_path).sha256,
        "seed": 23,
        "task_cap": 6,
        "bootstrap_replicates": 40,
        "contrasts": [
            {
                "name": "good-minus-bad",
                "left_system": "good",
                "right_system": "bad",
            }
        ],
    }
    manifest_path.write_text(canonical_json_dumps(manifest) + "\n", encoding="utf-8")
    output_path = tmp_path / "analysis-output.json"
    assert main([str(manifest_path), str(records_path), str(output_path)]) == 0
    output_text = output_path.read_text(encoding="utf-8")
    output = json.loads(output_text)
    assert output_text == canonical_json_dumps(output) + "\n"
    assert output["warning"] == ANALYSIS_WARNING
    assert output["paper_exact"] is False
    assert output["provenance"]["records"]["sha256"] == manifest["records_sha256"]
    assert len(output["provenance"]["implementation_sha256"]) == 64
    assert output["cohort"]["tasks"]["task-a"]["instances_after"] == 6
    assert canonical_json_dumps(output) in capsys.readouterr().out

    bad_manifest = dict(manifest, records_sha256="0" * 64)
    manifest_path.write_text(canonical_json_dumps(bad_manifest) + "\n", encoding="utf-8")
    with pytest.raises(AnalysisInputError, match="records_sha256"):
        analyze_binary_results(manifest_path, records_path, tmp_path / "bad.json")

    bad_record = _record(_panel()[0])
    bad_record["unexpected"] = True
    with pytest.raises(AnalysisInputError, match="unexpected"):
        BinaryPrediction.from_dict(bad_record)
