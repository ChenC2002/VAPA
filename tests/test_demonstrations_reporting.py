from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vapa.artifacts import strict_json_loads
from vapa.data import load_episode_objects, prepare_dataset
from vapa.data.adapters import LatestFieldEventAdapter
from vapa.data.io import DataValidationError
from vapa.evaluation.analysis import analyze_binary_results
from vapa.evaluation.reporting import (
    ExperimentMetricRecord,
    ExperimentReportError,
    _factorial_report,
    _horizon_report,
    _profile_report,
    _resampled_task_macro,
    analyze_experiments,
    load_experiment_records,
)
from vapa.evaluation.statistics import _t_critical, paired_t_test
from vapa.training.demonstrations import (
    DEMONSTRATION_MANIFEST_SCHEMA_VERSION,
    PUBLIC_REFERENCE_PROGRAM_ID,
    generate_sft_demonstrations,
)
from vapa.training.sft_train import load_sft_demonstrations

ROOT = Path(__file__).resolve().parents[1]


def test_public_reference_generator_feeds_sft_loader_and_is_deterministic(tmp_path: Path) -> None:
    first = generate_sft_demonstrations(
        ROOT / "examples" / "tiny_episode.json",
        tmp_path / "first.jsonl",
        seed=7,
        content_kind="public",
        repository_root=ROOT,
    )
    second = generate_sft_demonstrations(
        ROOT / "examples" / "tiny_episode.json",
        tmp_path / "second.jsonl",
        seed=7,
        content_kind="public",
        repository_root=ROOT,
    )
    assert first.output_sha256 == second.output_sha256
    assert first.examples == 3
    examples = load_sft_demonstrations(first.output_path)
    assert [example.action_text.split("(", 1)[0] for example in examples] == [
        "QueryField",
        "UpdateMemory",
        "Answer",
    ]
    assert all(example.group_id.startswith(PUBLIC_REFERENCE_PROGRAM_ID) for example in examples)
    rendered = "\n".join(message["content"] for example in examples for message in example.messages)
    assert "e-future" not in rendered
    manifest = strict_json_loads(first.manifest_path.read_bytes())
    assert manifest["schema_version"] == DEMONSTRATION_MANIFEST_SCHEMA_VERSION
    assert manifest["paper_exact"] is False


def test_public_reference_generator_refuses_public_patient_output(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    source = ROOT / "examples" / "tiny_episode.json"
    with pytest.raises(ValueError, match="refusing to write"):
        generate_sft_demonstrations(
            source,
            repository / "docs" / "patient.jsonl",
            content_kind="credentialed",
            repository_root=repository,
        )


def test_latest_field_adapter_constructs_tasks_directly_from_events(tmp_path: Path) -> None:
    prepared = prepare_dataset(
        ROOT / "examples" / "tiny_latest_field_manifest.json",
        tmp_path / "prepared",
    )
    episodes = load_episode_objects(prepared.split_paths["train"])
    assert len(episodes) == 3
    assert all(
        episode.task.metadata["task_builder"] == "latest_field_events" for episode in episodes
    )
    by_patient = {episode.task.patient_id: episode for episode in episodes}
    assert by_patient["p1"].gold_answer == 9.9
    assert by_patient["p1"].reference_evidence == ("e-p1-future",)
    assert all(
        event.timestamp <= episode.task.cutoff for episode in episodes for event in episode.events
    )


def test_paired_t_test_returns_two_sided_probability() -> None:
    result = paired_t_test([1.0, 2.0, 3.0])
    assert result.estimate == 2.0
    assert 0.07 < result.p_value < 0.08


def test_experiment_report_runs_all_sections_and_holm(tmp_path: Path) -> None:
    records = ROOT / "examples" / "tiny_experiment_metrics.jsonl"
    assert len(load_experiment_records(records)) == 24
    result = analyze_experiments(
        ROOT / "examples" / "tiny_experiment_report_manifest.json",
        records,
        tmp_path / "report.json",
        content_kind="public",
        repository_root=ROOT,
    )
    contrasts = result["factorial"]["success"]["contrasts"]
    assert contrasts["joint"]["estimate"] == 4.0
    assert contrasts["interaction"]["estimate"] == 1.0
    horizon = result["horizon"]["success"]
    assert horizon["axis"] == "decision_depth"
    assert horizon["systems"]["sys_a1"]["slope_per_step"]["estimate"] < 0
    assert horizon["systems"]["sys_a4"]["slope_per_step"]["estimate"] > 0
    assert result["profile"]["short"]["cells"]["sys_a4:success"]["estimate"] == 1.0
    assert result["holm_family"]
    assert json.loads((tmp_path / "report.json").read_text()) == result


def test_reference_generator_preserves_inputs_and_existing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "episode.json"
    original = (ROOT / "examples" / "tiny_episode.json").read_bytes()
    source.write_bytes(original)
    with pytest.raises(ValueError, match="aliases an input"):
        generate_sft_demonstrations(source, source, overwrite=True, content_kind="public")
    assert source.read_bytes() == original
    output = tmp_path / "sft.jsonl"
    generated = generate_sft_demonstrations(source, output, content_kind="public")
    prior = output.read_bytes(), generated.manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        generate_sft_demonstrations(source, output, content_kind="public")
    assert (output.read_bytes(), generated.manifest_path.read_bytes()) == prior
    generate_sft_demonstrations(source, output, overwrite=True, content_kind="public")
    assert (output.read_bytes(), generated.manifest_path.read_bytes()) == prior


def test_reference_generator_validates_before_publishing(tmp_path: Path, monkeypatch) -> None:
    import vapa.training.demonstrations as generator

    def reject(_payload: str) -> None:
        raise ValueError("invalid generated examples")

    monkeypatch.setattr(generator, "parse_sft_demonstrations", reject)
    with pytest.raises(ValueError, match="invalid generated examples"):
        generate_sft_demonstrations(
            ROOT / "examples" / "tiny_episode.json", tmp_path / "sft.jsonl", content_kind="public"
        )
    assert list(tmp_path.iterdir()) == []


def test_latest_field_ties_match_query_pointer_and_reject_conflicts() -> None:
    episode = load_episode_objects(ROOT / "examples" / "tiny_episode.json")[0]
    event = next(item for item in episode.events if item.field == "hba1c")
    config = LatestFieldEventAdapter._parse_config({"fields": ["hba1c"]})
    first, second = replace(event, pointer="a"), replace(event, pointer="b")
    built = LatestFieldEventAdapter._episodes([second, first], **config)
    assert built[0].reference_evidence == ("a",)
    for invalid in (replace(second, value=100), replace(second, unit="different")):
        with pytest.raises(DataValidationError, match="ambiguous latest value"):
            LatestFieldEventAdapter._episodes([first, invalid], **config)
    with pytest.raises(DataValidationError, match="is null"):
        LatestFieldEventAdapter._episodes([replace(first, value=None)], **config)


@pytest.mark.parametrize("section_name", ["factorial", "horizon", "profile"])
def test_report_rejects_unpaired_patient_panels(section_name: str) -> None:
    manifest = json.loads((ROOT / "examples" / "tiny_experiment_report_manifest.json").read_text())
    records = list(load_experiment_records(ROOT / "examples" / "tiny_experiment_metrics.jsonl"))
    index = next(
        i
        for i, row in enumerate(records)
        if row.system_id == "sys_a4"
        and (
            row.profile is not None
            if section_name == "profile"
            else row.decision_depth is not None
            if section_name == "horizon"
            else row.profile is None and row.decision_depth is None
        )
    )
    records[index] = replace(records[index], patient_id="unmatched")
    function = {
        "factorial": _factorial_report,
        "horizon": _horizon_report,
        "profile": _profile_report,
    }
    with pytest.raises(ExperimentReportError, match="patient panels"):
        function[section_name](records, manifest[section_name])


def test_profile_bootstrap_keeps_frozen_tasks_and_counts_undefined_draws() -> None:
    assert _resampled_task_macro({"a": {"p1": 0.0}, "b": {"p2": 1.0}}, {"p1": 2}) is None
    records = [
        ExperimentMetricRecord(system, seed, task, patient, "score", value, None, "short")
        for system in ("left", "right")
        for seed in (0, 1)
        for task, patient, value in (("a", "p1", 0.0), ("b", "p2", 1.0))
    ]
    section = {
        "systems": ["left", "right"],
        "metrics": ["score"],
        "profiles": ["short"],
        "bootstrap_draws": 200,
        "bootstrap_seed": 7,
        "contrasts": [{"name": "difference", "left": "left", "right": "right"}],
    }
    result = _profile_report(records, section)["short"]
    assert 0 < result["undefined_bootstrap_draws"] < 200
    assert result["valid_bootstrap_draws"] + result["undefined_bootstrap_draws"] == 200
    assert result["cells"]["left:score"]["lower"] == 0.5
    assert result["cells"]["left:score"]["upper"] == 0.5
    assert result["contrasts"]["difference:score"]["estimate"] == 0.0
    # Unrequested systems must not affect the bootstrap population or random stream.
    extra = replace(records[0], system_id="ignored", patient_id="other")
    assert _profile_report([*records, extra], section)["short"] == result


def test_profile_order_does_not_change_bootstrap_intervals() -> None:
    manifest = json.loads((ROOT / "examples" / "tiny_experiment_report_manifest.json").read_text())
    records = load_experiment_records(ROOT / "examples" / "tiny_experiment_metrics.jsonl")
    section = manifest["profile"]
    expected = _profile_report(records, section)
    assert (
        _profile_report(records, {**section, "profiles": list(reversed(section["profiles"]))})
        == expected
    )


def test_student_t_intervals_use_the_same_distribution_as_p_values() -> None:
    import math

    assert _t_critical(1) == pytest.approx(1 / math.tan(math.pi * 0.025), rel=1e-12)
    assert _t_critical(2) == pytest.approx(math.sqrt(2 * 0.95**2 / (1 - 0.95**2)), rel=1e-12)
    assert _t_critical(31) == pytest.approx(2.0395134464, rel=1e-9)


@pytest.mark.parametrize("kind", ["experiment", "binary"])
def test_analysis_refuses_to_overwrite_or_alias_inputs(tmp_path: Path, kind: str) -> None:
    function, manifest_name, records_name = (
        (
            analyze_experiments,
            "tiny_experiment_report_manifest.json",
            "tiny_experiment_metrics.jsonl",
        )
        if kind == "experiment"
        else (
            analyze_binary_results,
            "tiny_analysis_manifest.json",
            "tiny_binary_predictions.jsonl",
        )
    )
    manifest = ROOT / "examples" / manifest_name
    records = ROOT / "examples" / records_name
    destination = tmp_path / "report.json"
    destination.write_text("existing")
    with pytest.raises(FileExistsError):
        function(manifest, records, destination, content_kind="public")
    assert destination.read_text() == "existing"
    with pytest.raises(ValueError, match="aliases an input"):
        function(manifest, records, records, overwrite=True, content_kind="public")


def test_decision_depth_allows_different_cohorts_but_pairs_systems_at_each_depth() -> None:
    manifest = json.loads((ROOT / "examples/tiny_experiment_report_manifest.json").read_text())
    records = load_experiment_records(ROOT / "examples/tiny_experiment_metrics.jsonl")
    records = [
        replace(row, patient_id=f"{row.patient_id}:depth{row.decision_depth}") for row in records
    ]
    report, _ = _horizon_report(records, manifest["horizon"])
    assert report["success"]["axis"] == "decision_depth"
    assert report["success"]["contrasts"]["vapa_minus_grpo"]["estimate"] == pytest.approx(0.8)
    assert (
        _factorial_report(records, manifest["factorial"])[0]["success"]["contrasts"]["joint"][
            "estimate"
        ]
        == 4.0
    )


def test_legacy_day_axis_remains_separate_from_decision_depth() -> None:
    manifest = json.loads((ROOT / "examples/tiny_experiment_report_manifest.json").read_text())
    records = load_experiment_records(ROOT / "examples/tiny_experiment_metrics.jsonl")
    converted = [
        replace(row, horizon_days={1: 30, 6: 90}.get(row.decision_depth), decision_depth=None)
        for row in records
    ]
    section = {key: value for key, value in manifest["horizon"].items() if key != "decision_depths"}
    report, _ = _horizon_report(converted, {**section, "horizons_days": [30, 90]})
    assert report["success"]["axis"] == "horizon_days"
    assert report["success"]["contrasts"]["vapa_minus_grpo"]["estimate"] == pytest.approx(4 / 60)


@pytest.mark.parametrize("depth", [True, 0, -1, 1.5])
def test_report_loader_rejects_invalid_depth(tmp_path: Path, depth: object) -> None:
    row = json.loads((ROOT / "examples/tiny_experiment_metrics.jsonl").read_text().splitlines()[0])
    row["decision_depth"] = depth
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ExperimentReportError, match="positive or null"):
        load_experiment_records(path)


def test_report_loader_refuses_mixed_axes(tmp_path: Path) -> None:
    row = json.loads((ROOT / "examples/tiny_experiment_metrics.jsonl").read_text().splitlines()[0])
    row.update(decision_depth=1, horizon_days=30)
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ExperimentReportError, match="cannot mix"):
        load_experiment_records(path)


def test_report_loader_preserves_unicode_in_record_identifiers(tmp_path: Path) -> None:
    row = json.loads((ROOT / "examples/tiny_experiment_metrics.jsonl").read_text().splitlines()[0])
    row["patient_id"] = "first\u0085second\u2028third\u2029last"
    path = tmp_path / "unicode.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_experiment_records(path)[0].patient_id == row["patient_id"]


def test_report_only_adjusts_declared_confirmatory_tests(tmp_path: Path) -> None:
    result = analyze_experiments(
        ROOT / "examples/tiny_experiment_report_manifest.json",
        ROOT / "examples/tiny_experiment_metrics.jsonl",
        tmp_path / "report.json",
        content_kind="public",
    )
    assert set(result["holm_family"]) == {
        "factorial:success:interaction",
        "horizon:success:contrast:vapa_minus_grpo",
    }
    contrasts = result["factorial"]["success"]["contrasts"]
    assert contrasts["joint"]["holm_p_value"] is None
    assert contrasts["interaction"]["holm_p_value"] == contrasts["interaction"]["p_value"]


@pytest.mark.parametrize("family", [["typo"], ["factorial:success:joint"] * 2, "not-a-list"])
def test_report_rejects_invalid_confirmatory_family(tmp_path: Path, family: object) -> None:
    manifest = json.loads((ROOT / "examples/tiny_experiment_report_manifest.json").read_text())
    manifest["confirmatory_tests"] = family
    path, output = tmp_path / "manifest.json", tmp_path / "result.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ExperimentReportError):
        analyze_experiments(
            path, ROOT / "examples/tiny_experiment_metrics.jsonl", output, content_kind="public"
        )
    assert not output.exists()
