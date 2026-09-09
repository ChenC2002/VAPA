from __future__ import annotations

import copy
import math
import runpy
from pathlib import Path

import pytest

from vapa.artifacts import artifact_fingerprint, guard_artifact_write_path
from vapa.data.io import load_json, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
PAPER = runpy.run_path(str(ROOT / "scripts/export_paper_results.py"))
TRAINING = runpy.run_path(str(ROOT / "scripts/train_tiny.py"))


def test_paper_export_matches_embedded_source_rows_and_log() -> None:
    result = load_json(ROOT / "results/paper_results.json")
    assert result["record_count"] == 100
    assert (ROOT / "logs/paper_results.jsonl").read_text() == PAPER["result_log"](result)
    main = [row for row in result["records"] if row["table"] == "tab:main"]
    assert [metric["value"] for metric in main[-1]["metrics"]] == [
        82.63,
        78.03,
        42.67,
        36.08,
        41.86,
        71.84,
    ]
    assert main[2]["dimensions"]["n_seeds"] is None  # SFT is a fixed system.
    assert main[13]["dimensions"]["n_seeds"] == 3
    assert main[12]["dimensions"]["label"] == "Tree-GRPO + process"
    assert main[12]["metrics"][0]["value"] == 82.18
    assert main[2]["metrics"][3]["value"] == 19.89
    assert main[10]["metrics"][0]["value"] == 81.53
    assert main[-1]["dimensions"]["n_seeds"] == 5
    assert main[0]["metrics"][0]["uncertainty_type"] == "patient_clustered_bootstrap_standard_error"
    assert main[-1]["metrics"][0]["uncertainty_type"] == "across_run_sample_standard_deviation"


@pytest.mark.parametrize(
    "cell", [r"\result{??.??}{?.??}", "NaN", "1e999", "1.0 extra", "[4, 1]", r"\result{1}{-1}"]
)
def test_paper_parser_rejects_placeholders_and_malformed_cells(cell: str) -> None:
    with pytest.raises(ValueError):
        PAPER["numeric_cell"](cell)


def test_paper_parser_preserves_printed_significance_markers() -> None:
    assert PAPER["numeric_cell"]("78.03±0.94‡") == {
        "value": 78.03,
        "uncertainty": 0.94,
        "reported_significance_marker": "ddagger",
    }
    assert PAPER["numeric_cell"]("−1.68±0.63†")["reported_significance_marker"] == "dagger"


@pytest.mark.parametrize("change", ["metric", "identity", "source", "duplicate", "cells"])
def test_paper_validation_rejects_changed_or_mislabeled_records(change: str) -> None:
    result = copy.deepcopy(load_json(ROOT / "results/paper_results.json"))
    if change == "metric":
        result["records"][0]["metrics"][0]["value"] = 100
    elif change == "identity":
        result["independently_reproduced"] = True
    elif change == "source":
        result["sources"]["manuscript.pdf"] = "0" * 64
    elif change == "cells":
        result["records"][0]["source"]["cells"][0] = "100.00±1.12"
        result["records"][0]["metrics"][0]["value"] = 100.0
    else:
        result["records"][-1] = result["records"][0]
    result["records_sha256"] = artifact_fingerprint(result["records"])
    with pytest.raises(ValueError):
        PAPER["validate_results"](result)


def test_horizon_units_and_confirmatory_family_are_explicit() -> None:
    result = load_json(ROOT / "results/paper_results.json")
    rows = {record["id"]: record for record in result["records"]}
    assert rows["horizon.01"]["dimensions"]["decision_depth"] == 1
    assert rows["horizon.07"]["metrics"][4]["unit"] == "percentage_points_per_depth_step"
    assert rows["interaction-tests.03"]["metrics"][0]["unit"] == "percentage_points_per_depth_step"
    assert rows["interaction-tests.06"]["dimensions"]["family"] == "secondary_descriptive"
    assert rows["interaction-tests.06"]["metrics"][-1]["value"] is None
    # Preserve printed paired contrasts rather than subtracting rounded table means.
    assert rows["arms.05"]["metrics"][2]["value"] == 10.13
    assert rows["baseline-contrasts.03"]["metrics"][1]["ci95"] == [-1.23, 7.63]
    assert rows["baseline-contrasts.09"]["metrics"][1]["ci95"] == [-1.85, 8.78]
    assert rows["baseline-contrasts.07"]["dimensions"]["comparison_status"] == "prespecified"
    assert rows["baseline-contrasts.13"]["dimensions"]["comparison_status"] == "exploratory"
    assert rows["baseline-contrasts.15"]["metrics"][0]["value"] == 2.13
    assert rows["baseline-contrasts.15"]["metrics"][-1]["value"] == 0.6282


def test_reported_compute_totals_and_grouping_shares_reconcile() -> None:
    records = load_json(ROOT / "results/paper_results.json")["records"]
    budget = [row for row in records if row["table"] == "tab:compute-budget"]
    for panel in (budget[:8], budget[8:]):
        assert panel[-1]["dimensions"]["is_subtotal"] is True
        for index in range(3):
            total = sum(row["metrics"][index].get("value") or 0 for row in panel[:-1])
            assert math.isclose(total, panel[-1]["metrics"][index]["value"], abs_tol=1e-9)
    assert budget[7]["metrics"][1]["value"] == 915
    assert budget[-1]["metrics"][1]["value"] == 232
    coverage = [row for row in records if row["table"] == "tab:grouping-coverage"]
    for index in range(3):
        # The nonzero-local-term row overlaps these mutually exclusive tiers.
        total = sum(row["metrics"][index]["value"] for row in coverage[:4])
        assert math.isclose(total, 1.0, abs_tol=0.0002)


@pytest.mark.parametrize("change", ["none", "trace", "tokens", "scope", "check", "missing_update"])
def test_training_trace_validation(change: str) -> None:
    result = copy.deepcopy(load_json(ROOT / "results/training_results.json"))
    events = copy.deepcopy(load_jsonl(ROOT / "logs/training_results.jsonl"))
    if change == "none":
        TRAINING["validate_results"](result, events)
        return
    if change == "trace":
        events[-1]["gradient_norm"] = 0
    elif change == "tokens":
        events[0]["cumulative_action_tokens"] += 1
    elif change == "scope":
        result["full_rl_rollout_training"] = True
    elif change == "check":
        result["checks"]["reference_unchanged"] = False
    else:
        events.pop(0)
    result["trace_sha256"] = artifact_fingerprint(events)
    with pytest.raises(ValueError):
        TRAINING["validate_results"](result, events)


@pytest.mark.parametrize("name", ["paper", "training"])
@pytest.mark.parametrize("directory,extension", [("results", "json"), ("logs", "jsonl")])
def test_credentialed_outputs_cannot_replace_public_results(
    name: str, directory: str, extension: str
) -> None:
    with pytest.raises(ValueError):
        guard_artifact_write_path(
            ROOT / directory / f"{name}_results.{extension}", content_kind="credentialed"
        )


@pytest.mark.model
def test_real_tiny_training_resume_and_frozen_reference(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    result, trace = TRAINING["run"](tmp_path / "tiny", steps=4)
    assert result["sft"]["optimizer_steps"] == 4
    assert all(result["checks"].values())
    assert len(trace) == 5
