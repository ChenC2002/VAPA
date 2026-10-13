from __future__ import annotations

import copy
import math

import pytest

from vapa.artifacts import canonical_json_dumps
from vapa.evaluation.seed_audit import audit_seeds, compare_paper_cells, main


def fixture_panel():
    manifest = {
        "schema_version": "vapa-seed-audit-v1",
        "evidence_kind": "test_fixture",
        "seeds": [13, 17],
        "cells": [
            {
                "id": system,
                "system": system,
                "backbone": "unit-model",
                "endpoint": "test",
                "metric": "accuracy",
                "unit": "percent",
            }
            for system in ("left", "right", "control")
        ],
        "comparisons": [
            {"id": "left-right", "left": "left", "right": "right", "family": "primary"},
            {"id": "left-control", "left": "left", "right": "control", "family": "primary"},
        ],
    }
    rows = [
        {
            "cell": cell,
            "seed": seed,
            "value": value,
            "run_id": f"{cell}-{seed}",
            "source_sha256": "a" * 64,
        }
        for cell, values in (("left", [2, 4]), ("right", [1, 1]), ("control", [0, 0]))
        for seed, value in zip(manifest["seeds"], values, strict=True)
    ]
    return manifest, rows


def test_independent_cauchy_reference_and_holm():
    report = audit_seeds(*fixture_panel())
    summary = report["summaries"][0]
    assert summary["mean"] == 3
    assert summary["sample_sd"] == pytest.approx(math.sqrt(2))
    assert summary["standard_error_of_mean"] == pytest.approx(1)
    first, second = report["comparisons"]
    # With two paired seeds the t distribution is exactly Cauchy. This reference
    # does not call the incomplete-beta implementation under test.
    p_first = 1 - 2 * math.atan(2) / math.pi
    p_second = 1 - 2 * math.atan(3) / math.pi
    half_width = math.tan(0.475 * math.pi)
    assert first["p_two_sided"] == pytest.approx(p_first)
    assert first["ci95"] == pytest.approx([2 - half_width, 2 + half_width])
    assert second["p_two_sided"] == pytest.approx(p_second)
    assert second["p_holm"] == pytest.approx(2 * p_second)
    assert first["p_holm"] == pytest.approx(max(p_first, 2 * p_second))
    assert report["evidence_kind"] == "test_fixture"
    assert report["paper_reproduction"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "seed",
        "nan",
        "bool",
        "run",
        "extra",
        "unit",
        "repeated_contrast",
        "hash",
    ],
)
def test_reject_invalid_panels(mutation):
    spec, rows = fixture_panel()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "seed":
        rows[0]["seed"] = 999
    elif mutation in {"nan", "bool"}:
        rows[0]["value"] = math.nan if mutation == "nan" else True
    elif mutation == "run":
        rows[1]["run_id"] = rows[0]["run_id"]
    elif mutation == "extra":
        rows[0]["patient_id"] = "not-for-export"
    elif mutation == "unit":
        spec["cells"][1]["unit"] = "fraction"
    elif mutation == "repeated_contrast":
        other = dict(spec["comparisons"][0], id="extra-test", family="another-family")
        spec["comparisons"].append(other)
    elif mutation == "hash":
        rows[0]["source_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        audit_seeds(spec, rows)


def test_zero_variance_is_json_safe_and_flagged():
    spec, rows = fixture_panel()
    for row in rows:
        row["value"] = 1 if row["cell"] == "left" else 0
    report = audit_seeds(spec, rows)
    assert report["comparisons"][0]["t_statistic"] is None
    assert report["comparisons"][0]["zero_variance"] is True
    assert "Infinity" not in canonical_json_dumps(report)


def test_compare_paper_does_not_mark_missing_or_bootstrap_cells_verified():
    report = audit_seeds(*fixture_panel())
    base = {
        "id": "main.01",
        "table": "tab:main",
        "dimensions": {"label": "left", "backbone": "unit-model", "n_seeds": 2},
        "metrics": [
            {
                "name": "accuracy",
                "unit": "percent",
                "value": 3.0,
                "uncertainty": 1.41,
                "uncertainty_type": "across_run_sample_standard_deviation",
            }
        ],
    }
    missing, bootstrap, mismatch = (copy.deepcopy(base) for _ in range(3))
    missing["dimensions"]["label"] = "missing"
    bootstrap["metrics"][0]["uncertainty_type"] = "patient_clustered_bootstrap_standard_error"
    mismatch["metrics"][0]["value"] = 3.1
    result = compare_paper_cells(report, {"records": [base, missing, bootstrap, mismatch]})
    assert [row["status"] for row in result["cells"]] == [
        "agrees_at_printed_precision",
        "not_checked_missing_seed_measurements",
        "not_checked_requires_patient_bootstrap",
        "mismatch",
    ]


def test_cli_and_input_overwrite_protection(tmp_path):
    spec, rows = fixture_panel()
    manifest, data, output = (
        tmp_path / name for name in ("manifest.json", "data.jsonl", "out.json")
    )
    manifest.write_text(canonical_json_dumps(spec))
    data.write_text("".join(canonical_json_dumps(row) + "\n" for row in rows))
    assert main([str(manifest), str(data), str(output)]) == 0
    with pytest.raises(FileExistsError):
        main([str(manifest), str(data), str(output)])
    with pytest.raises(ValueError, match="aliases"):
        main([str(manifest), str(data), str(data), "--overwrite"])
