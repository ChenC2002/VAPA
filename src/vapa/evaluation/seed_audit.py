"""Recalculate aggregate, paired-seed evidence without publishing patient records.

The manifest fixes all expected cells, seeds, and multiplicity families before the
measurements are read. This verifies arithmetic, not the authenticity of an export
or the underlying patient-level metric. Bootstrap SEs require a different audit.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections.abc import Mapping
from pathlib import Path

from vapa.artifacts import (
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    guard_artifact_write_path,
    strict_json_loads,
    strict_jsonl_loads,
    validate_output_paths,
)
from vapa.evaluation.statistics import holm_adjust, paired_t_test


def _object(value: object, keys: set[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must have exactly these fields: {sorted(keys)}")
    return value


def _identifier(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:+-]+", value) is None:
        raise ValueError("IDs must be nonempty opaque labels, not paths or personal information")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("measurements must be finite numbers, not booleans")
    return float(value)


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("source_sha256 must identify the original aggregate export")
    return value


def audit_seeds(manifest: object, rows: list[object]) -> dict[str, object]:
    """Audit a complete panel of scalar seed metrics using paired, two-sided tests.

    A cell identifies one system/backbone/endpoint/metric/unit combination. Values
    must already be in the declared unit: the audit never guesses percentages or
    re-aggregates patients. One observation per declared seed is mandatory.
    """
    spec = _object(
        manifest,
        {"schema_version", "evidence_kind", "seeds", "cells", "comparisons"},
        "seed audit manifest",
    )
    if spec["schema_version"] != "vapa-seed-audit-v1":
        raise ValueError("unsupported seed audit schema")
    if spec["evidence_kind"] not in {"original_measurements", "test_fixture"}:
        raise ValueError("evidence_kind must distinguish original measurements from test fixtures")
    seeds = spec["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("declare at least two distinct nonnegative integer seeds")
    if not isinstance(spec["cells"], list) or not spec["cells"]:
        raise ValueError("declare a nonempty cell panel")
    cells = {}
    for item in spec["cells"]:
        cell = _object(item, {"id", "system", "backbone", "endpoint", "metric", "unit"}, "cell")
        for key, value in cell.items():
            if key == "system":
                if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:+ -]+", value) is None:
                    raise ValueError("system must be a public method label")
            else:
                _identifier(value)
        if cell["id"] in cells:
            raise ValueError("duplicate cell ID")
        cells[cell["id"]] = cell
    # Use fixed key ordering, independent of JSON object insertion order.
    dimensions = [
        tuple(cell[key] for key in ("system", "backbone", "endpoint", "metric", "unit"))
        for cell in cells.values()
    ]
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("two cell IDs describe the same metric")
    panel: dict[str, dict[int, Mapping]] = {name: {} for name in cells}
    for item in rows:
        row = _object(item, {"cell", "seed", "value", "run_id", "source_sha256"}, "seed row")
        if not isinstance(row["cell"], str) or row["cell"] not in panel:
            raise ValueError("undeclared metric cell")
        if type(row["seed"]) is not int or row["seed"] not in seeds:
            raise ValueError("undeclared seed")
        _number(row["value"])
        _identifier(row["run_id"])
        _digest(row["source_sha256"])
        if row["seed"] in panel[row["cell"]]:
            raise ValueError("duplicate cell/seed observation")
        panel[row["cell"]][row["seed"]] = row
    for name, observations in panel.items():
        if set(observations) != set(seeds):
            raise ValueError(f"incomplete seed panel for {name}")
        if len({row["run_id"] for row in observations.values()}) != len(seeds):
            raise ValueError(f"independent training seeds need distinct run IDs for {name}")

    summaries = []
    for name, cell in cells.items():
        values = [float(panel[name][seed]["value"]) for seed in seeds]
        deviation = statistics.stdev(values)
        summaries.append(
            {
                **cell,
                "n": len(seeds),
                "mean": statistics.fmean(values),
                "sample_sd": deviation,
                "standard_error_of_mean": deviation / math.sqrt(len(seeds)),
                "observations": [dict(panel[name][seed]) for seed in sorted(seeds)],
            }
        )
    if not isinstance(spec["comparisons"], list):
        raise ValueError("comparisons must be an array (empty means no significance claims)")
    contrasts = {}
    families: dict[str, dict[str, float]] = {}
    tested_pairs = set()
    for item in spec["comparisons"]:
        comparison = _object(item, {"id", "left", "right", "family"}, "comparison")
        name, family = _identifier(comparison["id"]), _identifier(comparison["family"])
        left, right = _identifier(comparison["left"]), _identifier(comparison["right"])
        if name in contrasts or left not in cells or right not in cells or left == right:
            raise ValueError("comparisons require unique IDs and two different declared cells")
        pair = frozenset((left, right))
        if pair in tested_pairs:
            raise ValueError("a contrast cannot be repeated or reassigned to another family")
        tested_pairs.add(pair)
        if any(cells[left][key] != cells[right][key] for key in ("endpoint", "metric", "unit")):
            raise ValueError("paired cells must share endpoint, metric, and unit")
        differences = [panel[left][seed]["value"] - panel[right][seed]["value"] for seed in seeds]
        test = paired_t_test(differences)
        families.setdefault(family, {})[name] = test.p_value
        contrasts[name] = {
            **comparison,
            "n": test.n,
            "mean_difference": test.estimate,
            "ci95": [test.lower, test.upper],
            "t_statistic": test.t_statistic if math.isfinite(test.t_statistic) else None,
            "zero_variance": statistics.stdev(differences) == 0.0,
            "p_two_sided": test.p_value,
            "seed_differences": [
                {"seed": seed, "difference": difference}
                for seed, difference in zip(seeds, differences, strict=True)
            ],
        }
    for family in families.values():
        for name, adjusted in holm_adjust(family).items():
            contrasts[name]["p_holm"] = adjusted
            contrasts[name]["significant_at_0_05"] = adjusted < 0.05
    return {
        "schema_version": "vapa-seed-audit-result-v1",
        "evidence_kind": spec["evidence_kind"],
        "verification_scope": "aggregate arithmetic only; source authenticity not established",
        "paper_reproduction": False,
        "manifest_sha256": artifact_fingerprint(spec),
        "observations_sha256": artifact_fingerprint(rows),
        "uncertainty": "sample SD across training seeds; not patient-bootstrap SE",
        "test": "two-sided paired Student t; 95% CI; Holm within each declared family",
        "summaries": summaries,
        "comparisons": list(contrasts.values()),
    }


def compare_paper_cells(audit: Mapping, paper: Mapping) -> dict[str, object]:
    """Compare main-table trained-system cells at the printed two-decimal precision.

    Does not overwrite manuscript transcriptions or promote them to reproduced data.
    Fixed systems and missing seed cells remain explicitly unverified.
    """
    lookup = {
        (cell["system"], cell["backbone"], cell["metric"], cell["unit"]): cell
        for cell in audit["summaries"]
    }
    if len(lookup) != len(audit["summaries"]):
        raise ValueError("paper comparison requires one endpoint per system/backbone/metric")
    comparisons = []
    for record in paper["records"]:
        if record["table"] != "tab:main":
            continue
        dimensions = record["dimensions"]
        for metric in record["metrics"]:
            result = {"record_id": record["id"], "metric": metric["name"]}
            key = (dimensions["label"], dimensions["backbone"], metric["name"], metric["unit"])
            actual = lookup.get(key)
            if metric.get("uncertainty_type") != "across_run_sample_standard_deviation":
                result["status"] = "not_checked_requires_patient_bootstrap"
            elif actual is None:
                result["status"] = "not_checked_missing_seed_measurements"
            else:
                matches = (
                    abs(actual["mean"] - metric["value"]) <= 0.005 + 1e-12
                    and abs(actual["sample_sd"] - metric["uncertainty"]) <= 0.005 + 1e-12
                    and actual["n"] == dimensions["n_seeds"]
                )
                result.update(
                    status="agrees_at_printed_precision" if matches else "mismatch",
                    calculated_mean=actual["mean"],
                    calculated_sample_sd=actual["sample_sd"],
                    calculated_n=actual["n"],
                )
            comparisons.append(result)
    if not comparisons:
        raise ValueError("paper snapshot contains no main-table cells")
    return {"cells": comparisons, "paper_sha256": artifact_fingerprint(paper)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--paper-results", type=Path)
    parser.add_argument(
        "--content-kind", choices=("credentialed", "derived", "public"), default="credentialed"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    inputs = [args.manifest, args.measurements]
    if args.paper_results:
        inputs.append(args.paper_results)
    destination = guard_artifact_write_path(args.output, content_kind=args.content_kind)
    validate_output_paths([destination], inputs=inputs, overwrite=args.overwrite)
    report = audit_seeds(
        strict_json_loads(args.manifest.read_bytes()),
        strict_jsonl_loads(args.measurements.read_text(encoding="utf-8")),
    )
    if args.paper_results:
        report["paper_comparison"] = compare_paper_cells(
            report, strict_json_loads(args.paper_results.read_bytes())
        )
    atomic_write_text(destination, canonical_json_dumps(report) + "\n", overwrite=args.overwrite)
    print(f"Seed arithmetic audited: {len(report['summaries'])} cells; {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
