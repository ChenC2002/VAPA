#!/usr/bin/env python3
"""Export reviewed numeric manuscript tables (never training logs).

Reviewed PDF cells are embedded with page/table/row locations. Validation reparses
those cells and checks the reviewed digest; it does not claim automatic PDF extraction
or independent reproduction. --source additionally verifies the original PDF hash.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vapa.artifacts import (  # noqa: E402
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    strict_json_loads,
    validate_output_paths,
)

ENDPOINTS = (
    "calculation_verified_accuracy",
    "retrieval_grounded_exact_match",
    "medagentbench_task_success",
    "medagentbench_cost_normalized_success",
    "ehrshot_mean_auprc",
    "ehrshot_mean_auroc",
)
# label: ((PDF page, table number), expected rows, ordered metric names, ordered units)
TABLES = {
    "main": ((6, 1), 16, ENDPOINTS, ("percent",) * 6),
    "arms": ((7, 2), 6, ENDPOINTS, ("percent",) * 6),
    "replay-effects": (
        (23, 11),
        8,
        ("replay", "no_fork", "gain"),
        ("percent", "percent", "percentage_points"),
    ),
    "grouping-coverage": (
        (24, 12),
        5,
        ("A2", "A4", "pooled"),
        ("fraction",) * 3,
    ),
    "compute-efficiency": (
        (25, 13),
        8,
        (
            "training_gpu_hours",
            "sampled_tokens_per_second",
            "peak_memory",
            "inference_actions_per_episode",
            "inference_tokens_per_episode",
            "inference_latency",
        ),
        ("device_hours", "tokens_per_second", "GiB", "actions", "tokens", "seconds"),
    ),
    "compute-budget": (
        (25, 14),
        12,
        ("runs", "gpu_hours", "rl_tokens"),
        ("count", "device_hours", "billion_tokens"),
    ),
    "interaction-tests": (
        (27, 15),
        7,
        ("estimate", "confidence_interval", "p_value", "holm_p_value"),
        ("percentage_points", "percentage_points", "probability", "probability"),
    ),
    "baseline-contrasts": (
        (28, 16),
        18,
        ("estimate", "confidence_interval", "p_value", "holm_p_value"),
        ("percentage_points", "percentage_points", "probability", "probability"),
    ),
    "horizon": (
        (33, 23),
        14,
        ("A1", "A2", "A3", "A4", "A4_minus_A1", "interaction"),
        ("percent",) * 4 + ("percentage_points",) * 2,
    ),
    "backbone-contrast": (
        (33, 24),
        6,
        ("qwen3_5_9b", "gpt_oss_20b"),
        ("percentage_points",) * 2,
    ),
}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
REVIEWED_SOURCES = {
    "manuscript.pdf": "5689aed86890a80516fd5e4d6462a4c73caa0fb2f8471da68397b58750340dc7",
}
REVIEWED_RECORDS_SHA256 = "504942bb73346c8a2288d0d695ef49deb1eec0d4b8bede4f4ad8ba99d41467cd"


def numeric_cell(cell: str) -> dict[str, object]:
    original = cell.strip()
    text = original.replace("−", "-")
    text = text.replace("†", "").replace("‡", "")
    if text in {"--", "–", "NA"}:
        return {"value": None, "missing_reason": "not_reported"}
    interval = re.fullmatch(
        r"(?:((?:"
        + NUMBER
        + r")(?:±(?:"
        + NUMBER
        + r"))?)\s*)?\[("
        + NUMBER
        + r"),\s*("
        + NUMBER
        + r")\]",
        text,
    )
    if interval:
        point, low, high = interval.groups()
        result = {} if point is None else numeric_cell(point)
        result["ci95"] = [float(low), float(high)]
        if float(low) > float(high):
            raise ValueError("reversed confidence interval")
        return result
    match = re.fullmatch(r"(" + NUMBER + r")(?:±(" + NUMBER + r"))?", text)
    if not match:
        raise ValueError(f"unsupported numeric cell: {original}")
    value, uncertainty = match.groups()
    result = {"value": float(value)}
    if uncertainty is not None:
        if float(uncertainty) < 0:
            raise ValueError("negative uncertainty")
        result["uncertainty"] = float(uncertainty)
    if "‡" in original or "†" in original:
        result["reported_significance_marker"] = "ddagger" if "‡" in original else "dagger"
    return result


def make_record(table: str, index: int, source: dict[str, object]) -> dict[str, object]:
    _, _, names, units = TABLES[table]
    cells = source["cells"]
    metrics = []
    for name, unit, cell in zip(names, units, cells, strict=True):
        metric = {"name": name, "unit": unit, **numeric_cell(cell)}
        if table == "arms" and index >= 4:
            metric["unit"] = "percentage_points"
        if table == "horizon" and index in {6, 13}:
            metric["unit"] = "percentage_points_per_depth_step"
        if (
            table == "interaction-tests"
            and index == 2
            and name in {"estimate", "confidence_interval"}
        ):
            metric["unit"] = "percentage_points_per_depth_step"
        if "uncertainty" in metric:
            metric["uncertainty_type"] = (
                "patient_clustered_bootstrap_standard_error"
                if table == "main" and index < 7
                else "across_run_sample_standard_deviation"
            )
        if "ci95" in metric or name == "confidence_interval":
            metric["interval_type"] = "paired_seed_t_95_percent"
        metrics.append(metric)
    dimensions: dict[str, object] = {"label": source["label"]}
    if table in {"main", "compute-efficiency"}:
        other = index in ({13, 14} if table == "main" else {5, 6, 7})
        dimensions["backbone"] = "gpt-oss-20b" if other else "Qwen3.5-9B"
        dimensions["n_seeds"] = (
            (None if index < 7 else 3 if other else 5)
            if table == "main"
            else (None if index in {0, 5} else 3 if other else 5)
        )
        if table == "compute-efficiency" and 1 <= index <= 4:
            dimensions["throughput_basis"] = "additive_model_estimate"
    elif table == "compute-budget":
        dimensions.update(
            panel="fixed_configuration_reproduction" if index < 8 else "search_and_screening",
            is_subtotal=index in {7, 11},
            accounting_basis="paper_reported_nominal_and_reconstructed",
        )
    elif table == "arms":
        dimensions = {
            "arm_or_contrast": ("A1", "A2", "A3", "A4", "A4_minus_A1", "interaction")[index],
            "n_seeds": 5,
        }
    elif table == "replay-effects":
        dimensions.update(arm="A2" if index < 4 else "A4", n_seeds=5)
    elif table == "horizon":
        dimensions.update(
            endpoint="task_success" if index < 7 else "cost_normalized_success",
            decision_depth=None if index % 7 == 6 else index % 7 + 1,
            n_seeds=5,
            episodes_per_depth=250,
        )
    elif table == "baseline-contrasts":
        dimensions.update(
            endpoint=ENDPOINTS[index % 6],
            family=("previously_selected_references", "adapted_vineppo", "tree_grpo_process")[
                index // 6
            ],
            comparison_status="prespecified" if 6 <= index < 12 else "exploratory",
            n_seeds=5,
        )
    elif table == "interaction-tests":
        dimensions.update(
            family="confirmatory" if index < 5 else "secondary_descriptive", n_seeds=5
        )
    elif table == "backbone-contrast":
        dimensions.update(
            endpoint=ENDPOINTS[index], contrast="A4_minus_A1", qwen_seeds=5, gpt_oss_seeds=3
        )
    return {
        "id": f"{table}.{index + 1:02d}",
        "table": f"tab:{table}",
        "kind": "paper_reported",
        "dimensions": dimensions,
        "metrics": metrics,
        "source": source,
    }


def build_results(source_rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Rebuild metrics from reviewed cells, not from an automated PDF/TeX parser."""

    records = []
    if set(source_rows) != set(TABLES):
        raise ValueError("unexpected reviewed table selection")
    for table in TABLES:
        records.extend(
            make_record(table, index, row) for index, row in enumerate(source_rows[table])
        )
    return {
        "schema_version": "vapa-paper-results-v1",
        "project": "VAPA",
        "paper_title": "Where Credit Lands: Step-Level Advantages for Bounded-Memory EHR Agents",
        "source_version": "manuscript-review-v2",
        "kind": "paper_reported",
        "independently_reproduced": False,
        "sources": dict(REVIEWED_SOURCES),
        "transcription_method": "visually_reviewed_pdf_cells",
        "coverage": {
            "included_tables": [f"tab:{name}" for name in TABLES],
            "excluded": [
                "Other appendix tables and narrative-only values",
                "Plot-only learning curves and diagnostics: original numeric arrays unavailable",
                "Superseded compute-ledger table from the previous manuscript version",
                "Per-seed observations, raw optimizer traces, and checkpoints: not supplied",
            ],
        },
        "caveats": [
            "Reviewed PDF transcription, not empirical verification or a run of this repository.",
            "Printed paired contrasts use unrounded per-seed values; "
            "subtracting rounded means can differ by 0.01 points.",
            "Fixed-system uncertainties are bootstrap standard errors; trained-system "
            "uncertainties are sample standard deviations, not interchangeable.",
            "Compute accounting mixes directly reported measurements and reconstructed "
            "accounting; underlying per-run compute artifacts were not supplied.",
            "Qwen factorial throughput entries are additive-model estimates. "
            "Compute Table 14 separates reproduction allowance from search and screening.",
            "Reported significance is transcribed, not recomputed from unavailable per-seed data.",
        ],
        "record_count": len(records),
        "records_sha256": artifact_fingerprint(records),
        "records": records,
    }


def validate_results(result: dict[str, object]) -> None:
    if (
        result.get("schema_version") != "vapa-paper-results-v1"
        or result.get("kind") != "paper_reported"
        or result.get("independently_reproduced") is not False
        or result.get("sources") != REVIEWED_SOURCES
        or result.get("source_version") != "manuscript-review-v2"
        or result.get("transcription_method") != "visually_reviewed_pdf_cells"
    ):
        raise ValueError("invalid paper-result identity or reproduction claim")
    records = result["records"]
    if (
        result["record_count"] != len(records)
        or result["records_sha256"] != artifact_fingerprint(records)
        or result["records_sha256"] != REVIEWED_RECORDS_SHA256
    ):
        raise ValueError("paper-result count or checksum mismatch")
    for table, (_, expected, _, _) in TABLES.items():
        selected = [record for record in records if record["table"] == f"tab:{table}"]
        if len(selected) != expected:
            raise ValueError(f"wrong record count for tab:{table}")
        for index, record in enumerate(selected):
            source = record["source"]
            if (
                set(source) != {"path", "page", "table", "row", "label", "cells"}
                or source["path"] != "manuscript.pdf"
                or (source["page"], source["table"]) != TABLES[table][0]
                or type(source["row"]) is not int
                or source["row"] != index + 1
                or not isinstance(source["label"], str)
                or not source["label"].strip()
                or not isinstance(source["cells"], list)
                or any(not isinstance(cell, str) for cell in source["cells"])
            ):
                raise ValueError("invalid paper-result source location")
            if record != make_record(table, index, record["source"]):
                raise ValueError(f"numeric/source mismatch in {record['id']}")
    if len(records) != sum(spec[1] for spec in TABLES.values()):
        raise ValueError("unexpected paper-result records")


def result_log(result: dict[str, object]) -> str:
    validate_results(result)
    metadata = {key: value for key, value in result.items() if key != "records"}
    events = [{"event": "manifest", **metadata}]
    events.extend({"event": "result", **record} for record in result["records"])
    events.append(
        {
            "event": "complete",
            "record_count": result["record_count"],
            "records_sha256": result["records_sha256"],
        }
    )
    return "".join(canonical_json_dumps(event) + "\n" for event in events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="reviewed manuscript PDF; verifies its original hash"
    )
    parser.add_argument(
        "--write", action="store_true", help="replace the reviewed paper-result snapshots"
    )
    args = parser.parse_args()
    if args.write and args.source is None:
        parser.error("--write requires --source")
    summary = ROOT / "results/paper_results.json"
    log = ROOT / "logs/paper_results.jsonl"
    saved = strict_json_loads(summary.read_bytes())
    if args.source is not None:
        if fingerprint_file(args.source).sha256 != REVIEWED_SOURCES["manuscript.pdf"]:
            raise ValueError("unreviewed manuscript revision; review table mappings before export")
    result = build_results(
        {
            table: [row["source"] for row in saved["records"] if row["table"] == f"tab:{table}"]
            for table in TABLES
        }
    )
    validate_results(result)
    if args.write:
        validate_output_paths(
            [summary, log],
            inputs=[args.source],
            overwrite=True,
        )
        atomic_write_text(log, result_log(result))
        atomic_write_text(
            summary, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    elif result != saved or log.read_text(encoding="utf-8") != result_log(result):
        raise ValueError("paper summary, log, or original manuscript disagree")
    print(
        f"validated {result['record_count']} paper-reported records; not independently reproduced"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
