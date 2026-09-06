#!/usr/bin/env python3
"""Export reviewed numeric manuscript tables (never training logs).

The deliberately narrow parser rejects a changed table shape or unsupported numeric
cell. Embedded source rows let release checks re-parse every published value without
requiring the private manuscript. Original source hashes bind external verification.
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
# label: (file, expected rows, ordered metric names, ordered units)
TABLES = {
    "main": ("experiments", 15, ENDPOINTS, ("percent",) * 6),
    "arms": ("experiments", 6, ENDPOINTS, ("percent",) * 6),
    "replay-effects": (
        "appendix",
        8,
        ("replay", "no_fork", "gain"),
        ("percent", "percent", "percentage_points"),
    ),
    "grouping-coverage": (
        "appendix",
        5,
        ("A2", "A4", "pooled"),
        ("fraction",) * 3,
    ),
    "compute-efficiency": (
        "appendix",
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
    "compute-ledger": (
        "appendix",
        12,
        ("runs", "sampled_tokens_per_second", "gpu_hours_per_run", "gpu_hours"),
        ("count", "tokens_per_second", "device_hours", "device_hours"),
    ),
    "compute-budget": (
        "appendix",
        11,
        ("runs", "gpu_hours", "rl_tokens"),
        ("count", "device_hours", "billion_tokens"),
    ),
    "interaction-tests": (
        "appendix",
        7,
        ("estimate", "confidence_interval", "p_value", "holm_p_value"),
        ("percentage_points", "percentage_points", "probability", "probability"),
    ),
    "baseline-contrasts": (
        "appendix",
        12,
        ("estimate", "confidence_interval", "p_value", "holm_p_value"),
        ("percentage_points", "percentage_points", "probability", "probability"),
    ),
    "horizon": (
        "appendix",
        14,
        ("A1", "A2", "A3", "A4", "A4_minus_A1", "interaction"),
        ("percent",) * 4 + ("percentage_points",) * 2,
    ),
    "backbone-contrast": (
        "appendix",
        6,
        ("qwen3_5_9b", "gpt_oss_20b"),
        ("percentage_points",) * 2,
    ),
}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
REVIEWED_SOURCES = {
    "Tex/experiments.tex": "4ac3fb1e8dbe7dff244d36c47665ba00dab7d41a8b42b89214d6f60e415ff833",
    "Tex/appendix.tex": "7dc25363ed73929aba47308ff7eb20486d7db79363c605a664bbccc9440a5b3e",
}


def strip_comment(line: str) -> str:
    # An even number of preceding backslashes means '%' starts a TeX comment.
    return re.split(r"(?<!\\)(?:\\\\)*%", line, maxsplit=1)[0].strip()


def numeric_cell(cell: str) -> dict[str, object]:
    original = cell.strip()
    text = original.replace("{,}", "")
    text = re.sub(
        r"\\(?:bestresult|baselineresult|result)\{(" + NUMBER + r")\}\{(" + NUMBER + r")\}",
        r"\1±\2",
        text,
    )
    for marker in (r"$", r"\(", r"\)"):
        text = text.replace(marker, "")
    text = re.sub(r"\^\{?\\(?:ddagger|dagger)\}?", "", text)
    text = text.replace(r"\pm", "±").replace(r"\mathrm{NA}", "NA").strip()
    if text in {"--", "NA"}:
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
    if r"\ddagger" in original or r"\dagger" in original:
        result["reported_significance_marker"] = "ddagger" if r"\ddagger" in original else "dagger"
    return result


def display_text(text: str) -> str:
    for macro, value in {
        "armone": "A1",
        "armtwo": "A2",
        "armthree": "A3",
        "armfour": "A4",
        "modelname": "VAPA",
    }.items():
        text = text.replace("\\" + macro + "{}", value)
    text = re.sub(r"\\citeyearpar\{[^}]+\}", "", text)
    text = text.replace(r"\-", "").replace(r"\(", "").replace(r"\)", "")
    return " ".join(text.replace("$", "").split()).strip()


def make_record(table: str, index: int, source: dict[str, object]) -> dict[str, object]:
    _, _, names, units = TABLES[table]
    cells = str(source["row_tex"]).removesuffix(r"\\").split("&")
    metrics = []
    for name, unit, cell in zip(names, units, cells[-len(names) :], strict=True):
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
    # Last label column is the actual method/endpoint, not a multirow family label.
    label = display_text(cells[-len(names) - 1])
    dimensions: dict[str, object] = {"label": label}
    if table in {"main", "compute-efficiency"}:
        other = index in ({12, 13} if table == "main" else {5, 6, 7})
        dimensions["backbone"] = "gpt-oss-20b" if other else "Qwen3.5-9B"
        dimensions["n_seeds"] = (
            (None if index < 7 else 3 if other else 5)
            if table == "main"
            else (None if index in {0, 5} else 3 if other else 5)
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
            family="strongest_reference" if index < 6 else "adapted_vineppo",
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


def extract_tables(source_root: Path) -> dict[str, object]:
    records = []
    sources = {}
    for file in ("experiments", "appendix"):
        path = source_root / "Tex" / f"{file}.tex"
        sources[f"Tex/{file}.tex"] = fingerprint_file(path).sha256
        if sources[f"Tex/{file}.tex"] != REVIEWED_SOURCES[f"Tex/{file}.tex"]:
            raise ValueError("unreviewed manuscript revision; review table mappings before export")
        lines = [strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for table, (source_file, count, names, _) in TABLES.items():
            if source_file != file:
                continue
            labels = [i for i, line in enumerate(lines) if line == f"\\label{{tab:{table}}}"]
            if len(labels) != 1:
                raise ValueError(f"expected exactly one tab:{table}")
            start = labels[0]
            stop = next(i for i in range(start, len(lines)) if lines[i].startswith(r"\end{table"))
            index = 0
            for line_number in range(start + 1, stop):
                line = lines[line_number]
                if not line.endswith(r"\\") or line.count("&") < len(names):
                    continue
                cells = line.removesuffix(r"\\").split("&")[-len(names) :]
                # Headers cannot start with a numeric literal/result macro. Once a
                # numeric body row is recognized every cell must parse, or fail closed.
                if not re.match(
                    r"^(?:[+\-\d.]|\$[+\-\d.]|\\(?:result|bestresult|baselineresult)|\\\((?:[+\-\d.]|\\mathrm\{NA\}))",
                    cells[0].strip(),
                ):
                    continue
                source = {"path": f"Tex/{file}.tex", "line": line_number + 1, "row_tex": line}
                records.append(make_record(table, index, source))
                index += 1
            if index != count:
                raise ValueError(f"tab:{table}: expected {count} numeric rows, found {index}")
    return {
        "schema_version": "vapa-paper-results-v1",
        "project": "VAPA",
        "paper_title": "Where Credit Lands: Step-Level Advantages for Bounded-Memory EHR Agents",
        "source_version": "manuscript-review-v1",
        "kind": "paper_reported",
        "independently_reproduced": False,
        "sources": sources,
        "coverage": {
            "included_tables": [f"tab:{name}" for name in TABLES],
            "excluded": [
                "Other appendix tables and narrative-only values",
                "Plot-only learning curves and diagnostics: original numeric arrays unavailable",
                "Commented-out draft scaffolds and unfinished experiments",
                "Per-seed observations, raw optimizer traces, and checkpoints: not supplied",
            ],
        },
        "caveats": [
            "Draft manuscript transcription, not empirical verification "
            "or a run of this repository.",
            "Printed paired contrasts use unrounded per-seed values; "
            "subtracting rounded means can differ by 0.01 points.",
            "Fixed-system uncertainties are bootstrap standard errors; trained-system "
            "uncertainties are sample standard deviations, not interchangeable.",
            "Compute accounting mixes directly reported measurements and reconstructed "
            "accounting; underlying per-run compute artifacts were not supplied.",
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
        or result.get("source_version") != "manuscript-review-v1"
    ):
        raise ValueError("invalid paper-result identity or reproduction claim")
    records = result["records"]
    if result["record_count"] != len(records) or result["records_sha256"] != artifact_fingerprint(
        records
    ):
        raise ValueError("paper-result count or checksum mismatch")
    for table, (_, expected, _, _) in TABLES.items():
        selected = [record for record in records if record["table"] == f"tab:{table}"]
        if len(selected) != expected:
            raise ValueError(f"wrong record count for tab:{table}")
        for index, record in enumerate(selected):
            source = record["source"]
            if (
                source["path"] != f"Tex/{TABLES[table][0]}.tex"
                or type(source["line"]) is not int
                or source["line"] < 1
                or source["row_tex"] != strip_comment(source["row_tex"])
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
        "--source", type=Path, help="review manuscript directory; verifies original files"
    )
    parser.add_argument(
        "--write", action="store_true", help="replace the reviewed paper-result snapshots"
    )
    args = parser.parse_args()
    if args.write and args.source is None:
        parser.error("--write requires --source")
    summary = ROOT / "results/paper_results.json"
    log = ROOT / "logs/paper_results.jsonl"
    saved = None if args.write else strict_json_loads(summary.read_bytes())
    result = extract_tables(args.source) if args.source is not None else saved
    validate_results(result)
    if args.write:
        validate_output_paths(
            [summary, log],
            inputs=[args.source / path for path in REVIEWED_SOURCES],
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
