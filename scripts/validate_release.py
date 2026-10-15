#!/usr/bin/env python3
"""Validate public configs and run the credential-free synthetic smoke path."""

from __future__ import annotations

import math
import runpy
import statistics
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate_cuda_results(result: dict, metrics: list[dict]) -> None:
    """Check published metadata and arithmetic without claiming execution attestation."""
    import hashlib

    from vapa.artifacts import artifact_fingerprint, canonical_json_dumps

    def sha(value):
        return hashlib.sha256((canonical_json_dumps(value) + "\n").encode()).hexdigest()

    def journal_sha(rows):
        text = "".join(canonical_json_dumps(row) + "\n" for row in rows)
        return hashlib.sha256(text.encode()).hexdigest()

    if (
        result["schema_version"] != "vapa-cuda-results-v1"
        or result["paper_reproduction"] is not False
        or not result["benchmarks"]
    ):
        raise ValueError("invalid CUDA result scope")
    for report in result["benchmarks"]:
        if not report["checks"] or any(value is not True for value in report["checks"].values()):
            raise ValueError("CUDA benchmark has an unsuccessful acceptance check")
        generation = report["generation"]
        for mode in ("serial", "batched"):
            trials = generation[mode]["trials"]
            if len(trials) != report["settings"]["repeats"] or not trials:
                raise ValueError("missing benchmark trials")
            for trial in trials:
                if (
                    trial["seconds"] <= 0
                    or trial["sampled_tokens"] <= 0
                    or not math.isclose(
                        trial["tokens_per_second"], trial["sampled_tokens"] / trial["seconds"]
                    )
                ):
                    raise ValueError("incorrect measured token throughput")
            if not math.isclose(
                generation[mode]["median_tokens_per_second"],
                statistics.median(t["tokens_per_second"] for t in trials),
            ):
                raise ValueError("incorrect median throughput")
        parity = report["full_precision_parity"]
        training = report["training"]
        if (
            not 0 <= generation["identical_sequences"] <= report["settings"]["requests"]
            or not 0 <= parity["gradient_relative_l2_error"] < 1e-3
            or abs(parity["loss"]["serial"] - parity["loss"]["batched"]) >= 1e-4
            or training["batch_gradient_within_5_percent"]
            != (training["gradient_relative_l2_error"] < 0.05)
        ):
            raise ValueError("inconsistent CUDA numerical diagnostics")
    rl = result["rl"]
    receipt, journal = rl["receipt"], rl["execution_journal"]
    if (
        not metrics
        or receipt["paper_reproduction"] is not False
        or receipt["stage"] != "rl"
        or metrics[-1]["global_step"] != receipt["optimizer_updates"]
        or metrics[-1]["sampled_tokens"] != receipt["sampled_tokens"]
        or sum(row["action_tokens"] for row in metrics) != receipt["trainable_tokens"]
        or [row["metric_index"] for row in metrics] != list(range(len(metrics)))
        or sha(rl["resolved_config"]) != receipt["config_sha256"]
        or sha(rl["run_manifest"]) != receipt["run_manifest_sha256"]
        or journal_sha(metrics) != receipt["metrics_sha256"]
        or journal_sha(journal) != receipt["execution_journal_sha256"]
        or journal[-1]["checkpoint"] != receipt["checkpoint"]
    ):
        raise ValueError("CUDA run receipt, configuration, or log linkage disagrees")
    previous = None
    for record in journal:
        if record["previous_sha256"] != previous:
            raise ValueError("CUDA execution journal hash chain is broken")
        previous = artifact_fingerprint(record)


def _assert_result_matches(expected: object, actual: object, path: str = "results") -> None:
    """Permit only floating-point roundoff across Python/libm platforms."""
    if isinstance(expected, float) and isinstance(actual, float):
        equal = math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)
    elif type(expected) is not type(actual):
        equal = False
    elif isinstance(expected, dict) and expected.keys() == actual.keys():
        for key in expected:
            _assert_result_matches(expected[key], actual[key], f"{path}.{key}")
        return
    elif isinstance(expected, list) and len(expected) == len(actual):
        for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_result_matches(left, right, f"{path}[{i}]")
        return
    else:
        equal = expected == actual
    if not equal:
        raise RuntimeError(f"stale demo result at {path}; rerun vapa demo --compact --publish")


def main() -> int:
    from vapa.artifacts import fingerprint_file
    from vapa.config import load_config
    from vapa.data import load_episode_objects, load_json
    from vapa.demo import demo_result_log, run_demo_suite, tiny_episode
    from vapa.environment.calculators import CalculatorRegistry
    from vapa.provenance import package_code_fingerprint
    from vapa.releases import load_public_core_release
    from vapa.verifiers import VerifierCatalog, paper_predicate_layout

    root = ROOT
    paper_predicate_layout()
    for path in sorted((root / "configs").glob("*.toml")):
        load_config(path)
    episodes = load_episode_objects(root / "examples" / "tiny_episode.json")
    if episodes != [tiny_episode()]:
        raise RuntimeError("public episode fixture drifted from the runnable demo")
    CalculatorRegistry.from_json(root / "examples" / "tiny_calculators.json")
    verifier_manifest = load_json(root / "examples" / "demo_verifier_catalog.json")
    if not isinstance(verifier_manifest, Mapping):
        raise RuntimeError("verifier manifest must be an object")
    catalog = VerifierCatalog.demo_default()
    families = list(dict.fromkeys(predicate.family.value for predicate in catalog.predicates))
    weights: dict[str, float] = {}
    for predicate in catalog.predicates:
        previous = weights.setdefault(predicate.reliability, predicate.weight)
        if previous != predicate.weight:
            raise RuntimeError("demo predicates use inconsistent reliability weights")
    expected_verifier_fields = {
        "schema_version": 1,
        "catalog_id": catalog.catalog_id,
        "paper_exact": catalog.paper_exact,
        "warning": (
            "The fourteen-predicate companion specification is not bundled; "
            "this is a seven-predicate demo catalog."
        ),
        "weights": weights,
        "families": families,
    }
    if dict(verifier_manifest) != expected_verifier_fields:
        raise RuntimeError("public verifier manifest drifted from the runnable catalog")
    task_catalog = load_json(root / "configs" / "task_catalog.json")
    if not isinstance(task_catalog, Mapping) or task_catalog.get("schema_version") != 1:
        raise RuntimeError("task catalog has an invalid schema")
    calculation = task_catalog.get("calculation")
    if not isinstance(calculation, Mapping):
        raise RuntimeError("task catalog calculation section must be an object")
    task_sections = (
        calculation.get("standard"),
        calculation.get("operationalized_proxies"),
        task_catalog.get("retrieval"),
        task_catalog.get("ehrshot_transfer"),
    )
    if any(
        not isinstance(section, list)
        or any(not isinstance(name, str) or not name for name in section)
        for section in task_sections
    ):
        raise RuntimeError("task catalog sections must be arrays of non-empty names")
    task_names = [name for section in task_sections for name in section]
    if len(task_names) != 42 or len(set(task_names)) != len(task_names):
        raise RuntimeError("task catalog must contain 42 unique task names")
    release = load_public_core_release()
    if release.paper_exact or release.release_id != "public_core_v1":
        raise RuntimeError("the bundled public-core release has the wrong identity")

    saved = load_json(root / "results/demo_results.json")
    with tempfile.TemporaryDirectory(prefix="vapa-release-validation-") as temporary:
        result = run_demo_suite(
            root / "examples", Path(temporary) / "run", seed=saved["seed"], compact=saved["compact"]
        )
    _assert_result_matches(saved, result)
    if (root / "logs/demo_results.jsonl").read_text(encoding="utf-8") != demo_result_log(saved):
        raise RuntimeError("public demo summary and result log disagree")
    paper_tools = runpy.run_path(str(root / "scripts/export_paper_results.py"))
    paper = load_json(root / "results/paper_results.json")
    if (root / "logs/paper_results.jsonl").read_text(encoding="utf-8") != paper_tools["result_log"](
        paper
    ):
        raise RuntimeError("public paper summary and result log disagree")
    training_tools = runpy.run_path(str(root / "scripts/train_tiny.py"))
    training = load_json(root / "results/training_results.json")
    from vapa.data.io import load_jsonl

    cuda = load_json(root / "results/cuda_results.json")
    validate_cuda_results(cuda, load_jsonl(root / "logs/cuda_results.jsonl"))
    for label, path in (("config", cuda["rl"]["config"]), ("episodes", cuda["rl"]["episodes"])):
        recorded = cuda["rl"]["run_manifest"]["input_artifacts"][label]["sha256"]
        if fingerprint_file(root / path).sha256 != recorded:
            raise RuntimeError(f"CUDA fixture changed: {label}")
    training_tools["validate_results"](training, load_jsonl(root / "logs/training_results.jsonl"))
    expected_provenance = {
        "fixture": "examples/tiny_sft.jsonl",
        "fixture_sha256": fingerprint_file(root / "examples/tiny_sft.jsonl").sha256,
        "implementation_sha256": package_code_fingerprint(),
        "runner_sha256": fingerprint_file(root / "scripts/train_tiny.py").sha256,
    }
    if training["provenance"] != expected_provenance:
        raise RuntimeError(
            "stale tiny-training results; rerun python scripts/train_tiny.py --publish"
        )
    print("release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
