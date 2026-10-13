"""Training-time provenance and metadata-only, integrity-checked run receipts.

Receipts establish file consistency, not independent execution attestation. They
never establish paper reproduction, even when every hash agrees. Raw training
artifacts remain on the run's approved storage; this exporter does not load model
pickle files, copy patient records, or infer clinical validity.
"""

from __future__ import annotations

import argparse
import platform
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from vapa.artifacts import (
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    guard_artifact_write_path,
    strict_json_loads,
    strict_jsonl_loads,
    validate_output_paths,
)
from vapa.inference import checkpoint_identity
from vapa.provenance import package_code_fingerprint
from vapa.training.checkpoint import read_manifest

_FINISHED_STATUSES = {"complete", "stopped", "max_optimizer_steps"}


def software_environment() -> dict[str, object]:
    packages = {}
    for name in ("vapa-ehr", "torch", "transformers", "peft", "tokenizers", "numpy", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    torch = sys.modules.get("torch")
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda_runtime": None if torch is None else torch.version.cuda,
    }


def _journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = strict_jsonl_loads(path.read_text(encoding="utf-8"))
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("execution journal rows must be objects")
    return rows


def _append(output: Path, record: dict) -> None:
    path = output / "execution.jsonl"
    records = _journal(path)
    record["previous_sha256"] = None if not records else artifact_fingerprint(records[-1])
    records.append(record)
    atomic_write_text(path, "".join(canonical_json_dumps(row) + "\n" for row in records))


def start_execution(output: Path, *, config: Mapping, stage: str, device: str) -> str:
    """Record runtime facts at training time, including separate resumed invocations."""
    if stage not in {"sft", "rl"}:
        raise ValueError("unsupported training stage")
    config_path = output / "resolved_config.json"
    encoded = canonical_json_dumps(config) + "\n"
    if config_path.exists():
        if strict_json_loads(config_path.read_bytes()) != strict_json_loads(encoded):
            raise ValueError("resolved configuration changed during the run")
    else:
        atomic_write_text(config_path, encoded, overwrite=False)
    execution_id = uuid.uuid4().hex
    _append(
        output,
        {
            "event": "start",
            "execution_id": execution_id,
            "time_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "environment": software_environment(),
            "device": device,
            "implementation_sha256": package_code_fingerprint(),
            "config_sha256": fingerprint_file(config_path).sha256,
            "run_manifest_sha256": fingerprint_file(output / "run_manifest.json").sha256,
        },
    )
    return execution_id


def finish_execution(output: Path, execution_id: str, *, status: str, checkpoint: Path) -> None:
    """Write only after metrics and the last checkpoint have successfully been saved."""
    if status not in _FINISHED_STATUSES:
        raise ValueError("only a successfully saved complete/stopped run can finish")
    _append(
        output,
        {
            "event": "finish",
            "execution_id": execution_id,
            "time_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "metrics_sha256": fingerprint_file(output / "metrics.jsonl").sha256,
            "checkpoint": checkpoint_identity(checkpoint),
        },
    )


def build_run_receipt(output: Path, checkpoint: Path, *, evaluation: Path | None = None) -> dict:
    """Verify a run/checkpoint/log chain without deserializing any model weights."""
    run_path = output / "run_manifest.json"
    config_path = output / "resolved_config.json"
    metrics_path = output / "metrics.jsonl"
    journal_path = output / "execution.jsonl"
    sources = [run_path, config_path, metrics_path, journal_path, checkpoint / "manifest.json"]
    before = {path: fingerprint_file(path) for path in sources}
    run = strict_json_loads(run_path.read_bytes())
    config = strict_json_loads(config_path.read_bytes())
    manifest = read_manifest(checkpoint)
    if manifest.contract.run_manifest_fingerprint != artifact_fingerprint(run):
        raise ValueError("checkpoint belongs to a different run manifest")
    if manifest.contract.config_fingerprint != artifact_fingerprint(config):
        raise ValueError("checkpoint configuration does not match resolved_config.json")
    for record in manifest.files.values():
        path = checkpoint / record.path
        if path.is_symlink():
            raise ValueError("checkpoint files cannot be symlinks")
        actual = fingerprint_file(path)
        if actual.sha256 != record.sha256 or actual.size_bytes != record.size:
            raise ValueError("checkpoint file failed integrity verification")
    if "runtime" not in manifest.files:
        raise ValueError("checkpoint has no runtime state")
    runtime_file = checkpoint / manifest.files["runtime"].path
    if strict_json_loads(runtime_file.read_bytes()) != manifest.runtime.to_dict():
        raise ValueError("checkpoint manifest and runtime file disagree")
    identity = checkpoint_identity(checkpoint)
    journal = _journal(journal_path)
    previous = None
    starts, finishes, finished_ids = {}, [], set()
    for record in journal:
        if record.get("previous_sha256") != previous:
            raise ValueError("execution journal hash chain is broken")
        previous = artifact_fingerprint(record)
        execution_id = record.get("execution_id")
        if not isinstance(execution_id, str):
            raise ValueError("execution journal lacks an invocation ID")
        if record.get("event") == "start":
            if execution_id in starts:
                raise ValueError("duplicate execution invocation ID")
            starts[execution_id] = record
        elif record.get("event") == "finish" and execution_id in starts:
            if execution_id in finished_ids or record.get("status") not in _FINISHED_STATUSES:
                raise ValueError("invalid or duplicate execution finish")
            finished_ids.add(execution_id)
            finishes.append(record)
        else:
            raise ValueError("execution finish has no corresponding start")
    if not finishes or journal[-1] != finishes[-1]:
        raise ValueError("latest execution did not finish; a partial run needs investigation")
    finish = finishes[-1]
    start = starts[finish["execution_id"]]
    if finish["checkpoint"] != identity or finish["metrics_sha256"] != before[metrics_path].sha256:
        raise ValueError("checkpoint or metrics changed after execution finished")
    if (
        start["config_sha256"] != before[config_path].sha256
        or start["run_manifest_sha256"] != before[run_path].sha256
    ):
        raise ValueError("configuration or run manifest changed after execution started")
    metrics = strict_jsonl_loads(metrics_path.read_text(encoding="utf-8"))
    runtime = manifest.runtime
    if not metrics or runtime.global_step < 1 or runtime.trainable_tokens < 1:
        raise ValueError("receipt requires actual optimizer updates and action-token training")
    if start["stage"] == "sft":
        if [row["step"] for row in metrics] != list(range(1, runtime.global_step + 1)):
            raise ValueError("SFT log has missing or duplicate optimizer steps")
        if metrics[-1]["cumulative_action_tokens"] != runtime.trainable_tokens:
            raise ValueError("SFT log/checkpoint token counters disagree")
    elif start["stage"] == "rl":
        if (
            metrics[-1]["global_step"] != runtime.global_step
            or metrics[-1]["sampled_tokens"] != runtime.sampled_tokens
            or len(metrics) != (runtime.extra or {}).get("metric_records")
            or [row["metric_index"] for row in metrics] != list(range(len(metrics)))
            or sum(row["action_tokens"] for row in metrics) != runtime.trainable_tokens
        ):
            raise ValueError("RL log/checkpoint counters disagree")
    else:
        raise ValueError("unsupported execution stage")
    evaluation_record = None
    if evaluation is not None:
        eval_path, aggregate_path = evaluation / "run_manifest.json", evaluation / "metrics.json"
        before.update({path: fingerprint_file(path) for path in (eval_path, aggregate_path)})
        eval_manifest = strict_json_loads(eval_path.read_bytes())
        aggregate = strict_json_loads(aggregate_path.read_bytes())
        if eval_manifest["checkpoint"] != identity:
            raise ValueError("evaluation used a different checkpoint")
        if (
            aggregate["data_sha256"] != eval_manifest["data_sha256"]
            or aggregate["metrics"]["instances"] != eval_manifest["instance_count"]
            or aggregate["metrics"]["instances"] < 1
            or aggregate["metrics"]["errors"] != 0
        ):
            raise ValueError("evaluation is incomplete, inconsistent, or contains errors")
        evaluation_record = {
            "manifest_sha256": fingerprint_file(eval_path).sha256,
            "metrics_sha256": fingerprint_file(aggregate_path).sha256,
            "instances": aggregate["metrics"]["instances"],
            "verification_scope": "checkpoint and summary linkage; predictions not re-scored",
        }
    if any(fingerprint_file(path) != digest for path, digest in before.items()):
        raise ValueError("run artifacts changed during receipt verification")
    if checkpoint_identity(checkpoint) != identity:
        raise ValueError("checkpoint changed during receipt verification")
    return {
        "schema_version": "vapa-run-receipt-v1",
        "verification_scope": "file integrity and linkage; not independent execution attestation",
        "paper_reproduction": False,
        "stage": start["stage"],
        "status": finish["status"],
        "device": start["device"],
        "environment": start["environment"],
        "implementation_sha256": start["implementation_sha256"],
        "config_sha256": before[config_path].sha256,
        "run_manifest_sha256": before[run_path].sha256,
        "execution_journal_sha256": before[journal_path].sha256,
        "metrics_sha256": before[metrics_path].sha256,
        "checkpoint": identity,
        "optimizer_updates": runtime.global_step,
        "sampled_tokens": runtime.sampled_tokens,
        "trainable_tokens": runtime.trainable_tokens,
        "seed": runtime.seed,
        "evaluation": evaluation_record,
        "limitations": [
            "No claim of pretrained initialization, original study data, or paper-exact catalogs.",
            "Data permissions and anonymous-release suitability require separate review.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args(argv)
    destination = guard_artifact_write_path(args.output, content_kind="derived")
    # Receipts cannot overwrite any source artifact, even with an explicit output path.
    validate_output_paths([destination])
    receipt = build_run_receipt(args.run, args.checkpoint, evaluation=args.evaluation)
    atomic_write_text(destination, canonical_json_dumps(receipt) + "\n", overwrite=False)
    print(f"Run receipt verified: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
