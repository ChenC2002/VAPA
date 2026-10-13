from __future__ import annotations

import json

import pytest

from vapa.artifacts import artifact_fingerprint, atomic_write_text, canonical_json_dumps
from vapa.training.checkpoint import (
    CheckpointContract,
    JsonStateStore,
    RuntimeState,
    read_manifest,
    save_checkpoint,
)
from vapa.training.evidence import build_run_receipt, finish_execution, start_execution


class FixtureState:
    fingerprint = "a" * 64

    def state_dict(self):
        return {"test_fixture": True}


def make_run(tmp_path):
    # Pure metadata fixture: deliberately NOT neural-model execution evidence.
    output, checkpoint = tmp_path / "run", tmp_path / "run" / "checkpoints" / "step-1"
    config = {"seed": 7}
    run = {"schema_version": "test-fixture", "run_id": "fixture"}
    atomic_write_text(output / "run_manifest.json", canonical_json_dumps(run))
    execution = start_execution(output, config=config, stage="sft", device="fixture")
    metrics = {"step": 1, "loss": 0.2, "cumulative_action_tokens": 3}
    atomic_write_text(output / "metrics.jsonl", canonical_json_dumps(metrics) + "\n")
    state = FixtureState()
    save_checkpoint(
        checkpoint,
        contract=CheckpointContract(
            "fixture",
            artifact_fingerprint(run),
            artifact_fingerprint(config),
            state.fingerprint,
            state.fingerprint,
            state.fingerprint,
            "fixture",
            "none",
            JsonStateStore().format_name,
        ),
        runtime=RuntimeState(1, 3, 3, 7),
        model=state,
        reference=state,
        tokenizer=state,
        optimizer=state,
        store=JsonStateStore(),
    )
    finish_execution(output, execution, status="complete", checkpoint=checkpoint)
    return output, checkpoint


def test_receipt_is_metadata_only_and_does_not_claim_real_model_execution(tmp_path):
    output, checkpoint = make_run(tmp_path)
    receipt = build_run_receipt(output, checkpoint)
    assert receipt["optimizer_updates"] == 1
    assert receipt["seed"] == 7
    assert receipt["paper_reproduction"] is False
    assert "not independent execution" in receipt["verification_scope"]
    assert str(tmp_path) not in json.dumps(receipt)


@pytest.mark.parametrize("target", ["metrics", "weights", "config", "journal", "unfinished"])
def test_receipt_rejects_broken_or_partial_evidence(tmp_path, target):
    output, checkpoint = make_run(tmp_path)
    if target == "metrics":
        (output / "metrics.jsonl").write_text('{"step":99}\n')
    elif target == "weights":
        manifest = read_manifest(checkpoint)
        (checkpoint / manifest.files["model"].path).write_text('{"tampered":true}')
    elif target == "config":
        (output / "resolved_config.json").write_text('{"seed":8}')
    elif target == "journal":
        path = output / "execution.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["device"] = "modified"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        start_execution(output, config={"seed": 7}, stage="sft", device="fixture")
    with pytest.raises(ValueError):
        build_run_receipt(output, checkpoint)


def test_receipt_evaluation_must_use_the_verified_checkpoint(tmp_path):
    output, checkpoint = make_run(tmp_path)
    evaluation = tmp_path / "evaluation"
    atomic_write_text(
        evaluation / "run_manifest.json",
        canonical_json_dumps(
            {
                "checkpoint": {"kind": "none"},
                "data_sha256": "b" * 64,
                "instance_count": 1,
            }
        ),
    )
    atomic_write_text(
        evaluation / "metrics.json",
        canonical_json_dumps(
            {
                "data_sha256": "b" * 64,
                "metrics": {"instances": 1, "errors": 0},
            }
        ),
    )
    with pytest.raises(ValueError, match="different checkpoint"):
        build_run_receipt(output, checkpoint, evaluation=evaluation)
