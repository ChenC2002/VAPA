from __future__ import annotations

import json
import math

import pytest

from vapa.data import (
    DataValidationError,
    IntegrityError,
    SplitFractions,
    assign_patient_split,
    load_episode_objects,
    load_episodes,
    load_events,
    load_json,
    patient_disjoint_hash_split,
    sha256_bytes,
    sha256_json,
    validate_patient_disjoint,
    verify_sha256_manifest,
    write_sha256_manifest,
)
from vapa.data.episodes import episode_from_record, episode_to_record
from vapa.demo import tiny_episode
from vapa.evaluation import (
    MetricInputError,
    any_reference_evidence,
    auprc,
    auroc,
    cost_normalized_success,
    grounded_exact_match,
    macro_cost_normalized_success,
    macro_task_average,
    macro_task_success,
    task_success,
    verified_accuracy,
    verified_match,
)


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"patient_id": None},
        {"instruction": None},
        {"family": False},
        {"requested_fields": "hba1c"},
        {"reference_evidence": "event"},
        {"metadata": []},
        {"events": [None]},
        {"episode_id": "conflicting-alias"},
    ],
)
def test_episode_contract_rejects_silent_coercions(updates: dict) -> None:
    with pytest.raises(DataValidationError):
        episode_from_record({**episode_to_record(tiny_episode()), **updates})


@pytest.mark.parametrize(
    "updates",
    [
        {"pointer": None},
        {"patient_id": True},
        {"field": None},
        {"timestamp": None},
        {"value": []},
        {"unit": 123},
        {"unexpected": "ignored-before"},
    ],
)
def test_episode_event_contract_rejects_malformed_records(updates: dict) -> None:
    record = episode_to_record(tiny_episode())
    record["events"][0].update(updates)
    with pytest.raises(DataValidationError):
        episode_from_record(record)


def test_strict_episode_and_event_loading(tmp_path):
    episodes_path = tmp_path / "episodes.json"
    episodes_path.write_text(
        json.dumps({"episodes": [{"episode_id": "ep-1", "patient_id": "p-1"}]}),
        encoding="utf-8",
    )
    assert load_episodes(episodes_path)[0]["episode_id"] == "ep-1"

    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        '{"pointer":"e#1","value":4.0}\n{"pointer":"e#2","value":5.0}\n',
        encoding="utf-8",
    )
    assert [event["pointer"] for event in load_events(events_path, id_field="pointer")] == [
        "e#1",
        "e#2",
    ]


def test_episode_object_loader_accepts_the_documented_episode_id_alias(tmp_path):
    path = tmp_path / "episodes.json"
    record = {
        "episode_id": "alias-1",
        "patient_id": "synthetic-patient",
        "instruction": "Return the value.",
        "family": "synthetic",
        "cutoff": "2026-01-01T00:00:00Z",
        "gold_answer": "ok",
        "events": [],
    }
    path.write_text(json.dumps([record]), encoding="utf-8")
    assert load_episode_objects(path)[0].task.instance_id == "alias-1"

    path.write_text(json.dumps([record, record]), encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicate episode ID"):
        load_episode_objects(path)


def test_loader_rejects_duplicate_ids_keys_and_nonfinite_values(tmp_path):
    duplicate_ids = tmp_path / "duplicate.json"
    duplicate_ids.write_text('[{"episode_id":"same"},{"episode_id":"same"}]', encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicate record ID"):
        load_episodes(duplicate_ids)

    duplicate_keys = tmp_path / "duplicate-keys.json"
    duplicate_keys.write_text('{"episode_id":"a","episode_id":"b"}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicate JSON object key"):
        load_json(duplicate_keys)

    for name, token in [("nan", "NaN"), ("infinity", "1e999")]:
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"episode_id":"ep", "value":{token}}}', encoding="utf-8")
        with pytest.raises(DataValidationError, match="non-finite"):
            load_json(path)

    blank_line = tmp_path / "blank.jsonl"
    blank_line.write_text('{"event_id":"a"}\n\n{"event_id":"b"}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="blank JSONL line"):
        load_events(blank_line)


def test_sha256_manifest_is_stable_and_detects_tampering(tmp_path):
    data_path = tmp_path / "events.jsonl"
    data_path.write_bytes(b"abc")
    expected_digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_bytes(b"abc") == expected_digest
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})

    manifest_path = tmp_path / "MANIFEST.json"
    manifest = write_sha256_manifest(manifest_path, [data_path], root=tmp_path)
    assert manifest["files"][0]["path"] == "events.jsonl"
    assert verify_sha256_manifest(manifest_path)

    # Keep the same byte length so verification reaches the digest check.
    data_path.write_bytes(b"abd")
    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        verify_sha256_manifest(manifest_path)


def test_patient_hash_split_is_deterministic_and_has_no_leakage():
    records = [
        {"episode_id": f"ep-{patient}-{visit}", "patient_id": f"p-{patient}"}
        for patient in range(40)
        for visit in range(2)
    ]
    fractions = SplitFractions(train=0.7, validation=0.1, test=0.2)
    first = patient_disjoint_hash_split(records, fractions=fractions, seed=17)
    second = patient_disjoint_hash_split(reversed(records), fractions=fractions, seed=17)

    def owners(splits):
        return {
            record["patient_id"]: split_name
            for split_name, split_records in splits.items()
            for record in split_records
        }

    assert owners(first) == owners(second)
    assert assign_patient_split("p-4", fractions=fractions, seed=17) == owners(first)["p-4"]
    validate_patient_disjoint(first, expected_patient_ids={f"p-{index}" for index in range(40)})
    for patient in {record["patient_id"] for record in records}:
        assigned_records = [
            record
            for split_records in first.values()
            for record in split_records
            if record["patient_id"] == patient
        ]
        assert len(assigned_records) == 2


def test_patient_leakage_validator_is_exact():
    leaking = {
        "train": [{"patient_id": "p-1"}],
        "validation": [{"patient_id": "p-1"}],
        "test": [],
    }
    with pytest.raises(DataValidationError, match="leaks across splits"):
        validate_patient_disjoint(leaking)
    with pytest.raises(DataValidationError, match="coverage mismatch"):
        validate_patient_disjoint(
            {"train": [{"patient_id": "p-1"}]},
            expected_patient_ids=["p-1", "p-2"],
        )


def test_grounded_exact_match_requires_explicit_grounding_semantics():
    assert grounded_exact_match("5.2", "5.2", True) == 1.0
    assert grounded_exact_match("5.2", "5.2", False) == 0.0
    assert grounded_exact_match("wrong", "5.2", True) == 0.0
    assert (
        grounded_exact_match(
            "present",
            "present",
            predicted_evidence=["e#2", "e#9"],
            reference_evidence=["e#2"],
            grounding_predicate=any_reference_evidence,
        )
        == 1.0
    )
    with pytest.raises(MetricInputError, match="grounded flag"):
        grounded_exact_match("present", "present")


def test_verified_accuracy_uses_maximum_absolute_or_relative_tolerance():
    assert verified_match(104.9, 100.0, abs_tolerance=2.0, relative_tolerance=0.05)
    assert not verified_match(105.1, 100.0, abs_tolerance=2.0, relative_tolerance=0.05)
    assert verified_match(0.4, 0.0, abs_tolerance=0.5, relative_tolerance=0.05)
    assert not verified_match(float("nan"), 1.0)
    assert verified_accuracy(
        [104.9, 105.1, None],
        [100.0, 100.0, 0.0],
        abs_tolerance=[2.0, 2.0, 0.5],
    ) == pytest.approx(100.0 / 3.0)


def test_task_and_cost_normalized_success_follow_paper_formula():
    successes = [1, 0, 1]
    costs = [0.0, 9.0, 5.0]
    maxima = [10.0, 10.0, 10.0]
    assert task_success(successes) == pytest.approx(200.0 / 3.0)
    assert cost_normalized_success(successes, costs, maxima) == 50.0
    assert cost_normalized_success([1], [0], [10]) == task_success([1])
    with pytest.raises(MetricInputError, match="exceeds"):
        cost_normalized_success([1], [11], [10])


def test_macro_task_weighting_does_not_overweight_large_task_families():
    successes = [1, 1, 1, 0]
    task_ids = ["frequent", "frequent", "frequent", "rare"]
    assert task_success(successes) == 75.0
    assert macro_task_success(successes, task_ids) == 50.0
    assert macro_task_average([1.0, 1.0, 1.0, 0.0], task_ids) == 0.5
    assert (
        macro_cost_normalized_success(
            successes,
            costs=[0, 0, 0, 0],
            max_costs=[10, 10, 10, 10],
            task_ids=task_ids,
        )
        == 50.0
    )


def test_auroc_and_auprc_group_tied_scores():
    labels = [0, 1, 0, 1]
    tied_scores = [0.5, 0.5, 0.5, 0.5]
    assert auroc(labels, tied_scores) == 0.5
    assert auprc(labels, tied_scores) == 0.5

    reversed_order = list(reversed(labels))
    assert auroc(reversed_order, tied_scores) == auroc(labels, tied_scores)
    assert auprc(reversed_order, tied_scores) == auprc(labels, tied_scores)
    assert auroc([0, 1], [0.1, 0.9]) == 1.0
    assert auprc([0, 1], [0.1, 0.9]) == 1.0
    with pytest.raises(MetricInputError, match="both positive and negative"):
        auroc([1, 1], [0.1, 0.2])
    with pytest.raises(MetricInputError, match="at least one positive"):
        auprc([0, 0], [0.1, 0.2])


def test_metric_inputs_reject_nonfinite_values():
    with pytest.raises(MetricInputError, match="finite"):
        auroc([0, 1], [0.0, math.inf])
    with pytest.raises(MetricInputError, match="finite"):
        task_success([1.0, math.nan])
