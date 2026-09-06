from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from vapa.artifacts import (
    ArtifactContentKind,
    ArtifactFingerprint,
    CheckpointContractMismatchError,
    RunManifest,
    artifact_fingerprint,
    atomic_write_text,
    build_checkpoint_contract,
    canonical_json_bytes,
    canonical_json_dumps,
    file_sha256,
    fingerprint_file,
    guard_artifact_write_path,
    load_manifest,
    save_manifest,
    strict_json_loads,
    strict_jsonl_loads,
    validate_checkpoint_contract,
    validate_output_paths,
)


def manifest(**updates: object) -> RunManifest:
    values = {
        "config_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "verifier_sha256": "c" * 64,
        "model_revision": "0123456789abcdef",
        "seed": 17,
        "code_version": "vapa-0.1.0+deadbeef",
    }
    values.update(updates)
    return RunManifest(**values)


def test_canonical_json_is_sorted_compact_and_utf8() -> None:
    left = {"z": [3, {"β": "patient-safe"}], "a": 1}
    right = {"a": 1, "z": [3, {"β": "patient-safe"}]}

    expected = '{"a":1,"z":[3,{"β":"patient-safe"}]}'
    assert canonical_json_dumps(left) == expected
    assert canonical_json_bytes(left) == expected.encode("utf-8")
    assert artifact_fingerprint(left) == artifact_fingerprint(right)


@pytest.mark.parametrize(
    "payload",
    [
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e9999}',
        '{"outer": {"key": 1, "key": 2}}',
    ],
)
def test_strict_json_rejects_nonfinite_numbers_and_duplicate_keys(payload: str) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(payload)


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_jsonl_preserves_unicode_inside_records(separator: str, ending: str) -> None:
    record = {"text": f"before{separator}after"}
    encoded = canonical_json_dumps(record)
    assert strict_jsonl_loads(encoded + "\r\n" + encoded + ending) == [record, record]


@pytest.mark.parametrize("payload", ["\n", "{}\n\n", '{}\n{"x":1,"x":2}', "{}\nNaN"])
def test_jsonl_rejects_corrupt_records_with_physical_line_numbers(payload: str) -> None:
    line = 1 if payload == "\n" else 2
    with pytest.raises(ValueError, match=f"records.jsonl:{line}:"):
        strict_jsonl_loads(payload, source="records.jsonl")


def test_jsonl_empty_input_requires_explicit_permission() -> None:
    with pytest.raises(ValueError, match="empty JSONL"):
        strict_jsonl_loads("")
    assert strict_jsonl_loads("", allow_empty=True) == []


def test_jsonl_does_not_accept_a_bare_carriage_return_as_a_record_separator() -> None:
    with pytest.raises(ValueError, match="invalid strict JSON"):
        strict_jsonl_loads("{}\r{}")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or Infinity"):
        canonical_json_dumps({"nested": [value]})


def test_canonical_json_rejects_non_string_keys_and_non_json_objects() -> None:
    with pytest.raises(TypeError, match="non-string object key"):
        canonical_json_dumps({1: "ambiguous"})
    with pytest.raises(TypeError, match="non-JSON value"):
        canonical_json_dumps({"path": Path("artifact.json")})


def test_file_fingerprint_records_digest_and_exact_size(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc\x00")
    expected = hashlib.sha256(b"abc\x00").hexdigest()

    assert file_sha256(artifact) == expected
    assert fingerprint_file(artifact) == ArtifactFingerprint(
        sha256=expected,
        size_bytes=4,
    )
    assert fingerprint_file(artifact, chunk_size=1) == fingerprint_file(artifact)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_file_fingerprint_rejects_invalid_chunk_sizes(tmp_path: Path, chunk_size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fingerprint_file(tmp_path / "unused", chunk_size=chunk_size)


def test_atomic_publication_preserves_existing_file_and_removes_temporary(tmp_path: Path) -> None:
    destination = tmp_path / "output.json"
    atomic_write_text(destination, "original", overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_write_text(destination, "replacement", overwrite=False)
    assert destination.read_text() == "original"
    assert list(tmp_path.iterdir()) == [destination]
    atomic_write_text(destination, "replacement")
    assert destination.read_text() == "replacement"


def test_output_validation_rejects_hardlink_and_symlink_input_aliases(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("input")
    hardlink, symlink = tmp_path / "hard.json", tmp_path / "sym.json"
    hardlink.hardlink_to(source)
    symlink.symlink_to(source)
    for output in (source, hardlink, symlink):
        with pytest.raises(ValueError, match="aliases an input"):
            validate_output_paths([output], inputs=[source], overwrite=True)


def test_sensitive_output_detects_destination_repo_when_called_elsewhere(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="refusing to write"):
        guard_artifact_write_path(
            repository / "examples" / "patient.jsonl", content_kind="credentialed"
        )


def test_sensitive_output_is_allowed_outside_any_repository(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "approved-storage" / "results.jsonl"
    assert guard_artifact_write_path(destination, content_kind="credentialed") == destination


@pytest.mark.parametrize("relative", ["results/demo_results.json", "logs/demo_results.jsonl"])
def test_public_result_snapshots_cannot_receive_credentialed_outputs(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(ValueError, match="refusing to write"):
        guard_artifact_write_path(
            tmp_path / relative, content_kind="credentialed", repository_root=tmp_path
        )


def test_manifest_save_load_round_trip_uses_canonical_json(tmp_path: Path) -> None:
    expected = manifest()
    destination = tmp_path / "nested" / "run-manifest.json"

    assert save_manifest(expected, destination) == destination
    assert load_manifest(destination) == expected
    assert destination.read_text(encoding="utf-8") == (
        canonical_json_dumps(expected.to_dict()) + "\n"
    )


def test_manifest_loader_rejects_duplicate_unknown_and_unsupported_fields(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    valid = manifest().to_dict()
    duplicate = canonical_json_dumps(valid).replace(
        '"seed":17',
        '"seed":17,"seed":18',
    )
    destination.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key 'seed'"):
        load_manifest(destination)

    unknown = dict(valid, accidental_field="not allowed")
    destination.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields: accidental_field"):
        load_manifest(destination)

    invalid_version = dict(valid, schema_version="2.0")
    destination.write_text(json.dumps(invalid_version), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported run manifest schema_version"):
        load_manifest(destination)


def test_manifest_validates_hashes_and_seed_types() -> None:
    with pytest.raises(ValueError, match="config_sha256"):
        manifest(config_sha256="A" * 64)
    with pytest.raises(TypeError, match="seed must be an integer"):
        manifest(seed=True)
    with pytest.raises(ValueError, match="seed cannot be negative"):
        manifest(seed=-1)


def test_checkpoint_contract_fingerprints_named_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_bytes(b'{"id":"derived-1"}\n')
    verifier = ArtifactFingerprint(sha256="d" * 64, size_bytes=123)

    contract = build_checkpoint_contract(
        manifest(),
        artifacts={"dataset": dataset, "verifier": verifier},
    )

    assert contract["run_manifest"] == manifest().to_dict()
    assert contract["artifacts"]["dataset"] == fingerprint_file(dataset).to_dict()
    assert contract["artifacts"]["verifier"] == verifier.to_dict()
    validate_checkpoint_contract(contract, deepcopy(contract))


def test_checkpoint_contract_reports_each_field_level_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"version one")
    expected = build_checkpoint_contract(manifest(), artifacts={"dataset": artifact})
    stored = deepcopy(expected)
    stored["run_manifest"]["seed"] = 99
    stored["artifacts"]["dataset"]["sha256"] = "e" * 64
    del stored["run_manifest"]["model_revision"]
    stored["unexpected"] = True

    with pytest.raises(CheckpointContractMismatchError) as caught:
        validate_checkpoint_contract(stored, expected)

    message = str(caught.value)
    assert "run_manifest.seed: expected 17, got 99" in message
    assert "artifacts.dataset.sha256" in message
    assert "run_manifest.model_revision: missing field" in message
    assert "unexpected: unexpected field" in message
    assert len(caught.value.mismatches) == 4


@pytest.mark.parametrize("directory", ["examples", "configs", "src", "docs", "tests", "."])
@pytest.mark.parametrize(
    "kind",
    [ArtifactContentKind.CREDENTIALED, ArtifactContentKind.RAW_RECORDS],
)
def test_path_guard_rejects_sensitive_content_in_tracked_public_locations(
    tmp_path: Path,
    directory: str,
    kind: ArtifactContentKind,
) -> None:
    repository = tmp_path / "repo"
    destination = repository / "runs" / ".." / directory / "patient-record.json"

    with pytest.raises(ValueError, match="tracked public location"):
        guard_artifact_write_path(
            destination,
            content_kind=kind,
            repository_root=repository,
        )


def test_path_guard_allows_safe_destinations_and_rejects_unknown_classification(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    raw_destination = repository / "data" / "raw" / "records.parquet"
    public_example = repository / "examples" / "synthetic.json"
    external = tmp_path / "private-runs" / "records.parquet"

    assert (
        guard_artifact_write_path(
            raw_destination,
            content_kind="credentialed",
            repository_root=repository,
        )
        == raw_destination.resolve()
    )
    assert (
        guard_artifact_write_path(
            public_example,
            content_kind="derived",
            repository_root=repository,
        )
        == public_example.resolve()
    )
    assert (
        guard_artifact_write_path(
            external,
            content_kind="raw_records",
            repository_root=repository,
        )
        == external.resolve()
    )
    with pytest.raises(ValueError, match="unknown content_kind"):
        guard_artifact_write_path(
            raw_destination,
            content_kind="probably-safe",
            repository_root=repository,
        )
