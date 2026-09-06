from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from vapa.releases import (
    ReleaseValidationError,
    load_public_core_release,
    validate_public_core_release,
)
from vapa.releases.bundle import (
    NON_PAPER_EXACT_WARNING,
    PROMPT_MARKER,
    RELEASE_ID,
    RELEASE_VERSION,
)

BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "src" / "vapa" / "releases" / "public_core_v1"


def test_result_validation_allows_roundoff_but_rejects_changed_metrics_and_identity() -> None:
    import runpy

    validator = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/validate_release.py")
    )["_assert_result_matches"]
    validator({"value": [0.12345]}, {"value": [0.12345 + 1e-14]})
    for expected, actual in [
        ({"value": 1.0}, {"value": 1.01}),
        ({"sha256": "a"}, {"sha256": "b"}),
        ({"seed": 1}, {"seed": True}),
    ]:
        with pytest.raises(RuntimeError, match="stale demo result"):
            validator(expected, actual)


def _copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "public_core_v1"
    shutil.copytree(BUNDLE_ROOT, destination)
    return destination


def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    (root / "manifest.json").write_text(payload + "\n", encoding="utf-8", newline="\n")


def _refresh_file_record(root: Path, relative: str) -> None:
    manifest = _manifest(root)
    payload = (root / relative).read_bytes()
    records = manifest["files"]
    assert isinstance(records, list)
    record = next(item for item in records if item["path"] == relative)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    record["size_bytes"] = len(payload)
    _write_manifest(root, manifest)


def test_bundled_release_loads_through_package_resources() -> None:
    release = load_public_core_release()

    assert release.release_id == RELEASE_ID
    assert release.version == RELEASE_VERSION
    assert release.paper_exact is False
    assert release.files_checked == 7
    assert set(release.prompts) == {"system", "user"}
    assert set(release.schemas) == {"calculator", "task", "verifier"}
    assert all(prompt.startswith(PROMPT_MARKER + "\n") for prompt in release.prompts.values())
    assert release.descriptor["warning"] == NON_PAPER_EXACT_WARNING
    assert release.schema("task")["properties"]["paper_exact"]["const"] is False


def test_validation_report_is_serializable_and_valid() -> None:
    report = validate_public_core_release(BUNDLE_ROOT)

    assert report.ok()
    report.require_valid()
    assert report.to_dict() == {
        "valid": True,
        "release_id": RELEASE_ID,
        "version": RELEASE_VERSION,
        "files_checked": 7,
        "issues": [],
    }


def test_manifest_is_canonical_and_inventory_is_sorted() -> None:
    raw = (BUNDLE_ROOT / "manifest.json").read_bytes()
    value = json.loads(raw)
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()

    assert raw == canonical
    paths = [record["path"] for record in value["files"]]
    assert paths == sorted(paths)
    assert "manifest.json" not in paths


def test_hash_drift_is_rejected(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    prompt_path = root / "prompts" / "system_prompt.txt"
    payload = prompt_path.read_text(encoding="utf-8")
    prompt_path.write_text(payload.replace("bounded-memory", "bounded_memory"), encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="SHA-256 drift"):
        load_public_core_release(root)


def test_size_drift_is_rejected(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    manifest = _manifest(root)
    manifest["files"][0]["size_bytes"] += 1
    _write_manifest(root, manifest)

    with pytest.raises(ReleaseValidationError, match="size drift"):
        load_public_core_release(root)


def test_path_traversal_is_rejected_before_file_access(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    manifest = _manifest(root)
    manifest["files"][0]["path"] = "../README.md"
    _write_manifest(root, manifest)

    with pytest.raises(ReleaseValidationError, match="unsafe release path"):
        load_public_core_release(root)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links are unavailable")
def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes((root / "prompts" / "system_prompt.txt").read_bytes())
    prompt = root / "prompts" / "system_prompt.txt"
    prompt.unlink()
    prompt.symlink_to(outside)

    with pytest.raises(ReleaseValidationError, match="inside the release"):
        load_public_core_release(root)


def test_duplicate_json_keys_are_rejected_even_with_matching_hash(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    descriptor = root / "release.json"
    descriptor.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}\n',
        encoding="utf-8",
    )
    _refresh_file_record(root, "release.json")

    with pytest.raises(ReleaseValidationError, match="duplicate JSON key"):
        load_public_core_release(root)


def test_nonfinite_json_is_rejected_even_with_matching_hash(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    descriptor = root / "release.json"
    descriptor.write_text('{"schema_version":NaN}\n', encoding="utf-8")
    _refresh_file_record(root, "release.json")

    with pytest.raises(ReleaseValidationError, match="non-finite JSON number"):
        load_public_core_release(root)


def test_release_version_drift_is_rejected_with_refreshed_hash(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    descriptor_path = root / "release.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["version"] = "1.0.1"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    _refresh_file_record(root, "release.json")

    with pytest.raises(ReleaseValidationError, match="version drifted"):
        load_public_core_release(root)


def test_schema_drift_is_rejected_with_refreshed_hash(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    schema_path = root / "schemas" / "task_manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["$schema"] = "https://json-schema.org/draft/2019-09/schema"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    _refresh_file_record(root, "schemas/task_manifest.schema.json")

    with pytest.raises(ReleaseValidationError, match="schema draft drifted"):
        load_public_core_release(root)


def test_prompt_contract_drift_is_rejected_with_refreshed_hash(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    prompt_path = root / "prompts" / "user_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8").replace("{{cutoff}}", "cutoff omitted")
    prompt_path.write_text(prompt, encoding="utf-8")
    _refresh_file_record(root, "prompts/user_prompt.txt")

    with pytest.raises(ReleaseValidationError, match="missing placeholders"):
        load_public_core_release(root)


def test_invalid_report_stays_non_throwing_until_required(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    (root / "release.json").write_text("{}\n", encoding="utf-8")
    report = validate_public_core_release(root)

    assert not report.ok()
    assert report.files_checked == 0
    assert report.issues
    with pytest.raises(ReleaseValidationError):
        report.require_valid()


def test_unknown_prompt_and_schema_names_are_explicit() -> None:
    release = load_public_core_release()

    with pytest.raises(KeyError, match="unknown public-core prompt"):
        release.prompt("missing")
    with pytest.raises(KeyError, match="unknown public-core schema"):
        release.schema("missing")
