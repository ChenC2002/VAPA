from __future__ import annotations

import os
import runpy
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGER = runpy.run_path(str(ROOT / "scripts/package_review.py"))


def fixture(root: Path) -> Path:
    root.mkdir()
    for name in PACKAGER["ROOT_FILES"] | PACKAGER["SNAPSHOTS"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("review fixture\n", encoding="utf-8")
    code = root / "src/vapa/data/adapter.py"
    code.parent.mkdir(parents=True)
    code.write_text("# public implementation\n", encoding="utf-8")
    return root


def test_archive_has_neutral_metadata_and_excludes_private_artifacts(tmp_path: Path) -> None:
    root = fixture(tmp_path / "source")
    for name in (".git/config", "runs/model.pt", "src/vapa/__pycache__/code.pyc", ".DS_Store"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private local artifact")
    output = tmp_path / "review.zip"
    count = PACKAGER["build_archive"](root, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == count
        assert "VAPA/src/vapa/data/adapter.py" in archive.namelist()
        assert all(name.startswith("VAPA/") for name in archive.namelist())
        assert all(".git/" not in name and "runs/" not in name for name in archive.namelist())
        for entry in archive.infolist():
            assert entry.date_time == (1980, 1, 1, 0, 0, 0)
            assert entry.extra == entry.comment == b""
            assert entry.external_attr >> 16 == 0o100644
    os.utime(root / "README.md", (1_500_000_000, 1_500_000_000))
    repeated = tmp_path / "repeated.zip"
    PACKAGER["build_archive"](root, repeated)
    assert repeated.read_bytes() == output.read_bytes()


@pytest.mark.parametrize("kind", ["email", "user_path", "account", "extra_name"])
def test_archive_rejects_identifiers_without_publishing(tmp_path: Path, kind: str) -> None:
    root = fixture(tmp_path / "source")
    # Construct test identifiers at runtime so test source itself remains uploadable.
    examples = {
        "email": "researcher" + "@" + "example.org",
        "user_path": "/" + "Users/" + "reviewer/project",
        "account": "https://" + "github.com/" + "personal-account/project",
        "extra_name": "Example Researcher",
    }
    (root / "README.md").write_text(examples[kind], encoding="utf-8")
    output = tmp_path / "review.zip"
    with pytest.raises(ValueError):
        PACKAGER["build_archive"](root, output, deny_text=("Example Researcher",))
    assert not output.exists()


def test_archive_rejects_symlinks_and_preserves_existing_output(tmp_path: Path) -> None:
    root = fixture(tmp_path / "source")
    link = root / "examples/linked.json"
    link.parent.mkdir()
    link.symlink_to(root / "README.md")
    with pytest.raises(ValueError, match="symlinks"):
        PACKAGER["build_archive"](root, tmp_path / "review.zip")
    link.unlink()
    output = tmp_path / "review.zip"
    output.write_bytes(b"keep this archive")
    with pytest.raises(FileExistsError):
        PACKAGER["build_archive"](root, output)
    assert output.read_bytes() == b"keep this archive"


def test_packager_preserves_results_disclosures() -> None:
    files = PACKAGER["review_files"](ROOT)
    assert b'"independently_reproduced": false' in files[ROOT / "results/paper_results.json"]
    assert b'"paper_reproduction": false' in files[ROOT / "results/training_results.json"]
    assert b'"kind": "synthetic_run"' in files[ROOT / "results/demo_results.json"]
