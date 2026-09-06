"""Content identities for the installed VAPA implementation."""

from __future__ import annotations

from pathlib import Path

from vapa.artifacts import artifact_fingerprint, fingerprint_file


def package_code_fingerprint(package_root: str | Path | None = None) -> str:
    """Hash every shipped Python source file by package-relative path and bytes."""

    root = (
        Path(__file__).resolve().parent
        if package_root is None
        else Path(package_root).expanduser().resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(f"VAPA package root does not exist: {root}")
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"VAPA source must be a regular non-symlink file: {path}")
        fingerprint = fingerprint_file(path)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": fingerprint.sha256,
                "size_bytes": fingerprint.size_bytes,
            }
        )
    if not records:
        raise ValueError("VAPA package contains no Python source files")
    return artifact_fingerprint(records)


__all__ = ["package_code_fingerprint"]
