#!/usr/bin/env python3
"""Build a metadata-minimized review archive while preserving result disclosures."""

from __future__ import annotations

import argparse
import re
import stat
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = frozenset({"README.md", "pyproject.toml", ".gitignore"})
PUBLIC_DIRECTORIES = frozenset({".github", "configs", "examples", "scripts", "src", "tests"})
SNAPSHOTS = frozenset(
    f"{directory}/{name}_results.{extension}"
    for directory, extension in (("results", "json"), ("logs", "jsonl"))
    for name in ("demo", "paper", "training")
)
EXCLUDED_PARTS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", "checkpoints", "build", "dist"}
)
TEXT_SUFFIXES = frozenset(
    {".py", ".json", ".jsonl", ".csv", ".md", ".toml", ".txt", ".yml", ".yaml"}
)
IDENTIFIERS = {
    "local user path": re.compile(r"(?:/(?:Users|home)/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)"),
    "private temporary path": re.compile(r"/(?:private/(?:tmp|var)/|var/folders/)[^\s]+"),
    "email address": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "account URL": re.compile(
        r"(?:https?://|git@)(?:github\.com|gitlab\.com|bitbucket\.org)[/:][\w.-]+/[\w.-]+",
        re.IGNORECASE,
    ),
    "private key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
}


def review_files(root: Path, *, deny_text: tuple[str, ...] = ()) -> dict[Path, bytes]:
    """Select only release text files; reject identifying content and symlinks."""
    if any(not item.strip() for item in deny_text):
        raise ValueError("deny-text values must not be empty")
    selected = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if EXCLUDED_PARTS.intersection(relative.parts) or any(
            part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        in_scope = (
            relative.as_posix() in ROOT_FILES | SNAPSHOTS or relative.parts[0] in PUBLIC_DIRECTORIES
        )
        if not in_scope:
            continue
        if path.is_symlink():
            raise ValueError(f"review archive cannot include symlinks: {relative}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"review archive requires regular files: {relative}")
        if path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        if relative.as_posix() not in ROOT_FILES and path.suffix not in TEXT_SUFFIXES:
            raise ValueError(f"unexpected release file type: {relative}")
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        searchable = relative.as_posix() + "\n" + text
        for kind, pattern in IDENTIFIERS.items():
            if pattern.search(searchable):
                raise ValueError(f"potential {kind} in {relative}; review before publishing")
        if any(value.casefold() in searchable.casefold() for value in deny_text):
            raise ValueError(f"denied identifier in {relative}; review before publishing")
        selected[path] = payload
    inventory = {path.relative_to(root).as_posix() for path in selected}
    missing = sorted((ROOT_FILES | SNAPSHOTS) - inventory)
    if missing:
        raise ValueError(f"review package is missing required files: {', '.join(missing)}")
    return selected


def build_archive(root: Path, output: Path, *, deny_text: tuple[str, ...] = ()) -> int:
    root = root.resolve()
    paths = review_files(root, deny_text=deny_text)
    output = output.absolute()
    if output.is_symlink() or output.exists():
        raise FileExistsError("archive output already exists; choose a new path")
    if output.suffix != ".zip":
        raise ValueError("archive output must end in .zip")
    if output.resolve().is_relative_to(root):
        raise ValueError("keep review archives outside the source tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an existing archive even if another process races.
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, payload in paths.items():
                entry = zipfile.ZipInfo(
                    "VAPA/" + path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0)
                )
                entry.create_system = 3
                entry.external_attr = (stat.S_IFREG | 0o644) << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.extra = b""
                entry.comment = b""
                archive.writestr(entry, payload)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or len(archive.infolist()) != len(paths):
            raise RuntimeError("review archive failed integrity verification")
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deny-text", action="append", default=[])
    args = parser.parse_args()
    count = build_archive(ROOT, args.output, deny_text=tuple(args.deny_text))
    print(f"review archive verified: {count} files; no history; normalized file timestamps")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
