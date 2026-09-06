#!/usr/bin/env python3
"""Source-checkout entrypoint for vapa evaluate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    from vapa.cli import main as cli_main

    return cli_main(["evaluate", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
