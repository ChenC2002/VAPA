#!/usr/bin/env python3
"""Analyze a strict binary-result panel under a frozen VAPA manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _main() -> int:
    from vapa.evaluation.analysis import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
