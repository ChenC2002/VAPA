#!/usr/bin/env python3
"""Analyze factorial, horizon, and named-profile experiment metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _main() -> int:
    from vapa.evaluation.reporting import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
