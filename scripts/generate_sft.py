#!/usr/bin/env python3
"""Generate strict public-reference SFT demonstrations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _main() -> int:
    from vapa.training.demonstrations import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
