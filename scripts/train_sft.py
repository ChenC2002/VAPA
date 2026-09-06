#!/usr/bin/env python3
"""Run or validate manifest-bound action-token supervised fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _main() -> int:
    from vapa.training.sft_train import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
