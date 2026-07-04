#!/usr/bin/env python3
"""Prepare Neural Latents Benchmark H5 files for LaDyS."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ladys.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["prepare-nlb", *sys.argv[1:]]))
