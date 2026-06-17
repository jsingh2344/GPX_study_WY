#!/usr/bin/env python3
"""Run the Strava GPX workflow in the same order as the Peakbagger workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STEPS = [
    "clean_gpx_10m.py",
    "create_activity_map.py",
    "create_slope_speed_dotplot.py",
    "train_ridge_speed_regression.py",
]


def main() -> int:
    for script in STEPS:
        path = SCRIPTS / script
        print(f"\n== {script} ==", flush=True)
        subprocess.run([sys.executable, str(path)], check=True)
    print("\nStrava workflow complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
