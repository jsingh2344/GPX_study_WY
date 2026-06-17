#!/usr/bin/env python3
"""Copy cleaned Strava activity GPX files that fall in Wyoming."""

from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "activities_cleaned_10m"
WYOMING_DIR = PROJECT_DIR / "activities_wyoming_10m"
SUMMARY_CSV = PROJECT_DIR / "products" / "activity_wyoming_filter_summary.csv"

WY_SOUTH = 40.994
WY_NORTH = 45.005
WY_WEST = -111.056
WY_EAST = -104.052
MIN_WYOMING_POINT_FRACTION = 0.5


def tag_name(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def point_counts(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    total = 0
    wyoming = 0
    for elem in root.iter():
        if tag_name(elem) != "trkpt":
            continue
        total += 1
        lat = float(elem.attrib["lat"])
        lon = float(elem.attrib["lon"])
        if WY_SOUTH <= lat <= WY_NORTH and WY_WEST <= lon <= WY_EAST:
            wyoming += 1
    return total, wyoming


def main() -> int:
    source_paths = sorted(SOURCE_DIR.glob("*.gpx"))
    if not source_paths:
        raise SystemExit(f"No GPX files found in {SOURCE_DIR}")

    WYOMING_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in WYOMING_DIR.glob("*.gpx"):
        stale_path.unlink()

    rows = []
    copied = 0
    for path in source_paths:
        total, wyoming = point_counts(path)
        fraction = wyoming / total if total else 0.0
        keep = fraction >= MIN_WYOMING_POINT_FRACTION
        if keep:
            shutil.copy2(path, WYOMING_DIR / path.name)
            copied += 1
        rows.append(
            {
                "file": path.name,
                "total_points": total,
                "wyoming_points": wyoming,
                "wyoming_point_fraction": round(fraction, 6),
                "copied_to_wyoming_folder": keep,
            }
        )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned {len(source_paths)} cleaned activity GPX files")
    print(f"Copied {copied} Wyoming activity GPX files to {WYOMING_DIR}")
    print(f"Wrote summary to {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
