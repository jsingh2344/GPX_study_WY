#!/usr/bin/env python3
"""Clean Strava GPX files with a 10 m minimum movement threshold."""

from __future__ import annotations

import csv
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "activities_wyoming_10m"
CLEAN_DIR = ROOT / "gpx_cleaned_10m"
SUMMARY = ROOT / "data" / "cleaning_10m_summary.csv"
MIN_MOVE_M = 10.0


def tag(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def distance_m(a: ET.Element, b: ET.Element) -> float:
    lat1, lon1 = float(a.attrib["lat"]), float(a.attrib["lon"])
    lat2, lon2 = float(b.attrib["lat"]), float(b.attrib["lon"])
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    x = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def kept_point_ids(points: list[ET.Element]) -> set[int]:
    if not points:
        return set()
    kept = {id(points[0])}
    previous = points[0]
    for point in points[1:]:
        if distance_m(previous, point) < MIN_MOVE_M:
            continue
        kept.add(id(point))
        previous = point
    return kept


def clean_file(source: Path, destination: Path) -> dict[str, object]:
    root = copy.deepcopy(ET.parse(source).getroot())
    points = [elem for elem in root.iter() if tag(elem) == "trkpt"]
    kept = kept_point_ids(points)
    for parent in root.iter():
        parent[:] = [child for child in list(parent) if tag(child) != "trkpt" or id(child) in kept]
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    kept_pct = len(kept) / len(points) * 100 if points else 0.0
    return {
        "file": source.name,
        "original_points": len(points),
        "cleaned_points": len(kept),
        "removed_points": len(points) - len(kept),
        "kept_pct": round(kept_pct, 3),
    }


def main() -> int:
    sources = sorted(SOURCE_DIR.glob("*.gpx"))
    if not sources:
        raise SystemExit(f"No GPX files found in {SOURCE_DIR}")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CLEAN_DIR.glob("*.gpx"):
        stale.unlink()
    rows = [clean_file(path, CLEAN_DIR / path.name) for path in sources]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Cleaned {len(rows)} GPX files to {CLEAN_DIR}")
    print(f"Original points: {sum(int(row['original_points']) for row in rows):,}")
    print(f"Cleaned points: {sum(int(row['cleaned_points']) for row in rows):,}")
    print(f"Wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
