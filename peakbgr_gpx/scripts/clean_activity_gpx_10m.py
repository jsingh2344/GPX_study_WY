#!/usr/bin/env python3
"""Clean Strava activity GPX files with a 10 m minimum movement threshold."""

from __future__ import annotations

import csv
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "activities"
CLEAN_DIR = PROJECT_DIR / "activities_cleaned_10m"
SUMMARY_CSV = PROJECT_DIR / "products" / "activity_gpx_10m_cleaning_summary.csv"
MIN_MOVE_M = 10.0


def tag_name(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_lat_lon(point: ET.Element) -> tuple[float, float]:
    return float(point.attrib["lat"]), float(point.attrib["lon"])


def should_keep_points(points: list[ET.Element], min_move_m: float) -> set[int]:
    if not points:
        return set()

    keep_ids = {id(points[0])}
    prev_lat, prev_lon = point_lat_lon(points[0])

    for point in points[1:]:
        lat, lon = point_lat_lon(point)
        distance = haversine_m(prev_lat, prev_lon, lat, lon)
        if distance < min_move_m:
            continue
        keep_ids.add(id(point))
        prev_lat, prev_lon = lat, lon

    return keep_ids


def clean_tree(root: ET.Element, min_move_m: float) -> dict[str, int]:
    points = [elem for elem in root.iter() if tag_name(elem) == "trkpt"]
    keep_ids = should_keep_points(points, min_move_m)

    for parent in root.iter():
        children = list(parent)
        if not children:
            continue
        parent[:] = [
            child
            for child in children
            if tag_name(child) != "trkpt" or id(child) in keep_ids
        ]

    return {
        "original_points": len(points),
        "cleaned_points": len(keep_ids),
        "removed_points": len(points) - len(keep_ids),
    }


def clean_file(source_path: Path, output_path: Path, min_move_m: float) -> dict[str, object]:
    tree = ET.parse(source_path)
    root = copy.deepcopy(tree.getroot())
    stats = clean_tree(root, min_move_m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    kept_pct = stats["cleaned_points"] / stats["original_points"] * 100 if stats["original_points"] else 0.0
    return {
        "file": source_path.name,
        "min_move_m": min_move_m,
        **stats,
        "kept_pct": round(kept_pct, 3),
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    source_paths = sorted(SOURCE_DIR.glob("*.gpx"))
    if not source_paths:
        raise SystemExit(f"No GPX files found in {SOURCE_DIR}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in CLEAN_DIR.glob("*.gpx"):
        stale_path.unlink()

    rows = [clean_file(path, CLEAN_DIR / path.name, MIN_MOVE_M) for path in source_paths]
    write_summary(rows)

    original_points = sum(int(row["original_points"]) for row in rows)
    cleaned_points = sum(int(row["cleaned_points"]) for row in rows)
    removed_points = sum(int(row["removed_points"]) for row in rows)
    print(f"Cleaned {len(rows)} GPX files with a {MIN_MOVE_M:g} m minimum movement threshold")
    print(f"Wrote cleaned files to {CLEAN_DIR}")
    print(f"Original points: {original_points:,}")
    print(f"Cleaned points: {cleaned_points:,}")
    print(f"Removed points: {removed_points:,}")
    print(f"Wrote summary to {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
