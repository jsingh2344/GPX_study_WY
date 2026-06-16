#!/usr/bin/env python3
"""Write GPX copies with duplicate-timestamp points removed."""

from __future__ import annotations

import argparse
import csv
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GPX_DIR = PROJECT_DIR / "gpx"
DEFAULT_CLEAN_DIR = PROJECT_DIR / "gpx_cleaned"
DEFAULT_PRODUCTS_DIR = PROJECT_DIR / "products"


def tag_name(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def child_text(elem: ET.Element, child_name: str) -> Optional[str]:
    for child in elem:
        if tag_name(child) == child_name and child.text:
            return child.text.strip()
    return None


def remove_duplicate_times(root: ET.Element) -> dict[str, int]:
    seen_times: set[str] = set()
    stats = {
        "original_points": 0,
        "cleaned_points": 0,
        "removed_duplicate_time_points": 0,
        "untimed_points_kept": 0,
    }

    for parent in root.iter():
        children = list(parent)
        if not children:
            continue

        kept_children: list[ET.Element] = []
        changed = False
        for child in children:
            if tag_name(child) not in {"trkpt", "rtept"}:
                kept_children.append(child)
                continue

            stats["original_points"] += 1
            timestamp = child_text(child, "time")
            if timestamp is None:
                stats["untimed_points_kept"] += 1
                stats["cleaned_points"] += 1
                kept_children.append(child)
            elif timestamp in seen_times:
                stats["removed_duplicate_time_points"] += 1
                changed = True
            else:
                seen_times.add(timestamp)
                stats["cleaned_points"] += 1
                kept_children.append(child)

        if changed:
            parent[:] = kept_children

    return stats


def clean_file(src: Path, dst: Path) -> dict[str, object]:
    tree = ET.parse(src)
    root = tree.getroot()
    clean_root = copy.deepcopy(root)
    stats = remove_duplicate_times(clean_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(clean_root).write(dst, encoding="utf-8", xml_declaration=True)
    return {
        "file": src.name,
        "original_points": stats["original_points"],
        "cleaned_points": stats["cleaned_points"],
        "removed_duplicate_time_points": stats["removed_duplicate_time_points"],
        "untimed_points_kept": stats["untimed_points_kept"],
    }


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpx-dir", default=str(DEFAULT_GPX_DIR), help="Directory containing source GPX files")
    parser.add_argument("--clean-dir", default=str(DEFAULT_CLEAN_DIR), help="Directory for cleaned GPX copies")
    parser.add_argument("--summary-csv", default=str(DEFAULT_PRODUCTS_DIR / "gpx_duplicate_timestamp_cleaning_summary.csv"))
    args = parser.parse_args()

    gpx_dir = Path(args.gpx_dir).expanduser().resolve()
    clean_dir = Path(args.clean_dir).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()

    rows = []
    for src in sorted(gpx_dir.glob("*.gpx")):
        rows.append(clean_file(src, clean_dir / src.name))

    if not rows:
        raise SystemExit(f"No GPX files found in {gpx_dir}")

    write_summary(rows, summary_csv)
    total_original = sum(int(row["original_points"]) for row in rows)
    total_cleaned = sum(int(row["cleaned_points"]) for row in rows)
    total_removed = sum(int(row["removed_duplicate_time_points"]) for row in rows)
    print(f"Wrote {len(rows)} cleaned GPX files to {clean_dir}")
    print(f"Original points: {total_original:,}")
    print(f"Cleaned points: {total_cleaned:,}")
    print(f"Removed duplicate-timestamp points: {total_removed:,}")
    print(f"Wrote summary to {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
