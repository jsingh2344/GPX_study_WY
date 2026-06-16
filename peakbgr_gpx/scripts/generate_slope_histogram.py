#!/usr/bin/env python3
"""Generate a 40 m segment slope histogram from Peakbagger GPX tracks."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GPX_DIR = PROJECT_DIR / "gpx"
DEFAULT_PRODUCTS_DIR = PROJECT_DIR / "products"
SEGMENT_LENGTH_M = 40.0
FT_PER_M = 3.28084


Point = tuple[float, float, Optional[float]]
Sample = tuple[float, float, Optional[float], float]


def tag_name(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def read_gpx(path: Path) -> list[Point]:
    root = ET.parse(path).getroot()
    points: list[Point] = []
    for elem in root.iter():
        if tag_name(elem) not in {"trkpt", "rtept"}:
            continue
        lat_text = elem.attrib.get("lat")
        lon_text = elem.attrib.get("lon")
        if lat_text is None or lon_text is None:
            continue
        ele = None
        for child in elem:
            if tag_name(child) == "ele" and child.text:
                try:
                    ele = float(child.text)
                except ValueError:
                    ele = None
        points.append((float(lat_text), float(lon_text), ele))
    return points


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def lerp(a: Optional[float], b: Optional[float], t: float) -> Optional[float]:
    if a is None or b is None:
        return None
    return a + (b - a) * t


def cumulative_samples(points: list[Point]) -> list[Sample]:
    samples: list[Sample] = []
    distance = 0.0
    previous: Optional[Point] = None
    for point in points:
        if previous is not None:
            distance += haversine_m(previous[0], previous[1], point[0], point[1])
        samples.append((point[0], point[1], point[2], distance))
        previous = point
    return samples


def sample_at_distance(samples: list[Sample], target_m: float, start_index: int) -> tuple[Sample, int]:
    index = start_index
    while index < len(samples) - 2 and samples[index + 1][3] < target_m:
        index += 1
    current = samples[index]
    next_sample = samples[index + 1]
    span = next_sample[3] - current[3]
    if span <= 0:
        return current, index
    t = (target_m - current[3]) / span
    return (
        current[0] + (next_sample[0] - current[0]) * t,
        current[1] + (next_sample[1] - current[1]) * t,
        lerp(current[2], next_sample[2], t),
        target_m,
    ), index


def split_segments(points: list[Point], segment_length_m: float) -> list[dict[str, float]]:
    samples = cumulative_samples(points)
    if len(samples) < 2 or samples[-1][3] < segment_length_m:
        return []
    segments = []
    index = 0
    distance = 0.0
    while distance + segment_length_m <= samples[-1][3]:
        start, index = sample_at_distance(samples, distance, index)
        end, index = sample_at_distance(samples, distance + segment_length_m, index)
        distance += segment_length_m
        if start[2] is None or end[2] is None:
            continue
        rise_m = end[2] - start[2]
        slope_angle = math.degrees(math.atan(abs(rise_m) / segment_length_m))
        midpoint_ele_ft = ((start[2] + end[2]) / 2) * FT_PER_M
        segments.append(
            {
                "slope_angle_deg": slope_angle,
                "midpoint_elevation_ft": midpoint_ele_ft,
                "rise_ft": rise_m * FT_PER_M,
            }
        )
    return segments


def slope_bin(angle: float) -> str:
    lower = min(80, int(angle // 10) * 10)
    upper = lower + 10
    return f"{lower}-{upper}"


def elevation_bin(ele_ft: float) -> str:
    lower = 5000 + int((ele_ft - 5000) // 1500) * 1500
    lower = max(5000, min(12500, lower))
    upper = lower + 1500
    return f"{lower}-{upper}"


def collect_segments(gpx_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(gpx_dir.glob("*.gpx")):
        points = read_gpx(path)
        for segment in split_segments(points, SEGMENT_LENGTH_M):
            angle = float(segment["slope_angle_deg"])
            ele = float(segment["midpoint_elevation_ft"])
            if ele < 5000 or ele > 14000:
                continue
            rows.append(
                {
                    "file": path.name,
                    "slope_bin": slope_bin(angle),
                    "elevation_bin": elevation_bin(ele),
                    "slope_angle_deg": round(angle, 3),
                    "midpoint_elevation_ft": round(ele),
                    "rise_ft": round(float(segment["rise_ft"]), 2),
                }
            )
    return rows


def write_segments_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_histogram(rows: list[dict[str, object]], out_path: Path) -> None:
    slope_bins = [f"{lower}-{lower + 10}" for lower in range(0, 90, 10)]
    elevation_bins = [f"{lower}-{lower + 1500}" for lower in range(5000, 14000, 1500)]
    colors = ["#254b8f", "#2d7fb8", "#31a6a6", "#70bf73", "#d8c94f", "#e6853a"]
    counts = {slope: {ele: 0 for ele in elevation_bins} for slope in slope_bins}
    for row in rows:
        counts[str(row["slope_bin"])][str(row["elevation_bin"])] += 1

    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    x_positions = list(range(len(slope_bins)))
    bottoms = [0] * len(slope_bins)
    for ele_bin, color in zip(elevation_bins, colors):
        values = [counts[slope][ele_bin] for slope in slope_bins]
        ax.bar(x_positions, values, bottom=bottoms, color=color, edgecolor="white", linewidth=0.6, label=f"{ele_bin} ft")
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    totals = bottoms
    for x, total in zip(x_positions, totals):
        if total:
            ax.text(x, total, f"{total:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Slope Angle Distribution Across 40 m GPX Segments", fontsize=16, weight="bold")
    ax.set_xlabel("Absolute slope angle over 40 m segment")
    ax.set_ylabel("Number of 40 m segments")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{label}°" for label in slope_bins])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(title="Midpoint elevation", ncols=2, frameon=False)
    fig.text(
        0.01,
        0.01,
        f"Source: {len(rows):,} valid 40 m segments from active Peakbagger GPX files. Segments outside 5000-14000 ft omitted.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpx-dir", default=str(DEFAULT_GPX_DIR), help="Directory containing GPX files")
    parser.add_argument("--out-dir", default=str(DEFAULT_PRODUCTS_DIR), help="Directory for generated products")
    parser.add_argument("--png", default="slope_angle_elevation_histogram.png", help="Output PNG filename")
    parser.add_argument("--csv", default="slope_angle_elevation_segments.csv", help="Output segment CSV filename")
    args = parser.parse_args()

    gpx_dir = Path(args.gpx_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_segments(gpx_dir)
    if not rows:
        raise SystemExit(f"No valid 40 m segments found in {gpx_dir}")
    png_path = out_dir / args.png
    csv_path = out_dir / args.csv
    write_segments_csv(rows, csv_path)
    plot_histogram(rows, png_path)
    print(f"Wrote {len(rows):,} segment rows to {csv_path}")
    print(f"Wrote histogram to {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
