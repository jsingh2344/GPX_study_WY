#!/usr/bin/env python3
"""Create a slope-vs-speed dotplot from timed 40 m GPX segments."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GPX_DIR = REPO_ROOT / "peakbgr_gpx" / "gpx"
DEFAULT_DATA_DIR = ANALYSIS_DIR / "data"
DEFAULT_PRODUCTS_DIR = ANALYSIS_DIR / "products"

SEGMENT_LENGTH_M = 40.0
MAX_SPEED_MPH = 15.0
FT_PER_M = 3.28084
M_PER_MILE = 1609.344
SECONDS_PER_HOUR = 3600.0

ELEVATION_BINS = [f"{lower}-{lower + 1500}" for lower in range(5000, 14000, 1500)]
ELEVATION_COLORS = ["#254b8f", "#2d7fb8", "#31a6a6", "#70bf73", "#d8c94f", "#e6853a"]


Point = tuple[float, float, Optional[float], Optional[datetime]]
Sample = tuple[float, float, Optional[float], Optional[datetime], float]


def tag_name(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def parse_time(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


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
        time = None
        for child in elem:
            name = tag_name(child)
            if name == "ele" and child.text:
                try:
                    ele = float(child.text)
                except ValueError:
                    ele = None
            elif name == "time" and child.text:
                time = parse_time(child.text)

        points.append((float(lat_text), float(lon_text), ele, time))
    return points


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def lerp_optional_float(a: Optional[float], b: Optional[float], t: float) -> Optional[float]:
    if a is None or b is None:
        return None
    return a + (b - a) * t


def lerp_optional_time(a: Optional[datetime], b: Optional[datetime], t: float) -> Optional[datetime]:
    if a is None or b is None:
        return None
    delta = (b - a).total_seconds()
    return a if delta < 0 else a + (b - a) * t


def cumulative_samples(points: list[Point]) -> list[Sample]:
    samples: list[Sample] = []
    distance = 0.0
    previous: Optional[Point] = None
    for point in points:
        if previous is not None:
            distance += haversine_m(previous[0], previous[1], point[0], point[1])
        samples.append((point[0], point[1], point[2], point[3], distance))
        previous = point
    return samples


def sample_at_distance(samples: list[Sample], target_m: float, start_index: int) -> tuple[Sample, int]:
    index = start_index
    while index < len(samples) - 2 and samples[index + 1][4] < target_m:
        index += 1
    current = samples[index]
    next_sample = samples[index + 1]
    span = next_sample[4] - current[4]
    if span <= 0:
        return current, index
    t = (target_m - current[4]) / span
    return (
        current[0] + (next_sample[0] - current[0]) * t,
        current[1] + (next_sample[1] - current[1]) * t,
        lerp_optional_float(current[2], next_sample[2], t),
        lerp_optional_time(current[3], next_sample[3], t),
        target_m,
    ), index


def elevation_bin(ele_ft: float) -> Optional[str]:
    if ele_ft < 5000 or ele_ft > 14000:
        return None
    lower = 5000 + int((ele_ft - 5000) // 1500) * 1500
    lower = max(5000, min(12500, lower))
    return f"{lower}-{lower + 1500}"


def split_segments(
    points: list[Point],
    segment_length_m: float,
    max_speed_mph: float,
) -> list[dict[str, object]]:
    samples = cumulative_samples(points)
    if len(samples) < 2 or samples[-1][4] < segment_length_m:
        return []

    rows: list[dict[str, object]] = []
    index = 0
    distance = 0.0
    segment_index = 0
    while distance + segment_length_m <= samples[-1][4]:
        start, index = sample_at_distance(samples, distance, index)
        end, index = sample_at_distance(samples, distance + segment_length_m, index)
        distance += segment_length_m
        segment_index += 1

        if start[2] is None or end[2] is None or start[3] is None or end[3] is None:
            continue
        elapsed_s = (end[3] - start[3]).total_seconds()
        if elapsed_s <= 0:
            continue

        rise_m = end[2] - start[2]
        midpoint_ele_ft = ((start[2] + end[2]) / 2) * FT_PER_M
        ele_bin = elevation_bin(midpoint_ele_ft)
        if ele_bin is None:
            continue

        speed_mph = (segment_length_m / M_PER_MILE) / (elapsed_s / SECONDS_PER_HOUR)
        if speed_mph > max_speed_mph:
            continue
        rows.append(
            {
                "segment_index": segment_index,
                "slope_angle_deg": math.degrees(math.atan(rise_m / segment_length_m)),
                "speed_mph": speed_mph,
                "elapsed_seconds": elapsed_s,
                "midpoint_elevation_ft": midpoint_ele_ft,
                "elevation_bin": ele_bin,
                "rise_ft": rise_m * FT_PER_M,
            }
        )
    return rows


def collect_segments(gpx_dir: Path, max_speed_mph: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(gpx_dir.glob("*.gpx")):
        for segment in split_segments(read_gpx(path), SEGMENT_LENGTH_M, max_speed_mph):
            rows.append(
                {
                    "file": path.name,
                    "segment_index": segment["segment_index"],
                    "slope_angle_deg": round(float(segment["slope_angle_deg"]), 3),
                    "speed_mph": round(float(segment["speed_mph"]), 3),
                    "elapsed_seconds": round(float(segment["elapsed_seconds"]), 2),
                    "midpoint_elevation_ft": round(float(segment["midpoint_elevation_ft"])),
                    "elevation_bin": segment["elevation_bin"],
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


def plot_dotplot(rows: list[dict[str, object]], path: Path, max_speed_mph: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=180)

    for ele_bin, color in zip(ELEVATION_BINS, ELEVATION_COLORS):
        matching = [row for row in rows if row["elevation_bin"] == ele_bin]
        if not matching:
            continue
        ax.scatter(
            [float(row["slope_angle_deg"]) for row in matching],
            [float(row["speed_mph"]) for row in matching],
            s=10,
            alpha=0.42,
            color=color,
            edgecolors="none",
            label=f"{ele_bin} ft",
            rasterized=True,
        )

    speeds = [float(row["speed_mph"]) for row in rows]
    max_abs_slope = max(abs(float(row["slope_angle_deg"])) for row in rows)
    slope_limit = min(90, max(10, math.ceil(max_abs_slope / 10) * 10))
    ax.set_xlim(-slope_limit, slope_limit)
    ax.set_ylim(0, max_speed_mph)
    ax.set_title("Slope vs. Speed Across 40 m GPX Segments", fontsize=16, weight="bold")
    ax.set_xlabel("Directional slope angle over 40 m segment")
    ax.set_ylabel("Speed over segment (mph)")
    ax.grid(alpha=0.22)
    ax.legend(title="Midpoint elevation", frameon=False, ncols=2, loc="upper right")
    fig.text(
        0.01,
        0.01,
        f"Source: {len(rows):,} timed 40 m segments from active Peakbagger GPX files. "
        f"Segments outside 5000-14000 ft or above {max_speed_mph:g} mph omitted.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpx-dir", default=str(DEFAULT_GPX_DIR), help="Directory containing timed GPX files")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory for generated CSV data")
    parser.add_argument("--products-dir", default=str(DEFAULT_PRODUCTS_DIR), help="Directory for generated figures")
    parser.add_argument("--max-speed-mph", type=float, default=MAX_SPEED_MPH, help="Discard faster artifact segments")
    parser.add_argument("--csv", default="slope_speed_40m_segments.csv", help="Output segment CSV filename")
    parser.add_argument("--png", default="slope_speed_dotplot.png", help="Output PNG filename")
    args = parser.parse_args()

    gpx_dir = Path(args.gpx_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    products_dir = Path(args.products_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    products_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_segments(gpx_dir, args.max_speed_mph)
    if not rows:
        raise SystemExit(f"No valid timed 40 m segments found in {gpx_dir}")

    csv_path = data_dir / args.csv
    png_path = products_dir / args.png
    write_segments_csv(rows, csv_path)
    plot_dotplot(rows, png_path, args.max_speed_mph)
    print(f"Wrote {len(rows):,} segment rows to {csv_path}")
    print(f"Wrote dotplot to {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
