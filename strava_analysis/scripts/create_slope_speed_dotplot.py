#!/usr/bin/env python3
"""Create Strava slope-vs-speed segment tables and dotplot."""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
GPX_DIR = ROOT / "gpx_cleaned_10m"
DATA_DIR = ROOT / "data"
PRODUCTS_DIR = ROOT / "products"
SEGMENTS_CSV = DATA_DIR / "strava_slope_speed_40m_segments.csv"
AVERAGES_CSV = DATA_DIR / "strava_slope_speed_10deg_average.csv"
PLOT = PRODUCTS_DIR / "strava_slope_speed_dotplot.png"

SEGMENT_M = 40.0
MIN_PLOT_SPEED_MPH = 0.3
MAX_SPEED_MPH = 15.0
FT_PER_M = 3.28084
M_PER_MI = 1609.344
ELEVATION_BINS = [f"{low}-{low + 1500}" for low in range(5000, 14000, 1500)]
COLORS = ["#254b8f", "#2d7fb8", "#31a6a6", "#70bf73", "#d8c94f", "#e6853a"]


def tag(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def parse_time(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def read_points(path: Path) -> list[tuple[float, float, float | None, datetime | None]]:
    points = []
    for elem in ET.parse(path).getroot().iter():
        if tag(elem) != "trkpt":
            continue
        ele, time = None, None
        for child in elem:
            if tag(child) == "ele" and child.text:
                ele = float(child.text)
            elif tag(child) == "time" and child.text:
                time = parse_time(child.text)
        points.append((float(elem.attrib["lat"]), float(elem.attrib["lon"]), ele, time))
    return points


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(a[0]), math.radians(b[0])
    dphi, dlambda = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
    x = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def cumulative(points):
    samples, distance = [], 0.0
    for previous, point in zip([None] + points[:-1], points):
        if previous:
            distance += haversine_m((previous[0], previous[1]), (point[0], point[1]))
        samples.append((*point, distance))
    return samples


def sample_at(samples, target_m, start_index):
    i = start_index
    while i < len(samples) - 2 and samples[i + 1][4] < target_m:
        i += 1
    a, b = samples[i], samples[i + 1]
    span = b[4] - a[4]
    if span <= 0:
        return a, i
    t = (target_m - a[4]) / span
    ele = None if a[2] is None or b[2] is None else a[2] + (b[2] - a[2]) * t
    time = None if a[3] is None or b[3] is None else a[3] + (b[3] - a[3]) * t
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, ele, time, target_m), i


def elevation_bin(ele_ft: float) -> str | None:
    if ele_ft < 5000 or ele_ft > 14000:
        return None
    low = max(5000, min(12500, 5000 + int((ele_ft - 5000) // 1500) * 1500))
    return f"{low}-{low + 1500}"


def file_segments(path: Path) -> list[dict[str, object]]:
    samples = cumulative(read_points(path))
    if len(samples) < 2 or samples[-1][4] < SEGMENT_M:
        return []
    rows, i, distance, segment_index = [], 0, 0.0, 0
    while distance + SEGMENT_M <= samples[-1][4]:
        start, i = sample_at(samples, distance, i)
        end, i = sample_at(samples, distance + SEGMENT_M, i)
        distance += SEGMENT_M
        segment_index += 1
        if start[2] is None or end[2] is None or start[3] is None or end[3] is None:
            continue
        elapsed = (end[3] - start[3]).total_seconds()
        if elapsed <= 0:
            continue
        speed = (SEGMENT_M / M_PER_MI) / (elapsed / 3600)
        if speed > MAX_SPEED_MPH:
            continue
        rise_m = end[2] - start[2]
        midpoint_ele_ft = (start[2] + end[2]) / 2 * FT_PER_M
        ele_bin = elevation_bin(midpoint_ele_ft)
        if ele_bin is None:
            continue
        rows.append({
            "file": path.name,
            "segment_index": segment_index,
            "slope_angle_deg": round(math.degrees(math.atan(rise_m / SEGMENT_M)), 3),
            "speed_mph": round(speed, 3),
            "elapsed_seconds": round(elapsed, 2),
            "midpoint_elevation_ft": round(midpoint_ele_ft),
            "elevation_bin": ele_bin,
            "rise_ft": round(rise_m * FT_PER_M, 2),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def averages(rows):
    out = []
    for low in range(-40, 40, 10):
        speeds = [float(row["speed_mph"]) for row in rows if low <= float(row["slope_angle_deg"]) < low + 10]
        if speeds:
            out.append({"slope_bin_lower_deg": low, "slope_bin_upper_deg": low + 10, "slope_bin_center_deg": low + 5, "mean_speed_mph": round(sum(speeds) / len(speeds), 3), "segment_count": len(speeds)})
    return out


def elevation_band_averages(rows, ele_bin):
    band_rows = [row for row in rows if row["elevation_bin"] == ele_bin]
    return averages(band_rows)


def plot(rows):
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=180)
    for ele_bin, color in zip(ELEVATION_BINS, COLORS):
        subset = [row for row in rows if row["elevation_bin"] == ele_bin]
        ax.scatter([float(row["slope_angle_deg"]) for row in subset], [float(row["speed_mph"]) for row in subset], s=9, alpha=0.38, color=color, edgecolors="none", label=f"{ele_bin} ft")
    for ele_bin, color in zip(ELEVATION_BINS, COLORS):
        band_avg = [row for row in elevation_band_averages(rows, ele_bin) if int(row["segment_count"]) >= 20]
        if not band_avg:
            continue
        ax.plot(
            [row["slope_bin_center_deg"] for row in band_avg],
            [row["mean_speed_mph"] for row in band_avg],
            color=color,
            linewidth=4.0,
            alpha=0.78,
            zorder=3,
        )
    avg = [row for row in averages(rows) if int(row["segment_count"]) >= 50]
    ax.plot([row["slope_bin_center_deg"] for row in avg], [row["mean_speed_mph"] for row in avg], color="black", marker="o", linewidth=2.4, label="Mean speed per 10 deg bin")
    for row in avg:
        ax.annotate(
            f"{float(row['mean_speed_mph']):.1f}",
            (float(row["slope_bin_center_deg"]), float(row["mean_speed_mph"])),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, MAX_SPEED_MPH)
    ax.set_title("Strava Slope vs. Speed Across 40 m GPX Segments", fontsize=16, weight="bold")
    ax.set_xlabel("Directional slope angle over 40 m segment")
    ax.set_ylabel("Speed over segment (mph)")
    ax.grid(alpha=0.22)
    ax.legend(title="Midpoint elevation", frameon=False, ncols=2, loc="upper right")
    fig.text(0.01, 0.01, f"Source: {len(rows):,} 40 m Strava moving segments. Segments outside 5000-14000 ft, below {MIN_PLOT_SPEED_MPH:g} mph, or above {MAX_SPEED_MPH:g} mph omitted.", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT)
    plt.close(fig)


def main() -> int:
    rows = [row for path in sorted(GPX_DIR.glob("*.gpx")) for row in file_segments(path)]
    if not rows:
        raise SystemExit(f"No valid segments found in {GPX_DIR}")
    moving_rows = [row for row in rows if float(row["speed_mph"]) >= MIN_PLOT_SPEED_MPH]
    write_csv(SEGMENTS_CSV, rows)
    write_csv(AVERAGES_CSV, averages(moving_rows))
    plot(moving_rows)
    print(f"Wrote {len(rows):,} segments to {SEGMENTS_CSV}")
    print(f"Wrote averages to {AVERAGES_CSV}")
    print(f"Wrote plot to {PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
