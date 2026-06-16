#!/usr/bin/env python3
"""Generate GPX elevation/mileage/point summaries and a Markdown report."""

from __future__ import annotations

import csv
import datetime as dt
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
GPX_DIR = REPO_ROOT / "peakbgr_gpx" / "gpx"
PRODUCTS_DIR = REPO_ROOT / "peakbgr_gpx" / "products"
MAP_PATH = PRODUCTS_DIR / "wyoming_peakbagger_gpx_map.html"
CSV_PATH = PRODUCTS_DIR / "track_elevation_mileage_points.csv"
MD_PATH = PRODUCTS_DIR / "track_summary.md"

Point = tuple[float, float, Optional[float], Optional[dt.datetime]]


def parse_time(text: str) -> Optional[dt.datetime]:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


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
        timestamp = None
        for child in elem:
            child_tag = tag_name(child)
            if child_tag == "ele" and child.text:
                try:
                    ele = float(child.text)
                except ValueError:
                    ele = None
            elif child_tag == "time" and child.text:
                timestamp = parse_time(child.text)
        points.append((float(lat_text), float(lon_text), ele, timestamp))
    return points


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_m(points: list[Point]) -> float:
    return sum(haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(points, points[1:]))


def elevation_gain_loss_m(points: list[Point]) -> tuple[float, float]:
    gain = 0.0
    loss = 0.0
    for a, b in zip(points, points[1:]):
        if a[2] is None or b[2] is None:
            continue
        diff = b[2] - a[2]
        if diff > 0:
            gain += diff
        else:
            loss += -diff
    return gain, loss


def clean_name(path: Path) -> str:
    name = path.stem
    for token in ["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017"]:
        name = name.replace(token, f" {token}")
    return name.replace("-", " ").strip()


def elapsed_hours(points: list[Point]) -> Optional[float]:
    times = [point[3] for point in points if point[3] is not None]
    if len(times) < 2:
        return None
    return (max(times) - min(times)).total_seconds() / 3600


def summarize_gpx(path: Path) -> dict[str, object]:
    points = read_gpx(path)
    if not points:
        raise ValueError(f"No GPX track or route points found in {path}")
    lats = [point[0] for point in points]
    lons = [point[1] for point in points]
    eles = [point[2] for point in points if point[2] is not None]
    times = [point[3] for point in points if point[3] is not None]
    gain_m, loss_m = elevation_gain_loss_m(points)
    dist_m = distance_m(points)
    elapsed = elapsed_hours(points)
    return {
        "track": clean_name(path),
        "file": path.name,
        "mileage": round(dist_m / 1609.344, 2),
        "gpx_points": len(points),
        "time_points": len(times),
        "min_elevation_ft": round(min(eles) * 3.28084) if eles else "",
        "max_elevation_ft": round(max(eles) * 3.28084) if eles else "",
        "elevation_gain_ft": round(gain_m * 3.28084),
        "elevation_loss_ft": round(loss_m * 3.28084),
        "elapsed_hours": round(elapsed, 2) if elapsed is not None else "",
        "centroid_lat": round(sum(lats) / len(lats), 6),
        "centroid_lon": round(sum(lons) / len(lons), 6),
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = [
        "track",
        "mileage",
        "gpx_points",
        "time_points",
        "min_elevation_ft",
        "max_elevation_ft",
        "elevation_gain_ft",
        "elevation_loss_ft",
        "elapsed_hours",
        "centroid_lat",
        "centroid_lon",
        "file",
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def distribution_summary(rows: list[dict[str, object]]) -> list[str]:
    west = [row for row in rows if float(row["centroid_lon"]) <= -110.0]
    central = [row for row in rows if -110.0 < float(row["centroid_lon"]) <= -107.0]
    east = [row for row in rows if float(row["centroid_lon"]) > -107.0]
    south = [row for row in rows if float(row["centroid_lat"]) < 42.5]
    north = [row for row in rows if float(row["centroid_lat"]) >= 44.0]
    return [
        f"- Western Wyoming / Teton-side tracks: {len(west)}",
        f"- Central Wyoming tracks: {len(central)}",
        f"- Eastern Wyoming tracks: {len(east)}",
        f"- Southern Wyoming tracks: {len(south)}",
        f"- Northern Wyoming tracks: {len(north)}",
    ]


def write_markdown(rows: list[dict[str, object]]) -> None:
    total_miles = sum(float(row["mileage"]) for row in rows)
    total_gain = sum(int(row["elevation_gain_ft"]) for row in rows)
    longest = max(rows, key=lambda row: float(row["mileage"]))
    most_gain = max(rows, key=lambda row: int(row["elevation_gain_ft"]))
    most_points = max(rows, key=lambda row: int(row["gpx_points"]))
    map_rel = MAP_PATH.relative_to(MD_PATH.parent)
    csv_rel = CSV_PATH.relative_to(MD_PATH.parent)
    body = [
        "# Peakbagger GPX Track Summary",
        "",
        "This report summarizes the timed Peakbagger GPX tracks in `peakbgr_gpx/gpx/`.",
        "",
        "## Map",
        "",
        f"The geographic distribution map is generated at [`{map_rel}`]({map_rel}). It uses a locally cached satellite background, dots at track centroids, dot size for mileage, and dot color for elevation gain.",
        "",
        "## Dataset Totals",
        "",
        f"- Tracks summarized: {len(rows)}",
        f"- Total mileage: {total_miles:.2f} mi",
        f"- Total elevation gain: {total_gain:,} ft",
        f"- Longest track: {longest['track']} ({longest['mileage']} mi)",
        f"- Highest cumulative gain: {most_gain['track']} ({int(most_gain['elevation_gain_ft']):,} ft)",
        f"- Most GPX points: {most_points['track']} ({int(most_points['gpx_points']):,} points)",
        "",
        "## Geographic Distribution",
        "",
        *distribution_summary(rows),
        "",
        "## Full Table",
        "",
        f"CSV version: [`{csv_rel}`]({csv_rel})",
        "",
        markdown_table(rows),
        "",
    ]
    MD_PATH.write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    rows = [summarize_gpx(path) for path in sorted(GPX_DIR.glob("*.gpx"))]
    rows.sort(key=lambda row: str(row["track"]))
    if not rows:
        raise SystemExit(f"No GPX files found in {GPX_DIR}")
    write_csv(rows)
    write_markdown(rows)
    print(f"Wrote {len(rows)} rows to {CSV_PATH}")
    print(f"Wrote summary to {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
