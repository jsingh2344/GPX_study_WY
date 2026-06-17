#!/usr/bin/env python3
"""Create a Wyoming map of cleaned Strava GPX activity centroids."""

from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPX_DIR = ROOT / "gpx_cleaned_10m"
PRODUCTS_DIR = ROOT / "products"
REF_DIR = ROOT / "ref"
BACKGROUND_SOURCE = ROOT.parents[0] / "peakbgr_gpx" / "ref" / "wyoming_satellite_background.png"
BACKGROUND = REF_DIR / "wyoming_satellite_background.png"
OUTPUT = PRODUCTS_DIR / "strava_wyoming_activity_map.html"

WEST, EAST, SOUTH, NORTH = -111.25, -104.0, 40.85, 45.05
WIDTH, HEIGHT = 800, 634
FT_PER_M = 3.28084
M_PER_MI = 1609.344


def tag(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def mercator(lat: float, lon: float) -> tuple[float, float]:
    x = (lon - WEST) / (EAST - WEST) * WIDTH
    north = math.log(math.tan(math.pi / 4 + math.radians(NORTH) / 2))
    south = math.log(math.tan(math.pi / 4 + math.radians(SOUTH) / 2))
    y_value = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    y = (north - y_value) / (north - south) * HEIGHT
    return x, y


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(a[0]), math.radians(b[0])
    dphi, dlambda = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def points(path: Path) -> list[tuple[float, float, float | None]]:
    out = []
    for elem in ET.parse(path).getroot().iter():
        if tag(elem) != "trkpt":
            continue
        ele = None
        for child in elem:
            if tag(child) == "ele" and child.text:
                ele = float(child.text)
        out.append((float(elem.attrib["lat"]), float(elem.attrib["lon"]), ele))
    return out


def summarize(path: Path) -> dict[str, object]:
    pts = points(path)
    distance = sum(haversine_m((a[0], a[1]), (b[0], b[1])) for a, b in zip(pts, pts[1:]))
    gain = 0.0
    for a, b in zip(pts, pts[1:]):
        if a[2] is not None and b[2] is not None and b[2] > a[2]:
            gain += b[2] - a[2]
    lat = sum(point[0] for point in pts) / len(pts)
    lon = sum(point[1] for point in pts) / len(pts)
    x, y = mercator(lat, lon)
    return {"file": path.name, "x": x, "y": y, "miles": distance / M_PER_MI, "gain_ft": gain * FT_PER_M}


def color(value: float, max_value: float) -> str:
    t = 0 if max_value <= 0 else min(1, value / max_value)
    r = round(45 + (230 - 45) * t)
    g = round(127 + (90 - 127) * t)
    b = round(184 + (50 - 184) * t)
    return f"rgb({r},{g},{b})"


def main() -> int:
    rows = [summarize(path) for path in sorted(GPX_DIR.glob("*.gpx"))]
    if not rows:
        raise SystemExit(f"No GPX files found in {GPX_DIR}")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    if BACKGROUND_SOURCE.exists() and not BACKGROUND.exists():
        BACKGROUND.write_bytes(BACKGROUND_SOURCE.read_bytes())
    if not BACKGROUND.exists():
        raise SystemExit(f"Missing background image: {BACKGROUND}")
    image = base64.b64encode(BACKGROUND.read_bytes()).decode("ascii")
    max_miles = max(float(row["miles"]) for row in rows)
    max_gain = max(float(row["gain_ft"]) for row in rows)
    circles = []
    for row in rows:
        radius = 4 + 12 * math.sqrt(float(row["miles"]) / max_miles)
        circles.append(f'<circle cx="{row["x"]:.1f}" cy="{row["y"]:.1f}" r="{radius:.1f}" fill="{color(float(row["gain_ft"]), max_gain)}" fill-opacity="0.82" stroke="white" stroke-width="1"><title>{row["file"]}: {row["miles"]:.1f} mi, {row["gain_ft"]:.0f} ft gain</title></circle>')
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Strava Wyoming Activity Map</title>
<style>body{{margin:0;background:#111;font-family:Arial,sans-serif}}.wrap{{position:relative;width:{WIDTH}px;margin:20px auto}}svg{{display:block}}.legend{{position:absolute;right:14px;top:14px;background:rgba(255,255,255,.88);padding:12px 14px;border-radius:6px;font-size:13px}}</style></head>
<body><div class="wrap"><svg width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<image href="data:image/png;base64,{image}" width="{WIDTH}" height="{HEIGHT}" opacity="0.72"/>
{''.join(circles)}
</svg><div class="legend"><b>Strava Wyoming activities</b><br>{len(rows)} cleaned GPX tracks<br>Dot size: mileage<br>Dot color: elevation gain</div></div></body></html>"""
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {len(rows)} activity pointers to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
