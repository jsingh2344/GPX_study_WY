#!/usr/bin/env python3
"""Create a Wyoming satellite map of Peakbagger GPX tracks.

The map uses an embedded static satellite background image. Each GPX file is represented by a
dot at its track centroid:

- marker size = track length
- marker color = elevation gain
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import datetime as dt
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen


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


def track_distance_m(points: list[Point]) -> float:
    return sum(haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(points, points[1:]))


def elevation_gain_m(points: list[Point]) -> float:
    gain = 0.0
    for a, b in zip(points, points[1:]):
        if a[2] is None or b[2] is None:
            continue
        diff = b[2] - a[2]
        if diff > 0:
            gain += diff
    return gain


def elapsed_hours(points: list[Point]) -> Optional[float]:
    times = [point[3] for point in points if point[3] is not None]
    if len(times) < 2:
        return None
    return (max(times) - min(times)).total_seconds() / 3600


def sample_line(points: list[Point], limit: int) -> list[list[float]]:
    if len(points) <= limit:
        sampled = points
    else:
        step = (len(points) - 1) / (limit - 1)
        sampled = [points[round(i * step)] for i in range(limit)]
    return [[point[0], point[1]] for point in sampled]


def clean_name(path: Path) -> str:
    name = path.stem
    for token in ["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017"]:
        name = name.replace(token, f" {token}")
    return name.replace("-", " ").strip()


def color_for_gain(gain_ft: float, max_gain_ft: float) -> str:
    del max_gain_ft
    t = max(0.0, min(1.0, gain_ft / 8000.0))
    hue = (225 - 220 * t) / 360
    saturation = 0.74
    value = 0.70 + 0.16 * math.sin(math.pi * t)
    rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
    rgb = tuple(round(channel * 255) for channel in rgb_float)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def marker_radius(distance_mi: float, min_distance_mi: float, max_distance_mi: float) -> float:
    if max_distance_mi <= min_distance_mi:
        return 10
    t = (distance_mi - min_distance_mi) / (max_distance_mi - min_distance_mi)
    return round(7 + math.sqrt(max(0, t)) * 17, 1)


MAP_WEST = -111.25
MAP_EAST = -104.0
MAP_SOUTH = 40.85
MAP_NORTH = 45.05
MAP_X = 0.0
MAP_Y = 0.0
MAP_WIDTH = 1400.0
MAP_HEIGHT = 900.0
SATELLITE_CACHE = "wyoming_satellite_background.png"


def project(lat: float, lon: float) -> tuple[float, float]:
    x = MAP_X + ((lon - MAP_WEST) / (MAP_EAST - MAP_WEST)) * MAP_WIDTH
    y = MAP_Y + ((MAP_NORTH - lat) / (MAP_NORTH - MAP_SOUTH)) * MAP_HEIGHT
    return x, y


def satellite_background_data_uri(cache_dir: Path) -> str:
    cache_path = cache_dir / SATELLITE_CACHE
    if not cache_path.exists():
        url = (
            "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
            f"?bbox={MAP_WEST},{MAP_SOUTH},{MAP_EAST},{MAP_NORTH}"
            "&bboxSR=4326&imageSR=4326&size=1400,900&format=png32&f=image"
        )
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=90) as response:
            cache_path.write_bytes(response.read())
    encoded = base64.b64encode(cache_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def summarize(path: Path, points: list[Point]) -> dict[str, object]:
    lats = [point[0] for point in points]
    lons = [point[1] for point in points]
    distance_mi = track_distance_m(points) / 1609.344
    gain_ft = elevation_gain_m(points) * 3.28084
    elapsed_hr = elapsed_hours(points)
    return {
        "file": path.name,
        "label": clean_name(path),
        "points": len(points),
        "lat": sum(lats) / len(lats),
        "lon": sum(lons) / len(lons),
        "start_lat": points[0][0],
        "start_lon": points[0][1],
        "distance_mi": round(distance_mi, 2),
        "gain_ft": round(gain_ft),
        "elapsed_hr": round(elapsed_hr, 2) if elapsed_hr is not None else None,
        "line": sample_line(points, 700),
    }


def load_tracks(gpx_dir: Path) -> list[dict[str, object]]:
    tracks = []
    for path in sorted(gpx_dir.glob("*.gpx")):
        points = read_gpx(path)
        if points:
            tracks.append(summarize(path, points))
    return tracks


def render_map(tracks: list[dict[str, object]]) -> str:
    distances = [float(track["distance_mi"]) for track in tracks]
    gains = [float(track["gain_ft"]) for track in tracks]
    min_distance = min(distances)
    max_distance = max(distances)
    max_gain = max(gains)
    for track in tracks:
        track["radius"] = marker_radius(float(track["distance_mi"]), min_distance, max_distance)
        track["color"] = color_for_gain(float(track["gain_ft"]), max_gain)

    background_uri = satellite_background_data_uri(Path(__file__).resolve().parent)
    sorted_tracks = sorted(tracks, key=lambda item: float(item["radius"]), reverse=True)
    marker_svg = []
    for track in sorted_tracks:
        x, y = project(float(track["lat"]), float(track["lon"]))
        radius = float(track["radius"])
        color = str(track["color"])
        label = str(track["label"])
        distance = float(track["distance_mi"])
        gain = int(track["gain_ft"])
        popup_text = f"{label} | {distance:.2f} mi | {gain:,} ft gain | {track['file']}"
        marker_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" '
            f'fill-opacity="0.86" stroke="#ffffff" stroke-width="2.1"><title>{popup_text}</title></circle>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wyoming Peakbagger GPX Track Map</title>
  <style>
    html, body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101416;
      color: #f4f1e8;
    }}
    .page {{
      min-height: 100vh;
      padding: 18px;
      box-sizing: border-box;
    }}
    .map-frame {{
      position: relative;
      width: min(100%, 1500px);
      margin: 0 auto;
      background: #f8faf6;
      border: 1px solid rgba(27, 37, 45, 0.2);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18);
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .static-map-bg {{
      opacity: 0.72;
      filter: saturate(0.78) contrast(0.92) brightness(0.74);
    }}
    .panel {{
      position: absolute;
      right: 16px;
      top: 16px;
      width: 285px;
      color: #f4f1e8;
      background: rgba(13, 18, 20, 0.80);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.42);
      padding: 12px 14px;
    }}
    .panel h1 {{
      margin: 0 0 7px;
      font-size: 16px;
      line-height: 1.2;
    }}
    .panel p {{
      margin: 5px 0;
      font-size: 12px;
      line-height: 1.35;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
      font-size: 12px;
    }}
    .ramp {{
      height: 10px;
      width: 132px;
      border-radius: 999px;
      background: linear-gradient(90deg, #2e55b2 0%, #2582bd 20%, #2fae9c 40%, #b6c94b 60%, #dc8c26 80%, #b22e35 100%);
    }}
    .size-dot {{
      display: inline-block;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.18);
      border: 2px solid #ffffff;
    }}
    .note {{
      margin-top: 9px;
      color: rgba(244, 241, 232, 0.72);
      font-size: 11px;
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="map-frame">
      <svg viewBox="0 0 1400 900" role="img" aria-label="Wyoming map with 29 GPX track pointers">
        <image class="static-map-bg" href="{background_uri}" x="0" y="0" width="1400" height="900"/>
        <g class="track-markers">
          {''.join(marker_svg)}
        </g>
      </svg>
      <section class="panel">
        <h1>Wyoming Peakbagger GPX Tracks</h1>
        <p>{len(tracks)} timed tracks. Dots are positioned at track centroids. Size shows track length; color shows cumulative elevation gain.</p>
        <div class="legend-row"><span class="ramp"></span><span>gain: continuous 0 to 8,000 ft</span></div>
        <div class="legend-row">
          <span class="size-dot" style="width:10px;height:10px"></span>
          <span class="size-dot" style="width:22px;height:22px"></span>
          <span>length: short to long</span>
        </div>
        <p class="note">Hover a dot for file, length, gain, and source name.</p>
        <p class="note">Satellite background: Esri World Imagery export embedded locally.</p>
      </section>
    </div>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpx-dir", default="gpx", help="Directory containing the 29 GPX files")
    parser.add_argument("--out", default="wyoming_peakbagger_gpx_map.html", help="Output HTML map")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    gpx_dir = (base_dir / args.gpx_dir).resolve()
    out_path = (base_dir / args.out).resolve()
    tracks = load_tracks(gpx_dir)
    if not tracks:
        raise SystemExit(f"No GPX tracks found in {gpx_dir}")
    out_path.write_text(render_map(tracks), encoding="utf-8")
    print(f"Wrote {len(tracks)} GPX pointers to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
