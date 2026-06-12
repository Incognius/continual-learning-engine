"""Build a cached table of NYC taxi-zone centroids (one-time, offline).

Downloads the official TLC taxi-zone shapefile and computes each zone's
area-weighted centroid. The shapefile is in NY State Plane (EPSG:2263, feet), a
locally conformal projection — so planar distances in feet are an excellent
approximation of true ground distance for NYC, and we avoid any reprojection
dependency (geopandas/pyproj). We only need pyshp (pure Python) here; the
serving path just reads the resulting CSV with pandas.

Output: data/zone_centroids.csv
    location_id, borough, zone, x_km, y_km, is_airport

Usage:
    python scripts/prepare_zones.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import shapefile  # pyshp

ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
RAW_DIR = Path("data/raw/taxi_zones")
OUT = Path("data/zone_centroids.csv")
FT_TO_KM = 0.0003048


def _find_shp() -> Path | None:
    hits = list(RAW_DIR.rglob("*.shp"))
    return hits[0] if hits else None


def _download_and_extract() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = _find_shp()
    if existing:
        print(f"  [cache] {existing}")
        return existing
    print(f"  [get ] {ZONES_URL}")
    data = urllib.request.urlopen(ZONES_URL).read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(RAW_DIR)
    shp = _find_shp()
    if not shp:
        raise FileNotFoundError("no .shp found in taxi_zones.zip")
    print(f"  [ok  ] extracted -> {shp}")
    return shp


def _ring_centroid_area(pts: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Area-weighted centroid of a single ring via the shoelace formula."""
    a = cx = cy = 0.0
    n = len(pts)
    for i in range(n - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-9:  # degenerate ring -> fall back to mean of points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return sum(xs) / n, sum(ys) / n, 0.0
    return cx / (6 * a), cy / (6 * a), abs(a)


def _shape_centroid(shape) -> tuple[float, float]:
    """Area-weighted centroid across all parts of a (multi)polygon, in feet."""
    parts = list(shape.parts) + [len(shape.points)]
    total_a = 0.0
    sx = sy = 0.0
    for i in range(len(parts) - 1):
        ring = shape.points[parts[i]:parts[i + 1]]
        if len(ring) < 3:
            continue
        cx, cy, area = _ring_centroid_area(ring)
        sx += cx * area
        sy += cy * area
        total_a += area
    if total_a == 0:
        pts = shape.points
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return sx / total_a, sy / total_a


def main() -> int:
    print("[zones] preparing taxi-zone centroids")
    shp_path = _download_and_extract()
    reader = shapefile.Reader(str(shp_path))

    # Locate fields by name (case-insensitive); dbf has LocationID/borough/zone.
    field_names = [f[0] for f in reader.fields[1:]]
    lower = {n.lower(): n for n in field_names}
    idx = {key: field_names.index(lower[key]) for key in ("locationid", "borough", "zone")}

    rows = []
    for sr in reader.shapeRecords():
        rec = sr.record
        loc = int(rec[idx["locationid"]])
        borough = str(rec[idx["borough"]]).strip()
        zone = str(rec[idx["zone"]]).strip()
        x_ft, y_ft = _shape_centroid(sr.shape)
        is_airport = 1 if "airport" in zone.lower() else 0
        rows.append((loc, borough, zone, round(x_ft * FT_TO_KM, 4),
                     round(y_ft * FT_TO_KM, 4), is_airport))

    rows.sort(key=lambda r: r[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        fh.write("location_id,borough,zone,x_km,y_km,is_airport\n")
        for loc, borough, zone, x, y, air in rows:
            zone_safe = zone.replace(",", " ")
            fh.write(f"{loc},{borough},{zone_safe},{x},{y},{air}\n")

    airports = [r for r in rows if r[5] == 1]
    print(f"[done ] {len(rows)} zones -> {OUT}")
    print(f"[check] airports flagged: {[r[2] for r in airports]}")
    # Sanity: distance between JFK (132) and LaGuardia (138) should be ~12-13 km.
    by_id = {r[0]: r for r in rows}
    if 132 in by_id and 138 in by_id:
        import math
        a, b = by_id[132], by_id[138]
        d = math.hypot(a[3] - b[3], a[4] - b[4])
        print(f"[check] JFK<->LGA centroid distance = {d:.1f} km (expect ~12-13)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
