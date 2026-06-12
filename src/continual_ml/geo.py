"""Zone geography — turns a (pickup_zone, dropoff_zone) pair into geographic
features that are knowable at request time (and therefore leakage-free).

Backed by the cached ``data/zone_centroids.csv`` (see scripts/prepare_zones.py).
Coordinates are in projected kilometres (NY State Plane), so a plain Euclidean
distance is an accurate ground distance for the NYC area — no geopandas/pyproj at
runtime.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional


class ZoneGeo:
    def __init__(self, centroids_path: str):
        self._coord: dict[str, tuple[float, float]] = {}
        self._borough: dict[str, str] = {}
        self._zone: dict[str, str] = {}
        self._airport: dict[str, int] = {}
        self._load(Path(centroids_path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Zone centroids not found at '{path}'. "
                f"Run: python scripts/prepare_zones.py"
            )
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                zid = str(row["location_id"])
                self._coord[zid] = (float(row["x_km"]), float(row["y_km"]))
                self._borough[zid] = row["borough"]
                self._zone[zid] = row["zone"]
                self._airport[zid] = int(row["is_airport"])

    @property
    def loaded_zones(self) -> int:
        return len(self._coord)

    def catalog(self) -> list[dict]:
        """Sorted list of zones for UI dropdowns: {id, zone, borough}."""
        out = [
            {"id": zid, "zone": self._zone.get(zid, zid), "borough": self._borough.get(zid, "")}
            for zid in self._coord
        ]
        return sorted(out, key=lambda z: (z["borough"], z["zone"]))

    def features(self, pu_zone: str, do_zone: str) -> dict:
        """Geographic features for a zone pair. Missing zones (e.g. 'Unknown'
        ids 264/265) degrade gracefully to neutral values."""
        pu = self._coord.get(str(pu_zone))
        do = self._coord.get(str(do_zone))
        pu_b = self._borough.get(str(pu_zone), "Unknown")
        do_b = self._borough.get(str(do_zone), "Unknown")

        if pu is None or do is None:
            distance = 0.0
            b_sin = b_cos = 0.0
        else:
            dx, dy = do[0] - pu[0], do[1] - pu[1]
            distance = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)  # radians; circular -> encode as sin/cos
            b_sin, b_cos = math.sin(bearing), math.cos(bearing)

        return {
            "gc_distance_km": round(distance, 4),
            "bearing_sin": round(b_sin, 4),
            "bearing_cos": round(b_cos, 4),
            "same_borough": 1.0 if pu_b == do_b and pu_b != "Unknown" else 0.0,
            "pu_airport": float(self._airport.get(str(pu_zone), 0)),
            "do_airport": float(self._airport.get(str(do_zone), 0)),
            "pu_borough": pu_b,
            "do_borough": do_b,
        }
