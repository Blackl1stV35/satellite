"""Area of interest definitions loaded from a JSON file."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AOI:
    name: str
    ticker: str
    kind: str  # "cane" or "reservoir"
    lat: float
    lon: float
    radius_m: float | None
    polygon: tuple[tuple[float, float], ...] | None  # (lon, lat) ring
    source: str
    resolution_m: float = 80.0
    notes: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def bbox_lonlat(self) -> tuple[float, float, float, float]:
        """West, south, east, north in degrees."""
        if self.polygon:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            return min(xs), min(ys), max(xs), max(ys)
        r = float(self.radius_m or 0)
        dlat = r / 111_320.0
        dlon = r / (111_320.0 * math.cos(math.radians(self.lat)))
        return self.lon - dlon, self.lat - dlat, self.lon + dlon, self.lat + dlat

    def area_km2(self) -> float:
        if self.polygon:
            lat0 = math.radians(self.lat)
            pts = [
                (p[0] * 111_320.0 * math.cos(lat0), p[1] * 110_574.0) for p in self.polygon
            ]
            s = 0.0
            for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True):
                s += x1 * y2 - x2 * y1
            return abs(s) / 2 / 1e6
        return math.pi * float(self.radius_m or 0) ** 2 / 1e6


def _parse(d: dict) -> AOI:
    poly = d.get("polygon")
    if not d.get("radius_m") and not poly:
        raise ValueError(f"AOI {d.get('name')} needs radius_m or polygon")
    if not d.get("source"):
        raise ValueError(f"AOI {d.get('name')} needs a source note")
    centre = d["centre"]
    return AOI(
        name=d["name"],
        ticker=d["ticker"],
        kind=d["kind"],
        lat=float(centre[0]),
        lon=float(centre[1]),
        radius_m=float(d["radius_m"]) if d.get("radius_m") else None,
        polygon=tuple((float(x), float(y)) for x, y in poly) if poly else None,
        source=d["source"],
        resolution_m=float(d.get("resolution_m", 80.0)),
        notes=d.get("notes", ""),
    )


def load_aois(path: str | Path) -> list[AOI]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_parse(d) for d in data["areas"]]
