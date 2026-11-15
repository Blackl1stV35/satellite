"""Sentinel-1 radar proxies from the Planetary Computer sentinel-1-rtc collection (gamma0, 10 m COG).

All network traffic goes through satellite.net (one request per 2 s per host). One scene of a fixed
relative orbit per calendar month, the one nearest the 15th.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

from . import net
from .aoi import AOI
from .sentinel import Grid, aoi_mask

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel1euwestrtc/sentinel1-grd-rtc"
COLLECTION = "sentinel-1-rtc"
WATER_DB = -16.0  # VV below this is open water (fixed in definitions.md)
MIN_VALID = 0.9
# Window reads here are a few MB, so the gate is a low floor (the 1.5 GB gate applies to full-scene work).
RAM_FLOOR_GB = float(os.environ.get("SAT_RAM_FLOOR_GB", "0.4"))
_io_lock = threading.Lock()
_tok = {"token": "", "expiry": 0.0}


def sas_token() -> str:
    """Anonymous read token issued by the public API; refreshed 10 minutes before expiry."""
    if _tok["token"] and _tok["expiry"] - time.time() > 600:
        return _tok["token"]
    d = net.get(SAS).json()
    exp = pd.Timestamp(d["msft:expiry"]).timestamp()
    _tok.update(token=d["token"], expiry=exp)
    return d["token"]


def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(x)


def search(bbox, start: str, end: str, orbit: int | None = None, limit: int = 200) -> list[dict]:
    body = {"collections": [COLLECTION], "bbox": list(bbox), "datetime": f"{start}/{end}", "limit": limit}
    if orbit is not None:
        body["query"] = {"sat:relative_orbit": {"eq": int(orbit)}}
    d = net.post_json(f"{STAC}/search", body).json()
    return d.get("features", [])


def contains(feature: dict, bbox) -> bool:
    return shape(feature["geometry"]).buffer(0).contains(box(*bbox))


def choose_track(features: list[dict], bbox) -> tuple[int, str] | None:
    """Most frequent (relative orbit, direction) among scenes fully covering bbox. Ties: lowest orbit."""
    cnt = Counter(
        (f["properties"]["sat:relative_orbit"], f["properties"]["sat:orbit_state"])
        for f in features
        if contains(f, bbox)
    )
    if not cnt:
        return None
    best = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0][0]))[0][0]
    return int(best[0]), str(best[1])


def nearest_to_mid(features: list[dict], bbox, period: pd.Period) -> dict | None:
    mid = pd.Timestamp(year=period.year, month=period.month, day=15, tz="UTC")
    ok = [f for f in features if contains(f, bbox)]
    ok = [f for f in ok if pd.Timestamp(f["properties"]["datetime"]).strftime("%Y-%m") == str(period)]
    if not ok:
        return None
    return min(ok, key=lambda f: abs(pd.Timestamp(f["properties"]["datetime"]) - mid))


def read_window(src, bbox_lonlat, res_m: float, full: bool = False):
    """Masked float32 linear-power array for the bbox at about res_m, with its Grid."""
    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox_lonlat)
    win = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, src.width, src.height))
    px = abs(src.transform.a)
    shp = (max(1, round(win.height * px / res_m)), max(1, round(win.width * px / res_m)))
    if full:
        shp = (int(win.height), int(win.width))
    arr = src.read(1, window=win, out_shape=shp, resampling=Resampling.average, masked=True)
    tr = src.window_transform(win) * rasterio.Affine.scale(win.width / shp[1], win.height / shp[0])
    data = np.where(np.ma.getmaskarray(arr), np.nan, np.ma.getdata(arr)).astype("float32")
    data[~(data > 0)] = np.nan
    return data, Grid(tr, shp, src.crs)


def open_pol(feature: dict, pol: str):
    href = feature["assets"][pol]["href"] + "?" + sas_token()
    return rasterio.open(href, opener=net.open_range)


def cane_metrics(vh: np.ndarray, vv: np.ndarray, mask: np.ndarray) -> dict:
    """Mean VH and VV in linear power over valid AOI pixels, returned in dB with VH-VV ratio."""
    both = mask & np.isfinite(vh) & np.isfinite(vv)
    n = int(mask.sum())
    frac = float(both.sum()) / n if n else 0.0
    if frac < MIN_VALID:
        return {"valid_frac": frac, "vh_db": math.nan, "vv_db": math.nan, "ratio_db": math.nan}
    a = float(db(np.mean(vh[both])))
    b = float(db(np.mean(vv[both])))
    return {"valid_frac": frac, "vh_db": a, "vv_db": b, "ratio_db": a - b}


def water_metrics(vv: np.ndarray, mask: np.ndarray, px_km2: float) -> dict:
    ok = mask & np.isfinite(vv)
    n = int(mask.sum())
    frac = float(ok.sum()) / n if n else 0.0
    if frac < MIN_VALID:
        return {"valid_frac": frac, "water_km2": math.nan}
    w = db(vv) < WATER_DB
    share = float((w & ok).sum()) / float(ok.sum())
    return {"valid_frac": frac, "water_km2": share * n * px_km2}


@dataclass
class Group:
    name: str
    aois: list[AOI]
    pols: tuple[str, ...]
    res_m: float

    def bbox(self):
        bs = [a.bbox_lonlat() for a in self.aois]
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def process_scene(group: Group, feature: dict) -> list[dict]:
    """One scene: read each polarisation once, compute the metrics for every AOI of the group."""
    arrs: dict[str, dict[str, tuple[np.ndarray, Grid]]] = {}
    for pol in group.pols:
        with open_pol(feature, pol) as src:
            arrs[pol] = {a.name: read_window(src, a.bbox_lonlat(), group.res_m) for a in group.aois}
        net.forget(feature["assets"][pol]["href"])
    out = []
    for a in group.aois:
        grid = arrs[group.pols[0]][a.name][1]
        mask = aoi_mask(a, grid)
        if a.kind == "cane":
            m = cane_metrics(arrs["vh"][a.name][0], arrs["vv"][a.name][0], mask)
        else:
            px = abs(grid.transform.a * grid.transform.e) / 1e6
            m = water_metrics(arrs["vv"][a.name][0], mask, px)
        out.append({"area": a.name, **m})
    return out


def load_done(path: Path) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not str(r.get("status", "")).startswith("error"):
                    done[(r["area"], r["month"])] = r
    return done


def run_group(group: Group, start: str, end: str, cache_dir: str | Path, track_year: int = 2022, log=print,
              part: tuple[int, int] = (0, 1)) -> None:
    """Resumable monthly run for one group; appends records to radar_records.jsonl."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "radar_records.jsonl"
    tpath = cache_dir / "radar_tracks.json"
    bbox = group.bbox()
    with _io_lock:
        tracks = json.loads(tpath.read_text()) if tpath.exists() else {}
    if group.name not in tracks:
        feats = search(bbox, f"{track_year}-01-01", f"{track_year}-12-31", limit=400)
        tr = choose_track(feats, bbox)
        with _io_lock:
            tracks = json.loads(tpath.read_text()) if tpath.exists() else {}
            tracks[group.name] = {"track": tr, "n_scenes_year": len(feats), "year": track_year}
            tpath.write_text(json.dumps(tracks, indent=1))
        log(f"track {group.name}: {tr}")
    tr = tracks[group.name]["track"]
    if tr is None:
        raise RuntimeError(f"no full-coverage track for {group.name}")
    done = load_done(path)
    for k, period in enumerate(pd.period_range(start, end, freq="M")):
        if k % part[1] != part[0]:
            continue
        if all((a.name, str(period)) in done for a in group.aois):
            continue
        if net.free_ram_gb() < RAM_FLOOR_GB:
            net.wait_for_ram(RAM_FLOOR_GB, log=log)
        nm = period + 1
        recs = []
        try:
            feats = search(bbox, f"{period.year}-{period.month:02d}-01", f"{nm.year}-{nm.month:02d}-01", orbit=tr[0])
            feats = [f for f in feats if f["properties"]["sat:orbit_state"] == tr[1]]
            f = nearest_to_mid(feats, bbox, period)
            if f is None:
                recs = [{"area": a.name, "month": str(period), "status": "no_scene"} for a in group.aois]
            else:
                meta = {
                    "month": str(period),
                    "scene_id": f["id"],
                    "scene_date": f["properties"]["datetime"][:10],
                    "rel_orbit": tr[0],
                    "orbit_state": tr[1],
                    "status": "ok",
                }
                recs = [m | meta for m in process_scene(group, f)]
                for r in recs:
                    if r.get("valid_frac", 0) < MIN_VALID:
                        r["status"] = "low_valid_frac"
        except Exception as exc:  # network or read failure: record and keep going, resumable later
            err = f"error: {type(exc).__name__}"
            recs = [{"area": a.name, "month": str(period), "status": err} for a in group.aois]
        with _io_lock, path.open("a", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
                done[(r["area"], r["month"])] = r
        log(f"{group.name} {period} {recs[0]['status']} requests={net.STATS['requests']}")


def default_groups(aois: list[AOI]) -> list[Group]:
    """One group per cane catchment (VH and VV at 80 m); one group for all EASTW reservoirs (VV at 40 m)."""
    groups = [Group(a.name, [a], ("vh", "vv"), 80.0) for a in aois if a.kind == "cane"]
    res = [a for a in aois if a.kind == "reservoir"]
    if res:
        groups.append(Group("EASTW reservoirs", res, ("vv",), 40.0))
    return groups


def run_all(groups: list[Group], start: str, end: str, cache_dir: str | Path, log=print, splits: int = 2) -> None:
    """Run groups (each split into interleaved month sets) in threads.

    Slow transfers keep several requests in flight; the shared per-host throttle still allows only one
    request start per 2 s per host.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(groups) * splits) as ex:
        futs = [ex.submit(run_group, g, start, end, cache_dir, 2022, log, (i, splits))
                for g in groups for i in range(splits)]
        for f in futs:
            f.result()
