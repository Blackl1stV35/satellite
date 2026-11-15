"""Sentinel-1 vessel counts over the Laem Chabang port and anchorage (documented simple threshold).

Pipeline (fixed in definitions.md): one RTC scene per month of a fixed track, VV at 10 m stored as dB
float16; sea mask from the multi-scene median; vessel pixels above a fixed dB threshold inside the sea;
static bright spots removed; vessels are small connected components.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import net, radar
from .aoi import AOI

PORT_BBOX = (100.80, 12.92, 100.98, 13.10)
PORT_RES_M = 20.0
SEA_MEDIAN_DB = -14.0  # sea: median VV below this
STRUCT_P90_DB = -5.0  # not sea if the 90th percentile is above this (fixed structures)
SHIP_DB = -5.0  # vessel pixel: VV above this inside the sea mask
MIN_PIX, MAX_PIX = 2, 150  # 20 m pixels: 800 to 60,000 square metres
STATIC_FREQ = 0.6  # bright in more than this share of scenes: fixed object, not a vessel
ERODE = 2
EIGHT = np.ones((3, 3), dtype=int)


def port_aoi() -> AOI:
    w, s, e, n = PORT_BBOX
    poly = ((w, s), (e, s), (e, n), (w, n), (w, s))
    return AOI("Laem Chabang port and anchorage", "PORT", "sea", (s + n) / 2, (w + e) / 2, None, poly,
               "Box around Laem Chabang port, channel and anchorage (about 20 km by 20 km)", PORT_RES_M)


def sea_mask(median_db: np.ndarray, p90_db: np.ndarray) -> np.ndarray:
    """Persistently dark pixels, away from fixed bright structures, shrunk by ERODE pixels."""
    m = (median_db < SEA_MEDIAN_DB) & (p90_db < STRUCT_P90_DB)
    return ndimage.binary_erosion(m, structure=EIGHT, iterations=ERODE)


def bright_in_sea(vv_db: np.ndarray, sea: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (vv_db > SHIP_DB) & sea


def count_vessels(bright: np.ndarray, static: np.ndarray | None = None) -> int:
    """Connected components (8-neighbour) of MIN_PIX to MAX_PIX pixels, outside the static mask."""
    b = bright & ~static if static is not None else bright
    lab, n = ndimage.label(b, structure=EIGHT)
    if n == 0:
        return 0
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return int(((sizes >= MIN_PIX) & (sizes <= MAX_PIX)).sum())


def static_mask(freq: np.ndarray) -> np.ndarray:
    """Pixels bright in more than STATIC_FREQ of scenes, grown by one pixel."""
    return ndimage.binary_dilation(freq > STATIC_FREQ, structure=EIGHT)


def fetch_month(feature: dict, out_dir: Path) -> dict:
    """Read VV at 20 m (COG overview, request budget) for the port box and store it as dB float16."""
    aoi = port_aoi()
    with radar.open_pol(feature, "vv") as src:
        arr, grid = radar.read_window(src, aoi.bbox_lonlat(), PORT_RES_M)
    net.forget(feature["assets"]["vv"]["href"])
    valid = float(np.isfinite(arr).mean())
    d = radar.db(arr).astype("float16")
    np.save(out_dir / f"{feature['properties']['datetime'][:7]}.npy", d)
    return {"valid_frac": valid, "shape": list(d.shape), "scene_id": feature["id"],
            "scene_date": feature["properties"]["datetime"][:10]}


def run_fetch(start: str, end: str, cache_dir: str | Path, log=print, workers: int = 6) -> None:
    """Resumable monthly download of the port box (throttled)."""
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / "port"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "port_scenes.jsonl"
    done = set()
    if meta_path.exists():
        done = {json.loads(line)["month"] for line in meta_path.read_text("utf-8").splitlines() if line.strip()}
    bbox = PORT_BBOX
    tpath = cache_dir / "radar_tracks.json"
    tracks = json.loads(tpath.read_text()) if tpath.exists() else {}
    if "port" not in tracks:
        tr = radar.choose_track(radar.search(bbox, "2022-01-01", "2022-12-31", limit=400), bbox)
        tracks["port"] = {"track": tr, "year": 2022}
        tpath.write_text(json.dumps(tracks, indent=1))
    tr = tracks["port"]["track"]
    log(f"port track {tr}")
    lock = threading.Lock()

    def one(period):
        nm = period + 1
        rec = {"month": str(period), "rel_orbit": tr[0], "orbit_state": tr[1]}
        try:
            feats = radar.search(bbox, f"{period.year}-{period.month:02d}-01",
                                 f"{nm.year}-{nm.month:02d}-01", orbit=tr[0])
            feats = [f for f in feats if f["properties"]["sat:orbit_state"] == tr[1]]
            f = radar.nearest_to_mid(feats, bbox, period)
            rec |= ({"status": "no_scene"} if f is None else fetch_month(f, out_dir) | {"status": "ok"})
        except Exception as exc:
            rec["status"] = f"error: {type(exc).__name__}"
        if not rec["status"].startswith("error"):
            with lock, meta_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + chr(10))
        log(f"port {period} {rec['status']} requests={net.STATS['requests']}")

    todo = [p for p in pd.period_range(start, end, freq="M") if str(p) not in done]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))


def _stack_stats(files: list[Path], step: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Median and 90th percentile over files, computed in row strips to limit memory."""
    shp = np.load(files[0], mmap_mode="r").shape
    med = np.empty(shp, dtype="float32")
    p90 = np.empty(shp, dtype="float32")
    arrs = [np.load(f, mmap_mode="r") for f in files]
    for r in range(0, shp[0], step):
        blk = np.stack([np.asarray(a[r : r + step], dtype="float32") for a in arrs])
        med[r : r + step] = np.nanmedian(blk, axis=0)
        p90[r : r + step] = np.nanpercentile(blk, 90, axis=0)
    return med, p90


def build_series(cache_dir: str | Path, mask_scenes: int = 25) -> pd.DataFrame:
    """Monthly mean vessel detections per scene from the stored port arrays."""
    cache_dir = Path(cache_dir)
    files = sorted((cache_dir / "port").glob("*.npy"))
    if not files:
        return pd.DataFrame(columns=["month", "vessels", "sea_px", "static_px", "status"])
    pick = np.linspace(0, len(files) - 1, min(mask_scenes, len(files))).round().astype(int)
    med, p90 = _stack_stats([files[i] for i in sorted(set(pick))])
    sea = sea_mask(med, p90)
    freq = np.zeros(sea.shape, dtype="float32")
    for f in files:
        freq += bright_in_sea(np.load(f).astype("float32"), sea)
    static = static_mask(freq / len(files))
    rows = []
    for f in files:
        v = np.load(f).astype("float32")
        n = count_vessels(bright_in_sea(v, sea), static)
        rows.append({"month": f.stem, "vessels": n, "sea_px": int(sea.sum()), "static_px": int(static.sum()),
                     "status": "ok"})
    df = pd.DataFrame(rows)
    return df.assign(vessels=df["vessels"].astype(float)).sort_values("month").reset_index(drop=True)



# ---- P2 (amendment 2026-11-11): time-local static mask for new construction -------------------------
LOCAL_HALF_MONTHS = 6  # window: scenes whose month is within 6 months of the scene
LOCAL_FREQ = 0.6  # same frequency as the global rule, applied inside the window
LOCAL_MIN_SCENES = 7  # windows with fewer scenes use no local mask (counted as P1 for that scene)


def local_static_masks(bright: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per scene: pixels bright in more than LOCAL_FREQ of the scenes within +-LOCAL_HALF_MONTHS, grown by 1 pixel.

    A fixed object that appears during the record (new quay, bund wall, pile-driving barge) is bright in nearly
    every scene after it appears, so it is removed from the window where it exists. A ship at anchor is bright in
    well under 60 percent of the scenes of a year at one spot and is kept. bright maps month -> bool array.
    """
    per = {m: pd.Period(m, "M") for m in bright}
    out = {}
    for m, p in per.items():
        ms = [k for k, q in per.items() if abs((q - p).n) <= LOCAL_HALF_MONTHS]
        if len(ms) < LOCAL_MIN_SCENES:
            out[m] = np.zeros_like(bright[m])
            continue
        freq = np.zeros(bright[m].shape, dtype="float32")
        for k in ms:
            freq += bright[k]
        out[m] = static_mask(freq / len(ms))
    return out


def build_series_p2(cache_dir: str | Path, mask_scenes: int = 25) -> pd.DataFrame:
    """P2 monthly vessel detections per scene: P1 rules plus the time-local static mask."""
    cache_dir = Path(cache_dir)
    files = sorted((cache_dir / "port").glob("*.npy"))
    pick = np.linspace(0, len(files) - 1, min(mask_scenes, len(files))).round().astype(int)
    med, p90 = _stack_stats([files[i] for i in sorted(set(pick))])
    sea = sea_mask(med, p90)
    bright = {f.stem: bright_in_sea(np.load(f).astype("float32"), sea) for f in files}
    glob = np.zeros(sea.shape, dtype="float32")
    for b in bright.values():
        glob += b
    gstatic = static_mask(glob / len(files))
    local = local_static_masks(bright)
    rows = [{"month": m, "vessels": float(count_vessels(b, gstatic | local[m])), "sea_px": int(sea.sum()),
             "static_px_global": int(gstatic.sum()), "static_px_local": int(local[m].sum()), "status": "ok"}
            for m, b in sorted(bright.items())]
    return pd.DataFrame(rows)
