"""Monthly series per area (cached, resumable) and quarterly aggregation."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from . import sentinel as sn
from .aoi import AOI


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def month_range(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="M"))


def process_month(aoi: AOI, client, period: pd.Period) -> dict:
    """Select the best scene for one month and compute metrics. One scene in memory."""
    rec = {
        "area": aoi.name,
        "month": str(period),
        "scene_id": "",
        "scene_date": "",
        "clear_frac": np.nan,
        "ndvi_mean": np.nan,
        "water_frac": np.nan,
        "water_km2": np.nan,
        "offset": np.nan,
        "n_candidates": 0,
        "status": "",
    }
    items = sn.search_month(client, aoi.bbox_lonlat(), period.year, period.month)
    if not items:
        return rec | {"status": "no_items"}
    best, cf, tried = sn.select_best_scene(aoi, items)
    rec["n_candidates"] = tried
    if best is None:
        return rec | {"status": "no_clear_scene", "clear_frac": cf}
    bands, grid, mask, off = sn.read_bands(aoi, best)
    px_km2 = abs(grid.transform.a * grid.transform.e) / 1e6
    m = sn.compute_metrics(bands, mask, px_km2)
    del bands
    return rec | {
        "scene_id": best.id,
        "scene_date": best.datetime.strftime("%Y-%m-%d"),
        "clear_frac": m["clear_frac"],
        "ndvi_mean": m["ndvi_mean"],
        "water_frac": m["water_frac"],
        "water_km2": m["water_km2"],
        "offset": off,
        "status": "ok",
    }


def _load_done(path: Path) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["area"], r["month"])] = r
    return done


def build_monthly(
    aois: list[AOI],
    start: str,
    end: str,
    cache_dir: str | Path,
    workers: int = 4,
    log=print,
) -> pd.DataFrame:
    """Compute monthly records for all areas. Resumable through a JSON lines cache."""
    from pystac_client import Client

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "monthly_records.jsonl"
    done = _load_done(path)
    tasks = [(a, p) for a in aois for p in month_range(start, end) if (a.name, str(p)) not in done]
    log(f"{len(done)} cached, {len(tasks)} to compute")
    lock = threading.Lock()
    local = threading.local()

    def work(task):
        aoi, period = task
        if not hasattr(local, "client"):
            local.client = Client.open(sn.STAC_URL, timeout=30)
        last = None
        for _ in range(3):
            try:
                with rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                    GDAL_HTTP_TIMEOUT="40",
                ):
                    rec = process_month(aoi, local.client, period)
                break
            except Exception as exc:  # network hiccup: retry, then record the failure
                last = exc
                rec = None
        if rec is None:
            err = f"error: {type(last).__name__}"
            rec = {"area": aoi.name, "month": str(period), "status": err}
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            done[(aoi.name, str(period))] = rec
            if len(done) % 5 == 0:
                log(f"{len(done)} records")
        return rec

    if workers <= 1:
        # Main thread only: GDAL HTTP reads from worker threads were seen to stall here.
        for t in tasks:
            work(t)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, tasks))
    df = pd.DataFrame(list(done.values()))
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df.sort_values(["area", "month"]).reset_index(drop=True)


def combine_areas(
    df: pd.DataFrame, areas: list[str], col: str, how: str = "mean", require_all: bool = True
) -> pd.Series:
    """Combine several areas into one monthly series indexed by Period (mean or sum)."""
    wide = df[df["area"].isin(areas)].pivot(index="month", columns="area", values=col)
    wide = wide.reindex(columns=areas)
    if require_all:
        wide = wide.where(wide.notna().all(axis=1))
    out = wide.sum(axis=1, min_count=1) if how == "sum" else wide.mean(axis=1)
    return out.dropna()


def quarter_end_months(first: pd.Period, last: pd.Period, fy_end_month: int = 12):
    """Quarter-end months (Period M) for a fiscal year ending in fy_end_month."""
    ends = []
    p = pd.Period(year=first.year, month=fy_end_month, freq="M")
    while p > first:
        p -= 3
    while p <= last:
        if p >= first:
            ends.append(p)
        p += 3
    return ends


def aggregate_periods(
    monthly: pd.Series,
    period_ends: list[pd.Period],
    min_months: int = 2,
    dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Mean of monthly values over the three months ending at each period end.

    Quarters with fewer than min_months monthly values are left empty. last_obs is the latest
    scene date inside the quarter (from dates, indexed by month) else the last month end.
    """
    rows = []
    for pe in period_ends:
        window = [pe - 2, pe - 1, pe]
        vals = monthly.reindex(window).dropna()
        n = len(vals)
        last = pd.NaT
        if n:
            if dates is not None:
                ds = pd.to_datetime(dates.reindex(vals.index)).dropna()
                last = ds.max() if len(ds) else pd.NaT
            else:
                last = vals.index.max().to_timestamp(how="end").normalize()
        rows.append(
            {
                "period_end": pe,
                "value": float(vals.mean()) if n >= min_months else np.nan,
                "n_months": n,
                "last_obs": last,
            }
        )
    return pd.DataFrame(rows)
