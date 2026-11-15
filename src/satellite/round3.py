"""Round 3: pixel-level proxies (Sentinel-2 and Sentinel-1) against filed company results.

Rules are pre-registered in E:/satellite-data/extension/round3/definitions.md. Scenes are the ones already selected
in earlier rounds (fetched by id). Per-scene results are cached under round3/cache, so analysis needs no download.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.enums import Resampling

from . import DATA_ROOT, battery, net, radar
from . import sentinel as sn
from .align import yoy_diff
from .aoi import AOI, load_aois
from .extension import ends_for, load_fundamentals, load_radar, result_series
from .round2 import pearson_boot_p, placebo_pass, shift_p
from .series import aggregate_periods, combine_areas, slug

EXT = Path(DATA_ROOT) / "extension"
OUT = EXT / "round3"
CACHE = OUT / "cache"
AREAS = Path(__file__).resolve().parents[2] / "data" / "areas.json"
GATE_GB = 0.3
MIN_N = 20
NDVI_BARE = 0.30
NDVI_DENSE = 0.60
RES_REF_SHARE = 0.5
S1_CANE_RES = 80.0
S1_RES_RES = 40.0

# id -> (tickers, label, expected sign (0 = two-sided))
PROXIES = {
    "Q1": (("KSL", "KTIS"), "Cane harvest-cleared pixel share (NDVI<0.30), quarter mean", 1),
    "Q2": (("KSL", "KTIS"), "Cane dense-canopy pixel share (NDVI>=0.60), trailing 12-month mean", 1),
    "Q3": (("KSL", "KTIS"), "Cane flooded-pixel fraction (VV<-16 dB), quarter mean", -1),
    "Q4": (("EASTW",), "Reservoir drawdown share of reference water pixels, trailing 12-month max", -1),
    "Q5": (("KSL", "KTIS"), "Cane VH pixel dispersion (SD dB), quarter mean", 0),
}
ROUND1_PATH = EXT / "series" / "tests_all.csv"
ROUND2_PATH = EXT / "round2" / "results_round2.csv"


# ---------------------------------------------------------------- RAM gate


def free_ram_ps() -> float:
    """Free physical RAM in GB from PowerShell (Win32_OperatingSystem.FreePhysicalMemory, in KB)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout.strip()
    return float(out) / 1048576.0


class Gate:
    """Waits until free RAM is at least `min_gb`; counts wait episodes and polls and logs them."""

    def __init__(self, min_gb: float = GATE_GB, probe=free_ram_ps, sleep=time.sleep, poll_s: float = 10.0,
                 log_path: Path | None = None, max_wait_s: float = 12 * 3600):
        self.min_gb, self.probe, self.sleep, self.poll_s = min_gb, probe, sleep, poll_s
        self.log_path, self.max_wait_s = log_path, max_wait_s
        self.events = 0
        self.polls = 0

    def _log(self, kind: str, gb: float) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.log_path.exists()
            with self.log_path.open("a", encoding="utf-8") as fh:
                if new:
                    fh.write("time,event,free_gb\n")
                fh.write(f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S},{kind},{gb:.2f}\n")

    def wait(self) -> float:
        gb = self.probe()
        if gb >= self.min_gb:
            return gb
        self.events += 1
        self._log("wait_start", gb)
        t0 = time.monotonic()
        while gb < self.min_gb:
            if time.monotonic() - t0 > self.max_wait_s:
                self._log("gave_up", gb)
                raise MemoryError(f"only {gb:.2f} GB free after {self.max_wait_s:.0f} s")
            self.sleep(self.poll_s)
            self.polls += 1
            gb = self.probe()
        self._log("wait_end", gb)
        return gb


# ---------------------------------------------------------------- pixel metrics (pure)


def s2_pixel_metrics(scl: np.ndarray, red: np.ndarray, nir: np.ndarray, mask: np.ndarray) -> dict:
    """Bare (NDVI < 0.30) and dense (NDVI >= 0.60) shares of land pixels in the AOI."""
    land = mask & ~sn.invalid_mask(scl) & (scl != sn.WATER_CLASS)
    nd = sn.ndvi(red, nir)
    land &= np.isfinite(nd)
    n = int(land.sum())
    if n == 0:
        return {"n_land": 0, "bare_frac": math.nan, "dense_frac": math.nan}
    v = nd[land]
    return {"n_land": n, "bare_frac": float((v < NDVI_BARE).mean()), "dense_frac": float((v >= NDVI_DENSE).mean())}


def s1_cane_pixel_metrics(vh: np.ndarray, vv: np.ndarray, mask: np.ndarray) -> dict:
    """Flooded-pixel fraction (VV dB < -16) and per-pixel VH dB standard deviation over valid AOI pixels."""
    both = mask & np.isfinite(vh) & np.isfinite(vv)
    n = int(mask.sum())
    frac = float(both.sum()) / n if n else 0.0
    if frac < radar.MIN_VALID:
        return {"valid_frac": frac, "flooded_frac": math.nan, "vh_sd_db": math.nan}
    return {
        "valid_frac": frac,
        "flooded_frac": float((radar.db(vv[both]) < radar.WATER_DB).mean()),
        "vh_sd_db": float(np.std(radar.db(vh[both]))),
    }


def water_state(vv: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """uint8 state per pixel: 0 invalid or outside AOI, 1 land, 2 open water (VV < -16 dB); and the valid share."""
    ok = mask & np.isfinite(vv)
    n = int(mask.sum())
    frac = float(ok.sum()) / n if n else 0.0
    st = np.zeros(vv.shape, dtype="uint8")
    st[ok] = 1
    st[ok & (radar.db(np.where(ok, vv, 1.0)) < radar.WATER_DB)] = 2
    return st, frac


def drawdown_series(stacks: dict[str, dict[pd.Period, np.ndarray]]) -> pd.Series:
    """Monthly pooled drawdown share over reference-water pixels.

    stacks[box][month] = uint8 state array (months failing the 90 percent rule are simply absent).
    Reference pixels of a box: water in at least RES_REF_SHARE of the months in which the pixel is valid.
    A month counts only if every box has an array for it.
    """
    ref = {}
    for box, d in stacks.items():
        arr = np.stack(list(d.values()))
        valid = (arr > 0).sum(axis=0)
        water = (arr == 2).sum(axis=0)
        ref[box] = (valid > 0) & (water >= RES_REF_SHARE * np.maximum(valid, 1))
    months = sorted(set.intersection(*[set(d) for d in stacks.values()]))
    out = {}
    for m in months:
        n_ref = n_wat = 0
        for box, d in stacks.items():
            st = d[m]
            r = ref[box] & (st > 0)
            n_ref += int(r.sum())
            n_wat += int((r & (st == 2)).sum())
        if n_ref > 0:
            out[m] = 1.0 - n_wat / n_ref
    return pd.Series(out, dtype=float).sort_index()


# ---------------------------------------------------------------- collection


def _load_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not str(r.get("status", "")).startswith("error"):
                    done[(r["area"], r["month"])] = r
    return done


def _append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def collect_s1(aois: list[AOI], kind: str, gate: Gate, log=print, cache: Path = CACHE) -> None:
    """Sentinel-1 pixel metrics for the cane areas (kind 'cane') or reservoir boxes (kind 'reservoir')."""
    cache.mkdir(parents=True, exist_ok=True)
    rad = load_radar(EXT / "cache")
    rad = rad[(rad["status"] == "ok") & rad["area"].isin([a.name for a in aois if a.kind == kind])]
    by_name = {a.name: a for a in aois}
    out = cache / ("s1_cane.jsonl" if kind == "cane" else "s1_res.jsonl")
    done = _load_jsonl(out)
    pols = ("vh", "vv") if kind == "cane" else ("vv",)
    res = S1_CANE_RES if kind == "cane" else S1_RES_RES
    stack_dir = cache / "res_stack"
    stack_dir.mkdir(exist_ok=True)
    for (month, sid), g in rad.groupby(["month", "scene_id"], sort=True):
        todo = [r for r in g.itertuples() if (r.area, str(month)) not in done]
        if not todo:
            continue
        gate.wait()
        try:
            feat = net.get(f"{radar.STAC}/collections/{radar.COLLECTION}/items/{sid}").json()
            arrs = {}
            for pol in pols:
                with radar.open_pol(feat, pol) as src:
                    arrs[pol] = {r.area: radar.read_window(src, by_name[r.area].bbox_lonlat(), res) for r in todo}
                net.forget(feat["assets"][pol]["href"])
            states = {}
            for r in todo:
                a = by_name[r.area]
                grid = arrs[pols[0]][r.area][1]
                mask = sn.aoi_mask(a, grid)
                base = {"area": r.area, "month": str(month), "scene_id": sid, "status": "ok"}
                if kind == "cane":
                    m = s1_cane_pixel_metrics(arrs["vh"][r.area][0], arrs["vv"][r.area][0], mask)
                    rec = base | m
                else:
                    st, frac = water_state(arrs["vv"][r.area][0], mask)
                    rec = base | {"valid_frac": frac}
                    if frac >= radar.MIN_VALID:
                        states[slug(r.area)] = st
                    else:
                        rec["status"] = "low_valid_frac"
                if kind == "cane" and rec["valid_frac"] < radar.MIN_VALID:
                    rec["status"] = "low_valid_frac"
                _append(out, rec)
            if states:
                np.savez_compressed(stack_dir / f"{month}.npz", **states)
            del arrs, states
            log(f"s1 {kind} {month} ok requests={net.STATS['requests']}")
        except Exception as exc:  # network or read failure: record, keep going, rerun resumes
            for r in todo:
                err = f"error: {type(exc).__name__}"
                _append(out, {"area": r.area, "month": str(month), "scene_id": sid, "status": err})
            log(f"s1 {kind} {month} error {type(exc).__name__}")


def collect_s2(aois: list[AOI], gate: Gate, log=print, cache: Path = CACHE) -> None:
    """Sentinel-2 pixel shares for the cane areas, one recorded scene at a time (red, NIR, SCL only)."""
    from pystac import Item

    cache.mkdir(parents=True, exist_ok=True)
    sn.OPENER = net.open_range
    opt = pd.DataFrame([json.loads(x) for x in (Path(DATA_ROOT) / "cache" / "monthly_records.jsonl")
                        .read_text("utf-8").splitlines() if x.strip()])
    opt = opt[(opt["status"] == "ok") & opt["area"].isin([a.name for a in aois if a.kind == "cane"])]
    opt = opt.drop_duplicates(["area", "month"], keep="last").sort_values(["month", "area"])
    by_name = {a.name: a for a in aois}
    out = cache / "s2_cane.jsonl"
    done = _load_jsonl(out)
    for r in opt.itertuples():
        if (r.area, r.month) in done:
            continue
        gate.wait()
        a = by_name[r.area]
        try:
            feat = net.get(f"{sn.STAC_URL}/collections/{sn.COLLECTION}/items/{r.scene_id}").json()
            item = Item.from_dict(feat)
            scl, grid, mask = sn.aoi_grid_for(a, item)
            band = {}
            for name in ("red", "nir"):
                dn, _ = sn.read_window(item.assets[name].href, a.bbox_lonlat(), a.resolution_m,
                                       Resampling.average, out_shape=grid.shape)
                net.forget(item.assets[name].href)
                band[name] = sn.scale_reflectance(dn, float(r.offset))
            net.forget(item.assets["scl"].href)
            m = s2_pixel_metrics(scl, band["red"], band["nir"], mask)
            _append(out, {"area": r.area, "month": r.month, "scene_id": r.scene_id, "status": "ok", **m})
            del scl, band, mask
            log(f"s2 {r.area} {r.month} ok requests={net.STATS['requests']}")
        except Exception as exc:
            _append(out, {"area": r.area, "month": r.month, "scene_id": r.scene_id,
                          "status": f"error: {type(exc).__name__}"})
            log(f"s2 {r.area} {r.month} error {type(exc).__name__}")


# ---------------------------------------------------------------- series and tests


def trailing(monthly: pd.Series, ends: list[pd.Period], how: str, min_months: int) -> pd.Series:
    """Statistic (mean, min or max) of the 12 monthly values ending at each quarter end."""
    fn = {"mean": np.mean, "min": np.min, "max": np.max}[how]
    out = {}
    for e in ends:
        w = monthly.reindex(pd.period_range(e - 11, e, freq="M")).dropna()
        if len(w) >= min_months:
            out[e] = float(fn(w.to_numpy()))
    return pd.Series(out, dtype=float).sort_index()


def quarter_mean(monthly: pd.Series, ends: list[pd.Period]) -> pd.Series:
    q = aggregate_periods(monthly, ends, min_months=2)
    return q.set_index("period_end")["value"].dropna()


def _recs(path: Path) -> pd.DataFrame:
    rows = list(_load_jsonl(path).values())
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["status"] == "ok"].copy()
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def monthly_inputs(areas_json: list[dict], cache: Path = CACHE) -> dict:
    """Monthly pixel series per company, from the per-scene cache."""
    s2, s1 = _recs(cache / "s2_cane.jsonl"), _recs(cache / "s1_cane.jsonl")
    inp: dict = {"bare": {}, "dense": {}, "flood": {}, "disp": {}}
    for t in ("KSL", "KTIS"):
        names = [a["name"] for a in areas_json if a["ticker"] == t]
        for key, df, col, mult in (("bare", s2, "bare_frac", 100.0), ("dense", s2, "dense_frac", 100.0),
                                   ("flood", s1, "flooded_frac", 100.0), ("disp", s1, "vh_sd_db", 1.0)):
            inp[key][t] = combine_areas(df, names, col, "mean") * mult if len(df) else pd.Series(dtype=float)
    res_dir = cache / "res_stack"
    stacks: dict[str, dict[pd.Period, np.ndarray]] = {}
    names = [slug(a["name"]) for a in areas_json if a["kind"] == "reservoir"]
    if res_dir.exists():
        for f in sorted(res_dir.glob("*.npz")):
            m = pd.Period(f.stem, freq="M")
            with np.load(f) as z:
                for n in names:
                    if n in z.files:
                        stacks.setdefault(n, {})[m] = z[n]
    inp["drawdown"] = (drawdown_series(stacks) * 100.0 if len(stacks) == len(names) else pd.Series(dtype=float))
    return inp


def build_proxy(pid: str, ticker: str, inp: dict, ends: list[pd.Period]) -> pd.Series:
    if pid == "Q1":
        return quarter_mean(inp["bare"][ticker], ends)
    if pid == "Q2":
        return trailing(inp["dense"][ticker], ends, "mean", 8)
    if pid == "Q3":
        return quarter_mean(inp["flood"][ticker], ends)
    if pid == "Q4":
        return trailing(inp["drawdown"], ends, "max", 9)
    if pid == "Q5":
        return quarter_mean(inp["disp"][ticker], ends)
    raise KeyError(pid)


def combined_adjusted(
    r3_raw: list[float], r1_path: Path = ROUND1_PATH, r2_path: Path = ROUND2_PATH
) -> tuple[list, list, int]:
    """Holm and Bonferroni over all rounds' raw p-values; returns the round 3 slices and the combined size."""
    p1 = pd.read_csv(r1_path)["perm_p"].tolist()
    p2 = pd.read_csv(r2_path)["raw_p"].tolist()
    allp = p1 + p2 + list(r3_raw)
    h = battery.holm(allp)
    m = sum(not math.isnan(p) for p in allp)
    h3 = h[len(p1) + len(p2):]
    b3 = [min(1.0, p * m) if not math.isnan(p) else math.nan for p in r3_raw]
    return h3, b3, len(allp)


def run(out: Path = OUT, cache: Path = CACHE, areas_path: str | Path = AREAS):
    areas_json = json.loads(Path(areas_path).read_text(encoding="utf-8"))["areas"]
    fund = load_fundamentals(EXT / "fundamentals" / "quarterly_primary.csv")
    inp = monthly_inputs(areas_json, cache)
    rows, tables = [], {}
    for pid, (tickers, label, sign) in PROXIES.items():
        for t in tickers:
            ends = ends_for(t, fund)
            q = build_proxy(pid, t, inp, ends)
            yoy = yoy_diff(q) if len(q) else q
            res = result_series(fund, t)
            tables[(pid, t)] = pd.concat([q.rename("proxy"), yoy.rename("proxy_yoy"), res], axis=1)
            degenerate = bool(len(q) == 0 or yoy.dropna().nunique() < 10)
            for target, col in (("revenue", "revenue_yoy"), ("profit", "profit_chg_scaled")):
                for lag in (0, 1):
                    row = {"proxy": pid, "company": t, "metric": target, "lag_q": lag, "label": label,
                           "expected_sign": sign, "degenerate": degenerate}
                    if degenerate:
                        row |= {"n": 0, "spearman_rho": math.nan, "pearson_r": math.nan, "pearson_boot_p": math.nan,
                                "raw_p": math.nan, "shift_p": math.nan, "boot_lo": math.nan, "boot_hi": math.nan,
                                "placebo_p": math.nan, "n_shifts": 0, "placebo_min_attainable": math.nan,
                                "placebo_pass": False, "loyo_stable": False, "rho_excl_2020_21": math.nan}
                        rows.append(row)
                        continue
                    px, ty = yoy.dropna(), res[col].dropna()
                    r = battery.run_test(px, ty, lag)
                    d = battery.paired(px, ty, lag)
                    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
                    keep = [p for p in px.index if p.year not in (2020, 2021)]
                    s = battery.paired(px.loc[keep], ty, lag)
                    row |= {
                        "n": r["n"], "spearman_rho": r["rho"], "pearson_r": r["r"],
                        "pearson_boot_p": pearson_boot_p(x, y) if len(d) >= 5 else math.nan,
                        "raw_p": r["perm_p"], "shift_p": shift_p(x, y),
                        "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"],
                        "placebo_p": r["placebo_p"], "n_shifts": r["n_placebo"],
                        "placebo_min_attainable": 1 / (r["n_placebo"] + 1) if r["n_placebo"] else math.nan,
                        "placebo_pass": placebo_pass(r), "loyo_stable": r["loyo_stable"],
                        "rho_excl_2020_21": battery._rho(s["x"].to_numpy(float), s["y"].to_numpy(float))
                        if len(s) >= 5 else math.nan,
                    }
                    rows.append(row)
    raw = [r["raw_p"] for r in rows]
    holm = battery.holm(raw)
    m = sum(not math.isnan(p) for p in raw)
    h_all, b_all, n_all = combined_adjusted(raw)
    for r, h, ha, ba in zip(rows, holm, h_all, b_all, strict=True):
        r["holm_p"] = h
        r["bonf_p"] = min(1.0, r["raw_p"] * m) if not math.isnan(r["raw_p"]) else math.nan
        r["holm_p_combined"] = ha
        r["bonf_p_combined"] = ba
        r["family_size"] = len(rows)
        r["combined_size"] = n_all
        r["n_ok"] = r["n"] >= MIN_N
        rho = r["spearman_rho"]
        r["sign_ok"] = bool(r["expected_sign"] == 0 or (not math.isnan(rho) and np.sign(rho) == r["expected_sign"]))
        r["signal"] = bool(
            r["n_ok"] and not r["degenerate"] and not math.isnan(h) and h < 0.05 and r["shift_p"] < 0.05
            and r["placebo_pass"] and r["loyo_stable"] and r["sign_ok"]
        )
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results_round3.csv", index=False)
    (out / "series").mkdir(exist_ok=True)
    for (pid, t), tab in tables.items():
        tab.to_csv(out / "series" / f"{pid}_{t}_quarterly.csv")
    return df, tables


def main_collect(which: tuple[str, ...] = ("s1_cane", "s1_res", "s2_cane")) -> Gate:
    aois = load_aois(AREAS)
    gate = Gate(log_path=OUT / "ram_gate_log.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "collect.log").open("a", encoding="utf-8", buffering=1)

    def log(msg: str) -> None:
        logf.write(f"{pd.Timestamp.now():%H:%M:%S} {msg}\n")

    log(f"start, free RAM {free_ram_ps():.2f} GB")
    if "s1_cane" in which:
        collect_s1(aois, "cane", gate, log)
    if "s1_res" in which:
        collect_s1(aois, "reservoir", gate, log)
    if "s2_cane" in which:
        collect_s2(aois, gate, log)
    log(f"done, gate wait episodes {gate.events}, polls {gate.polls}")
    return gate


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        g = main_collect()
        print("gate events", g.events, "polls", g.polls)
    else:
        from .round2 import chart_signals

        d, tabs = run()
        print(len(d), "comparisons;", int(d["signal"].sum()), "signals;", int((~d["n_ok"]).sum()), "underpowered")
        print(chart_signals(d, tabs, OUT, "round3_signal.png"))
