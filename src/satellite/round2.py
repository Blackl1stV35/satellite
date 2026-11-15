"""Round 2: new proxies derived from cached monthly series, tested against filed results.

Rules are pre-registered in E:/satellite-data/extension/round2/definitions.md. Reuses battery.py.
No downloads: inputs are the monthly series and radar records already on disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import DATA_ROOT, battery
from .align import yoy_diff, yoy_pct
from .extension import ends_for, load_fundamentals, load_radar, result_series
from .series import aggregate_periods, combine_areas

EXT = Path(DATA_ROOT) / "extension"
OUT = EXT / "round2"
MIN_N = 20

# id -> (tickers, label, yoy kind)
PROXIES = {
    "N1": (("KSL", "KTIS"), "Cane NDVI, trailing 12-month mean", "pct"),
    "N2": (("KSL", "KTIS"), "Cane radar VH dB, trailing 12-month mean", "diff"),
    "N3": (("KSL", "KTIS"), "Cane-area radar VV dB (wetness), quarter mean", "diff"),
    "N4": (("EASTW",), "Reservoir water km2, trailing 12-month minimum", "pct"),
    "N5": (("EASTW",), "Laem Chabang vessels per scene, quarter mean", "pct"),
}


def trailing(monthly: pd.Series, ends: list[pd.Period], how: str, min_months: int) -> pd.Series:
    out = {}
    for e in ends:
        w = monthly.reindex(pd.period_range(e - 11, e, freq="M")).dropna()
        if len(w) >= min_months:
            out[e] = float(w.mean() if how == "mean" else w.min())
    return pd.Series(out, dtype=float).sort_index()


def quarter_mean(monthly: pd.Series, ends: list[pd.Period]) -> pd.Series:
    q = aggregate_periods(monthly, ends, min_months=2)
    return q.set_index("period_end")["value"].dropna()


def monthly_inputs(areas_json: list[dict], ext: Path = EXT) -> dict:
    s = ext / "series"

    def load(name: str) -> pd.Series:
        x = pd.read_csv(s / f"{name}_monthly.csv", index_col=0).iloc[:, 0]
        return x.set_axis(pd.PeriodIndex(x.index, freq="M"))

    radar = load_radar(ext / "cache")
    radar = radar[radar["status"] == "ok"]
    vv = {}
    for t in ("KSL", "KTIS"):
        names = [a["name"] for a in areas_json if a["ticker"] == t]
        vv[t] = combine_areas(radar, names, "vv_db", "mean").dropna()
    port = pd.read_csv(s / "P1_port_vessels_monthly.csv")
    port = port[port["status"] == "ok"]
    p1 = port.set_index(pd.PeriodIndex(port["month"], freq="M"))["vessels"].astype(float)
    return {
        "O1": {t: load(f"O1_{t}") for t in ("KSL", "KTIS")},
        "R1": {t: load(f"R1_{t}") for t in ("KSL", "KTIS")},
        "VV": vv,
        "R3": load("R3_EASTW"),
        "P1": p1,
    }


def build_proxy(pid: str, ticker: str, inp: dict, ends: list[pd.Period]) -> pd.Series:
    """Quarterly proxy level at the given quarter-end months."""
    if pid == "N1":
        return trailing(inp["O1"][ticker], ends, "mean", 8)
    if pid == "N2":
        return trailing(inp["R1"][ticker], ends, "mean", 9)
    if pid == "N3":
        return quarter_mean(inp["VV"][ticker], ends)
    if pid == "N4":
        return trailing(inp["R3"], ends, "min", 9)
    if pid == "N5":
        return quarter_mean(inp["P1"], ends)
    raise KeyError(pid)


def shift_p(x: np.ndarray, y: np.ndarray) -> float:
    """Circular-shift permutation p for Spearman rho (n-1 shifts plus the observed pairing)."""
    obs = battery._rho(x, y)
    n = len(x)
    if math.isnan(obs) or n < 5:
        return math.nan
    cnt = 1
    for k in range(1, n):
        v = battery._rho(np.roll(x, k), y)
        if not math.isnan(v) and abs(v) >= abs(obs) - 1e-12:
            cnt += 1
    return cnt / n


def pearson_boot_p(x: np.ndarray, y: np.ndarray, seed: int = battery.SEED, n_boot: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            vals.append(float(stats.pearsonr(x[i], y[i])[0]))
    if len(vals) < 50:
        return math.nan
    v = np.array(vals)
    return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))


def placebo_pass(res: dict) -> bool:
    """Registered rule: observed |rho| exceeds every shifted |rho| (placebo p at its attainable minimum)."""
    n_pl = res.get("n_placebo", 0)
    p = res.get("placebo_p", math.nan)
    return bool(n_pl > 0 and not math.isnan(p) and abs(p - 1 / (n_pl + 1)) < 1e-12)


def run(ext: Path = EXT, out: Path = OUT, areas_path: str | Path | None = None):
    areas_path = areas_path or Path(__file__).resolve().parents[2] / "data" / "areas.json"
    areas_json = json.loads(Path(areas_path).read_text(encoding="utf-8"))["areas"]
    fund = load_fundamentals(ext / "fundamentals" / "quarterly_primary.csv")
    inp = monthly_inputs(areas_json, ext)
    rows, tables = [], {}
    for pid, (tickers, label, kind) in PROXIES.items():
        for t in tickers:
            ends = ends_for(t, fund)
            q = build_proxy(pid, t, inp, ends)
            yoy = yoy_diff(q) if kind == "diff" else yoy_pct(q)
            res = result_series(fund, t)
            tables[(pid, t)] = pd.concat([q.rename("proxy"), yoy.rename("proxy_yoy"), res], axis=1)
            for target, col in (("revenue", "revenue_yoy"), ("profit", "profit_chg_scaled")):
                for lag in (0, 1):
                    px, ty = yoy.dropna(), res[col].dropna()
                    r = battery.run_test(px, ty, lag)
                    d = battery.paired(px, ty, lag)
                    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
                    keep = [p for p in px.index if p.year not in (2020, 2021)]
                    s = battery.paired(px.loc[keep], ty, lag)
                    rows.append(
                        {
                            "proxy": pid, "company": t, "metric": target, "lag_q": lag, "label": label,
                            "n": r["n"], "spearman_rho": r["rho"], "pearson_r": r["r"],
                            "pearson_boot_p": pearson_boot_p(x, y) if len(d) >= 5 else math.nan,
                            "raw_p": r["perm_p"], "shift_p": shift_p(x, y),
                            "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"],
                            "placebo_p": r["placebo_p"], "n_shifts": r["n_placebo"],
                            "placebo_min_attainable": 1 / (r["n_placebo"] + 1) if r["n_placebo"] else math.nan,
                            "placebo_pass": placebo_pass(r),
                            "loyo_stable": r["loyo_stable"],
                            "rho_excl_2020_21": battery._rho(s["x"].to_numpy(float), s["y"].to_numpy(float))
                            if len(s) >= 5 else math.nan,
                        }
                    )
    holm_p = battery.holm([r["raw_p"] for r in rows])
    m = sum(not math.isnan(r["raw_p"]) for r in rows)
    for r, h in zip(rows, holm_p, strict=True):
        r["holm_p"] = h
        r["bonf_p"] = min(1.0, r["raw_p"] * m) if not math.isnan(r["raw_p"]) else math.nan
        r["family_size"] = len(rows)
        r["n_ok"] = r["n"] >= MIN_N
        r["signal"] = bool(
            r["n_ok"] and not math.isnan(h) and h < 0.05 and r["shift_p"] < 0.05
            and r["placebo_pass"] and r["loyo_stable"]
        )
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results_round2.csv", index=False)
    (out / "series").mkdir(exist_ok=True)
    for (pid, t), tab in tables.items():
        tab.to_csv(out / "series" / f"{pid}_{t}_quarterly.csv")
    return df, tables


def chart_signals(df: pd.DataFrame, tables: dict, out: Path = OUT, name: str = "round2_signal.png") -> Path | None:
    """One chart, only when at least one comparison meets the registered signal rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = df[df["signal"]]
    if sig.empty:
        return None
    fig, axes = plt.subplots(len(sig), 1, figsize=(5.2, 2.6 * len(sig)), squeeze=False)
    for ax, (_, r) in zip(axes[:, 0], sig.iterrows(), strict=True):
        t = tables[(r["proxy"], r["company"])]
        col = "revenue_yoy" if r["metric"] == "revenue" else "profit_chg_scaled"

        def z(s):
            return (s - s.mean()) / s.std()

        idx = t.index.to_timestamp(how="end")
        ax.plot(idx, z(t["proxy_yoy"]).to_numpy(), color="#1b6ca8", lw=1.6, label=f"{r['proxy']} YoY")
        y = t[col].copy()
        y.index = y.index - 3 * int(r["lag_q"])
        ax.plot(idx, z(y.reindex(t.index)).to_numpy(), color="#d9822b", lw=1.6,
                label=f"{r['metric']} YoY (lag {r['lag_q']})")
        ax.set_title(f"{r['company']} {r['label']}, rho {r['spearman_rho']:.2f}, Holm p {r['holm_p']:.3f}",
                     fontsize=8, loc="left")
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


if __name__ == "__main__":
    d, tabs = run()
    print(len(d), "comparisons;", int(d["signal"].sum()), "signals;", int((~d["n_ok"]).sum()), "underpowered")
    print(chart_signals(d, tabs))
