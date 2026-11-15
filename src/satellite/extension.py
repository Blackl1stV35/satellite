"""Extension analysis: radar and long-history optical proxies against primary-document results.

Outputs go to E:/satellite-data/extension (series CSVs, test table, charts). Definitions are fixed in
definitions.md before any test is run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import DATA_ROOT, battery  # noqa: E402
from .align import yoy_diff, yoy_pct  # noqa: E402
from .report import load_monthly  # noqa: E402
from .series import aggregate_periods, combine_areas  # noqa: E402

EXT = Path(DATA_ROOT) / "extension"
MIN_PAIRS = 8

# Proxy registry fixed before testing: id -> (ticker, source, column, areas filter, combine, yoy kind, label)
PROXIES = {
    "R1_KSL": ("KSL", "radar", "vh_db", "KSL", "mean", "diff", "Radar VH dB, KSL cane"),
    "R2_KSL": ("KSL", "radar", "ratio_db", "KSL", "mean", "diff", "Radar VH-VV dB, KSL cane"),
    "R1_KTIS": ("KTIS", "radar", "vh_db", "KTIS", "mean", "diff", "Radar VH dB, KTIS cane"),
    "R2_KTIS": ("KTIS", "radar", "ratio_db", "KTIS", "mean", "diff", "Radar VH-VV dB, KTIS cane"),
    "R3_EASTW": ("EASTW", "radar", "water_km2", "reservoir", "sum", "pct", "Radar water km2, 4 reservoirs"),
    "O1_KSL": ("KSL", "optical", "ndvi_mean", "KSL", "mean", "pct", "Optical NDVI, KSL cane"),
    "O1_KTIS": ("KTIS", "optical", "ndvi_mean", "KTIS", "mean", "pct", "Optical NDVI, KTIS cane"),
    "O2_EASTW": ("EASTW", "optical", "water_km2", "reservoir", "sum", "pct", "Optical water km2, 4 reservoirs"),
}
STEP_OF = {k: (1 if k.startswith("R") else 2) for k in PROXIES}


def load_radar(cache_dir: str | Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in (Path(cache_dir) / "radar_records.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(rows)
    df = df[~df["status"].astype(str).str.startswith("error")]
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df.drop_duplicates(["area", "month"], keep="last")


def monthly_proxy(pid: str, radar: pd.DataFrame, optical: pd.DataFrame, areas_json: list[dict]) -> pd.Series:
    ticker, source, col, flt, how, _, _ = PROXIES[pid]
    names = [a["name"] for a in areas_json if a["ticker"] == ticker]
    df = (radar if source == "radar" else optical)
    df = df[df["status"] == "ok"]
    s = combine_areas(df, names, col, how)
    return s.dropna()


def quarterly(monthly: pd.Series, ends: list[pd.Period], min_months: int = 2) -> pd.Series:
    q = aggregate_periods(monthly, ends, min_months=min_months)
    return q.set_index("period_end")["value"]


def proxy_yoy(q: pd.Series, kind: str) -> pd.Series:
    return yoy_diff(q) if kind == "diff" else yoy_pct(q)


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["period_end"] = pd.PeriodIndex(df["period_end"], freq="M")
    return df


def result_series(fund: pd.DataFrame, ticker: str) -> pd.DataFrame:
    f = fund[fund["ticker"] == ticker].set_index("period_end").sort_index()
    rev = f["revenue_mthb"]
    prof = f["net_profit_mthb"]
    return pd.DataFrame(
        {
            "revenue": rev,
            "net_profit": prof,
            "revenue_yoy": yoy_pct(rev),
            "profit_chg_scaled": yoy_diff(prof, scale=rev),
        }
    )


def ends_for(ticker: str, fund: pd.DataFrame) -> list[pd.Period]:
    """Quarter-end months seen in the filings, plus the year-earlier ones needed for year-on-year."""
    pe = sorted(fund[fund["ticker"] == ticker]["period_end"].unique())
    allp = sorted(set(pe) | {p - 12 for p in pe})
    return allp


def run_all(radar: pd.DataFrame, optical: pd.DataFrame, fund: pd.DataFrame, areas_json: list[dict]):
    """Build every proxy, run the battery on proxy x target x lag, apply the family correction."""
    rows, tables = [], {}
    for pid, (ticker, _, _, _, _, kind, label) in PROXIES.items():
        mon = monthly_proxy(pid, radar, optical, areas_json)
        ends = ends_for(ticker, fund)
        q = quarterly(mon, ends)
        yoy = proxy_yoy(q, kind)
        res = result_series(fund, ticker)
        tab = pd.concat([q.rename("proxy"), yoy.rename("proxy_yoy"), res], axis=1)
        tables[pid] = {"monthly": mon, "table": tab}
        for target, tcol in (("revenue", "revenue_yoy"), ("profit", "profit_chg_scaled")):
            for lag in (0, 1):
                r = battery.run_test(yoy.dropna(), res[tcol].dropna(), lag)
                keep = [p for p in yoy.dropna().index if p.year not in (2020, 2021)]
                sens = battery.paired(yoy.dropna().loc[keep], res[tcol].dropna(), lag)
                sens_rho = battery._rho(sens["x"].to_numpy(), sens["y"].to_numpy()) if len(sens) >= 5 else math.nan
                rows.append(
                    {
                        "proxy": pid,
                        "step": STEP_OF[pid],
                        "ticker": ticker,
                        "label": label,
                        "target": target,
                        "lag_q": lag,
                        **r,
                        "rho_excl_2020_21": sens_rho,
                        "n_excl_2020_21": len(sens),
                    }
                )
    rows = battery.apply_family(rows)
    return pd.DataFrame(rows), tables


def lead_days(fund: pd.DataFrame) -> pd.DataFrame:
    """Median days from quarter end to filing, per ticker, where the filing timestamp is known."""
    f = fund.dropna(subset=["filing_date"]).copy()
    f["days"] = (pd.to_datetime(f["filing_date"]) - f["period_end"].dt.to_timestamp(how="end").dt.normalize()).dt.days
    return f.groupby("ticker")["days"].describe()[["count", "min", "50%", "max"]]


def chart_step(tables: dict, pids: list[str], path: Path, title: str) -> None:
    """One phone-width chart: standardised proxy YoY and result YoY per proxy, stacked panels."""
    fig, axes = plt.subplots(len(pids), 1, figsize=(5.2, 2.5 * len(pids)), squeeze=False)
    for ax, pid in zip(axes[:, 0], pids, strict=True):
        t = tables[pid]["table"]
        x = t["proxy_yoy"]
        y = t["revenue_yoy"]
        z = lambda s: (s - s.mean()) / s.std() if s.notna().sum() > 2 else s  # noqa: E731
        idx = t.index.to_timestamp(how="end")
        ax.plot(idx, z(x).to_numpy(), color="#1b6ca8", lw=1.6, label="proxy YoY")
        ax.plot(idx, z(y).to_numpy(), color="#d9822b", lw=1.6, label="revenue YoY")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(PROXIES[pid][6], fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_xlim(pd.Timestamp("2017-12-31"), pd.Timestamp("2026-12-31"))
        ax.legend(fontsize=6, loc="upper left", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_port(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if len(df):
        idx = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
        ax.plot(idx, df["vessels"].to_numpy(), color="#1b6ca8", lw=1.2, marker="o", ms=2.5)
        roll = df["vessels"].rolling(3, min_periods=2).mean()
        ax.plot(idx, roll.to_numpy(), color="#d9822b", lw=1.8, label="3-month mean")
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("Laem Chabang: vessel detections per radar scene", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)



def coverage(mon: pd.Series, start: str = "2017-01", end: str = "2026-09") -> dict:
    full = pd.period_range(start, end, freq="M")
    got = mon.reindex(full).notna()
    first = str(mon.index.min()) if len(mon) else ""
    last = str(mon.index.max()) if len(mon) else ""
    return {"months_total": len(full), "months_with_value": int(got.sum()), "first_month": first, "last_month": last}


def main(areas_path: str | Path, out_dir: str | Path = EXT) -> pd.DataFrame:
    """Run the whole extension analysis and write series, test table and charts to out_dir."""
    out_dir = Path(out_dir)
    (out_dir / "series").mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)
    areas_json = json.loads(Path(areas_path).read_text(encoding="utf-8"))["areas"]
    radar = load_radar(out_dir / "cache")
    optical = load_monthly(Path(DATA_ROOT) / "cache")
    fund = load_fundamentals(out_dir / "fundamentals" / "quarterly_primary.csv")
    res, tables = run_all(radar, optical, fund, areas_json)
    res.to_csv(out_dir / "series" / "tests_all.csv", index=False)
    cov = []
    for pid, d in tables.items():
        d["monthly"].rename(pid).to_csv(out_dir / "series" / f"{pid}_monthly.csv")
        d["table"].to_csv(out_dir / "series" / f"{pid}_quarterly.csv")
        cov.append({"proxy": pid, **coverage(d["monthly"])})
    pd.DataFrame(cov).to_csv(out_dir / "series" / "coverage.csv", index=False)
    r_ids = [p for p in PROXIES if p.startswith("R")]
    o_ids = [p for p in PROXIES if p.startswith("O")]
    chart_step(tables, r_ids, out_dir / "charts" / "step1_radar.png", "Step 1: radar proxy vs revenue, year on year")
    chart_step(tables, o_ids, out_dir / "charts" / "step2_optical_long.png",
               "Step 2: optical proxy vs revenue, 2017 to 2026")
    return res


def extra_checks(out_dir: str | Path = EXT) -> dict:
    """Cross-sensor agreement, publication timing and minimum detectable correlation (for the report)."""
    from scipy import stats

    out_dir = Path(out_dir)
    s = out_dir / "series"
    res: dict = {}

    def load(pid):
        return pd.read_csv(s / f"{pid}_monthly.csv", index_col=0).iloc[:, 0].pipe(
            lambda x: x.set_axis(pd.PeriodIndex(x.index, freq="M")))

    def anomaly(x):
        return x - x.groupby(x.index.month).transform("mean")

    for a, b in (("R3_EASTW", "O2_EASTW"), ("R1_KSL", "O1_KSL"), ("R1_KTIS", "O1_KTIS")):
        xa, xb = load(a), load(b)
        d = pd.concat([xa, xb], axis=1).dropna()
        da = pd.concat([anomaly(xa), anomaly(xb)], axis=1).dropna()
        res[f"{a} vs {b}"] = {
            "n_months": len(d),
            "spearman_levels": float(stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]),
            "spearman_seasonally_adjusted": float(stats.spearmanr(da.iloc[:, 0], da.iloc[:, 1])[0]),
        }
    fund = load_fundamentals(out_dir / "fundamentals" / "quarterly_primary.csv")
    f = fund[fund["filing_date"].astype(str).str.len() == 10].copy()
    f["days"] = (pd.to_datetime(f["filing_date"]) - f["period_end"].dt.to_timestamp(how="end").dt.normalize()).dt.days
    days = f.groupby("ticker")["days"].agg(["count", "min", "median", "max"])
    res["filing_days_after_quarter_end"] = days.to_dict("index")
    crit = {}
    for n in (20, 25, 30, 34):
        t = stats.t.ppf(1 - 0.05 / 32 / 2, n - 2)
        crit[n] = float(t / math.sqrt(n - 2 + t * t))
    res["rho_needed_after_32_test_correction"] = crit
    return res
