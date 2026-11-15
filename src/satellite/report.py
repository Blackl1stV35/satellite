"""Series CSVs, charts and the short markdown report for the pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import DATA_ROOT  # noqa: E402
from .align import (
    align_quarters,
    correlation,
    detrended_correlation,
    lagged_correlation,
    load_results,
)  # noqa: E402
from .aoi import AOI  # noqa: E402
from .series import aggregate_periods, combine_areas, quarter_end_months  # noqa: E402

PILOT = Path(DATA_ROOT) / "pilot"

# Company set-up: proxy column, how areas combine, fiscal year end month, label.
COMPANIES = {
    "KSL": {"col": "ndvi_mean", "how": "mean", "fy_end": 10, "label": "Mean NDVI, cane catchments"},
    "KTIS": {"col": "ndvi_mean", "how": "mean", "fy_end": 12, "label": "Mean NDVI, cane catchment"},
    "EASTW": {"col": "water_km2", "how": "sum", "fy_end": 12, "label": "Water area km2, 4 reservoirs"},
}


def load_monthly(cache_dir: str | Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in (Path(cache_dir) / "monthly_records.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(rows)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df.drop_duplicates(["area", "month"], keep="last")


def company_proxy(df: pd.DataFrame, aois: list[AOI], ticker: str):
    """Monthly proxy series and scene dates for one company."""
    cfg = COMPANIES[ticker]
    names = [a.name for a in aois if a.ticker == ticker]
    ok = df[df["status"] == "ok"]
    s = combine_areas(ok, names, cfg["col"], cfg["how"])
    dates = ok[ok["area"].isin(names)].groupby("month")["scene_date"].max()
    return s, dates, names


def analyse(df: pd.DataFrame, aois: list[AOI], results_csv: str | Path):
    """Return per-company dict with monthly, aligned quarterly table and statistics."""
    out = {}
    for ticker, cfg in COMPANIES.items():
        monthly, dates, names = company_proxy(df, aois, ticker)
        res = load_results(results_csv, ticker, cfg["fy_end"])
        ends = quarter_end_months(res["period_end"].min() - 12, res["period_end"].max(), cfg["fy_end"])
        pq = aggregate_periods(monthly, ends, min_months=2, dates=dates)
        al = align_quarters(pq, res)
        stats_ = {}
        for lag in (0, 1):
            stats_[f"rev_lag{lag}"] = lagged_correlation(al, "proxy_yoy", "revenue_yoy", lag)
            stats_[f"profit_lag{lag}"] = lagged_correlation(al, "proxy_yoy", "profit_chg_scaled", lag)
        stats_["rev_detrended"] = detrended_correlation(al["proxy_yoy"], al["revenue_yoy"])
        keep = al[(al["period_end"] < pd.Period("2020-01", "M")) | (al["period_end"] > pd.Period("2021-12", "M"))]
        stats_["rev_excl_2020_21"] = correlation(keep["proxy_yoy"], keep["revenue_yoy"])
        out[ticker] = {"monthly": monthly, "dates": dates, "aligned": al, "stats": stats_,
                       "areas": names, "results": res}
    return out


def _fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def company_chart(ticker: str, d: dict, path: Path) -> None:
    cfg = COMPANIES[ticker]
    al = d["aligned"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), height_ratios=[1.3, 1])
    m = d["monthly"]
    ax1.plot(m.index.to_timestamp(), m.values, "o-", ms=3, lw=1, color="#2a6f97")
    ann = al["announce_date"].dropna()
    ann = ann[(ann >= m.index.min().to_timestamp()) & (ann <= m.index.max().to_timestamp(how="end"))]
    for i, a in enumerate(ann):
        ax1.axvline(a, color="#c1121f", lw=0.7, alpha=0.6, label="results date (quarter end + 45d)" if i == 0 else None)
    ax1.set_ylabel(cfg["label"])
    ax1.set_title(f"{ticker}: monthly satellite proxy with results dates")
    ax1.legend(loc="upper right")
    a2 = al.dropna(subset=["proxy_yoy", "revenue_yoy"], how="all")
    x = a2["period_end"].dt.to_timestamp(how="end")
    ax2.plot(x, a2["proxy_yoy"] * 100, "o-", color="#2a6f97", label="proxy year-on-year %")
    ax2.plot(x, a2["revenue_yoy"] * 100, "s-", color="#6a994e", label="revenue year-on-year %")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("percent")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_chart(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (ticker, d) in zip(axes, res.items(), strict=True):
        al = d["aligned"].dropna(subset=["proxy_yoy", "revenue_yoy"])
        ax.scatter(al["proxy_yoy"] * 100, al["revenue_yoy"] * 100, s=28, color="#2a6f97")
        st = d["stats"]["rev_lag0"]
        ax.set_title(f"{ticker}: n={st['n']}, r={_fmt(st['r'])}")
        ax.set_xlabel("proxy year-on-year %")
        ax.set_ylabel("revenue year-on-year %")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
    fig.suptitle("Proxy change versus revenue change, same quarter (descriptive, tiny samples)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(res: dict, aois: list[AOI], df: pd.DataFrame, path: Path, extra: str = "") -> None:
    rows = []
    for ticker, d in res.items():
        s0, sp = d["stats"]["rev_lag0"], d["stats"]["profit_lag0"]
        s1 = d["stats"]["rev_lag1"]
        al = d["aligned"]
        lead = al["lead_days"].dropna()
        rows.append(
            f"| {ticker} | {COMPANIES[ticker]['label']} | {s0['n']} | "
            f"{_fmt(s0['r'])} ({_fmt(s0['ci_low'])} to {_fmt(s0['ci_high'])}) | "
            f"{_fmt(sp['r'])} ({_fmt(sp['ci_low'])} to {_fmt(sp['ci_high'])}) | "
            f"{_fmt(s1['r'])} (n={s1['n']}) | "
            f"{_fmt(d['stats']['rev_detrended']['r'])} | "
            f"{int(lead.median()) if len(lead) else 'n/a'} |"
        )
    ok = df[df["status"] == "ok"]
    cov = ok.groupby("area").size()
    tot = df.groupby("area").size()
    cov_lines = [f"- {a}: {cov.get(a, 0)} of {tot[a]} months with a usable scene" for a in tot.index]
    src_lines = [f"- {a.name} ({a.ticker}): {a.source}" for a in aois]
    per = "; ".join(
        f"{t} r={_fmt(d['stats']['rev_lag0']['r'])} (95% CI {_fmt(d['stats']['rev_lag0']['ci_low'])} to "
        f"{_fmt(d['stats']['rev_lag0']['ci_high'])}, n={d['stats']['rev_lag0']['n']})"
        for t, d in res.items()
    )
    ns = [d["stats"]["rev_lag0"]["n"] for d in res.values()]
    md = f"""# Satellite proxies versus Thai quarterly results: descriptive pilot

Research only. This is not investment advice and gives no buy, sell or hold view.

## Bottom line

A free Sentinel-2 proxy (vegetation index over sugar catchments, water area over reservoirs) was lined up against year-on-year revenue changes for three SET companies, using {min(ns)} to {max(ns)} quarters each. Same-quarter correlations with revenue: {per}. With samples this small, and with overlapping quarters that share seasonality, weather, prices and trends, these numbers are not evidence of a relationship, and a nonzero coefficient can come from a shared trend alone. The two sugar companies show no clear same-quarter alignment with revenue. The reservoir proxy shows a positive coefficient that survives removing a straight-line trend (see the trend-removed column), but it rests on {ns[2]} overlapping quarters and cannot be separated from shared drivers such as drought years or the 2020 demand shock, so it is a lead worth checking on a larger sample, not a finding. With the 2020 and 2021 quarters left out, EASTW r is {_fmt(res['EASTW']['stats']['rev_excl_2020_21']['r'])} on n={res['EASTW']['stats']['rev_excl_2020_21']['n']}. The proxy is observable about two months before the assumed results date by construction, because the quarter ends 45 days before it.

## Results

Same-quarter year-on-year change of the proxy against year-on-year change of revenue, and against net profit change scaled by year-earlier revenue (profit crosses zero, so a percent change is unstable). Pearson r with a Fisher 95% interval. The lead column is the median number of days between the last usable scene in the quarter and the assumed results date, so a positive number means the proxy was observable before the announcement.

| Ticker | Proxy | Quarters (n) | r vs revenue (95% CI) | r vs profit change (95% CI) | r proxy(t) vs revenue(t+1) | r vs revenue, trend removed | Median lead days |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Three risks

- Sample size: each coefficient rests on a few dozen quarters at most, with serial overlap. Intervals are wide and several quarters were dropped for cloud. No inference about predictive value is justified.
- Proxy validity: catchment radii (12 km) and reservoir boxes are assumptions, mill centres are accurate to a few km, cane land mixes with other crops, sugar revenue depends on prices, export quotas and inventory timing, and EASTW reservoirs are state owned and only partly drive its sales.
- Data quality: results dates are quarter end plus 45 days, not actual SET filing dates (the SET site blocks automated access). Quarterly results come from a public aggregator that republishes SET filings, cross-checked against yfinance for the latest five quarters only.

## Coverage

{chr(10).join(cov_lines)}

## Areas and sources

{chr(10).join(src_lines)}

## Method notes

- Scenes: Sentinel-2 L2A via Earth Search STAC. One best scene per month and area, chosen by the clear share of the area from the scene classification layer (classes 0, 1, 3, 8, 9, 10 treated as unusable), at least 50 percent clear.
- Reflectance: DN * 0.0001, plus -0.1 only for items whose DN still contains the +1000 offset (processing baseline 04.00 or later and earthsearch:boa_offset_applied false). Items flagged true carry no offset in DN, although their asset metadata still lists -0.1; this was confirmed on water pixels.
- NDVI = (NIR - red) / (NIR + red) over clear, non-water pixels. Water = MNDWI = (green - SWIR1) / (green + SWIR1) above 0.
- Quarterly proxy = mean of the monthly values in the company's fiscal quarter (KSL fiscal year ends in October), needing at least two months. Year-on-year change is against the same quarter a year earlier.
{extra}
"""
    path.write_text(md, encoding="utf-8")


def crosscheck_yfinance(results_csv: str | Path) -> str:
    """Compare the aggregator quarters with yfinance for the quarters yfinance still holds."""
    try:
        import yfinance as yf
    except ImportError:
        return "- yfinance cross-check skipped (not installed)."
    lines = []
    for ticker, cfg in COMPANIES.items():
        try:
            inc = yf.Ticker(f"{ticker}.BK").quarterly_income_stmt
            res = load_results(results_csv, ticker, cfg["fy_end"]).set_index("period_end")
            diffs_np, diffs_rev = [], []
            for col in inc.columns:
                pe = pd.Period(col, freq="M")
                if pe not in res.index:
                    continue
                npf = inc.loc["Net Income Common Stockholders", col] / 1e6
                rev = inc.loc["Total Revenue", col] / 1e6
                if pd.notna(npf):
                    diffs_np.append(abs(npf - res.loc[pe, "net_profit_mthb"]))
                if pd.notna(rev):
                    diffs_rev.append(abs(rev - res.loc[pe, "revenue_mthb"]))
            lines.append(
                f"- {ticker}: {len(diffs_np)} quarters compared with yfinance, max net profit difference "
                f"{max(diffs_np):.1f} million baht, max revenue difference {max(diffs_rev):.1f} million baht."
                if diffs_np else f"- {ticker}: no overlapping quarters in yfinance."
            )
        except Exception as exc:
            lines.append(f"- {ticker}: yfinance cross-check failed ({type(exc).__name__}).")
    return "\n".join(lines)


def run_pilot(aois: list[AOI], cache_dir: str | None = None, results_csv: str | None = None):
    cache_dir = cache_dir or f"{DATA_ROOT}/cache"
    results_csv = results_csv or str(PILOT / "fundamentals" / "finnomena_quarterly.csv")
    (PILOT / "series").mkdir(parents=True, exist_ok=True)
    (PILOT / "charts").mkdir(parents=True, exist_ok=True)
    df = load_monthly(cache_dir)
    res = analyse(df, aois, results_csv)
    df.assign(month=df["month"].astype(str)).to_csv(PILOT / "series" / "monthly_all_areas.csv", index=False)
    for ticker, d in res.items():
        d["monthly"].rename("proxy").to_frame().assign(
            scene_date=d["dates"].reindex(d["monthly"].index)
        ).to_csv(PILOT / "series" / f"{ticker}_monthly_proxy.csv", index_label="month")
        d["aligned"].to_csv(PILOT / "series" / f"{ticker}_quarterly_aligned.csv", index=False)
        company_chart(ticker, d, PILOT / "charts" / f"{ticker}_proxy.png")
    summary_chart(res, PILOT / "charts" / "summary_scatter.png")
    stats_rows = []
    for ticker, d in res.items():
        for k, v in d["stats"].items():
            stats_rows.append({"ticker": ticker, "test": k, **v})
    pd.DataFrame(stats_rows).to_csv(PILOT / "series" / "correlations.csv", index=False)
    extra = "\n## Results data cross-check\n\n" + crosscheck_yfinance(results_csv) + "\n"
    write_report(res, aois, df, PILOT / "report.md", extra)
    return res


__all__ = ["run_pilot", "analyse", "correlation"]
