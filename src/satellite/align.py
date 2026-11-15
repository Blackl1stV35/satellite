"""Company results loading and year-on-year alignment of proxy versus results."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ANNOUNCE_LAG_DAYS = 45


def load_results(path: str | Path, ticker: str, fy_end_month: int = 12) -> pd.DataFrame:
    """Quarterly revenue and net profit (million baht) with period end and announce date.

    Fiscal quarter q of fiscal year f ends in month fy_end_month - 3 * (4 - q).
    The announce date is the quarter end plus 45 days (SET dates were not retrievable).
    """
    df = pd.read_csv(path)
    df = df[df["ticker"] == ticker].copy()
    ends = []
    for f, q in zip(df["fiscal"], df["quarter"], strict=True):
        month = fy_end_month - 3 * (4 - int(q))
        year = int(f)
        while month <= 0:
            month += 12
            year -= 1
        ends.append(pd.Period(year=year, month=month, freq="M"))
    df["period_end"] = ends
    df["announce_date"] = [
        p.to_timestamp(how="end").normalize() + pd.Timedelta(days=ANNOUNCE_LAG_DAYS) for p in ends
    ]
    out = df[["period_end", "announce_date", "revenue_mthb", "net_profit_mthb"]]
    return out.sort_values("period_end").reset_index(drop=True)


def yoy_pct(s: pd.Series) -> pd.Series:
    """(x_t - x_{t-4}) / |x_{t-4}| on a quarterly series indexed by Period M."""
    prev = s.copy()
    prev.index = prev.index + 12
    prev = prev.reindex(s.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (s - prev) / prev.abs()
    return out.replace([np.inf, -np.inf], np.nan)


def yoy_diff(s: pd.Series, scale: pd.Series | None = None) -> pd.Series:
    """x_t - x_{t-4}, optionally divided by the absolute year-earlier scale series."""
    prev = s.copy()
    prev.index = prev.index + 12
    prev = prev.reindex(s.index)
    d = s - prev
    if scale is not None:
        sc = scale.copy()
        sc.index = sc.index + 12
        sc = sc.reindex(s.index).abs()
        d = d / sc
    return d.replace([np.inf, -np.inf], np.nan)


def align_quarters(proxy_q: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Merge quarterly proxy and results and add year-on-year changes.

    proxy_q has columns period_end, value, n_months, last_obs (a Timestamp or NaT).
    """
    p = proxy_q.set_index("period_end")
    r = results.set_index("period_end")
    idx = r.index.union(p.index).sort_values()
    out = pd.DataFrame(index=idx)
    out["proxy"] = p["value"].reindex(idx)
    out["proxy_months"] = p["n_months"].reindex(idx)
    out["proxy_last_obs"] = p["last_obs"].reindex(idx)
    out["revenue"] = r["revenue_mthb"].reindex(idx)
    out["net_profit"] = r["net_profit_mthb"].reindex(idx)
    out["announce_date"] = r["announce_date"].reindex(idx)
    out["proxy_yoy"] = yoy_pct(out["proxy"])
    out["revenue_yoy"] = yoy_pct(out["revenue"])
    # Profit change is scaled by year-earlier revenue because profit crosses zero.
    out["profit_chg_scaled"] = yoy_diff(out["net_profit"], scale=out["revenue"])
    out["lead_days"] = (out["announce_date"] - out["proxy_last_obs"]).dt.days
    return out.reset_index(names="period_end")


def correlation(x: pd.Series, y: pd.Series, level: float = 0.95) -> dict:
    """Pearson r with Fisher z interval, Spearman rho, and n. NaNs dropped pairwise."""
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    res = {"n": n, "r": math.nan, "ci_low": math.nan, "ci_high": math.nan,
           "p": math.nan, "rho": math.nan}
    if n < 3:
        return res
    if d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return res
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    res.update(r=float(r), p=float(p), rho=float(stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]))
    if n > 3 and abs(r) < 1:
        z = math.atanh(r)
        se = 1 / math.sqrt(n - 3)
        zc = stats.norm.ppf(0.5 + level / 2)
        res["ci_low"] = math.tanh(z - zc * se)
        res["ci_high"] = math.tanh(z + zc * se)
    return res


def lagged_correlation(
    aligned: pd.DataFrame, proxy_col: str, result_col: str, lag_quarters: int
) -> dict:
    """Correlate proxy change in quarter t with the result change lag_quarters later."""
    a = aligned.set_index("period_end")
    y = a[result_col].copy()
    y.index = y.index - 3 * lag_quarters  # result of quarter t+k placed at t
    return correlation(a[proxy_col], y.reindex(a.index))


def detrended_correlation(x: pd.Series, y: pd.Series) -> dict:
    """Correlation after removing a straight-line time trend from each series (sensitivity)."""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 4:
        return correlation(x.iloc[:0], y.iloc[:0])
    t = np.arange(len(d), dtype=float)
    res = []
    for c in d.columns:
        coef = np.polyfit(t, d[c].to_numpy(dtype=float), 1)
        res.append(pd.Series(d[c].to_numpy(dtype=float) - np.polyval(coef, t), index=d.index))
    return correlation(res[0], res[1])
