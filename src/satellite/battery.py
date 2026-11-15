"""Test battery for proxy versus fundamental year-on-year changes (see extension definitions.md).

Every proxy-target-lag pair gets: n, Spearman, Pearson, permutation p, bootstrap interval,
whole-year-shift placebo, leave-one-year-out stability. A family-wide Holm correction is applied.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261111
N_PERM = 10_000
N_BOOT = 2_000


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(stats.spearmanr(x, y)[0])


def paired(proxy: pd.Series, target: pd.Series, lag_q: int) -> pd.DataFrame:
    """Pairs (proxy at quarter t, target at t + lag_q quarters), both indexed by quarter-end Period M."""
    y = target.copy()
    y.index = y.index - 3 * lag_q
    return pd.concat([proxy.rename("x"), y.rename("y")], axis=1).dropna()


def perm_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """Two-sided permutation p for Spearman rho (ranks shuffled)."""
    obs = _rho(x, y)
    if math.isnan(obs):
        return math.nan
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    n = len(x)
    cnt = 1
    for _ in range(n_perm):
        r = float(np.dot(rx, rng.permutation(ry)) / n)
        if abs(r) >= abs(obs) - 1e-12:
            cnt += 1
    return cnt / (n_perm + 1)


def boot_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    n = len(x)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = _rho(x[i], y[i])
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < 50:
        return math.nan, math.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def placebo_p(proxy: pd.Series, target: pd.Series, lag_q: int, max_years: int = 4, min_pairs: int = 8) -> dict:
    """Shift the proxy by whole-year offsets; share of shifted |rho| at least the observed |rho|."""
    base = paired(proxy, target, lag_q)
    obs = _rho(base["x"].to_numpy(), base["y"].to_numpy())
    vals = []
    for k in [o for o in range(-max_years, max_years + 1) if o != 0]:
        sh = proxy.copy()
        sh.index = sh.index + 12 * k
        d = paired(sh, target, lag_q)
        if len(d) < min_pairs:
            continue
        v = _rho(d["x"].to_numpy(), d["y"].to_numpy())
        if not math.isnan(v):
            vals.append(v)
    if math.isnan(obs) or not vals:
        return {"placebo_p": math.nan, "n_placebo": len(vals), "placebo_med_abs": math.nan}
    p = (1 + sum(abs(v) >= abs(obs) - 1e-12 for v in vals)) / (len(vals) + 1)
    return {"placebo_p": p, "n_placebo": len(vals), "placebo_med_abs": float(np.median(np.abs(vals)))}


def loyo(d: pd.DataFrame, full_rho: float) -> dict:
    """Leave-one-calendar-year-out (by proxy quarter end) stability of Spearman rho."""
    yrs = sorted({p.year for p in d.index})
    rhos = []
    for y in yrs:
        s = d[[p.year != y for p in d.index]]
        v = _rho(s["x"].to_numpy(), s["y"].to_numpy())
        if not math.isnan(v):
            rhos.append(v)
    if math.isnan(full_rho) or not rhos:
        return {"loyo_min": math.nan, "loyo_max": math.nan, "loyo_stable": False}
    same = all(np.sign(v) == np.sign(full_rho) for v in rhos)
    stable = bool(same and min(abs(v) for v in rhos) >= 0.5 * abs(full_rho))
    return {"loyo_min": float(min(rhos)), "loyo_max": float(max(rhos)), "loyo_stable": stable}


def run_test(proxy: pd.Series, target: pd.Series, lag_q: int, seed: int = SEED) -> dict:
    """Full battery on one proxy-target-lag pair. proxy and target are YoY series indexed by Period M."""
    rng = np.random.default_rng(seed)
    d = paired(proxy, target, lag_q)
    x, y = d["x"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float)
    res = {"n": len(d), "rho": _rho(x, y), "r": math.nan, "perm_p": math.nan, "boot_lo": math.nan,
           "boot_hi": math.nan}
    if len(d) >= 5 and np.ptp(x) > 0 and np.ptp(y) > 0:
        res["r"] = float(stats.pearsonr(x, y)[0])
        res["perm_p"] = perm_p(x, y, rng)
        res["boot_lo"], res["boot_hi"] = boot_ci(x, y, rng)
    res.update(placebo_p(proxy, target, lag_q))
    res.update(loyo(d, res["rho"]))
    return res


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN kept as NaN, not counted in the family size)."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [math.nan] * len(pvals)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def apply_family(rows: list[dict]) -> list[dict]:
    """Add Holm and Bonferroni adjusted p and the final verdict across the whole family of tests."""
    ps = [r["perm_p"] for r in rows]
    h = holm(ps)
    m = sum(not math.isnan(p) for p in ps)
    for r, hp in zip(rows, h, strict=True):
        r["family_size"] = m
        r["holm_p"] = hp
        r["bonf_p"] = min(1.0, r["perm_p"] * m) if not math.isnan(r["perm_p"]) else math.nan
        r["survives"] = bool(
            not math.isnan(hp) and hp < 0.05 and r["placebo_p"] < 0.05 and r["loyo_stable"]
        )
    return rows
