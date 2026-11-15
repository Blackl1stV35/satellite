import json
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.warp import transform

from satellite import sentinel as sn
from satellite.align import (
    align_quarters,
    correlation,
    lagged_correlation,
    load_results,
    yoy_diff,
    yoy_pct,
)
from satellite.aoi import AOI, load_aois
from satellite.series import aggregate_periods, combine_areas, quarter_end_months

# ---------- scaling ----------


def test_scale_reflectance_with_offset_and_nodata():
    dn = np.array([0, 1000, 2000, 10000], dtype="uint16")
    out = sn.scale_reflectance(dn, -0.1)
    assert np.isnan(out[0])
    assert out[1:] == pytest.approx([0.0, 0.1, 0.9], abs=1e-6)


def test_scale_reflectance_no_offset():
    out = sn.scale_reflectance(np.array([500, 3000], dtype="uint16"), 0.0)
    assert out == pytest.approx([0.05, 0.3], abs=1e-6)


def _item(baseline, applied):
    props = {"s2:processing_baseline": baseline}
    if applied is not None:
        props["earthsearch:boa_offset_applied"] = applied
    return SimpleNamespace(properties=props)


def test_asset_offset_rules():
    assert sn.asset_offset(_item("02.11", False)) == 0.0
    assert sn.asset_offset(_item("04.00", False)) == -0.1
    assert sn.asset_offset(_item("05.12", True)) == 0.0
    assert sn.asset_offset(_item("05.00", None)) == -0.1


def test_indices():
    red = np.array([0.1, 0.2], dtype="float32")
    nir = np.array([0.5, 0.2], dtype="float32")
    assert sn.ndvi(red, nir) == pytest.approx([0.6667, 0.0], abs=1e-3)
    green = np.array([0.1], dtype="float32")
    swir = np.array([0.05], dtype="float32")
    assert sn.mndwi(green, swir)[0] == pytest.approx(0.3333, abs=1e-3)


# ---------- cloud masking ----------


def test_invalid_mask_classes():
    scl = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    inv = sn.invalid_mask(scl)
    expected = np.array([[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 0]], dtype=bool)
    assert (inv == expected).all()


def test_clear_fraction_uses_aoi_mask_only():
    scl = np.full((4, 4), 4)
    scl[:2, :] = 9  # top half cloud
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True  # left half of the area
    assert sn.scl_clear_fraction(scl, mask) == pytest.approx(0.5)
    assert sn.scl_clear_fraction(scl, np.zeros((4, 4), dtype=bool)) == 0.0


def test_compute_metrics_masks_cloud_and_separates_water():
    scl = np.array([[4, 4, 6, 9]])
    red = np.array([[0.1, 0.1, 0.02, 0.5]], dtype="float32")
    nir = np.array([[0.5, 0.3, 0.01, 0.5]], dtype="float32")
    green = np.array([[0.1, 0.1, 0.05, 0.5]], dtype="float32")
    swir = np.array([[0.2, 0.2, 0.01, 0.5]], dtype="float32")
    bands = {"scl": scl, "red": red, "nir": nir, "green": green, "swir16": swir}
    m = sn.compute_metrics(bands, np.ones((1, 4), dtype=bool), pixel_area_km2=0.01)
    assert m["n_clear"] == 3  # cloud pixel dropped
    assert m["clear_frac"] == pytest.approx(0.75)
    assert m["ndvi_mean"] == pytest.approx((0.6667 + 0.5) / 2, abs=1e-3)  # water excluded
    assert m["water_frac"] == pytest.approx(1 / 3)
    assert m["water_km2"] == pytest.approx(1 / 3 * 4 * 0.01)


def test_compute_metrics_all_cloud_is_nan():
    bands = {k: np.ones((2, 2), dtype="float32") for k in ("red", "nir", "green", "swir16")}
    bands["scl"] = np.full((2, 2), 9)
    m = sn.compute_metrics(bands, np.ones((2, 2), dtype=bool), 0.01)
    assert m["n_clear"] == 0 and math.isnan(m["ndvi_mean"])


# ---------- AOI ----------


def test_aoi_mask_circle_area():
    aoi = AOI("c", "T", "cane", 13.5, 100.0, 5000.0, None, "test")
    x, y = transform("EPSG:4326", "EPSG:32647", [100.0], [13.5])
    tr = rasterio.Affine(100.0, 0, x[0] - 7500, 0, -100.0, y[0] + 7500)
    grid = sn.Grid(tr, (150, 150), rasterio.crs.CRS.from_epsg(32647))
    mask = sn.aoi_mask(aoi, grid)
    assert mask.sum() * 0.01 == pytest.approx(math.pi * 25, rel=0.02)


def test_aoi_mask_polygon_and_bbox():
    x, y = transform("EPSG:4326", "EPSG:32647", [100.0], [13.5])
    d = 0.02
    poly = ((100 - d, 13.5 - d), (100 + d, 13.5 - d), (100 + d, 13.5 + d), (100 - d, 13.5 + d))
    aoi = AOI("p", "T", "reservoir", 13.5, 100.0, None, poly + (poly[0],), "test")
    assert aoi.bbox_lonlat() == pytest.approx((99.98, 13.48, 100.02, 13.52))
    assert 15 < aoi.area_km2() < 20
    tr = rasterio.Affine(100.0, 0, x[0] - 5000, 0, -100.0, y[0] + 5000)
    grid = sn.Grid(tr, (100, 100), rasterio.crs.CRS.from_epsg(32647))
    assert sn.aoi_mask(aoi, grid).sum() * 0.01 == pytest.approx(aoi.area_km2(), rel=0.08)


def test_load_aois_validation(tmp_path):
    good = {"areas": [{"name": "a", "ticker": "T", "kind": "cane", "centre": [1, 2],
                       "radius_m": 1000, "source": "s"}]}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(good))
    assert load_aois(p)[0].radius_m == 1000.0
    bad = {"areas": [{"name": "a", "ticker": "T", "kind": "cane", "centre": [1, 2], "radius_m": 1000}]}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_aois(p)


# ---------- scene month selection (no network) ----------


def test_select_best_scene_prefers_clear(monkeypatch):
    aoi = AOI("c", "T", "cane", 13.5, 100.0, 1000.0, None, "t")
    mk = lambda i, d: SimpleNamespace(  # noqa: E731
        id=i, bbox=[99, 13, 101, 14], datetime=datetime(2024, 2, d, tzinfo=UTC),
        properties={"s2:mgrs_tile": "47PPT"})
    items = [mk("a", 1), mk("b", 6), mk("c", 11)]
    fractions = {"a": 0.4, "b": 0.9, "c": 0.7}
    monkeypatch.setattr(sn, "aoi_grid_for", lambda aoi, it: (it.id, None, None))
    monkeypatch.setattr(sn, "scl_clear_fraction", lambda scl, mask: fractions[scl])
    best, cf, tried = sn.select_best_scene(aoi, items)
    assert best.id == "b" and cf == 0.9 and tried == 3
    fractions.update({"a": 0.1, "b": 0.2, "c": 0.3})
    best, cf, _ = sn.select_best_scene(aoi, items)
    assert best is None and cf == 0.3


# ---------- quarterly aggregation ----------


def test_quarter_end_months_calendar_and_october_fy():
    cal = quarter_end_months(pd.Period("2020-01", "M"), pd.Period("2020-12", "M"), 12)
    assert [str(p) for p in cal] == ["2020-03", "2020-06", "2020-09", "2020-12"]
    fy = quarter_end_months(pd.Period("2020-01", "M"), pd.Period("2020-12", "M"), 10)
    assert [str(p) for p in fy] == ["2020-01", "2020-04", "2020-07", "2020-10"]


def test_aggregate_periods_mean_and_min_months():
    idx = pd.period_range("2020-01", "2020-06", freq="M")
    s = pd.Series([1.0, 3.0, np.nan, 5.0, np.nan, np.nan], index=idx).dropna()
    dates = pd.Series(pd.to_datetime(["2020-01-10", "2020-02-12", "2020-04-20"]), index=s.index)
    out = aggregate_periods(s, [pd.Period("2020-03", "M"), pd.Period("2020-06", "M")], 2, dates)
    assert out.loc[0, "value"] == pytest.approx(2.0) and out.loc[0, "n_months"] == 2
    assert out.loc[0, "last_obs"] == pd.Timestamp("2020-02-12")
    assert math.isnan(out.loc[1, "value"]) and out.loc[1, "n_months"] == 1


def test_combine_areas_requires_all():
    df = pd.DataFrame({
        "area": ["a", "b", "a"],
        "month": pd.PeriodIndex(["2020-01", "2020-01", "2020-02"], freq="M"),
        "v": [1.0, 3.0, 5.0],
    })
    s = combine_areas(df, ["a", "b"], "v", "sum")
    assert list(s.index.astype(str)) == ["2020-01"] and s.iloc[0] == 4.0
    s2 = combine_areas(df, ["a", "b"], "v", "mean", require_all=False)
    assert len(s2) == 2


# ---------- alignment ----------


def _quarters(n, start="2019-03"):
    first = pd.Period(start, freq="M")
    return pd.PeriodIndex([first + 3 * i for i in range(n)], freq="M")


def test_yoy_pct_and_diff():
    s = pd.Series([10.0, 20, 30, 40, 15, 10, 60, 20], index=_quarters(8))
    y = yoy_pct(s)
    assert y.iloc[:4].isna().all()
    assert y.iloc[4:].tolist() == pytest.approx([0.5, -0.5, 1.0, -0.5])
    d = yoy_diff(s, scale=s)
    assert d.iloc[4] == pytest.approx(0.5)
    z = pd.Series([0.0, 1, 1, 1, 2, 2, 2, 2], index=_quarters(8))
    assert np.isnan(yoy_pct(z).iloc[4])  # division by zero becomes NaN


def test_load_results_period_mapping(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text(
        "ticker,fiscal,quarter,revenue_mthb,net_profit_mthb\n"
        "K,2026,1,10,1\nK,2025,4,20,2\nC,2026,1,5,5\n"
    )
    r = load_results(csv, "K", fy_end_month=10)
    assert [str(p) for p in r["period_end"]] == ["2025-10", "2026-01"]
    assert r["announce_date"].iloc[1] == pd.Timestamp("2026-01-31") + pd.Timedelta(days=45)
    c = load_results(csv, "C", fy_end_month=12)
    assert str(c["period_end"].iloc[0]) == "2026-03"


def test_align_and_correlation_synthetic():
    q = _quarters(12)
    rng = np.random.default_rng(1)
    proxy = pd.Series(rng.uniform(0.3, 0.7, 12), index=q)
    pq = pd.DataFrame({
        "period_end": q, "value": proxy.values, "n_months": 3,
        "last_obs": [p.to_timestamp(how="end").normalize() - pd.Timedelta(days=5) for p in q],
    })
    py = yoy_pct(proxy)
    rev = pd.Series(100.0, index=q)
    for i in range(4, 12):  # revenue year-on-year equals 2 * proxy year-on-year
        rev.iloc[i] = rev.iloc[i - 4] * (1 + 2 * py.iloc[i])
    res = pd.DataFrame({
        "period_end": q, "announce_date": [p.to_timestamp(how="end").normalize()
                                           + pd.Timedelta(days=45) for p in q],
        "revenue_mthb": rev.values, "net_profit_mthb": rev.values * 0.1,
    })
    al = align_quarters(pq, res)
    assert (al["lead_days"].dropna() == 50).all()
    st = correlation(al["proxy_yoy"], al["revenue_yoy"])
    assert st["n"] == 8 and st["r"] == pytest.approx(1.0, abs=1e-6)
    lag = lagged_correlation(al, "proxy_yoy", "revenue_yoy", 1)
    assert lag["n"] == 7  # shifting by one quarter loses one of the 8 pairs


def test_correlation_interval_and_small_n():
    x = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    y = pd.Series([1.2, 1.9, 3.5, 3.9, 5.8, 5.9, 7.4, 8.8])
    st = correlation(x, y)
    assert st["ci_low"] < st["r"] < st["ci_high"] and st["r"] > 0.95
    tiny = correlation(pd.Series([1.0, 2]), pd.Series([2.0, 3]))
    assert tiny["n"] == 2 and math.isnan(tiny["r"])
    flat = correlation(pd.Series([1.0, 1, 1, 1]), pd.Series([1.0, 2, 3, 4]))
    assert math.isnan(flat["r"])


def test_verify_offset_uses_data_not_flag():
    valid = np.ones((4, 4), dtype=bool)
    with_offset = np.full((4, 4), 1300)  # +1000 present: dark pixels still at 1000 or above
    without = np.full((4, 4), 300)
    assert sn.verify_offset(-0.1, with_offset, valid) == -0.1
    assert sn.verify_offset(-0.1, without, valid) == 0.0
    assert sn.verify_offset(0.0, with_offset, valid) == 0.0


def test_detrended_correlation_removes_common_trend():
    from satellite.align import detrended_correlation

    t = np.arange(20, dtype=float)
    rng = np.random.default_rng(3)
    x = pd.Series(t + rng.normal(0, 0.5, 20))
    y = pd.Series(-2 * t + rng.normal(0, 0.5, 20))
    assert correlation(x, y)["r"] < -0.95
    assert abs(detrended_correlation(x, y)["r"]) < 0.6
