"""Tests for round 3 pixel metrics, RAM gate and series helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from satellite import round3


def test_s2_pixel_metrics_shares_and_masks():
    scl = np.full((4, 5), 4, dtype="uint8")
    scl[0, :] = 9  # cloud row, excluded
    scl[1, 0] = 6  # water, excluded
    red = np.full((4, 5), 0.1, dtype="float32")
    nir = np.full((4, 5), 0.1, dtype="float32")
    nir[2, :] = 0.5  # NDVI 0.667, dense
    nir[3, :3] = 0.12  # NDVI ~0.09, bare
    nir[3, 3:] = 0.2  # NDVI 0.333, neither
    mask = np.ones((4, 5), dtype=bool)
    m = round3.s2_pixel_metrics(scl, red, nir, mask)
    assert m["n_land"] == 14  # 20 - 5 cloud - 1 water
    assert math.isclose(m["dense_frac"], 5 / 14)
    assert math.isclose(m["bare_frac"], (4 + 3) / 14)  # row1 cols1-4 (NDVI 0), row 3 cols 0-2


def test_s2_pixel_metrics_all_invalid():
    scl = np.full((2, 2), 9, dtype="uint8")
    z = np.ones((2, 2), dtype="float32")
    m = round3.s2_pixel_metrics(scl, z, z, np.ones((2, 2), dtype=bool))
    assert m["n_land"] == 0 and math.isnan(m["bare_frac"])


def test_s1_cane_pixel_metrics():
    vv = np.full((10, 10), 0.1, dtype="float32")  # -10 dB
    vv[:2, :] = 0.01  # -20 dB, flooded: 20 pixels
    vh = np.full((10, 10), 0.01, dtype="float32")
    vh[5:, :] = 0.1
    mask = np.ones((10, 10), dtype=bool)
    m = round3.s1_cane_pixel_metrics(vh, vv, mask)
    assert math.isclose(m["flooded_frac"], 0.2)
    assert math.isclose(m["vh_sd_db"], 5.0)  # half -20 dB, half -10 dB
    vv2 = vv.copy()
    vv2[:5, :] = np.nan
    assert math.isnan(round3.s1_cane_pixel_metrics(vh, vv2, mask)["flooded_frac"])


def test_water_state_and_drawdown():
    mask = np.ones((2, 2), dtype=bool)
    vv = np.array([[0.01, 0.01], [0.1, np.nan]], dtype="float32")
    st, frac = round3.water_state(vv, mask)
    assert st.tolist() == [[2, 2], [1, 0]] and math.isclose(frac, 0.75)
    wet = np.array([[2, 2], [2, 1]], dtype="uint8")
    dry = np.array([[2, 1], [1, 1]], dtype="uint8")
    stacks = {"a": {pd.Period("2020-01", "M"): wet, pd.Period("2020-02", "M"): wet, pd.Period("2020-03", "M"): dry}}
    s = round3.drawdown_series(stacks)
    # reference pixels: water in >= 50 percent of months -> (0,0) 3/3, (0,1) 2/3, (1,0) 2/3; (1,1) 0/3
    assert math.isclose(s[pd.Period("2020-01", "M")], 0.0)
    assert math.isclose(s[pd.Period("2020-03", "M")], 2 / 3)


def test_trailing_max_and_quarter_mean():
    idx = pd.period_range("2020-01", "2020-12", freq="M")
    m = pd.Series(range(1, 13), index=idx, dtype=float)
    e = [pd.Period("2020-12", freq="M")]
    assert round3.trailing(m, e, "max", 9).iloc[0] == 12.0
    assert round3.trailing(m.iloc[:5], e, "max", 9).empty
    q = round3.quarter_mean(m, [pd.Period("2020-03", "M"), pd.Period("2020-06", "M")])
    assert q.iloc[0] == 2.0 and q.iloc[1] == 5.0


def test_gate_counts_waits_without_sleeping_for_real(tmp_path):
    seq = iter([0.4, 0.5, 1.6, 2.0])
    slept = []
    g = round3.Gate(min_gb=1.5, probe=lambda: next(seq), sleep=slept.append, poll_s=1.0, log_path=tmp_path / "g.csv")
    assert g.wait() == 1.6
    assert g.events == 1 and g.polls == 2 and slept == [1.0, 1.0]
    assert g.wait() == 2.0 and g.events == 1  # enough RAM: no new wait episode
    assert "wait_start" in (tmp_path / "g.csv").read_text()


def test_gate_gives_up():
    g = round3.Gate(probe=lambda: 0.1, sleep=lambda s: None, max_wait_s=-1.0)
    try:
        g.wait()
    except MemoryError:
        return
    raise AssertionError("expected MemoryError")


def test_combined_adjusted_uses_all_rounds(tmp_path):
    pd.DataFrame({"perm_p": [0.5, 0.01]}).to_csv(tmp_path / "r1.csv", index=False)
    pd.DataFrame({"raw_p": [0.2]}).to_csv(tmp_path / "r2.csv", index=False)
    h, b, n = round3.combined_adjusted([0.001, math.nan], tmp_path / "r1.csv", tmp_path / "r2.csv")
    assert n == 5
    assert math.isclose(h[0], 0.004) and math.isclose(b[0], 0.004)  # 4 valid p-values: 0.001 * 4
    assert math.isnan(h[1])
