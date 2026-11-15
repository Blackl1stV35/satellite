"""Tests for the extension: throttle, range file, radar metrics, test battery."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from satellite import battery, net, radar


def test_throttle_spacing():
    t = [0.0]
    slept = []

    def clock():
        return t[0]

    def sleep(s):
        slept.append(s)
        t[0] += s

    net._last.clear()
    net.wait_turn("h", 2.0, clock=clock, sleep=sleep)
    t[0] += 0.5
    waited = net.wait_turn("h", 2.0, clock=clock, sleep=sleep)
    assert abs(waited - 1.5) < 1e-9
    assert net.wait_turn("other", 2.0, clock=clock, sleep=sleep) == 0.0


def test_range_file_reads_across_blocks(monkeypatch):
    data = bytes(range(256)) * 40  # 10240 bytes
    calls = []

    class R:
        def __init__(self, start, end):
            self.status_code = 206
            self.content = data[start : end + 1]
            self.headers = {"Content-Range": f"bytes {start}-{min(end, len(data) - 1)}/{len(data)}"}

    def fake_get(url, headers=None, **kw):
        a, b = headers["Range"].split("=")[1].split("-")
        calls.append((int(a), int(b)))
        return R(int(a), int(b))

    monkeypatch.setattr(net, "get", fake_get)
    net.clear_cache()
    f = net.RangeFile("http://x/y.tif", block=4096)
    assert f.size == len(data)
    f.seek(4000)
    assert f.read(200) == data[4000:4200]
    f.seek(0, 2)
    assert f.tell() == len(data)
    f.seek(0)
    assert f.read(32 << 20) == data[:16384]  # a probe-sized read is cut short, never fetches the whole file
    n_calls = len(calls)
    f.seek(100)
    f.read(50)
    assert len(calls) == n_calls  # cached block, no new request
    net.clear_cache()


def test_cane_metrics_in_linear_power():
    vh = np.array([[0.01, 0.03], [0.02, np.nan]], dtype="float32")
    vv = np.array([[0.1, 0.1], [0.1, 0.1]], dtype="float32")
    mask = np.ones((2, 2), dtype=bool)
    m = radar.cane_metrics(vh, vv, mask)
    assert m["valid_frac"] == 0.75 and math.isnan(m["vh_db"])  # below 0.9 valid share
    m = radar.cane_metrics(vh, vv, np.array([[1, 1], [1, 0]], dtype=bool))
    assert abs(m["vh_db"] - 10 * math.log10(0.02)) < 1e-6
    assert abs(m["ratio_db"] - (m["vh_db"] - m["vv_db"])) < 1e-9


def test_water_metrics_threshold():
    vv = np.full((10, 10), 0.1, dtype="float32")  # -10 dB land
    vv[:3, :] = 0.01  # -20 dB water, 30 percent
    m = radar.water_metrics(vv, np.ones((10, 10), dtype=bool), px_km2=0.01)
    assert abs(m["water_km2"] - 0.3) < 1e-6
    assert radar.db(np.float32(0.0253)) > radar.WATER_DB  # -15.99 dB is not water


def _series(vals, start="2019-03"):
    idx = pd.period_range(start, periods=len(vals) * 3, freq="M")[::3]
    return pd.Series(vals, index=idx)


def test_battery_detects_planted_relation_and_rejects_noise():
    rng = np.random.default_rng(1)
    n = 28
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.3, size=n)
    px, ty = _series(x), _series(y)
    res = battery.run_test(px, ty, 0)
    assert res["n"] == n and res["rho"] > 0.8 and res["perm_p"] < 0.01
    z = _series(rng.normal(size=n))
    res2 = battery.run_test(px, z, 0)
    assert res2["perm_p"] > 0.05


def test_lag_one_pairs_proxy_with_next_quarter():
    p = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t = _series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    d = battery.paired(p, t, 1)
    # proxy of quarter k pairs with the target of quarter k+1
    assert list(d["x"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(d["y"]) == [20.0, 30.0, 40.0, 50.0, 60.0]


def test_holm_and_family_verdict():
    adj = battery.holm([0.01, 0.04, 0.03, math.nan])
    assert abs(adj[0] - 0.03) < 1e-12 and abs(adj[2] - 0.06) < 1e-12 and abs(adj[1] - 0.06) < 1e-12
    assert math.isnan(adj[3])
    rows = [
        {"perm_p": 0.001, "placebo_p": 0.01, "loyo_stable": True},
        {"perm_p": 0.001, "placebo_p": 0.5, "loyo_stable": True},
        {"perm_p": 0.2, "placebo_p": 0.01, "loyo_stable": True},
    ]
    out = battery.apply_family(rows)
    assert [r["survives"] for r in out] == [True, False, False]
    assert out[0]["family_size"] == 3


def test_vessel_count_threshold_size_and_static():
    from satellite import ships

    sea = np.ones((60, 60), dtype=bool)
    vv = np.full((60, 60), -22.0, dtype="float32")
    vv[10:12, 10:12] = 0.0  # 4 px vessel
    vv[30, 30] = 0.0  # 1 px speckle, below the size limit
    vv[40:44, 40:44] = 0.0  # 16 px vessel, will be marked static
    vv[50:52, 5:7] = -6.0  # below the -5 dB threshold
    bright = ships.bright_in_sea(vv, sea)
    assert ships.count_vessels(bright) == 2
    static = np.zeros_like(bright)
    static[40:44, 40:44] = True
    assert ships.count_vessels(bright, static) == 1
    big = np.zeros((100, 100), dtype=bool)
    big[10:50, 10:50] = True  # 1600 px: a structure, not a vessel
    assert ships.count_vessels(big) == 0


def test_sea_mask_excludes_land_and_structures():
    from satellite import ships

    med = np.full((30, 30), -22.0, dtype="float32")
    p90 = np.full((30, 30), -15.0, dtype="float32")
    med[:, :10] = -8.0  # land
    p90[15, 20] = 0.0  # fixed bright structure in the water
    m = ships.sea_mask(med, p90)
    assert not m[:, :12].any()
    assert not m[15, 20] and m[5, 25]


def test_parse_income_unlabelled_subtotal_and_owners():
    from satellite import fundamentals as F

    rows = [
        ["PROFIT OR LOSS"], ["UNIT : THOUSAND BAHT"],
        ["", "", "", "", "", "", 2018.0, "", 2017.0],
        ["Revenues from sales and rendering of services"],
        ["", "Revenues from sales", "", "", "", "", 5000.0, "", 4000.0],
        ["", "Revenues from services", "", "", "", "", 100.0, "", 90.0],
        ["", "", "", "", "", "", 5100.0, "", 4090.0],
        ["Costs of sales", "", "", "", "", "", -4000.0, "", -3000.0],
        ["Other income", "", "", "", "", "", 77.0, "", 1.0],
        ["", "Owners of the parent", "", "", "", "", -94.0, "", 58.0],
        ["Basic earnings per share"], ["Owners of the parent", "", "", "", "", "", -0.02, "", 0.01],
    ]
    p = F.parse_income(rows)
    assert abs(p["revenue_mthb"] - 5.1) < 1e-9 and abs(p["net_profit_mthb"] + 0.094) < 1e-9


def test_local_static_masks_remove_late_fixed_object_only():
    import numpy as np
    from satellite import ships
    months = [f"2024-{m:02d}" for m in range(1, 13)]
    bright = {m: np.zeros((9, 9), dtype=bool) for m in months}
    for m in months:
        bright[m][4, 4] = True  # fixed object present in every scene of the window
    bright["2024-03"][1, 1] = True  # a ship seen once
    loc = ships.local_static_masks(bright)
    assert loc["2024-06"][4, 4] and not loc["2024-06"][1, 1]
