"""Tests for round 2 helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from satellite import round2


def test_trailing_mean_and_min():
    idx = pd.period_range("2020-01", "2020-12", freq="M")
    m = pd.Series(range(1, 13), index=idx, dtype=float)
    e = [pd.Period("2020-12", freq="M")]
    assert round2.trailing(m, e, "mean", 9).iloc[0] == 6.5
    assert round2.trailing(m, e, "min", 9).iloc[0] == 1.0
    assert round2.trailing(m.iloc[:5], e, "mean", 9).empty


def test_shift_p_bounds_and_placebo_rule():
    x = np.arange(20, dtype=float)
    assert abs(round2.shift_p(x, x) - 1 / 20) < 1e-12
    assert round2.placebo_pass({"n_placebo": 8, "placebo_p": 1 / 9})
    assert not round2.placebo_pass({"n_placebo": 8, "placebo_p": 0.33})
