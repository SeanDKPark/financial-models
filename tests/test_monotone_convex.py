"""Offline tests for the Hagan-West monotone-convex interpolator."""

import numpy as np
import pytest
from scipy.integrate import quad

from finmodels.curves.base import BaseYieldCurve
from finmodels.curves.bootstrapping import bootstrap_par_curve
from finmodels.curves.data_models import CurveMarketData
from finmodels.curves.interpolation import LinearZeroInterpolator
from finmodels.curves.monotone_convex import HaganWestInterpolator

PILLARS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
ZEROS = np.array(
    [0.0385, 0.0398, 0.0413, 0.0438, 0.0450, 0.0455, 0.0467, 0.0482, 0.0548, 0.0538]
)


@pytest.fixture
def hw():
    return HaganWestInterpolator(PILLARS, ZEROS)


def test_is_base_yield_curve(hw):
    assert isinstance(hw, BaseYieldCurve)


def test_accepts_discount_factors():
    dfs = np.exp(-ZEROS * PILLARS)
    a = HaganWestInterpolator(PILLARS, ZEROS)
    b = HaganWestInterpolator(PILLARS, discount_factors=dfs)
    np.testing.assert_allclose(
        a.instantaneous_forward_rate(np.linspace(0.5, 25, 50)),
        b.instantaneous_forward_rate(np.linspace(0.5, 25, 50)),
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        HaganWestInterpolator(PILLARS, ZEROS, discount_factors=dfs)


def test_pillar_invariance(hw):
    dfs_in = np.exp(-ZEROS * PILLARS)
    np.testing.assert_allclose(hw.discount_factor(PILLARS), dfs_in, atol=1e-10)
    np.testing.assert_allclose(hw.zero_rate(PILLARS), ZEROS, atol=1e-10)


def test_integral_repricing_per_interval(hw):
    t = np.concatenate(([0.0], PILLARS))
    p = np.concatenate(([1.0], np.exp(-ZEROS * PILLARS)))
    f_disc = (np.log(p[:-1]) - np.log(p[1:])) / np.diff(t)
    for i in range(f_disc.size):
        area, _ = quad(
            lambda u: float(hw.instantaneous_forward_rate(u)),
            t[i], t[i + 1], limit=200,
        )
        assert area == pytest.approx(f_disc[i] * (t[i + 1] - t[i]), abs=1e-7)


def test_flat_curve_invariance():
    flat = HaganWestInterpolator(PILLARS, np.full(PILLARS.shape, 0.05))
    grid = np.linspace(0.0, 30.0, 501)
    np.testing.assert_allclose(
        flat.instantaneous_forward_rate(grid), 0.05, atol=1e-12
    )
    np.testing.assert_allclose(flat.zero_rate(grid[1:]), 0.05, atol=1e-12)
    assert flat.zero_rate(0.0) == pytest.approx(0.05)


def test_origin_values(hw):
    assert hw.discount_factor(0.0) == pytest.approx(1.0)
    assert hw.zero_rate(0.0) == pytest.approx(hw.instantaneous_forward_rate(0.0))


def test_scalar_and_array_types(hw):
    assert isinstance(hw.zero_rate(3.0), float)
    assert isinstance(hw.discount_factor(3.0), float)
    assert isinstance(hw.instantaneous_forward_rate(3.0), float)
    out = hw.instantaneous_forward_rate(np.array([1.0, 5.0, 12.0]))
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_positivity_on_volatile_positive_forwards():
    pil = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30], dtype=float)
    dt = np.diff(np.concatenate(([0.0], pil)))
    f_disc = np.array(
        [0.02, 0.085, 0.015, 0.09, 0.02, 0.10, 0.01, 0.07, 0.03, 0.06, 0.02, 0.05]
    )  # jagged but strictly positive
    zeros = np.cumsum(f_disc * dt) / pil
    curve = HaganWestInterpolator(pil, zeros)
    grid = np.linspace(0.0, 30.0, 1000)
    f = np.asarray(curve.instantaneous_forward_rate(grid))
    assert np.all(f >= -1e-12)


def test_no_10y_20y_shark_fin():
    """HW removes the *discontinuity* linear puts at the 20Y knot, and does not
    peak higher than linear does over the 10Y-20Y stretch."""
    pil = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    par = np.array(
        [0.039, 0.040, 0.041, 0.044, 0.045, 0.046, 0.047, 0.048, 0.055, 0.054]
    )
    md = CurveMarketData("2026-09-04", "T", pil, par)
    zeros = np.asarray(
        bootstrap_par_curve(md, compounding="semi-annual").zero_rate(pil)
    )
    linear = LinearZeroInterpolator(pil, zeros)
    hw = HaganWestInterpolator(pil, zeros)

    grid = np.linspace(10.0, 22.0, 600)  # straddles the 20Y knot
    lin_f = np.asarray(
        linear.forward_rate_rolling(grid, tenor=0.5, compounding="semi-annual")
    )
    hw_f = np.asarray(
        hw.forward_rate_rolling(grid, tenor=0.5, compounding="semi-annual")
    )

    # Largest step between adjacent dense points: linear has a cliff at 20Y,
    # HW is smooth.
    assert np.max(np.abs(np.diff(hw_f))) < 0.2 * np.max(np.abs(np.diff(lin_f)))
    # And HW never peaks above the linear artifact.
    assert np.max(hw_f) <= np.max(lin_f)


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        HaganWestInterpolator(np.array([1.0]), np.array([0.04]))
    with pytest.raises(ValueError):
        HaganWestInterpolator(np.array([2.0, 1.0]), np.array([0.04, 0.05]))
    with pytest.raises(ValueError):
        HaganWestInterpolator(PILLARS, ZEROS[:-1])
    with pytest.raises(ValueError):
        HaganWestInterpolator(PILLARS)  # neither zeros nor dfs
