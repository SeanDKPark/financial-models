"""Offline tests for the par-to-zero bootstrapping engine."""

import numpy as np
import pytest

from finmodels.curves.bootstrapping import (
    ParToZeroBootstrapper,
    bootstrap_par_curve,
)
from finmodels.curves.data_models import CurveMarketData
from finmodels.curves.interpolation import LinearZeroInterpolator

# Standard Treasury benchmark pillars (years).
PILLARS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def make_curve(rates, rate_type="PAR_YIELD"):
    return CurveMarketData(
        as_of_date="2025-01-03",
        curve_name="US_TREASURY_PAR_YIELD",
        pillars=PILLARS,
        rates=np.asarray(rates, dtype=float),
        rate_type=rate_type,
    )


def reprice_par_bond(interp: LinearZeroInterpolator, y: float, maturity: float, m: int) -> float:
    """PV of a par bond: semi-annual coupons y/m plus principal, using DFs from interp."""
    n = int(round(m * maturity))
    taus = np.array([j / m for j in range(1, n + 1)])
    dfs = interp.discount_factor(taus)
    return float((y / m) * dfs.sum() + dfs[-1])


def test_flat_curve_invariance():
    """Flat par curve at 5% -> semi-annually compounded zeros are exactly 5%."""
    md = make_curve(np.full(PILLARS.shape, 0.05))
    interp = bootstrap_par_curve(md, compounding="semi-annual", frequency=2)
    np.testing.assert_allclose(interp.rates, 0.05, atol=1e-12)


def test_repricing_par_check():
    """Bootstrapped DFs reprice every input long-end par bond to exactly 1.0."""
    par = np.array([0.043, 0.044, 0.045, 0.046, 0.047, 0.049, 0.050, 0.051, 0.053, 0.052])
    md = make_curve(par)
    interp = bootstrap_par_curve(md, compounding="continuous", frequency=2)
    for t, y in zip(PILLARS, par):
        if t <= 1.0:
            continue
        pv = reprice_par_bond(interp, y, float(t), m=2)
        assert pv == pytest.approx(1.0, abs=1e-7)


def test_upward_sloping_zero_above_par():
    par = np.array([0.030, 0.032, 0.035, 0.038, 0.040, 0.043, 0.045, 0.047, 0.050, 0.052])
    md = make_curve(par)
    # Compare like-for-like: par yields are semi-annual, so ask for semi-annual zeros.
    bs = ParToZeroBootstrapper(compounding="semi-annual")
    interp = bs.bootstrap(md)
    for t, y in zip(PILLARS, par):
        if t <= 1.0:
            continue
        assert interp.zero_rate(float(t)) > y


def test_inverted_curve_zero_below_par():
    par = np.array([0.055, 0.053, 0.050, 0.047, 0.045, 0.042, 0.040, 0.038, 0.036, 0.035])
    md = make_curve(par)
    interp = bootstrap_par_curve(md, compounding="semi-annual")
    for t, y in zip(PILLARS, par):
        if t <= 1.0:
            continue
        assert interp.zero_rate(float(t)) < y


def test_short_end_passes_through():
    par = np.array([0.030, 0.032, 0.035, 0.038, 0.040, 0.043, 0.045, 0.047, 0.050, 0.052])
    md = make_curve(par)
    interp = bootstrap_par_curve(md, compounding="continuous")
    # For continuous compounding a single-cashflow instrument's zero rate == par yield.
    for t, y in zip(PILLARS, par):
        if t <= 1.0:
            assert interp.zero_rate(float(t)) == pytest.approx(y)


def test_returns_interpolator():
    interp = bootstrap_par_curve(make_curve(np.full(PILLARS.shape, 0.04)))
    assert isinstance(interp, LinearZeroInterpolator)
    # Output nodes span the dense semi-annual grid out to the last market pillar.
    assert interp.pillars[-1] == pytest.approx(30.0)
    assert np.all(np.diff(interp.pillars) > 0)
    for t in PILLARS[PILLARS > 1.0]:
        assert np.any(np.isclose(interp.pillars, t))


def test_rejects_non_par_yield():
    md = make_curve(np.full(PILLARS.shape, 0.04), rate_type="ZERO")
    with pytest.raises(ValueError, match="PAR_YIELD"):
        ParToZeroBootstrapper().bootstrap(md)


def test_rejects_non_positive_frequency():
    with pytest.raises(ValueError, match="frequency"):
        ParToZeroBootstrapper(frequency=0)
    with pytest.raises(ValueError, match="frequency"):
        ParToZeroBootstrapper(frequency=-2)


def test_rejects_bad_compounding():
    with pytest.raises(ValueError, match="compounding"):
        ParToZeroBootstrapper(compounding="annual")
