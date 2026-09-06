"""Offline tests for the linear zero-rate interpolator."""

import numpy as np
import pytest

from finmodels.curves.base import BaseYieldCurve
from finmodels.curves.data_models import CurveMarketData
from finmodels.curves.interpolation import LinearZeroInterpolator

PILLARS = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
RATES = np.array([0.0430, 0.0416, 0.0425, 0.0438, 0.0457])


@pytest.fixture
def interp():
    return LinearZeroInterpolator(PILLARS, RATES)


def test_linear_is_base_yield_curve(interp):
    assert issubclass(LinearZeroInterpolator, BaseYieldCurve)
    assert isinstance(interp, BaseYieldCurve)


def test_exact_pillar_hits(interp):
    for t, r in zip(PILLARS, RATES):
        assert interp.zero_rate(float(t)) == pytest.approx(r)


def test_midpoint_arithmetic(interp):
    # Halfway between t=1.0 (0.0416) and t=2.0 (0.0425) -> 0.04205
    assert interp.zero_rate(1.5) == pytest.approx(0.04205)
    # One quarter of the way from t=2.0 to t=5.0
    expected = 0.0425 + (0.0438 - 0.0425) / (5.0 - 2.0) * (2.75 - 2.0)
    assert interp.zero_rate(2.75) == pytest.approx(expected)


def test_flat_extrapolation_left(interp):
    assert interp.zero_rate(0.05) == pytest.approx(RATES[0])
    assert interp.zero_rate(0.0) == pytest.approx(RATES[0])


def test_flat_extrapolation_right(interp):
    assert interp.zero_rate(30.0) == pytest.approx(RATES[-1])


def test_array_matches_scalar(interp):
    ts = np.array([0.1, 0.25, 1.5, 10.0, 50.0])
    arr = interp.zero_rate(ts)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == ts.shape
    for i, t in enumerate(ts):
        assert arr[i] == pytest.approx(interp.zero_rate(float(t)))


def test_scalar_return_type(interp):
    assert isinstance(interp.zero_rate(1.0), float)
    assert isinstance(interp.discount_factor(1.0), float)


def test_discount_factor(interp):
    t = 2.0
    expected = np.exp(-interp.zero_rate(t) * t)
    assert interp.discount_factor(t) == pytest.approx(expected)
    assert interp.discount_factor(0.0) == pytest.approx(1.0)


def test_from_market_data():
    md = CurveMarketData(
        as_of_date="2025-01-02",
        curve_name="US_TREASURY_PAR_YIELD",
        pillars=PILLARS,
        rates=RATES,
    )
    interp = LinearZeroInterpolator.from_market_data(md)
    assert interp.zero_rate(1.0) == pytest.approx(0.0416)


def test_rejects_both_sources():
    md = CurveMarketData("2025-01-02", "X", PILLARS, RATES)
    with pytest.raises(ValueError):
        LinearZeroInterpolator(PILLARS, RATES, market_data=md)


def test_rejects_unsorted_pillars():
    with pytest.raises(ValueError):
        LinearZeroInterpolator(np.array([1.0, 0.5, 2.0]), np.array([0.04, 0.04, 0.04]))


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        LinearZeroInterpolator(np.array([1.0, 2.0, 3.0]), np.array([0.04, 0.04]))


# --------------------------------------------------------------- forward rates

FLAT = LinearZeroInterpolator(np.array([0.25, 1.0, 5.0, 30.0]), np.full(4, 0.05))
UPWARD = LinearZeroInterpolator(np.array([1.0, 2.0]), np.array([0.03, 0.04]))


@pytest.mark.parametrize("t1,t2", [(0.0, 1.0), (0.5, 2.0), (1.0, 1.25), (3.0, 30.0)])
def test_forward_flat_curve_is_exact(t1, t2):
    assert FLAT.forward_rate(t1, t2, "continuous") == pytest.approx(0.05, abs=1e-13)


@pytest.mark.parametrize("t", [0.1, 0.25, 1.0, 1.5, 7.3, 30.0])
def test_forward_from_origin_equals_spot(interp, t):
    assert interp.forward_rate(0.0, t, "continuous") == pytest.approx(
        interp.zero_rate(t), abs=1e-12
    )


@pytest.mark.parametrize("t1,t2", [(0.3, 1.7), (1.0, 2.0), (2.5, 9.0)])
def test_forward_no_arbitrage_consistency(interp, t1, t2):
    f = interp.forward_rate(t1, t2, "continuous")
    ratio = interp.discount_factor(t1) / interp.discount_factor(t2)
    assert ratio == pytest.approx(np.exp(f * (t2 - t1)), abs=1e-12)


def test_forward_upward_sloping_above_long_spot():
    z1, z2 = UPWARD.zero_rate(1.0), UPWARD.zero_rate(2.0)
    f = UPWARD.forward_rate(1.0, 2.0, "continuous")
    assert f > z2 > z1
    assert f == pytest.approx(0.05)  # (0.04*2 - 0.03*1) / (2 - 1)


def test_forward_simple_and_semiannual_ordering(interp):
    # Same discount-factor ratio, less frequent compounding -> higher nominal rate:
    # simple > semi-annual > continuous.
    t1, t2 = 1.0, 3.0
    f_cont = interp.forward_rate(t1, t2, "continuous")
    f_simple = interp.forward_rate(t1, t2, "simple")
    f_semi = interp.forward_rate(t1, t2, "semi-annual")
    assert f_simple > f_semi > f_cont > 0.0


def test_forward_semiannual_matches_definition(interp):
    t1, t2 = 2.0, 5.0
    p1, p2 = interp.discount_factor(t1), interp.discount_factor(t2)
    expected = 2 * ((p1 / p2) ** (1 / (2 * (t2 - t1))) - 1)
    assert interp.forward_rate(t1, t2, "semi-annual") == pytest.approx(expected)


def test_forward_rate_rejects_bad_horizons(interp):
    with pytest.raises(ValueError):
        interp.forward_rate(2.0, 2.0)
    with pytest.raises(ValueError):
        interp.forward_rate(3.0, 1.0)
    with pytest.raises(ValueError):
        interp.forward_rate(-1.0, 1.0)


def test_forward_rate_rejects_bad_compounding(interp):
    with pytest.raises(ValueError):
        interp.forward_rate(1.0, 2.0, "annual")


def test_forward_rolling_scalar_and_array(interp):
    single = interp.forward_rate_rolling(2.0, tenor=0.5)
    assert isinstance(single, float)

    ts = np.array([0.0, 1.0, 2.0, 9.5])
    arr = interp.forward_rate_rolling(ts, tenor=0.5)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == ts.shape
    for i, t in enumerate(ts):
        assert arr[i] == pytest.approx(interp.forward_rate_rolling(float(t), tenor=0.5))


def test_forward_rolling_matches_forward_rate(interp):
    assert interp.forward_rate_rolling(1.5, tenor=0.5, compounding="continuous") == (
        pytest.approx(interp.forward_rate(1.5, 2.0, "continuous"))
    )


def test_forward_rolling_flat_curve_continuous(interp):
    flat = LinearZeroInterpolator(np.array([1.0, 10.0]), np.array([0.05, 0.05]))
    out = flat.forward_rate_rolling(np.array([0.5, 3.0, 7.0]), tenor=0.5, compounding="continuous")
    np.testing.assert_allclose(out, 0.05, atol=1e-13)


def test_forward_rolling_rejects_bad_input(interp):
    with pytest.raises(ValueError):
        interp.forward_rate_rolling(1.0, tenor=0.0)
    with pytest.raises(ValueError):
        interp.forward_rate_rolling(-0.5, tenor=0.5)
    with pytest.raises(ValueError):
        interp.forward_rate_rolling(np.array([1.0, -2.0]), tenor=0.5)
    with pytest.raises(ValueError):
        interp.forward_rate_rolling(1.0, tenor=0.5, compounding="daily")
