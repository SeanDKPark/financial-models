"""Offline tests for the Nelson-Siegel-Svensson parametric curve."""

import numpy as np
import pytest

from finmodels.curves.base import BaseYieldCurve
from finmodels.curves.parametric import SvenssonCurve

# A realistic upward-sloping-then-flattening parameter set.
PARAMS = dict(beta0=0.052, beta1=-0.015, beta2=-0.02, beta3=0.015, tau1=1.6, tau2=8.0)


@pytest.fixture
def curve():
    return SvenssonCurve(**PARAMS)


def test_is_base_yield_curve(curve):
    assert isinstance(curve, BaseYieldCurve)


def test_constructor_rejects_bad_tau():
    with pytest.raises(ValueError):
        SvenssonCurve(0.05, -0.01, 0.0, 0.0, 0.0, 5.0)
    with pytest.raises(ValueError):
        SvenssonCurve(0.05, -0.01, 0.0, 0.0, 1.0, -1.0)


def test_zero_rate_origin_is_beta0_plus_beta1(curve):
    assert curve.zero_rate(0.0) == pytest.approx(PARAMS["beta0"] + PARAMS["beta1"])
    assert curve.instantaneous_forward_rate(0.0) == pytest.approx(
        PARAMS["beta0"] + PARAMS["beta1"]
    )


def test_zero_rate_matches_closed_form(curve):
    t = 3.0
    x1, x2 = t / PARAMS["tau1"], t / PARAMS["tau2"]
    l1 = (1 - np.exp(-x1)) / x1
    expected = (
        PARAMS["beta0"]
        + PARAMS["beta1"] * l1
        + PARAMS["beta2"] * (l1 - np.exp(-x1))
        + PARAMS["beta3"] * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))
    )
    assert curve.zero_rate(t) == pytest.approx(expected)


def test_long_end_tends_to_beta0(curve):
    assert curve.zero_rate(200.0) == pytest.approx(PARAMS["beta0"], abs=1e-3)
    assert curve.instantaneous_forward_rate(200.0) == pytest.approx(
        PARAMS["beta0"], abs=1e-3
    )


def test_flat_when_only_level():
    flat = SvenssonCurve(0.04, 0.0, 0.0, 0.0, 1.0, 5.0)
    ts = np.array([0.0, 0.5, 2.0, 10.0, 30.0])
    np.testing.assert_allclose(flat.zero_rate(ts), 0.04, atol=1e-12)
    np.testing.assert_allclose(flat.instantaneous_forward_rate(ts), 0.04, atol=1e-12)
    assert flat.forward_rate(1.0, 5.0, "continuous") == pytest.approx(0.04)


def test_discount_factor(curve):
    t = 4.0
    assert curve.discount_factor(t) == pytest.approx(np.exp(-curve.zero_rate(t) * t))
    assert curve.discount_factor(0.0) == pytest.approx(1.0)


def test_analytical_forward_matches_finite_difference(curve):
    t, h = 5.0, 1e-6
    lnp = lambda s: np.log(curve.discount_factor(s))
    fd = -(lnp(t + h) - lnp(t - h)) / (2 * h)
    assert curve.instantaneous_forward_rate(t) == pytest.approx(fd, rel=1e-5)


def test_scalar_and_array_types(curve):
    assert isinstance(curve.zero_rate(2.0), float)
    assert isinstance(curve.discount_factor(2.0), float)
    assert isinstance(curve.instantaneous_forward_rate(2.0), float)
    out = curve.zero_rate(np.array([1.0, 5.0, 20.0]))
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_rejects_negative_t(curve):
    with pytest.raises(ValueError):
        curve.zero_rate(-0.5)
    with pytest.raises(ValueError):
        curve.instantaneous_forward_rate(np.array([1.0, -1.0]))


def test_nelson_siegel_factory():
    ns = SvenssonCurve.nelson_siegel(0.05, -0.01, 0.02, 2.0)
    assert ns.beta3 == 0.0
    t = 4.0
    x = t / 2.0
    l1 = (1 - np.exp(-x)) / x
    expected = 0.05 + -0.01 * l1 + 0.02 * (l1 - np.exp(-x))
    assert ns.zero_rate(t) == pytest.approx(expected)


def test_calibrate_recovers_a_known_curve():
    truth = SvenssonCurve(**PARAMS)  # all params inside the economic bounds
    pillars = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    target = np.asarray(truth.zero_rate(pillars))

    fitted = SvenssonCurve.calibrate(pillars, target)
    assert isinstance(fitted, SvenssonCurve)
    # Parameters are not uniquely identifiable, but the fitted rates must match.
    np.testing.assert_allclose(fitted.zero_rate(pillars), target, atol=1e-4)


def test_calibrate_rejects_too_few_points():
    with pytest.raises(ValueError):
        SvenssonCurve.calibrate(np.array([1.0, 2.0, 3.0]), np.array([0.04, 0.045, 0.05]))


def test_calibrate_respects_economic_bounds():
    # A curve whose unconstrained NSS fit is known to send beta0 very negative.
    pillars = np.array([0.08, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    zeros = np.array(
        [0.038, 0.039, 0.040, 0.041, 0.044, 0.045, 0.046, 0.047, 0.048, 0.055, 0.054]
    )
    c = SvenssonCurve.calibrate(pillars, zeros)
    p = c.params
    assert 0.01 <= p["beta0"] <= 0.15
    assert -0.15 <= p["beta1"] <= 0.15
    assert -0.20 <= p["beta2"] <= 0.20
    assert -0.20 <= p["beta3"] <= 0.20
    assert 0.2 <= p["tau1"] <= 5.0
    assert 3.0 <= p["tau2"] <= 25.0
    assert p["beta0"] + p["beta1"] >= -1e-9          # z(0) non-negative
    assert c.zero_rate(0.0) >= 0.0
    rmse_bp = np.sqrt(np.mean((c.zero_rate(pillars) - zeros) ** 2)) * 1e4
    assert rmse_bp < 15.0                             # still a good fit


def test_calibrate_custom_bounds_are_honoured():
    pillars = np.array([0.25, 1, 2, 5, 10, 30], dtype=float)
    zeros = np.array([0.040, 0.043, 0.045, 0.048, 0.050, 0.052])
    c = SvenssonCurve.calibrate(pillars, zeros, beta0_bounds=(0.048, 0.052))
    assert 0.048 <= c.params["beta0"] <= 0.052


def test_calibrate_rejects_misaligned_weights():
    pillars = np.array([0.25, 1, 2, 5, 10], dtype=float)
    zeros = np.array([0.040, 0.043, 0.045, 0.048, 0.050])
    with pytest.raises(ValueError):
        SvenssonCurve.calibrate(pillars, zeros, weights=np.array([1.0, 1.0]))
