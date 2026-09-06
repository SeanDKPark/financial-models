"""Offline tests for the spline zero-rate interpolators."""

import numpy as np
import pytest

from finmodels.curves.base import BaseYieldCurve
from finmodels.curves.interpolation import LinearZeroInterpolator
from finmodels.curves.splines import CubicZeroInterpolator, PchipZeroInterpolator

PILLARS = np.array([0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 30.0])
ZEROS = np.array([0.0398, 0.0413, 0.0438, 0.0455, 0.0467, 0.0482, 0.0548, 0.0538])

SPLINE_CLASSES = [CubicZeroInterpolator, PchipZeroInterpolator]


@pytest.fixture(params=SPLINE_CLASSES)
def spline(request):
    return request.param(PILLARS, ZEROS)


def test_is_base_yield_curve(spline):
    assert isinstance(spline, BaseYieldCurve)


def test_interpolates_pillars_exactly(spline):
    for t, z in zip(PILLARS, ZEROS):
        assert spline.zero_rate(float(t)) == pytest.approx(z, abs=1e-12)


def test_scalar_and_array_return_types(spline):
    assert isinstance(spline.zero_rate(3.0), float)
    assert isinstance(spline.discount_factor(3.0), float)
    out = spline.zero_rate(np.array([1.0, 3.0, 8.0]))
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_discount_factor_matches_zero(spline):
    t = 4.0
    assert spline.discount_factor(t) == pytest.approx(
        np.exp(-spline.zero_rate(t) * t)
    )
    assert spline.discount_factor(0.0) == pytest.approx(1.0)


def test_flat_extrapolation(spline):
    assert spline.zero_rate(0.1) == pytest.approx(ZEROS[0], abs=1e-12)
    assert spline.zero_rate(50.0) == pytest.approx(ZEROS[-1], abs=1e-12)


def test_inherited_forward_rate_origin_identity(spline):
    for t in (1.0, 3.0, 9.0):
        assert spline.forward_rate(0.0, t, "continuous") == pytest.approx(
            spline.zero_rate(t), abs=1e-12
        )


def test_analytical_forward_flat_curve():
    flat = CubicZeroInterpolator(np.array([1.0, 5.0, 10.0]), np.full(3, 0.05))
    f = flat.instantaneous_forward_rate(np.array([2.0, 6.0, 9.0]))
    np.testing.assert_allclose(f, 0.05, atol=1e-10)


def test_analytical_forward_matches_z_plus_t_zprime():
    cubic = CubicZeroInterpolator(PILLARS, ZEROS)
    t = 4.0
    z = cubic.spline(t)
    dz = cubic.spline.derivative()(t)
    assert cubic.instantaneous_forward_rate(t) == pytest.approx(z + t * dz)


def test_analytical_forward_rejects_negative_t(spline):
    with pytest.raises(ValueError):
        spline.instantaneous_forward_rate(-1.0)


def test_spline_forward_is_smoother_than_linear():
    """PCHIP / Cubic 6M forwards have far smaller jumps at pillar kinks than linear."""
    grid = np.linspace(0.5, 29.5, 400)
    linear = LinearZeroInterpolator(PILLARS, ZEROS)
    cubic = CubicZeroInterpolator(PILLARS, ZEROS)
    pchip = PchipZeroInterpolator(PILLARS, ZEROS)

    def max_abs_2nd_diff(curve):
        f = np.asarray(curve.forward_rate_rolling(grid, tenor=0.5, compounding="semi-annual"))
        return np.max(np.abs(np.diff(f, 2)))

    lin_roughness = max_abs_2nd_diff(linear)
    assert max_abs_2nd_diff(cubic) < lin_roughness
    assert max_abs_2nd_diff(pchip) < lin_roughness


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        CubicZeroInterpolator(np.array([1.0]), np.array([0.04]))  # need >= 2
    with pytest.raises(ValueError):
        PchipZeroInterpolator(np.array([2.0, 1.0]), np.array([0.04, 0.05]))  # unsorted
    with pytest.raises(ValueError):
        CubicZeroInterpolator(np.array([1.0, 2.0, 3.0]), np.array([0.04, 0.05]))  # mismatch
