"""Spline zero-rate interpolators.

Both curves interpolate a set of zero-rate pillars with a smoother scheme than
piecewise-linear, which removes the slope discontinuities that make the
linear-interpolated forward curve look like a shark fin. Each exposes an
*analytical* instantaneous forward built from the spline's own derivative:

    f_inst(t) = z(t) + t * z'(t)          (continuously compounded)

Values are flat-extrapolated outside the pillar range; the instantaneous forward
is flat there too (``z'`` is taken as 0).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from finmodels.curves.base import BaseYieldCurve


class _ScipyZeroCurve(BaseYieldCurve):
    """Shared machinery for scipy-backed zero-rate interpolators."""

    def __init__(self, pillars: np.ndarray, zero_rates: np.ndarray, curve) -> None:
        self.pillars = np.asarray(pillars, dtype=float)
        self.zero_rates = np.asarray(zero_rates, dtype=float)

        if self.pillars.shape != self.zero_rates.shape:
            raise ValueError("pillars and zero_rates length mismatch")
        if self.pillars.ndim != 1 or self.pillars.size < 2:
            raise ValueError("need at least two pillars")
        if np.any(np.diff(self.pillars) <= 0.0):
            raise ValueError("pillars must be strictly ascending")
        if not (np.all(np.isfinite(self.pillars)) and np.all(np.isfinite(self.zero_rates))):
            raise ValueError("pillars and zero_rates must be finite")

        self._curve = curve
        self._deriv = curve.derivative()
        self._t_min = float(self.pillars[0])
        self._t_max = float(self.pillars[-1])

    def _clip(self, t_arr: np.ndarray) -> np.ndarray:
        return np.clip(t_arr, self._t_min, self._t_max)

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        z = np.asarray(self._curve(self._clip(t_arr)), dtype=float)
        if np.isscalar(t) or t_arr.ndim == 0:
            return float(z)
        return z

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        z = np.asarray(self._curve(self._clip(t_arr)), dtype=float)
        df = np.exp(-z * t_arr)
        if np.isscalar(t) or t_arr.ndim == 0:
            return float(df)
        return df

    def instantaneous_forward_rate(
        self, t: float | np.ndarray
    ) -> float | np.ndarray:
        """Analytical instantaneous forward ``f(t) = z(t) + t * z'(t)``."""
        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")
        clipped = self._clip(t_arr)
        z = np.asarray(self._curve(clipped), dtype=float)
        dz = np.asarray(self._deriv(clipped), dtype=float)
        outside = (t_arr < self._t_min) | (t_arr > self._t_max)
        dz = np.where(outside, 0.0, dz)
        f = z + t_arr * dz
        if np.isscalar(t) or t_arr.ndim == 0:
            return float(f)
        return f


class CubicZeroInterpolator(_ScipyZeroCurve):
    """Natural cubic spline through the zero-rate pillars (C2, may overshoot)."""

    def __init__(self, pillars: np.ndarray, zero_rates: np.ndarray) -> None:
        p = np.asarray(pillars, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        super().__init__(p, z, CubicSpline(p, z, bc_type="natural"))
        self.spline = self._curve


class PchipZeroInterpolator(_ScipyZeroCurve):
    """Shape-preserving monotone cubic (PCHIP): C1, no overshoot between pillars."""

    def __init__(self, pillars: np.ndarray, zero_rates: np.ndarray) -> None:
        p = np.asarray(pillars, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        super().__init__(p, z, PchipInterpolator(p, z))
        self.pchip = self._curve
