"""Nelson-Siegel-Svensson parametric yield curve.

The Svensson (1994) extension of Nelson-Siegel models the continuously
compounded zero rate as a level, a slope, and two "hump" terms::

    z(t) = b0
         + b1 * L1(t, tau1)
         + b2 * (L1(t, tau1) - exp(-t/tau1))
         + b3 * (L1(t, tau2) - exp(-t/tau2))

    L1(t, tau) = (1 - exp(-t/tau)) / (t/tau)          # -> 1 as t -> 0

The instantaneous forward rate has the matching closed form::

    f(t) = b0
         + b1 * exp(-t/tau1)
         + b2 * (t/tau1) * exp(-t/tau1)
         + b3 * (t/tau2) * exp(-t/tau2)

Both ``z`` and ``f`` tend to ``b0`` as ``t -> infinity`` and to ``b0 + b1`` at
``t = 0``. Nelson-Siegel is the special case ``b3 = 0``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from finmodels.curves.base import BaseYieldCurve


def _l1_and_hump(t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(L1, L1 - exp(-t/tau))`` with the correct ``t = 0`` limits."""
    x = t / tau
    safe_x = np.where(x == 0.0, 1.0, x)
    l1 = np.where(x == 0.0, 1.0, -np.expm1(-safe_x) / safe_x)
    hump = l1 - np.exp(-x)  # at x = 0: 1 - 1 = 0
    return l1, hump


class SvenssonCurve(BaseYieldCurve):
    """Six-parameter Nelson-Siegel-Svensson zero curve."""

    __slots__ = ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2")

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        tau1: float,
        tau2: float,
    ) -> None:
        if tau1 <= 0.0 or tau2 <= 0.0:
            raise ValueError(f"tau1, tau2 must be positive, got {tau1}, {tau2}")
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.beta3 = float(beta3)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)

    # ------------------------------------------------------------------ factories
    @classmethod
    def nelson_siegel(
        cls, beta0: float, beta1: float, beta2: float, tau1: float
    ) -> "SvenssonCurve":
        """Four-parameter Nelson-Siegel curve (``beta3 = 0``, ``tau2`` inert)."""
        return cls(beta0, beta1, beta2, 0.0, tau1, tau1)

    @classmethod
    def calibrate(
        cls,
        pillars: np.ndarray,
        zero_rates: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        beta0_bounds: tuple[float, float] = (0.01, 0.15),
        beta1_bounds: tuple[float, float] = (-0.15, 0.15),
        beta_curv_bounds: tuple[float, float] = (-0.20, 0.20),
        tau1_bounds: tuple[float, float] = (0.2, 5.0),
        tau2_bounds: tuple[float, float] = (3.0, 25.0),
    ) -> "SvenssonCurve":
        """Least-squares fit of the six parameters to observed zero rates.

        Fits ``z(pillar_i)`` to ``zero_rates_i`` inside economically plausible
        box bounds:

        * ``beta0`` (long-term level) in ``[1%, 15%]`` - non-negative,
        * ``beta1`` (slope loading) in ``[-15%, 15%]``,
        * ``beta2``, ``beta3`` (curvature loadings) in ``[-20%, 20%]``,
        * ``tau1`` in ``[0.2, 5.0]`` (short/medium hump),
        * ``tau2`` in ``[3.0, 25.0]`` (medium/long hump).

        A penalty residual keeps the fitted instantaneous short rate
        ``z(0) = beta0 + beta1`` non-negative (box bounds alone cannot express
        that cross-parameter inequality). A short multi-start over the decay
        parameters mitigates the multimodality of the NSS objective.
        ``weights`` (aligned with ``pillars``) scale the fit residuals.
        """
        pillars = np.asarray(pillars, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)
        if pillars.shape != zero_rates.shape or pillars.ndim != 1 or pillars.size < 4:
            raise ValueError("need >= 4 aligned (pillar, zero_rate) points")
        if np.any(pillars <= 0.0):
            raise ValueError("pillars must be positive")

        w = np.ones_like(pillars) if weights is None else np.asarray(weights, float)
        if w.shape != pillars.shape:
            raise ValueError("weights must align with pillars")

        lo = [
            beta0_bounds[0], beta1_bounds[0], beta_curv_bounds[0],
            beta_curv_bounds[0], tau1_bounds[0], tau2_bounds[0],
        ]
        hi = [
            beta0_bounds[1], beta1_bounds[1], beta_curv_bounds[1],
            beta_curv_bounds[1], tau1_bounds[1], tau2_bounds[1],
        ]

        short_rate_penalty = 100.0  # >> residual scale, so z(0) < 0 is avoided

        def residual(p: np.ndarray) -> np.ndarray:
            fit = (cls(*p)._z(pillars) - zero_rates) * w
            neg_short_rate = max(-(p[0] + p[1]), 0.0)
            return np.append(fit, short_rate_penalty * neg_short_rate)

        level = float(np.clip(zero_rates[-1], *beta0_bounds))
        slope = float(np.clip(zero_rates[0] - zero_rates[-1], *beta1_bounds))
        if level + slope < 0.0:  # initial-guess feasibility for z(0) >= 0
            slope = -level

        best: tuple[float, np.ndarray] | None = None
        for tau1_0, tau2_0 in ((0.5, 4.0), (1.5, 7.0), (2.5, 12.0), (4.0, 20.0)):
            x0 = np.clip(
                np.array([level, slope, 0.0, 0.0, tau1_0, tau2_0]), lo, hi
            )
            sol = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf")
            cost = float(sol.cost)
            if best is None or cost < best[0]:
                best = (cost, sol.x)

        assert best is not None
        params = best[1]
        if params[0] + params[1] < -1e-6:
            raise ValueError(
                f"calibration produced a negative short rate z(0) = "
                f"{params[0] + params[1]:.4%}"
            )
        return cls(*params)

    # --------------------------------------------------------------------- params
    @property
    def params(self) -> dict[str, float]:
        return {
            "beta0": self.beta0,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "beta3": self.beta3,
            "tau1": self.tau1,
            "tau2": self.tau2,
        }

    # ---------------------------------------------------------------- curve values
    def _z(self, t: np.ndarray) -> np.ndarray:
        l1a, humpa = _l1_and_hump(t, self.tau1)
        _, humpb = _l1_and_hump(t, self.tau2)
        return self.beta0 + self.beta1 * l1a + self.beta2 * humpa + self.beta3 * humpb

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")
        z = self._z(t_arr)
        if np.isscalar(t) or t_arr.ndim == 0:
            return float(z)
        return z

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")
        df = np.exp(-self._z(t_arr) * t_arr)
        if np.isscalar(t) or t_arr.ndim == 0:
            return float(df)
        return df

    def instantaneous_forward_rate(
        self, t: float | np.ndarray
    ) -> float | np.ndarray:
        """Analytical instantaneous forward ``f(t)`` (continuously compounded)."""
        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")
        x1 = t_arr / self.tau1
        x2 = t_arr / self.tau2
        e1 = np.exp(-x1)
        f = (
            self.beta0
            + self.beta1 * e1
            + self.beta2 * x1 * e1
            + self.beta3 * x2 * np.exp(-x2)
        )
        if np.isscalar(t) or t_arr.ndim == 0:
            return float(f)
        return f
