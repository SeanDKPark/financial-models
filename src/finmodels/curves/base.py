"""Common interface for yield-curve objects.

A concrete curve only has to answer two questions - the zero rate and the
discount factor at a tenor. Everything derivable from discount factors (discrete
forward rates in the supported compounding conventions) lives here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_FWD_COMPOUNDING = ("continuous", "simple", "semi-annual")
_SEMI_M = 2  # compounding frequency for the "semi-annual" convention


def _forward_from_dfs(p1, p2, dt, compounding: str):
    """Forward rate implied by discount factors ``p1 = P(0, t1)``, ``p2 = P(0, t2)``.

    ``dt = t2 - t1``. Works elementwise on scalars or numpy arrays.
    """
    if compounding == "continuous":
        return (np.log(p1) - np.log(p2)) / dt
    if compounding == "simple":
        return (p1 / p2 - 1.0) / dt
    if compounding == "semi-annual":
        m = _SEMI_M
        return m * ((p1 / p2) ** (1.0 / (m * dt)) - 1.0)
    raise ValueError(
        f"compounding must be one of {_FWD_COMPOUNDING}, got {compounding!r}"
    )


class BaseYieldCurve(ABC):
    """Abstract yield curve. Subclasses implement ``zero_rate`` / ``discount_factor``."""

    @abstractmethod
    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Continuously-compounded zero rate at tenor ``t`` (years)."""

    @abstractmethod
    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        """Discount factor ``P(0, t)``."""

    # ------------------------------------------------------------------ forwards
    def forward_rate(
        self, t1: float, t2: float, compounding: str = "continuous"
    ) -> float:
        """Forward rate over ``[t1, t2]`` implied by the curve's discount factors.

        ``0 <= t1 < t2``. ``compounding`` is ``"continuous"`` (default),
        ``"simple"`` or ``"semi-annual"`` (``m = 2``). With ``t1 == 0`` the
        continuous forward equals the spot zero ``z(t2)``.
        """
        t1 = float(t1)
        t2 = float(t2)
        if t1 < 0.0:
            raise ValueError(f"t1 must be non-negative, got {t1}")
        if t2 <= t1:
            raise ValueError(f"require t1 < t2, got t1={t1}, t2={t2}")
        if compounding not in _FWD_COMPOUNDING:
            raise ValueError(
                f"compounding must be one of {_FWD_COMPOUNDING}, got {compounding!r}"
            )

        p1 = self.discount_factor(t1)
        p2 = self.discount_factor(t2)
        return float(_forward_from_dfs(p1, p2, t2 - t1, compounding))

    def forward_rate_rolling(
        self,
        t: float | np.ndarray,
        tenor: float = 0.5,
        compounding: str = "semi-annual",
    ) -> float | np.ndarray:
        """Forward rate from ``t`` to ``t + tenor`` (e.g. the 6-month rolling forward).

        Vectorized: scalar in -> ``float`` out, numpy array in -> array out.
        Requires ``t >= 0`` and ``tenor > 0``.
        """
        tenor = float(tenor)
        if tenor <= 0.0:
            raise ValueError(f"tenor must be positive, got {tenor}")
        if compounding not in _FWD_COMPOUNDING:
            raise ValueError(
                f"compounding must be one of {_FWD_COMPOUNDING}, got {compounding!r}"
            )

        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")

        p1 = np.asarray(self.discount_factor(t_arr), dtype=float)
        p2 = np.asarray(self.discount_factor(t_arr + tenor), dtype=float)
        fwd = _forward_from_dfs(p1, p2, tenor, compounding)

        if np.isscalar(t) or t_arr.ndim == 0:
            return float(fwd)
        return np.asarray(fwd, dtype=float)
