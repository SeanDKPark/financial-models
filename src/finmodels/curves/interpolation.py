"""Pure-math curve interpolation, decoupled from any market data source."""

from __future__ import annotations

import numpy as np

from finmodels.curves.base import BaseYieldCurve
from finmodels.curves.data_models import CurveMarketData


class LinearZeroInterpolator(BaseYieldCurve):
    """Linear interpolation of zero rates in the tenor dimension.

    Between adjacent pillars the rate is a straight line::

        r(t) = r_1 + (r_2 - r_1) / (t_2 - t_1) * (t - t_1)

    Outside the pillar domain the nearest pillar rate is held flat. Forward-rate
    helpers are inherited from :class:`BaseYieldCurve`.
    """

    def __init__(
        self,
        pillars: np.ndarray | None = None,
        rates: np.ndarray | None = None,
        *,
        market_data: CurveMarketData | None = None,
    ) -> None:
        if market_data is not None:
            if pillars is not None or rates is not None:
                raise ValueError(
                    "provide either market_data or (pillars, rates), not both"
                )
            pillars, rates = market_data.pillars, market_data.rates

        if pillars is None or rates is None:
            raise ValueError("provide either market_data or both pillars and rates")

        self.pillars = np.asarray(pillars, dtype=float)
        self.rates = np.asarray(rates, dtype=float)

        if self.pillars.shape != self.rates.shape:
            raise ValueError("pillars and rates length mismatch")
        if self.pillars.ndim != 1 or self.pillars.size == 0:
            raise ValueError("pillars and rates must be non-empty 1-D arrays")
        if np.any(np.diff(self.pillars) <= 0.0):
            raise ValueError("pillars must be strictly ascending")

    @classmethod
    def from_market_data(cls, market_data: CurveMarketData) -> "LinearZeroInterpolator":
        return cls(market_data=market_data)

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolated zero rate at tenor ``t`` (years), flat-extrapolated."""
        t_arr = np.asarray(t, dtype=float)
        # np.interp already applies flat extrapolation using the end values.
        result = np.interp(t_arr, self.pillars, self.rates)
        if np.isscalar(t) or (t_arr.ndim == 0):
            return float(result)
        return result

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        """Continuously-compounded discount factor ``P(0, t) = exp(-r(t) * t)``."""
        t_arr = np.asarray(t, dtype=float)
        r = np.interp(t_arr, self.pillars, self.rates)
        df = np.exp(-r * t_arr)
        if np.isscalar(t) or (t_arr.ndim == 0):
            return float(df)
        return df
