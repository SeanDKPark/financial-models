"""Par-to-zero bootstrapping for Treasury CMT par yield curves.

Strips discount factors ``P(0, t)`` and zero-coupon spot rates ``z(t)`` from a
:class:`~finmodels.curves.data_models.CurveMarketData` holding par yields.

Method
------
* Tenors ``t <= short_end_cutoff`` are treated as single-cashflow (money-market /
  T-Bill) instruments and discounted directly from the par yield.
* Tenors ``t > short_end_cutoff`` are par coupon bonds paying ``frequency`` times
  per year. Discount factors are solved by a **sequential closed-form** recursion
  on a dense coupon grid (steps of ``1 / frequency`` years), so no non-linear
  root finder is required::

      P(0, T_k) = (1 - (y_k / m) * sum_{j < N_k} P(0, tau_j)) / (1 + y_k / m)

  Par yields at intermediate coupon dates that fall between sparse market pillars
  are obtained by linear interpolation of the par curve.
"""

from __future__ import annotations

import math

import numpy as np

from finmodels.curves.data_models import CurveMarketData
from finmodels.curves.interpolation import LinearZeroInterpolator

_VALID_COMPOUNDING = ("continuous", "semi-annual")
_PAR_YIELD = "PAR_YIELD"


class ParToZeroBootstrapper:
    """Configurable par-yield -> zero-rate bootstrapper.

    Parameters
    ----------
    frequency:
        Coupon payments per year for the long-end par bonds (``m``). Default 2.
    short_end_cutoff:
        Tenors (in years) at or below this are treated as zero-coupon
        instruments. Default 1.0.
    compounding:
        Convention for the returned zero rates: ``"continuous"`` (default) or
        ``"semi-annual"``.

    Notes
    -----
    The returned :class:`LinearZeroInterpolator` stores zero rates in the chosen
    ``compounding`` convention. Its ``discount_factor`` helper assumes continuous
    compounding, so it is only self-consistent when ``compounding="continuous"``.
    """

    def __init__(
        self,
        frequency: int = 2,
        short_end_cutoff: float = 1.0,
        compounding: str = "continuous",
    ) -> None:
        if frequency <= 0:
            raise ValueError(f"frequency must be positive, got {frequency}")
        if compounding not in _VALID_COMPOUNDING:
            raise ValueError(
                f"compounding must be one of {_VALID_COMPOUNDING}, got {compounding!r}"
            )
        if short_end_cutoff < 0.0:
            raise ValueError("short_end_cutoff must be non-negative")

        self.frequency = int(frequency)
        self.short_end_cutoff = float(short_end_cutoff)
        self.compounding = compounding

    # ------------------------------------------------------------------ helpers
    def _zero_coupon_df(self, y: float, t: float) -> float:
        """Discount factor for a single-cashflow instrument with par yield ``y``."""
        if self.compounding == "continuous":
            return math.exp(-y * t)
        m = self.frequency
        return (1.0 + y / m) ** (-m * t)

    def _df_to_zero(self, df: float, t: float) -> float:
        """Convert a discount factor to a zero rate in the configured convention."""
        if self.compounding == "continuous":
            return -math.log(df) / t
        m = self.frequency
        return m * (df ** (-1.0 / (m * t)) - 1.0)

    def _validate(self, market_data: CurveMarketData) -> None:
        if market_data.rate_type != _PAR_YIELD:
            raise ValueError(
                f"expected rate_type={_PAR_YIELD!r}, got {market_data.rate_type!r}"
            )

    # -------------------------------------------------------------------- solve
    def bootstrap(self, market_data: CurveMarketData) -> LinearZeroInterpolator:
        """Bootstrap ``market_data`` and return a zero-rate interpolator."""
        self._validate(market_data)

        m = self.frequency
        pillars = np.asarray(market_data.pillars, dtype=float)
        par_rates = np.asarray(market_data.rates, dtype=float)
        t_max = float(pillars[-1])

        def par_at(t: float) -> float:
            return float(np.interp(t, pillars, par_rates))

        # Dense semi-annual coupon grid: tau_j = j / m for j = 1 .. m * t_max.
        n_max = int(round(m * t_max))
        grid_tenors = np.array([j / m for j in range(1, n_max + 1)], dtype=float)
        grid_df = np.empty(n_max, dtype=float)

        running_sum = 0.0  # sum of P(0, tau_i) over already-solved grid nodes
        for i, tau in enumerate(grid_tenors):
            y = par_at(tau)
            if tau <= self.short_end_cutoff + 1e-12:
                df = self._zero_coupon_df(y, float(tau))
            else:
                c = y / m
                df = (1.0 - c * running_sum) / (1.0 + c)
            grid_df[i] = df
            running_sum += df

        # Assemble the output tenor set: the dense coupon grid (so every coupon
        # date is a node and discount factors reprice par bonds exactly) plus any
        # short-end market pillars that fall off that grid (e.g. 1M, 3M).
        off_grid = [
            float(t)
            for t in pillars
            if not np.any(np.isclose(grid_tenors, t))
        ]
        out_tenors = np.array(sorted(set(off_grid) | set(grid_tenors.tolist())))

        out_df = np.empty(out_tenors.size, dtype=float)
        for k, t in enumerate(out_tenors):
            hit = np.where(np.isclose(grid_tenors, t))[0]
            if hit.size:
                out_df[k] = grid_df[hit[0]]
            else:
                out_df[k] = self._zero_coupon_df(par_at(t), t)

        zero_rates = np.array(
            [self._df_to_zero(df, float(t)) for df, t in zip(out_df, out_tenors)],
            dtype=float,
        )
        return LinearZeroInterpolator(out_tenors, zero_rates)


def bootstrap_par_curve(
    market_data: CurveMarketData, **kwargs
) -> LinearZeroInterpolator:
    """Convenience wrapper: ``ParToZeroBootstrapper(**kwargs).bootstrap(market_data)``."""
    return ParToZeroBootstrapper(**kwargs).bootstrap(market_data)
