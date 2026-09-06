"""One-off demo: fetch the live Treasury par curve, bootstrap it, print a table.

    python scripts/demo_treasury_curve.py

Hits the network (home.treasury.gov). Not a pytest test; nothing here is asserted.
"""

from __future__ import annotations

import numpy as np

from finmodels.curves.bootstrapping import bootstrap_par_curve
from finmodels.curves.data_models import CurveMarketData
from finmodels.market_data.treasury_par import fetch_treasury_par_curve

# label -> year fraction, for the benchmark tenors we want in the table.
_BENCHMARKS: list[tuple[str, float]] = [
    ("1M", 1 / 12),
    ("3M", 3 / 12),
    ("6M", 6 / 12),
    ("1Y", 1.0),
    ("2Y", 2.0),
    ("5Y", 5.0),
    ("10Y", 10.0),
    ("30Y", 30.0),
]


def main() -> None:
    par_curve = fetch_treasury_par_curve()
    zero_curve = bootstrap_par_curve(par_curve, compounding="semi-annual")

    print(f"US Treasury par curve as of {par_curve.as_of_date} ({par_curve.curve_name})")
    print()
    header = f"{'Tenor':<6}{'Par Yield':>12}{'Zero Rate':>14}{'Discount Factor':>18}"
    print(header)
    print("-" * len(header))

    for label, t in _BENCHMARKS:
        par_yield = _par_at(par_curve, t)
        zero_rate = zero_curve.zero_rate(t)
        # Discount factor consistent with the semi-annual zero rate, not the
        # interpolator's exp(-r t) helper (which assumes continuous compounding).
        df = (1.0 + zero_rate / 2.0) ** (-2.0 * t)
        print(f"{label:<6}{par_yield:>11.3%}{zero_rate:>13.3%}{df:>18.6f}")


def _par_at(par_curve: CurveMarketData, t: float) -> float:
    return float(np.interp(t, par_curve.pillars, par_curve.rates))


if __name__ == "__main__":
    main()
