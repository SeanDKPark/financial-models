"""Offline-friendly data containers for curve construction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

# Plausible domain for an interest rate expressed as a decimal (e.g. 0.045 = 4.5%).
# Wide enough to cover historical extremes without admitting percentage-point inputs.
_RATE_MIN = -0.10
_RATE_MAX = 1.00


@dataclass
class CurveMarketData:
    """A snapshot of quoted curve pillars for a single valuation date.

    Attributes
    ----------
    as_of_date:
        Valuation date of the quotes. Either a ``datetime.date`` or an
        ISO-8601 string (``"YYYY-MM-DD"``).
    curve_name:
        Identifier for the curve, e.g. ``"US_TREASURY_PAR_YIELD"``.
    pillars:
        Tenors in years, strictly positive and strictly ascending.
    rates:
        Rates as decimals aligned with ``pillars`` (``0.045`` for 4.5%).
    rate_type:
        Quote convention, e.g. ``"PAR_YIELD"`` or ``"ZERO"``.
    """

    as_of_date: date | str
    curve_name: str
    pillars: np.ndarray
    rates: np.ndarray
    rate_type: str = "PAR_YIELD"

    def __post_init__(self) -> None:
        self.pillars = np.asarray(self.pillars, dtype=float)
        self.rates = np.asarray(self.rates, dtype=float)

        if self.pillars.ndim != 1 or self.rates.ndim != 1:
            raise ValueError("pillars and rates must be one-dimensional")

        if self.pillars.shape != self.rates.shape:
            raise ValueError(
                f"pillars and rates length mismatch: "
                f"{self.pillars.shape[0]} vs {self.rates.shape[0]}"
            )

        if self.pillars.size == 0:
            raise ValueError("at least one pillar is required")

        if not np.all(np.isfinite(self.pillars)) or not np.all(np.isfinite(self.rates)):
            raise ValueError("pillars and rates must all be finite")

        if np.any(self.pillars <= 0.0):
            raise ValueError("pillars must be strictly positive")

        if np.any(np.diff(self.pillars) <= 0.0):
            raise ValueError("pillars must be strictly ascending")

        if np.any((self.rates < _RATE_MIN) | (self.rates > _RATE_MAX)):
            raise ValueError(
                f"rates must be decimals within [{_RATE_MIN}, {_RATE_MAX}]; "
                "did you pass percentage points?"
            )
