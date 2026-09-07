"""Fixed-coupon bond contract and cashflow schedule generation.

This module is deliberately decoupled from yield curves. It owns the calendar
math: backward-rolled coupon schedules (with End-of-Month handling), day-count
conventions, accrued interest, and the settlement-relative discounting times
(``year_fraction``) that a pricing layer later feeds to a ``BaseYieldCurve``.

Scope is US Treasury notes/bonds: regular semi-annual schedules anchored at
maturity. Irregular first/last (stub) periods are not modelled; every coupon
carries ``accrual_factor = 1 / frequency``.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

_ALLOWED_FREQUENCIES = frozenset({1, 2, 4, 12})


class DayCountConvention(str, Enum):
    """Day-count basis for the accrued-interest fraction."""

    ACT_ACT_ICMA = "ACT/ACT_ICMA"  # standard US Treasury note/bond convention
    THIRTY_360_US = "30/360_US"  # standard bond basis / corporate fallback


@dataclass(frozen=True)
class CashFlow:
    """A single scheduled payment, viewed from a settlement date.

    Attributes
    ----------
    payment_date:
        Calendar date the cashflow is paid.
    year_fraction:
        Discounting time ``t_i`` in years measured from settlement, on the
        semi-annual Treasury convention (see :meth:`FixedCouponBond.cashflows`).
    accrual_factor:
        Fraction of the annual coupon this payment represents; exactly
        ``1 / frequency`` for a regular period.
    coupon:
        Coupon amount in currency units, ``face_value * coupon_rate / frequency``.
    principal:
        Face value at maturity, ``0.0`` for intermediate coupons.
    amount:
        ``coupon + principal``.
    """

    payment_date: date
    year_fraction: float
    accrual_factor: float
    coupon: float
    principal: float
    amount: float


def _is_month_end(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def _add_months(year: int, month: int, months: int) -> tuple[int, int]:
    """Return the (year, month) that is ``months`` away from ``year-month``."""
    total = year * 12 + (month - 1) + months
    y, m = divmod(total, 12)
    return y, m + 1


def _days_30_360_us(d1: date, d2: date) -> int:
    """Signed day count between ``d1`` and ``d2`` under the US 30/360 rules."""
    day1, day2 = d1.day, d2.day
    if day1 == 31:
        day1 = 30
    if day2 == 31 and day1 == 30:
        day2 = 30
    return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (day2 - day1)


class FixedCouponBond:
    """A fixed-rate coupon bond with a regular backward-rolled schedule."""

    def __init__(
        self,
        maturity_date: date,
        coupon_rate: float,
        face_value: float = 100.0,
        frequency: int = 2,
        day_count: DayCountConvention = DayCountConvention.ACT_ACT_ICMA,
        issue_date: Optional[date] = None,
        is_eom: Optional[bool] = None,
    ) -> None:
        if frequency not in _ALLOWED_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(_ALLOWED_FREQUENCIES)}, got {frequency}"
            )
        if coupon_rate < 0.0:
            raise ValueError(f"coupon_rate must be non-negative, got {coupon_rate}")
        if face_value <= 0.0:
            raise ValueError(f"face_value must be positive, got {face_value}")
        if issue_date is not None and issue_date >= maturity_date:
            raise ValueError(
                f"issue_date {issue_date} must precede maturity_date {maturity_date}"
            )

        self.maturity_date = maturity_date
        self.coupon_rate = float(coupon_rate)
        self.face_value = float(face_value)
        self.frequency = int(frequency)
        self.day_count = DayCountConvention(day_count)
        self.issue_date = issue_date
        self.is_eom = _is_month_end(maturity_date) if is_eom is None else bool(is_eom)

        self._step_months = 12 // self.frequency

    def __repr__(self) -> str:
        return (
            f"FixedCouponBond(maturity={self.maturity_date.isoformat()}, "
            f"coupon={self.coupon_rate * 100:g}%, freq={self.frequency}, "
            f"day_count={self.day_count.value})"
        )

    # -- schedule ---------------------------------------------------------------

    def _coupon_date(self, i: int) -> date:
        """The ``i``-th coupon date rolling backward from maturity (``i = 0`` is maturity)."""
        y, m = _add_months(
            self.maturity_date.year, self.maturity_date.month, -i * self._step_months
        )
        last = calendar.monthrange(y, m)[1]
        day = last if self.is_eom else min(self.maturity_date.day, last)
        return date(y, m, day)

    def coupon_dates(self) -> list[date]:
        """All coupon payment dates in ``(issue_date, maturity_date]``, ascending.

        Requires ``issue_date``; rolling backward has no natural stopping point
        without it.
        """
        if self.issue_date is None:
            raise ValueError("coupon_dates() requires issue_date")
        dates: list[date] = []
        i = 0
        while True:
            d = self._coupon_date(i)
            if d <= self.issue_date:
                break
            dates.append(d)
            i += 1
        dates.sort()
        return dates

    def coupon_period(self, settlement: date) -> tuple[date, date]:
        """The ``(T_0, T_1)`` coupon period bracketing ``settlement``.

        ``T_0 <= settlement < T_1``. ``T_0`` is the last coupon date on or before
        settlement (or ``issue_date`` for the first period); ``T_1`` is the next
        coupon date. Raises if ``settlement`` is outside ``[issue_date, maturity_date]``.
        """
        self._check_settlement(settlement)
        return self._bracketing_dates(settlement)

    def _bracketing_dates(self, settlement: date) -> tuple[date, date]:
        """Return ``(T_0, T_1)`` with ``T_0 <= settlement < T_1``."""
        i = 0
        nxt = self.maturity_date
        while True:
            d = self._coupon_date(i)
            if d <= settlement:
                return d, nxt
            if self.issue_date is not None and d <= self.issue_date:
                # rolled past the dated date without bracketing: stub first period
                return self.issue_date, nxt
            if self.issue_date is None and i > 1200:
                raise ValueError(
                    "settlement precedes the first coupon period; provide issue_date"
                )
            nxt = d
            i += 1

    # -- accrual --------------------------------------------------------------

    def _check_settlement(self, settlement: date) -> None:
        if settlement > self.maturity_date:
            raise ValueError(
                f"settlement {settlement} is after maturity {self.maturity_date}"
            )
        if self.issue_date is not None and settlement < self.issue_date:
            raise ValueError(
                f"settlement {settlement} is before issue_date {self.issue_date}"
            )

    def accrual_fraction(self, settlement: date) -> float:
        """Fraction ``w`` of the current coupon period elapsed at ``settlement``."""
        self._check_settlement(settlement)
        if settlement == self.maturity_date:
            return 0.0
        t0, t1 = self._bracketing_dates(settlement)
        if self.day_count is DayCountConvention.ACT_ACT_ICMA:
            return (settlement - t0).days / (t1 - t0).days
        # 30/360_US
        return _days_30_360_us(t0, settlement) / (360.0 / self.frequency)

    def accrued_interest(self, settlement: date) -> float:
        """Accrued interest in currency units at ``settlement``."""
        w = self.accrual_fraction(settlement)
        return w * self.face_value * self.coupon_rate / self.frequency

    # -- cashflows ----------------------------------------------------------

    def cashflows(self, settlement: date) -> list[CashFlow]:
        """Remaining cashflows (``payment_date > settlement``), ascending.

        ``year_fraction`` for the ``i``-th remaining coupon (``i = 1..K``) is
        ``t_i = (1 - w) / frequency + (i - 1) / frequency`` where ``w`` is the
        accrual fraction; when settlement falls on a coupon date ``w = 0`` and
        ``t_1 = 1 / frequency``.
        """
        self._check_settlement(settlement)
        if self.issue_date is None:
            raise ValueError("cashflows() requires issue_date")

        w = self.accrual_fraction(settlement)
        future = [d for d in self.coupon_dates() if d > settlement]

        coupon_amt = self.face_value * self.coupon_rate / self.frequency
        accrual_factor = 1.0 / self.frequency

        flows: list[CashFlow] = []
        for i, pay_date in enumerate(future, start=1):
            t_i = (1.0 - w) / self.frequency + (i - 1) / self.frequency
            principal = self.face_value if pay_date == self.maturity_date else 0.0
            flows.append(
                CashFlow(
                    payment_date=pay_date,
                    year_fraction=t_i,
                    accrual_factor=accrual_factor,
                    coupon=coupon_amt,
                    principal=principal,
                    amount=coupon_amt + principal,
                )
            )
        return flows
