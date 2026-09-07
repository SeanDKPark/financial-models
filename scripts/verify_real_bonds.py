"""Verify `finmodels.instruments` against two real on-the-run US Treasuries.

    .venv/Scripts/python.exe scripts/verify_real_bonds.py

Checks schedule generation, End-of-Month handling (including the 2028 leap-year
month-end), and ACT/ACT_ICMA accrual against hand-computed day counts for the
benchmark 10Y note (CUSIP 91282CJZ5) and 5Y note (CUSIP 91282CKD2).

Offline, no network, nothing curve-related. Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from datetime import date

from finmodels.instruments import DayCountConvention, FixedCouponBond

_failures = 0


def check(label: str, got, expected) -> None:
    global _failures
    ok = got == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         expected: {expected!r}")
        print(f"         got:      {got!r}")
        _failures += 1


def check_close(label: str, got: float, expected: float, tol: float = 1e-12) -> None:
    global _failures
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         expected: {expected!r}")
        print(f"         got:      {got!r}  (|diff| = {abs(got - expected):.3e})")
        _failures += 1


def verify_10y_note() -> None:
    print("Benchmark 10Y Note  (CUSIP 91282CJZ5, 4.000% due 2034-02-15)")
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.040,
        frequency=2,
        day_count=DayCountConvention.ACT_ACT_ICMA,
        issue_date=date(2024, 2, 15),
    )

    check("is_eom auto-detected False", bond.is_eom, False)

    dates = bond.coupon_dates()
    check("20 coupon dates", len(dates), 20)
    check(
        "every coupon on the 15th of Feb / Aug",
        all(d.day == 15 and d.month in (2, 8) for d in dates),
        True,
    )
    check("final coupon is maturity", dates[-1], date(2034, 2, 15))

    settlement = date(2024, 10, 10)
    t0, t1 = bond.coupon_period(settlement)
    check("T_0 (last coupon)", t0, date(2024, 8, 15))
    check("T_1 (next coupon)", t1, date(2025, 2, 15))

    days_in_period = (t1 - t0).days
    days_accrued = (settlement - t0).days
    check("days_in_period == 184", days_in_period, 184)
    check("days_accrued == 56", days_accrued, 56)

    check_close("accrual_fraction w == 56/184", bond.accrual_fraction(settlement), 56 / 184)
    check_close(
        "accrued_interest == (56/184) * 2.0",
        bond.accrued_interest(settlement),
        (56 / 184) * 2.0,
    )


def verify_5y_note() -> None:
    print("\nBenchmark 5Y Note  (CUSIP 91282CKD2, 4.250% due 2029-02-28)")
    bond = FixedCouponBond(
        maturity_date=date(2029, 2, 28),
        coupon_rate=0.0425,
        frequency=2,
        day_count=DayCountConvention.ACT_ACT_ICMA,
        issue_date=date(2024, 2, 29),  # leap day
    )

    check("is_eom auto-detected True", bond.is_eom, True)

    expected_schedule = [
        date(2024, 8, 31),
        date(2025, 2, 28),
        date(2025, 8, 31),
        date(2026, 2, 28),
        date(2026, 8, 31),
        date(2027, 2, 28),
        date(2027, 8, 31),
        date(2028, 2, 29),  # leap-year month-end
        date(2028, 8, 31),
        date(2029, 2, 28),
    ]
    check("coupon_dates() preserves all 10 month-ends", bond.coupon_dates(), expected_schedule)

    settlement = date(2028, 1, 15)
    t0, t1 = bond.coupon_period(settlement)
    check("T_0", t0, date(2027, 8, 31))
    check("T_1 (spans leap February)", t1, date(2028, 2, 29))
    check("days_in_period == 182", (t1 - t0).days, 182)


def main() -> int:
    verify_10y_note()
    verify_5y_note()

    print()
    if _failures:
        print(f"FAILED: {_failures} check(s) did not pass.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
