"""Offline tests for the fixed-coupon bond contract and schedule generation."""

from datetime import date

import pytest

from finmodels.instruments.bond import (
    CashFlow,
    DayCountConvention,
    FixedCouponBond,
)

# --------------------------------------------------------------------------- #
# End-of-month schedule handling
# --------------------------------------------------------------------------- #


def test_eom_autodetect_and_backward_roll_across_leap_year():
    # 5Y note maturing on the last day of Feb; is_eom should auto-detect.
    bond = FixedCouponBond(
        maturity_date=date(2029, 2, 28),
        coupon_rate=0.0425,
        issue_date=date(2024, 2, 29),
    )
    assert bond.is_eom is True

    dates = bond.coupon_dates()
    assert len(dates) == 10  # 5 years * 2 coupons

    # Each unadjusted coupon date snaps to month-end.
    assert date(2028, 8, 31) in dates  # Aug -> 31
    assert date(2028, 2, 29) in dates  # leap February -> 29
    assert date(2027, 2, 28) in dates  # non-leap February -> 28
    assert dates[-1] == date(2029, 2, 28)


def test_mid_month_cycle_preserves_day_of_month():
    # 10Y note on the 15th: classic mid-month Treasury cycle.
    bond = FixedCouponBond(
        maturity_date=date(2034, 11, 15),
        coupon_rate=0.045,
        issue_date=date(2024, 11, 15),
    )
    assert bond.is_eom is False

    dates = bond.coupon_dates()
    assert len(dates) == 20
    assert all(d.day == 15 for d in dates)
    assert {d.month for d in dates} == {5, 11}


def test_is_eom_can_be_forced():
    bond = FixedCouponBond(
        maturity_date=date(2030, 6, 15),
        coupon_rate=0.03,
        issue_date=date(2025, 6, 15),
        is_eom=True,
    )
    dates = bond.coupon_dates()
    assert date(2029, 12, 31) in dates
    assert date(2030, 6, 30) in dates


# --------------------------------------------------------------------------- #
# Accrual boundary conditions
# --------------------------------------------------------------------------- #


def test_accrual_zero_on_coupon_date():
    bond = FixedCouponBond(
        maturity_date=date(2034, 11, 15),
        coupon_rate=0.045,
        issue_date=date(2024, 11, 15),
    )
    settle = date(2030, 5, 15)  # exactly on a coupon date
    assert bond.accrual_fraction(settle) == 0.0
    assert bond.accrued_interest(settle) == 0.0

    # First remaining cashflow discounts at exactly one period.
    flows = bond.cashflows(settle)
    assert flows[0].payment_date == date(2030, 11, 15)
    assert flows[0].year_fraction == pytest.approx(0.5)


def test_accrual_one_day_before_coupon_actact():
    bond = FixedCouponBond(
        maturity_date=date(2034, 11, 15),
        coupon_rate=0.045,
        issue_date=date(2024, 11, 15),
    )
    t0, t1 = date(2029, 11, 15), date(2030, 5, 15)
    settle = date(2030, 5, 14)  # one day before the coupon date
    days_in_period = (t1 - t0).days
    expected_w = (settle - t0).days / days_in_period
    assert bond.accrual_fraction(settle) == pytest.approx(expected_w)
    assert bond.accrued_interest(settle) == pytest.approx(
        expected_w * 100.0 * 0.045 / 2
    )

    # year_fraction of the next coupon is a small positive number.
    flows = bond.cashflows(settle)
    assert 0.0 < flows[0].year_fraction < 0.01
    assert flows[0].payment_date == date(2030, 5, 15)


def test_repr():
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.040,
        issue_date=date(2024, 2, 15),
    )
    assert repr(bond) == (
        "FixedCouponBond(maturity=2034-02-15, coupon=4%, freq=2, "
        "day_count=ACT/ACT_ICMA)"
    )


def test_coupon_period_accessor():
    bond = FixedCouponBond(
        maturity_date=date(2029, 2, 28),
        coupon_rate=0.0425,
        issue_date=date(2024, 2, 29),
    )
    assert bond.coupon_period(date(2028, 1, 15)) == (
        date(2027, 8, 31),
        date(2028, 2, 29),
    )
    # First period brackets from the dated date.
    assert bond.coupon_period(date(2024, 5, 1)) == (
        date(2024, 2, 29),
        date(2024, 8, 31),
    )
    # On a coupon date, that date is T_0.
    assert bond.coupon_period(date(2028, 2, 29)) == (
        date(2028, 2, 29),
        date(2028, 8, 31),
    )
    with pytest.raises(ValueError):
        bond.coupon_period(date(2024, 1, 1))


def test_accrual_zero_at_maturity():
    bond = FixedCouponBond(
        maturity_date=date(2029, 2, 28),
        coupon_rate=0.0425,
        issue_date=date(2024, 2, 29),
    )
    assert bond.accrual_fraction(date(2029, 2, 28)) == 0.0
    assert bond.cashflows(date(2029, 2, 28)) == []


def test_thirty_360_accrual():
    bond = FixedCouponBond(
        maturity_date=date(2030, 1, 31),
        coupon_rate=0.06,
        issue_date=date(2025, 1, 31),
        day_count=DayCountConvention.THIRTY_360_US,
        is_eom=True,
    )
    # Period 2029-07-31 -> 2030-01-31; settle 2029-10-31.
    # 30/360: D1=31->30, D2=31->30  =>  90 days of 180.
    settle = date(2029, 10, 31)
    assert bond.accrual_fraction(settle) == pytest.approx(0.5)
    assert bond.accrued_interest(settle) == pytest.approx(0.5 * 100.0 * 0.06 / 2)


# --------------------------------------------------------------------------- #
# Cashflow totals & structure
# --------------------------------------------------------------------------- #


def test_cashflow_totals_match_contract():
    bond = FixedCouponBond(
        maturity_date=date(2029, 2, 28),
        coupon_rate=0.0425,
        face_value=100.0,
        issue_date=date(2024, 2, 29),
    )
    flows = bond.cashflows(date(2024, 2, 29))
    assert len(flows) == 10

    total_coupons = sum(f.coupon for f in flows)
    total_principal = sum(f.principal for f in flows)
    assert total_principal == pytest.approx(100.0)
    assert total_coupons == pytest.approx(10 * (100.0 * 0.0425 / 2))
    assert sum(f.amount for f in flows) == pytest.approx(
        total_coupons + total_principal
    )

    # Only the final flow repays principal.
    assert flows[-1].principal == pytest.approx(100.0)
    assert all(f.principal == 0.0 for f in flows[:-1])
    assert flows[-1].payment_date == date(2029, 2, 28)

    # Discounting times are an arithmetic sequence at 1/frequency spacing.
    yfs = [f.year_fraction for f in flows]
    assert yfs[0] == pytest.approx(0.5)
    assert all(b - a == pytest.approx(0.5) for a, b in zip(yfs, yfs[1:]))


def test_cashflow_is_frozen_dataclass():
    cf = CashFlow(date(2030, 1, 1), 1.0, 0.5, 2.125, 0.0, 2.125)
    with pytest.raises(Exception):
        cf.coupon = 9.9  # type: ignore[misc]


def test_quarterly_frequency_schedule():
    bond = FixedCouponBond(
        maturity_date=date(2027, 3, 15),
        coupon_rate=0.04,
        frequency=4,
        issue_date=date(2025, 3, 15),
    )
    dates = bond.coupon_dates()
    assert len(dates) == 8
    assert {d.month for d in dates} == {3, 6, 9, 12}
    flows = bond.cashflows(date(2025, 3, 15))
    assert flows[0].accrual_factor == pytest.approx(0.25)
    assert flows[0].year_fraction == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(maturity_date=date(2030, 1, 1), coupon_rate=-0.01),
        dict(maturity_date=date(2030, 1, 1), coupon_rate=0.03, frequency=3),
        dict(maturity_date=date(2030, 1, 1), coupon_rate=0.03, face_value=0.0),
        dict(
            maturity_date=date(2030, 1, 1),
            coupon_rate=0.03,
            issue_date=date(2030, 1, 1),
        ),
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        FixedCouponBond(**kwargs)


def test_settlement_out_of_range_rejected():
    bond = FixedCouponBond(
        maturity_date=date(2030, 1, 15),
        coupon_rate=0.03,
        issue_date=date(2025, 1, 15),
    )
    with pytest.raises(ValueError):
        bond.accrual_fraction(date(2030, 6, 1))  # after maturity
    with pytest.raises(ValueError):
        bond.accrual_fraction(date(2024, 1, 1))  # before issue
