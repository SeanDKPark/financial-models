"""Offline tests for bond valuation: yield analytics and curve pricing."""

from datetime import date

import numpy as np
import pytest

from finmodels.curves.base import BaseYieldCurve
from finmodels.curves.bootstrapping import bootstrap_par_curve
from finmodels.curves.data_models import CurveMarketData
from finmodels.instruments.bond import DayCountConvention, FixedCouponBond
from finmodels.pricing.bond import (
    DEFAULT_KRD_PILLARS,
    bond_effective_duration_convexity,
    bond_key_rate_durations,
    bond_yield_risk,
    bond_ytm,
    clean_price_from_ytm,
    dirty_price_from_ytm,
    key_rate_weight,
    price_bond,
)

_BP = 1.0e-4


class FlatContinuousCurve(BaseYieldCurve):
    """P(t) = exp(-z t) for a constant continuously-compounded zero z."""

    def __init__(self, z: float) -> None:
        self.z = float(z)

    def zero_rate(self, t):
        return np.full_like(np.asarray(t, dtype=float), self.z)

    def discount_factor(self, t):
        return np.exp(-self.z * np.asarray(t, dtype=float))


# --------------------------------------------------------------------------- #
# 1. Par invariance on a bootstrapped curve
# --------------------------------------------------------------------------- #


def test_par_bond_prices_to_par_on_bootstrapped_curve():
    md = CurveMarketData(
        as_of_date="2024-02-15",
        curve_name="TEST",
        pillars=np.array([2.0, 5.0, 10.0]),
        rates=np.array([0.040, 0.042, 0.045]),
    )
    curve = bootstrap_par_curve(md, compounding="continuous", frequency=2)

    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.045,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    res = price_bond(bond, curve, settlement=date(2024, 2, 15))

    assert res.accrued_interest == 0.0
    assert res.clean_price == pytest.approx(100.0, abs=1e-6)
    assert res.dirty_price == pytest.approx(100.0, abs=1e-6)
    assert res.ytm == pytest.approx(0.045, abs=1e-6)


# --------------------------------------------------------------------------- #
# 2. YTM solver precision
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("y", [-0.01, 0.00, 0.02, 0.05, 0.12])
def test_ytm_round_trip(y):
    bond = FixedCouponBond(
        maturity_date=date(2029, 2, 15),
        coupon_rate=0.03,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2024, 2, 15)  # on issue -> w = 0
    clean = clean_price_from_ytm(bond, y, settle)
    assert bond_ytm(bond, clean, settle) == pytest.approx(y, abs=1e-9)


def test_ytm_round_trip_mid_period():
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.04,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2027, 6, 20)  # between coupons -> w > 0
    assert bond.accrued_interest(settle) > 0.0
    for y in (-0.005, 0.03, 0.088):
        clean = clean_price_from_ytm(bond, y, settle)
        assert bond_ytm(bond, clean, settle) == pytest.approx(y, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. Analytic vs finite-difference risk
# --------------------------------------------------------------------------- #


def test_dv01_matches_central_difference():
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.04,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2027, 6, 20)
    y = 0.05
    _, _, dv01, _ = bond_yield_risk(bond, y, settle)

    fd = dirty_price_from_ytm(bond, y - 0.5 * _BP, settle) - dirty_price_from_ytm(
        bond, y + 0.5 * _BP, settle
    )
    assert dv01 == pytest.approx(fd, abs=1e-6)


def test_convexity_matches_second_difference():
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.04,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2024, 2, 15)
    y = 0.045
    _, _, _, convexity = bond_yield_risk(bond, y, settle)

    h = 1e-4
    p0 = dirty_price_from_ytm(bond, y, settle)
    p_up = dirty_price_from_ytm(bond, y + h, settle)
    p_dn = dirty_price_from_ytm(bond, y - h, settle)
    fd_convexity = (p_up + p_dn - 2.0 * p0) / (p0 * h * h)
    assert convexity == pytest.approx(fd_convexity, rel=1e-4)


def test_effective_duration_matches_modified_on_flat_curve():
    # The effective-duration bump shifts the semi-annually compounded zero curve,
    # so on a flat curve it recovers the analytic modified duration and yield
    # convexity.
    curve = FlatContinuousCurve(z=0.045)
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.05,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2024, 2, 15)
    res = price_bond(bond, curve, settle)

    eff_dur, eff_cvx = bond_effective_duration_convexity(bond, curve, settle)
    assert eff_dur == pytest.approx(res.modified_duration, abs=1e-4)
    assert eff_cvx == pytest.approx(res.convexity, rel=1e-4)
    assert res.modified_duration == pytest.approx(
        res.macaulay_duration / (1.0 + res.ytm / 2.0), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# 4. Clean / dirty consistency
# --------------------------------------------------------------------------- #


def test_clean_equals_dirty_on_coupon_date():
    curve = FlatContinuousCurve(z=0.04)
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.045,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2028, 8, 15)  # exactly a coupon date
    res = price_bond(bond, curve, settle)
    assert res.accrued_interest == 0.0
    assert res.clean_price == pytest.approx(res.dirty_price, abs=1e-12)


def test_dirty_equals_clean_plus_accrued_between_coupons():
    curve = FlatContinuousCurve(z=0.04)
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.045,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2028, 11, 3)
    res = price_bond(bond, curve, settle)
    assert res.accrued_interest > 0.0
    assert res.clean_price + res.accrued_interest == pytest.approx(
        res.dirty_price, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# 5. Key-rate durations
# --------------------------------------------------------------------------- #


def test_key_rate_weights_partition_unity():
    grid = np.linspace(0.01, 35.0, 2000)
    total = sum(
        key_rate_weight(grid, DEFAULT_KRD_PILLARS, k)
        for k in range(len(DEFAULT_KRD_PILLARS))
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


def test_krd_sum_matches_effective_duration():
    md = CurveMarketData(
        as_of_date="2024-02-15",
        curve_name="TEST",
        pillars=np.array([2.0, 5.0, 10.0, 30.0]),
        rates=np.array([0.040, 0.042, 0.045, 0.047]),
    )
    curve = bootstrap_par_curve(md, compounding="continuous", frequency=2)
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.04,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2027, 6, 20)

    krds = bond_key_rate_durations(bond, curve, settle)
    eff_dur, _ = bond_effective_duration_convexity(bond, curve, settle)
    assert abs(sum(krds.values()) - eff_dur) < 1e-4


def test_krd_risk_concentration_for_10y_bond():
    curve = FlatContinuousCurve(z=0.045)
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.05,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2024, 2, 15)
    krds = bond_key_rate_durations(bond, curve, settle)

    assert max(krds, key=krds.get) == 10.0
    assert krds[20.0] == 0.0
    assert krds[30.0] == 0.0
    assert krds[10.0] > 0.0


def test_krd_custom_pillars():
    curve = FlatContinuousCurve(z=0.04)
    bond = FixedCouponBond(
        maturity_date=date(2031, 2, 15),
        coupon_rate=0.035,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2025, 3, 10)
    krds = bond_key_rate_durations(bond, curve, settle, pillars=(1.0, 5.0, 10.0))

    assert list(krds) == [1.0, 5.0, 10.0]
    eff_dur, _ = bond_effective_duration_convexity(bond, curve, settle)
    assert abs(sum(krds.values()) - eff_dur) < 1e-4


def test_krd_rejects_unsorted_pillars():
    curve = FlatContinuousCurve(z=0.04)
    bond = FixedCouponBond(
        maturity_date=date(2030, 2, 15),
        coupon_rate=0.03,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    with pytest.raises(ValueError):
        bond_key_rate_durations(bond, curve, date(2024, 2, 15), pillars=(5.0, 1.0, 10.0))


def test_bondsupermart_5y_note_snapshot_2026_09_07():
    # Live Bondsupermart snapshot for US91282CKD29 (CUSIP 91282CKD2),
    # 4.25% UST due 2029-02-28, as of settlement 2026-09-07.
    bond = FixedCouponBond(
        maturity_date=date(2029, 2, 28),
        coupon_rate=0.0425,
        frequency=2,
        day_count=DayCountConvention.ACT_ACT_ICMA,
        issue_date=date(2024, 2, 29),
        is_eom=True,
    )
    settle = date(2026, 9, 7)

    t0, t1 = bond.coupon_period(settle)
    assert (t0, t1) == (date(2026, 8, 31), date(2027, 2, 28))
    assert (t1 - t0).days == 181
    assert (settle - t0).days == 7
    assert bond.accrued_interest(settle) == pytest.approx(0.082182, abs=1e-4)

    for clean_price, vendor_ytm in [(99.556, 0.04441), (99.703, 0.04377)]:
        ytm = bond_ytm(bond, clean_price, settle)
        assert ytm == pytest.approx(vendor_ytm, abs=2e-4)

        _, modified, _, _ = bond_yield_risk(bond, ytm, settle)
        assert modified == pytest.approx(2.322, abs=0.02)

        assert clean_price_from_ytm(bond, ytm, settle) == pytest.approx(
            clean_price, abs=1e-9
        )


def test_ytm_fallback_to_brentq_on_extreme_price():
    # A deeply distressed price still solves via the bracketed fallback.
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.04,
        frequency=2,
        issue_date=date(2024, 2, 15),
    )
    settle = date(2024, 2, 15)
    y = bond_ytm(bond, clean_price=25.0, settlement=settle)
    assert dirty_price_from_ytm(bond, y, settle) == pytest.approx(25.0, abs=1e-8)
