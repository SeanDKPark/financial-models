"""Fixed-coupon bond valuation: yield/price analytics and curve pricing.

Two entry points:

* Pure yield math - :func:`dirty_price_from_ytm`, :func:`clean_price_from_ytm`,
  :func:`bond_ytm`, :func:`bond_yield_risk` - all on the semi-annual
  (``m = frequency``) compounding convention, using the bond's own
  settlement-relative ``year_fraction`` times.
* Curve pricing - :func:`price_bond` discounts the cashflows on a
  ``BaseYieldCurve`` and back-solves the implied yield, and
  :func:`bond_effective_duration_convexity` measures a symmetric parallel
  zero-rate bump.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from finmodels.curves.base import BaseYieldCurve
from finmodels.instruments.bond import FixedCouponBond

_BP = 1.0e-4

DEFAULT_KRD_PILLARS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


@dataclass(frozen=True)
class BondPriceResult:
    """Valuation of a bond on a curve, with the implied-yield risk profile."""

    clean_price: float
    dirty_price: float
    accrued_interest: float
    ytm: float  # annual, semi-annual compounding (decimal)
    macaulay_duration: float  # years
    modified_duration: float  # years
    dv01: float  # price change per 1bp yield decrease, per face_value
    convexity: float  # d2P/dy2 / P


def _flow_arrays(bond: FixedCouponBond, settlement: date) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, amounts)`` for the remaining cashflows."""
    flows = bond.cashflows(settlement)
    if not flows:
        raise ValueError("bond has no remaining cashflows at this settlement date")
    times = np.array([f.year_fraction for f in flows], dtype=float)
    amounts = np.array([f.amount for f in flows], dtype=float)
    return times, amounts


# --------------------------------------------------------------------------- #
# Pure yield analytics
# --------------------------------------------------------------------------- #


def dirty_price_from_ytm(bond: FixedCouponBond, ytm: float, settlement: date) -> float:
    """Present value of all remaining cashflows at ``ytm`` (semi-annual)."""
    m = bond.frequency
    times, amounts = _flow_arrays(bond, settlement)
    return float(np.sum(amounts * (1.0 + ytm / m) ** (-m * times)))


def clean_price_from_ytm(bond: FixedCouponBond, ytm: float, settlement: date) -> float:
    """Dirty price at ``ytm`` less accrued interest."""
    return dirty_price_from_ytm(bond, ytm, settlement) - bond.accrued_interest(settlement)


def _dirty_and_derivative(
    m: int, times: np.ndarray, amounts: np.ndarray, ytm: float
) -> tuple[float, float]:
    """``(dirty_price, d(dirty)/dy)`` at ``ytm``."""
    v = 1.0 / (1.0 + ytm / m)
    tau = m * times
    disc = v**tau
    dirty = float(np.sum(amounts * disc))
    deriv = float(-(v / m) * np.sum(tau * amounts * disc))
    return dirty, deriv


def bond_ytm(
    bond: FixedCouponBond,
    clean_price: float,
    settlement: date,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Solve for the semi-annual yield that reproduces ``clean_price``.

    Newton-Raphson with an analytic first derivative; falls back to
    ``scipy.optimize.brentq`` over ``[-0.10, 1.00]`` if the iteration leaves the
    economic range or fails to converge.
    """
    m = bond.frequency
    accrued = bond.accrued_interest(settlement)
    target_dirty = clean_price + accrued
    times, amounts = _flow_arrays(bond, settlement)
    face = bond.face_value
    t_k = float(times[-1])

    y = (
        bond.coupon_rate * face + (face - clean_price) / max(t_k, 0.5)
    ) / ((face + clean_price) / 2.0)

    for _ in range(max_iter):
        dirty, deriv = _dirty_and_derivative(m, times, amounts, y)
        if deriv == 0.0 or not np.isfinite(deriv):
            break
        y_new = y - (dirty - target_dirty) / deriv
        if y_new <= -m or abs(y_new) > 1.0:
            break
        if abs(y_new - y) < tol:
            return float(y_new)
        y = y_new
    else:
        # exhausted max_iter without converging
        pass

    def objective(rate: float) -> float:
        return dirty_price_from_ytm(bond, rate, settlement) - target_dirty

    return float(brentq(objective, -0.10, 1.00, xtol=tol, maxiter=200))


def bond_yield_risk(
    bond: FixedCouponBond, ytm: float, settlement: date
) -> tuple[float, float, float, float]:
    """``(macaulay_duration, modified_duration, dv01, convexity)`` at ``ytm``.

    Durations in years; ``dv01`` is the price gain per 1bp yield *decrease*, in
    the same currency units as the price (i.e. per ``face_value``).
    """
    m = bond.frequency
    times, amounts = _flow_arrays(bond, settlement)
    v = 1.0 / (1.0 + ytm / m)
    tau = m * times
    pv = amounts * v**tau
    dirty = float(np.sum(pv))

    macaulay = float(np.sum(times * pv) / dirty)
    modified = macaulay * v
    dv01 = modified * dirty * _BP
    convexity = float(np.sum(tau * (tau + 1.0) * amounts * v ** (tau + 2.0)) / (dirty * m * m))
    return macaulay, modified, dv01, convexity


# --------------------------------------------------------------------------- #
# Curve pricing
# --------------------------------------------------------------------------- #


def price_bond(
    bond: FixedCouponBond, curve: BaseYieldCurve, settlement: date
) -> BondPriceResult:
    """Discount the bond on ``curve`` and back out the implied yield and risk."""
    times, amounts = _flow_arrays(bond, settlement)
    dfs = np.asarray(curve.discount_factor(times), dtype=float)
    dirty = float(np.sum(amounts * dfs))
    accrued = bond.accrued_interest(settlement)
    clean = dirty - accrued

    ytm = bond_ytm(bond, clean, settlement)
    macaulay, modified, dv01, convexity = bond_yield_risk(bond, ytm, settlement)

    return BondPriceResult(
        clean_price=clean,
        dirty_price=dirty,
        accrued_interest=accrued,
        ytm=ytm,
        macaulay_duration=macaulay,
        modified_duration=modified,
        dv01=dv01,
        convexity=convexity,
    )


def bond_effective_duration_convexity(
    bond: FixedCouponBond,
    curve: BaseYieldCurve,
    settlement: date,
    bump_bps: float = 1.0,
) -> tuple[float, float]:
    """Effective duration and convexity from a symmetric parallel yield bump.

    The bump ``h = bump_bps * 1e-4`` is applied to the *semi-annually compounded*
    zero curve. Each cashflow's zero rate ``z2(t) = 2 (P(t)**(-1/(2t)) - 1)`` is
    shifted by ``+/- h`` and the cashflows repriced; on a flat curve this
    recovers the analytic modified duration and yield convexity.
    """
    m = bond.frequency
    h = bump_bps * _BP
    times, amounts = _flow_arrays(bond, settlement)
    dfs = np.asarray(curve.discount_factor(times), dtype=float)

    z2 = m * (dfs ** (-1.0 / (m * times)) - 1.0)

    def reprice(shift: float) -> float:
        bumped = (1.0 + (z2 + shift) / m) ** (-m * times)
        return float(np.sum(amounts * bumped))

    p0 = float(np.sum(amounts * dfs))
    p_up = reprice(h)
    p_down = reprice(-h)

    eff_duration = (p_down - p_up) / (2.0 * p0 * h)
    eff_convexity = (p_down + p_up - 2.0 * p0) / (p0 * h * h)
    return eff_duration, eff_convexity


def key_rate_weight(t, pillars: Sequence[float], k: int) -> np.ndarray:
    """Triangular ("tent") perturbation ``w_k(t)`` for pillar ``k``.

    Boundary pillars extrapolate flat (``w_0 = 1`` below ``T_0``,
    ``w_{P-1} = 1`` above ``T_{P-1}``). For any ``t >= 0`` the weights across all
    pillars sum to exactly 1.
    """
    t = np.asarray(t, dtype=float)
    tt = np.atleast_1d(t)
    ladder = np.asarray(pillars, dtype=float)
    n = ladder.size
    w = np.zeros_like(tt)
    tk = ladder[k]

    if k == 0:
        w[tt <= tk] = 1.0
        seg = (tt > tk) & (tt <= ladder[1])
        w[seg] = (ladder[1] - tt[seg]) / (ladder[1] - tk)
    elif k == n - 1:
        w[tt > tk] = 1.0
        seg = (tt > ladder[k - 1]) & (tt <= tk)
        w[seg] = (tt[seg] - ladder[k - 1]) / (tk - ladder[k - 1])
    else:
        up = (tt > ladder[k - 1]) & (tt <= tk)
        w[up] = (tt[up] - ladder[k - 1]) / (tk - ladder[k - 1])
        down = (tt > tk) & (tt <= ladder[k + 1])
        w[down] = (ladder[k + 1] - tt[down]) / (ladder[k + 1] - tk)

    return w.reshape(t.shape) if t.ndim else w


def bond_key_rate_durations(
    bond: FixedCouponBond,
    curve: BaseYieldCurve,
    settlement: date,
    pillars: Sequence[float] = DEFAULT_KRD_PILLARS,
    bump_bps: float = 1.0,
) -> dict[float, float]:
    """Compute key-rate durations across market tenors via semi-annual zero shifts.

    Each pillar's zero rate is bumped by ``+/- h * w_k(t)`` (tent-weighted, same
    ``z2`` shift as :func:`bond_effective_duration_convexity`) and the cashflows
    repriced. Because the tent weights partition unity, the KRDs sum to the
    effective (parallel-shift) duration.
    """
    ladder = np.asarray(pillars, dtype=float)
    if ladder.ndim != 1 or ladder.size < 2:
        raise ValueError("pillars must be a 1-D sequence of at least two tenors")
    if np.any(np.diff(ladder) <= 0.0):
        raise ValueError("pillars must be strictly ascending")

    m = bond.frequency
    h = bump_bps * _BP
    times, amounts = _flow_arrays(bond, settlement)
    dfs = np.asarray(curve.discount_factor(times), dtype=float)
    z2 = m * (dfs ** (-1.0 / (m * times)) - 1.0)
    p0 = float(np.sum(amounts * dfs))

    krds: dict[float, float] = {}
    for k in range(ladder.size):
        dz = h * key_rate_weight(times, ladder, k)
        p_up = float(np.sum(amounts * (1.0 + (z2 + dz) / m) ** (-m * times)))
        p_down = float(np.sum(amounts * (1.0 + (z2 - dz) / m) ** (-m * times)))
        krds[float(ladder[k])] = (p_down - p_up) / (2.0 * p0 * h)
    return krds
