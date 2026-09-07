"""Check finmodels bond analytics against external market-quote snapshots.

    .venv/Scripts/python.exe scripts/verify_market_quotes.py
    .venv/Scripts/python.exe scripts/verify_market_quotes.py --cusip 91282CJZ5 --settle 2024-10-10 --clean 98.50
    .venv/Scripts/python.exe scripts/verify_market_quotes.py --cusip 91282CJZ5 --settle 2024-10-10 --ytm 0.0418

Given a clean price (or a yield), it derives the full analytic profile - dirty
price, accrued interest, YTM, Macaulay / modified duration, DV01, convexity - and
prints it for eyeball comparison against a quote source (Bondsupermart, WSJ, the
TreasuryDirect auction sheet, ...). Offline; no curve involved.
"""

from __future__ import annotations

import argparse
from datetime import date

from finmodels.instruments.bond import DayCountConvention, FixedCouponBond
from finmodels.pricing.bond import (
    bond_yield_risk,
    bond_ytm,
    clean_price_from_ytm,
)

# --------------------------------------------------------------------------- #
# Benchmark presets, keyed by CUSIP
# --------------------------------------------------------------------------- #

PRESETS: dict[str, dict] = {
    "91282CJZ5": dict(
        name="US Treasury Note 4.000% due 2034-02-15 (10Y benchmark)",
        maturity_date=date(2034, 2, 15),
        issue_date=date(2024, 2, 15),
        coupon_rate=0.040,
        frequency=2,
        day_count=DayCountConvention.ACT_ACT_ICMA,
        default_settlement=date(2024, 10, 10),
        default_clean=98.50,
    ),
    "91282CKD2": dict(
        name="US Treasury Note 4.250% due 2029-02-28 (5Y benchmark)",
        maturity_date=date(2029, 2, 28),
        issue_date=date(2024, 2, 29),
        coupon_rate=0.0425,
        frequency=2,
        day_count=DayCountConvention.ACT_ACT_ICMA,
        is_eom=True,
        default_settlement=date(2024, 10, 10),
        default_clean=99.75,
    ),
}


def _bond_from_preset(preset: dict) -> FixedCouponBond:
    return FixedCouponBond(
        maturity_date=preset["maturity_date"],
        coupon_rate=preset["coupon_rate"],
        frequency=preset["frequency"],
        day_count=preset["day_count"],
        issue_date=preset["issue_date"],
        is_eom=preset.get("is_eom"),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def inspect_market_quote(
    bond: FixedCouponBond,
    settlement: date,
    clean_price: float | None = None,
    ytm: float | None = None,
    name: str = "",
    cusip: str = "",
) -> None:
    if (clean_price is None) == (ytm is None):
        raise ValueError("provide exactly one of clean_price or ytm")

    if clean_price is not None:
        ytm = bond_ytm(bond, clean_price, settlement)
    else:
        clean_price = clean_price_from_ytm(bond, ytm, settlement)

    accrued = bond.accrued_interest(settlement)
    dirty_price = clean_price + accrued
    macaulay, modified, dv01, convexity = bond_yield_risk(bond, ytm, settlement)

    t0, t1 = bond.coupon_period(settlement)
    days_accrued = (settlement - t0).days
    period_days = (t1 - t0).days
    w = bond.accrual_fraction(settlement)

    label = name or repr(bond)
    print("=" * 72)
    print(label)
    if cusip:
        print(f"CUSIP            : {cusip}")
    print(f"Coupon / Freq    : {bond.coupon_rate * 100:.3f}%  x{bond.frequency}/yr"
          f"   ({bond.day_count.value})")
    print(f"Maturity         : {bond.maturity_date.isoformat()}")
    print("-" * 72)
    print(f"Settlement Date  : {settlement.isoformat()}")
    print(f"Coupon Period    : {t0.isoformat()} -> {t1.isoformat()}")
    print(f"Days Accrued     : {days_accrued} / {period_days}   (w = {w:.8f})")
    print("-" * 72)
    rows = [
        ("Clean Price ($)", f"{clean_price:.6f}"),
        ("Accrued Interest ($)", f"{accrued:.6f}"),
        ("Dirty Price ($)", f"{dirty_price:.6f}"),
        ("YTM (s.a., %)", f"{ytm * 100:.6f}"),
        ("Macaulay Duration (yrs)", f"{macaulay:.6f}"),
        ("Modified Duration (yrs)", f"{modified:.6f}"),
        ("DV01 (per 100 face)", f"{dv01:.6f}"),
        ("Convexity", f"{convexity:.6f}"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, val in rows:
        print(f"  {k:<{width}} : {val:>16}")
    print("=" * 72)
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cusip", choices=sorted(PRESETS), help="benchmark preset to inspect")
    p.add_argument("--settle", type=date.fromisoformat, help="settlement date (YYYY-MM-DD)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--clean", type=float, help="observed clean price")
    group.add_argument("--ytm", type=float, help="observed yield to maturity (decimal)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.cusip is None:
        # No CUSIP: run both presets with their default quotes.
        for cusip, preset in PRESETS.items():
            inspect_market_quote(
                _bond_from_preset(preset),
                settlement=preset["default_settlement"],
                clean_price=preset["default_clean"],
                name=preset["name"],
                cusip=cusip,
            )
        return 0

    preset = PRESETS[args.cusip]
    bond = _bond_from_preset(preset)
    settlement = args.settle or preset["default_settlement"]

    kwargs: dict = {}
    if args.ytm is not None:
        kwargs["ytm"] = args.ytm
    elif args.clean is not None:
        kwargs["clean_price"] = args.clean
    else:
        kwargs["clean_price"] = preset["default_clean"]

    inspect_market_quote(
        bond, settlement=settlement, name=preset["name"], cusip=args.cusip, **kwargs
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
