"""Pretty-print the full structure of a FixedCouponBond to stdout.

    .venv/Scripts/python.exe scripts/inspect_bond.py

Run directly, it inspects the benchmark 10Y note (CUSIP 91282CJZ5) viewed from
settlement 2024-10-10: security terms, current-period accrual, and the full
remaining cashflow schedule as an ASCII table. Offline; nothing curve-related.
"""

from __future__ import annotations

from datetime import date

from finmodels.instruments import DayCountConvention, FixedCouponBond


def _rule(widths: list[int]) -> str:
    return "+".join("-" * (w + 2) for w in widths)


def _row(cells: list[str], widths: list[int], aligns: list[str]) -> str:
    out = []
    for cell, w, a in zip(cells, widths, aligns):
        out.append(f" {cell:>{w}} " if a == "r" else f" {cell:<{w}} ")
    return "|".join(out)


def inspect_bond(bond: FixedCouponBond, settlement: date) -> None:
    coupon_amt = bond.face_value * bond.coupon_rate / bond.frequency
    t0, t1 = bond.coupon_period(settlement)
    days_elapsed = (settlement - t0).days
    days_in_period = (t1 - t0).days
    w = bond.accrual_fraction(settlement)
    ai = bond.accrued_interest(settlement)

    print(repr(bond))
    print("=" * 72)
    print("SECURITY TERMS")
    print(f"  Maturity        : {bond.maturity_date.isoformat()}")
    print(f"  Issue Date      : {bond.issue_date.isoformat() if bond.issue_date else '-'}")
    print(f"  Coupon Rate     : {bond.coupon_rate * 100:.3f}%  (${coupon_amt:.4f} / period)")
    print(f"  Frequency       : {bond.frequency} / year")
    print(f"  Day Count       : {bond.day_count.value}")
    print(f"  is_eom          : {bond.is_eom}")
    print(f"  Face Value      : {bond.face_value:.2f}")
    print()
    print("SETTLEMENT")
    print(f"  Settlement Date : {settlement.isoformat()}")
    print(f"  Coupon Period   : {t0.isoformat()} -> {t1.isoformat()}")
    print(f"  Days Elapsed    : {days_elapsed} / {days_in_period}")
    print(f"  Accrual Frac (w): {w:.10f}")
    print(f"  Accrued Interest: ${ai:.6f}")
    print()

    flows = bond.cashflows(settlement)
    headers = [
        "#",
        "Payment Date",
        "Tenor (yrs)",
        "Accr. Factor",
        "Coupon ($)",
        "Principal ($)",
        "Total ($)",
    ]
    aligns = ["r", "l", "r", "r", "r", "r", "r"]
    rows = [
        [
            str(i),
            cf.payment_date.isoformat(),
            f"{cf.year_fraction:.6f}",
            f"{cf.accrual_factor:.6f}",
            f"{cf.coupon:.4f}",
            f"{cf.principal:.4f}",
            f"{cf.amount:.4f}",
        ]
        for i, cf in enumerate(flows, start=1)
    ]
    widths = [
        max(len(headers[c]), *(len(r[c]) for r in rows)) if rows else len(headers[c])
        for c in range(len(headers))
    ]

    print(f"REMAINING CASHFLOWS  ({len(flows)})")
    print(_row(headers, widths, aligns))
    print(_rule(widths))
    for r in rows:
        print(_row(r, widths, aligns))
    print(_rule(widths))

    tot_cpn = sum(cf.coupon for cf in flows)
    tot_prin = sum(cf.principal for cf in flows)
    print(
        _row(
            ["", "TOTAL", "", "", f"{tot_cpn:.4f}", f"{tot_prin:.4f}",
             f"{tot_cpn + tot_prin:.4f}"],
            widths,
            aligns,
        )
    )


def main() -> None:
    bond = FixedCouponBond(
        maturity_date=date(2034, 2, 15),
        coupon_rate=0.040,
        frequency=2,
        day_count=DayCountConvention.ACT_ACT_ICMA,
        issue_date=date(2024, 2, 15),
    )
    inspect_bond(bond, settlement=date(2024, 10, 10))


if __name__ == "__main__":
    main()
