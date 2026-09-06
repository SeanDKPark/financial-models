"""Market data ingestion adapters."""

from finmodels.market_data.treasury_par import (
    TreasuryFetchError,
    TreasuryParHistory,
    fetch_treasury_par_curve,
    fetch_treasury_par_history,
    parse_treasury_par_curve,
    parse_treasury_par_history,
)

# Unqualified aliases: the US Treasury feed is always CMT par yields.
fetch_treasury_curve = fetch_treasury_par_curve
parse_treasury_curve = parse_treasury_par_curve

__all__ = [
    "TreasuryFetchError",
    "TreasuryParHistory",
    "fetch_treasury_par_curve",
    "parse_treasury_par_curve",
    "fetch_treasury_par_history",
    "parse_treasury_par_history",
    "fetch_treasury_curve",
    "parse_treasury_curve",
]
