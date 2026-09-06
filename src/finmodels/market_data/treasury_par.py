"""Fetch the daily US Treasury CMT **par yield** curve from home.treasury.gov.

The US Treasury publishes Constant Maturity Treasury (CMT) *par yields*, not
zero-coupon rates:

* Tenors <= 1Y are quoted as **T-Bill coupon-equivalent (investment) yields**.
* Tenors >= 2Y are **semi-annual coupon** CMT par yields (the coupon that prices
  a notional Treasury of that maturity at par).

A bootstrapping step (par -> zero) is therefore required before these rates may
be treated as continuously-compounded zero rates for discounting.

Data source (human-readable tables and methodology):
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve
Par yield methodology:
    https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology
"""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import requests

from finmodels.curves.data_models import CurveMarketData

# Machine-readable XML feed. Browse the same data as HTML at:
#   https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=<YEAR>
_FEED_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml"
)
_FEED_PARAMS_DATASET = "daily_treasury_yield_curve"

# Atom / OData namespaces used by the Treasury feed.
_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
}

# Feed field name -> (tenor label, year fraction). Order defines the output curve.
# BC_1_5MONTH (the 6-week CMT) was added by Treasury in 2024 and is absent from
# older payloads; the parser treats every tenor as optional.
_TENOR_MAP: list[tuple[str, str, float]] = [
    ("BC_1MONTH", "1M", 1 / 12),
    ("BC_1_5MONTH", "1.5M", 1.5 / 12),
    ("BC_2MONTH", "2M", 2 / 12),
    ("BC_3MONTH", "3M", 3 / 12),
    ("BC_4MONTH", "4M", 4 / 12),
    ("BC_6MONTH", "6M", 6 / 12),
    ("BC_1YEAR", "1Y", 1.0),
    ("BC_2YEAR", "2Y", 2.0),
    ("BC_3YEAR", "3Y", 3.0),
    ("BC_5YEAR", "5Y", 5.0),
    ("BC_7YEAR", "7Y", 7.0),
    ("BC_10YEAR", "10Y", 10.0),
    ("BC_20YEAR", "20Y", 20.0),
    ("BC_30YEAR", "30Y", 30.0),
]

# Direct CSV download of a full year's daily par yield history. This is the same
# data the Excel workbook's Power Query connection consumes, so its column
# headers ("1 Mo", "1.5 Month", "2 Mo", ... "30 Yr") are taken verbatim.
_CSV_URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)

_REQUEST_TIMEOUT = 30
_CURVE_NAME = "US_TREASURY_PAR_YIELD"
_RATE_TYPE = "PAR_YIELD"


class TreasuryFetchError(RuntimeError):
    """Raised when the Treasury par curve cannot be fetched or parsed."""


def fetch_treasury_par_curve(year: int | None = None) -> CurveMarketData:
    """Return the most recent business day's Treasury CMT par yield curve.

    The result is a :class:`CurveMarketData` with ``rate_type="PAR_YIELD"`` and
    ``curve_name="US_TREASURY_PAR_YIELD"``. These are par yields (bills quoted as
    investment yields, notes/bonds as semi-annual coupon yields) and must be
    bootstrapped to zero rates before use in discounting.

    Parameters
    ----------
    year:
        Calendar year of the feed to query. Defaults to the current year.

    Raises
    ------
    TreasuryFetchError:
        On network failure, timeout, non-200 response, or unparseable payload.
    """
    if year is None:
        year = date.today().year

    params = {
        "data": _FEED_PARAMS_DATASET,
        "field_tdr_date_value": str(year),
    }
    try:
        response = requests.get(_FEED_URL, params=params, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:  # network, timeout, HTTP error
        raise TreasuryFetchError(
            f"failed to fetch Treasury par curve for {year}: {exc}"
        ) from exc

    return parse_treasury_par_curve(response.content)


def parse_treasury_par_curve(payload: bytes | str) -> CurveMarketData:
    """Parse a Treasury yield-curve XML payload into ``CurveMarketData``.

    Selects the latest ``NEW_DATE`` entry and extracts the standard CMT tenors
    (1M, 1.5M, 2M, 3M, 4M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y). Every tenor
    is optional: tenors missing on the selected date - including ``BC_1_5MONTH``
    in payloads predating its 2024 introduction - are simply dropped.

    The returned rates are **par yields** as decimals and require a par->zero
    bootstrap before use as zero rates.
    """
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise TreasuryFetchError(f"could not parse Treasury XML: {exc}") from exc

    latest_props: ET.Element | None = None
    latest_dt: datetime | None = None

    for props in root.iterfind(".//m:properties", _NS):
        date_el = props.find("d:NEW_DATE", _NS)
        if date_el is None or not (date_el.text or "").strip():
            continue
        raw = date_el.text.strip()
        try:
            entry_dt = datetime.fromisoformat(raw.replace("Z", ""))
        except ValueError:
            continue
        if latest_dt is None or entry_dt > latest_dt:
            latest_dt, latest_props = entry_dt, props

    if latest_props is None or latest_dt is None:
        raise TreasuryFetchError("no dated entries found in Treasury payload")

    tenors: list[float] = []
    rates: list[float] = []
    for field_name, _label, year_fraction in _TENOR_MAP:
        el = latest_props.find(f"d:{field_name}", _NS)
        if el is None or not (el.text or "").strip():
            continue
        try:
            pct = float(el.text.strip())
        except ValueError as exc:
            raise TreasuryFetchError(
                f"non-numeric rate for {field_name}: {el.text!r}"
            ) from exc
        tenors.append(year_fraction)
        rates.append(pct / 100.0)  # percentage points -> decimal

    if not tenors:
        raise TreasuryFetchError(
            f"no tenor rates present for {latest_dt.date().isoformat()}"
        )

    return CurveMarketData(
        as_of_date=latest_dt.date(),
        curve_name=_CURVE_NAME,
        pillars=np.asarray(tenors, dtype=float),
        rates=np.asarray(rates, dtype=float),
        rate_type=_RATE_TYPE,
    )


@dataclass
class TreasuryParHistory:
    """A full year of daily CMT par yields, newest date first.

    Unlike :class:`CurveMarketData`, rates here are kept in **percentage points**
    exactly as Treasury publishes them (``3.79`` means 3.79%), missing tenors are
    ``None``, and the tenor labels are the raw CSV headers (``"1 Mo"``,
    ``"1.5 Month"``, ... ``"30 Yr"``). This mirrors the CSV feed the Excel
    workbook's Power Query connection loads.
    """

    tenor_labels: list[str]
    rows: list[tuple[date, list[float | None]]]


def fetch_treasury_par_history(year: int | None = None) -> TreasuryParHistory:
    """Download one calendar year of daily Treasury par yields as CSV.

    Parameters
    ----------
    year:
        Calendar year to download. Defaults to the current year.

    Raises
    ------
    TreasuryFetchError:
        On network failure, timeout, non-200 response, or unparseable CSV.
    """
    if year is None:
        year = date.today().year

    url = _CSV_URL_TEMPLATE.format(year=year)
    try:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TreasuryFetchError(
            f"failed to fetch Treasury par history for {year}: {exc}"
        ) from exc

    return parse_treasury_par_history(response.text)


def parse_treasury_par_history(csv_text: str) -> TreasuryParHistory:
    """Parse the Treasury daily-rates CSV into a :class:`TreasuryParHistory`.

    The first column is the observation date (``MM/DD/YYYY``); the remaining
    columns are par yields in percentage points. Blank cells become ``None``.
    Row order is preserved (Treasury serves newest first).
    """
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration:
        raise TreasuryFetchError("empty Treasury CSV payload") from None

    if not header or header[0].strip().lower() != "date":
        raise TreasuryFetchError(f"unexpected Treasury CSV header: {header!r}")

    tenor_labels = [h.strip() for h in header[1:]]
    n = len(tenor_labels)
    rows: list[tuple[date, list[float | None]]] = []

    for lineno, raw in enumerate(reader, start=2):
        if not raw or not raw[0].strip():
            continue
        try:
            obs = datetime.strptime(raw[0].strip(), "%m/%d/%Y").date()
        except ValueError as exc:
            raise TreasuryFetchError(
                f"bad date on Treasury CSV line {lineno}: {raw[0]!r}"
            ) from exc

        values: list[float | None] = []
        for cell in raw[1 : 1 + n]:
            cell = cell.strip()
            if not cell:
                values.append(None)
                continue
            try:
                values.append(float(cell))
            except ValueError as exc:
                raise TreasuryFetchError(
                    f"non-numeric rate on Treasury CSV line {lineno}: {cell!r}"
                ) from exc
        values.extend([None] * (n - len(values)))
        rows.append((obs, values))

    if not rows:
        raise TreasuryFetchError("no data rows in Treasury CSV")

    return TreasuryParHistory(tenor_labels=tenor_labels, rows=rows)


# Backward-compatible aliases. The Treasury feed is always par yields, so the
# unqualified names are retained as thin aliases of the explicit ones.
fetch_treasury_curve = fetch_treasury_par_curve
parse_treasury_curve = parse_treasury_par_curve
