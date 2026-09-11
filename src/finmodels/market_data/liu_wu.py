"""Fetch the NY Fed / Liu-Wu smoothed Treasury zero-coupon yield curve via FRED.

Liu & Wu (2021), "Reconstructing the Yield Curve," refines the Gurkaynak-Sack-
Wright methodology with a kernel-weighted local regression fit to off-the-run
Treasury quotes. The Federal Reserve Bank of New York publishes the resulting
daily smoothed zero-coupon curve, and the series are mirrored on FRED
(Federal Reserve Economic Data), which gives us a real queryable REST API
instead of a static file.

Unlike the Treasury CMT par curve, these are already **continuously-compounded
zero rates** at each tenor -- no bootstrap step is needed before using them in
a discounting/pricing curve.

Requires a free FRED API key: create an account at https://fred.stlouisfed.org,
then request a key from the account's "API Keys" page. Pass it explicitly or
set it in the ``FRED_API_KEY`` environment variable.

IMPORTANT -- series ID verification
------------------------------------
The default ``_TENOR_SERIES_MAP`` below is a **best-effort guess** (a
``THREEFYn`` naming pattern recalled from NY Fed research references) and has
not been verified against the live FRED catalog. Before relying on it, call
:func:`search_liu_wu_series` to confirm the correct series IDs for the tenors
you need, and override the mapping via the ``tenor_series`` argument if the
defaults are wrong.
"""

from __future__ import annotations

import os
from datetime import date, datetime

import numpy as np
import requests

from finmodels.curves.data_models import CurveMarketData

_FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
_FRED_SEARCH_URL = "https://api.stlouisfed.org/fred/series/search"
_REQUEST_TIMEOUT = 30
_CURVE_NAME = "US_TREASURY_LIU_WU_SMOOTHED_ZERO"
_RATE_TYPE = "ZERO"
_API_KEY_ENV_VAR = "FRED_API_KEY"

# Best-guess default mapping (tenor label, year fraction) -> FRED series ID.
# UNVERIFIED -- confirm with search_liu_wu_series() before trusting in production.
_TENOR_SERIES_MAP: dict[str, tuple[str, float]] = {
    "1Y": ("THREEFY1", 1.0),
    "2Y": ("THREEFY2", 2.0),
    "3Y": ("THREEFY3", 3.0),
    "5Y": ("THREEFY5", 5.0),
    "7Y": ("THREEFY7", 7.0),
    "10Y": ("THREEFY10", 10.0),
    "20Y": ("THREEFY20", 20.0),
    "30Y": ("THREEFY30", 30.0),
}


class FredFetchError(RuntimeError):
    """Raised when FRED data cannot be fetched or parsed."""


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get(_API_KEY_ENV_VAR)
    if not key:
        raise FredFetchError(
            f"no FRED API key provided; pass api_key= or set the "
            f"{_API_KEY_ENV_VAR} environment variable "
            f"(get a free key at https://fred.stlouisfed.org)"
        )
    return key


def search_liu_wu_series(
    keyword: str = "smoothed treasury yield",
    *,
    api_key: str | None = None,
) -> list[dict]:
    """Search FRED for candidate series matching ``keyword``.

    Use this to verify/discover the correct series IDs for the Liu-Wu (or
    Gurkaynak-Sack-Wright) smoothed Treasury yield curve before trusting
    :data:`_TENOR_SERIES_MAP`'s defaults. Returns the raw list of series
    metadata dicts from FRED's ``series/search`` endpoint (id, title,
    observation_start, observation_end, frequency, ...).
    """
    key = _resolve_api_key(api_key)
    params = {
        "search_text": keyword,
        "api_key": key,
        "file_type": "json",
    }
    try:
        response = requests.get(_FRED_SEARCH_URL, params=params, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FredFetchError(f"FRED series search failed for {keyword!r}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise FredFetchError(f"could not parse FRED search response as JSON: {exc}") from exc

    return payload.get("seriess", [])


def fetch_liu_wu_curve(
    as_of_date: date | str | None = None,
    *,
    api_key: str | None = None,
    tenor_series: dict[str, tuple[str, float]] | None = None,
) -> CurveMarketData:
    """Fetch the smoothed zero-coupon Treasury curve for a single date.

    Parameters
    ----------
    as_of_date:
        Date to fetch. Defaults to the most recent available observation.
        Accepts a ``datetime.date`` or an ISO-8601 string.
    api_key:
        FRED API key. Falls back to the ``FRED_API_KEY`` environment
        variable if omitted.
    tenor_series:
        Override for the tenor -> (series_id, year_fraction) mapping.
        Defaults to :data:`_TENOR_SERIES_MAP`, which is unverified -- see
        the module docstring.

    Raises
    ------
    FredFetchError:
        On a missing API key, network failure, non-200 response, or
        unparseable/empty payload for every requested series.
    """
    key = _resolve_api_key(api_key)
    mapping = tenor_series if tenor_series is not None else _TENOR_SERIES_MAP

    if as_of_date is None:
        date_str = None
    elif isinstance(as_of_date, str):
        date_str = as_of_date
    else:
        date_str = as_of_date.isoformat()

    tenors: list[float] = []
    rates: list[float] = []
    resolved_date: date | None = None

    for _label, (series_id, year_fraction) in mapping.items():
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        if date_str is not None:
            params["observation_end"] = date_str

        try:
            response = requests.get(
                _FRED_OBSERVATIONS_URL, params=params, timeout=_REQUEST_TIMEOUT
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FredFetchError(
                f"failed to fetch FRED series {series_id!r}: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise FredFetchError(
                f"could not parse FRED response for {series_id!r} as JSON: {exc}"
            ) from exc

        observations = payload.get("observations", [])
        if not observations:
            raise FredFetchError(f"no observations returned for series {series_id!r}")

        obs = observations[0]
        raw_value = obs.get("value")
        if raw_value is None or raw_value == ".":
            raise FredFetchError(
                f"missing value for series {series_id!r} on {obs.get('date')!r}"
            )
        try:
            pct = float(raw_value)
        except ValueError as exc:
            raise FredFetchError(
                f"non-numeric value for series {series_id!r}: {raw_value!r}"
            ) from exc

        obs_date = datetime.strptime(obs["date"], "%Y-%m-%d").date()
        if resolved_date is None or obs_date > resolved_date:
            resolved_date = obs_date

        tenors.append(year_fraction)
        rates.append(pct / 100.0)  # FRED reports these as percent, not decimal

    if resolved_date is None:
        raise FredFetchError("no observations resolved for any requested series")

    order = np.argsort(tenors)
    tenors_arr = np.asarray(tenors, dtype=float)[order]
    rates_arr = np.asarray(rates, dtype=float)[order]

    return CurveMarketData(
        as_of_date=resolved_date,
        curve_name=_CURVE_NAME,
        pillars=tenors_arr,
        rates=rates_arr,
        rate_type=_RATE_TYPE,
    )
