"""Tests for the Treasury par-yield curve fetcher/parser (offline by default)."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import requests

from finmodels.curves.data_models import CurveMarketData
from finmodels.market_data import (
    TreasuryFetchError,
    TreasuryParHistory,
    fetch_treasury_curve,
    fetch_treasury_par_curve,
    fetch_treasury_par_history,
    parse_treasury_curve,
    parse_treasury_par_curve,
    parse_treasury_par_history,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "treasury_sample.xml"          # includes BC_1_5MONTH on latest date
LEGACY_FIXTURE = FIXTURES / "treasury_sample_legacy.xml"  # pre-2024, no BC_1_5MONTH
CSV_FIXTURE = FIXTURES / "treasury_rates_sample.csv"


@pytest.fixture
def sample_xml() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture
def legacy_xml() -> bytes:
    return LEGACY_FIXTURE.read_bytes()


def test_aliases_point_to_par_functions():
    assert fetch_treasury_curve is fetch_treasury_par_curve
    assert parse_treasury_curve is parse_treasury_par_curve


def test_parse_selects_latest_date(sample_xml):
    md = parse_treasury_par_curve(sample_xml)
    assert isinstance(md, CurveMarketData)
    assert md.as_of_date == date(2025, 1, 3)
    assert md.curve_name == "US_TREASURY_PAR_YIELD"
    assert md.rate_type == "PAR_YIELD"


def test_parse_includes_1_5m_tenor(sample_xml):
    md = parse_treasury_par_curve(sample_xml)
    # 14 tenors when the 6-week pillar is present.
    assert md.pillars.shape == (14,)
    assert md.pillars[1] == pytest.approx(1.5 / 12)
    # Sits chronologically between 1M and 2M.
    assert md.pillars[0] == pytest.approx(1 / 12)
    assert md.pillars[2] == pytest.approx(2 / 12)
    # 2025-01-03 1.5M = 4.32 percent -> 0.0432 decimal
    assert md.rates[1] == pytest.approx(0.0432)
    assert np.all(np.diff(md.pillars) > 0)


def test_parse_tenor_mapping_and_decimals(sample_xml):
    md = parse_treasury_par_curve(sample_xml)
    assert md.pillars[-1] == 30.0
    assert md.rates[-1] == pytest.approx(0.0482)
    assert np.all(md.rates < 0.10)


def test_parse_legacy_fixture_without_1_5m(legacy_xml):
    """Backward compatibility: payloads predating BC_1_5MONTH still parse."""
    md = parse_treasury_par_curve(legacy_xml)
    assert md.as_of_date == date(2019, 6, 14)
    assert md.pillars.shape == (12,)
    assert not np.any(np.isclose(md.pillars, 1.5 / 12))
    assert np.all(np.diff(md.pillars) > 0)


def test_parse_missing_tenor_is_dropped(sample_xml):
    xml = sample_xml.replace(
        b'<d:BC_4MONTH m:type="Edm.Double">4.35</d:BC_4MONTH>', b""
    )
    md = parse_treasury_par_curve(xml)
    assert md.pillars.shape == (13,)
    assert not np.any(np.isclose(md.pillars, 4 / 12))


def test_parse_bad_xml_raises():
    with pytest.raises(TreasuryFetchError):
        parse_treasury_par_curve(b"<feed><not-closed>")


def test_parse_no_entries_raises():
    empty = (
        b'<feed xmlns="http://www.w3.org/2005/Atom" '
        b'xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata" '
        b'xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"></feed>'
    )
    with pytest.raises(TreasuryFetchError):
        parse_treasury_par_curve(empty)


def test_fetch_uses_mocked_response(sample_xml):
    class FakeResponse:
        content = sample_xml

        def raise_for_status(self):
            return None

    with patch(
        "finmodels.market_data.treasury_par.requests.get", return_value=FakeResponse()
    ) as mock_get:
        md = fetch_treasury_par_curve(2025)

    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["field_tdr_date_value"] == "2025"
    assert md.as_of_date == date(2025, 1, 3)
    assert md.rate_type == "PAR_YIELD"


def test_fetch_network_error_wrapped():
    with patch(
        "finmodels.market_data.treasury_par.requests.get",
        side_effect=requests.ConnectionError("boom"),
    ):
        with pytest.raises(TreasuryFetchError):
            fetch_treasury_par_curve(2025)


@pytest.mark.network
def test_fetch_live_treasury_par_curve():
    md = fetch_treasury_par_curve()
    assert isinstance(md, CurveMarketData)
    assert md.curve_name == "US_TREASURY_PAR_YIELD"
    assert md.rate_type == "PAR_YIELD"
    assert md.pillars.size >= 5
    assert np.all((md.rates > -0.05) & (md.rates < 0.25))


# --------------------------------------------------------------------- CSV history


@pytest.fixture
def sample_csv() -> str:
    return CSV_FIXTURE.read_text()


def test_parse_history_headers_and_order(sample_csv):
    hist = parse_treasury_par_history(sample_csv)
    assert isinstance(hist, TreasuryParHistory)
    assert hist.tenor_labels[:3] == ["1 Mo", "1.5 Month", "2 Mo"]
    assert hist.tenor_labels[-1] == "30 Yr"
    assert len(hist.tenor_labels) == 14
    # Row order preserved: newest first.
    assert [d for d, _ in hist.rows] == [
        date(2026, 9, 4), date(2026, 9, 3), date(2019, 1, 2)
    ]


def test_parse_history_values_are_percentage_points(sample_csv):
    hist = parse_treasury_par_history(sample_csv)
    d, vals = hist.rows[0]
    assert d == date(2026, 9, 4)
    assert vals[0] == 3.79           # 1 Mo, still in percentage points
    assert vals[-1] == 5.24          # 30 Yr
    assert len(vals) == 14


def test_parse_history_blank_cells_become_none(sample_csv):
    hist = parse_treasury_par_history(sample_csv)
    d, vals = hist.rows[-1]
    assert d == date(2019, 1, 2)
    assert vals[1] is None           # 1.5 Month absent pre-2024
    assert vals[4] is None           # 4 Mo blank in this row
    assert vals[0] == 2.40


def test_parse_history_rejects_bad_header():
    with pytest.raises(TreasuryFetchError):
        parse_treasury_par_history("Something,Else\n1,2\n")


def test_parse_history_rejects_empty():
    with pytest.raises(TreasuryFetchError):
        parse_treasury_par_history("")


def test_fetch_history_uses_mocked_response(sample_csv):
    class FakeResponse:
        text = sample_csv

        def raise_for_status(self):
            return None

    with patch(
        "finmodels.market_data.treasury_par.requests.get", return_value=FakeResponse()
    ) as mock_get:
        hist = fetch_treasury_par_history(2026)

    mock_get.assert_called_once()
    assert "2026" in mock_get.call_args[0][0]
    assert hist.rows[0][0] == date(2026, 9, 4)


def test_fetch_history_network_error_wrapped():
    with patch(
        "finmodels.market_data.treasury_par.requests.get",
        side_effect=requests.Timeout("slow"),
    ):
        with pytest.raises(TreasuryFetchError):
            fetch_treasury_par_history(2026)


@pytest.mark.network
def test_fetch_live_treasury_par_history():
    hist = fetch_treasury_par_history()
    assert len(hist.rows) > 20
    assert hist.tenor_labels[0] == "1 Mo"
    latest_date, latest_vals = hist.rows[0]
    assert latest_date > date(2025, 1, 1)
    assert any(v is not None for v in latest_vals)
