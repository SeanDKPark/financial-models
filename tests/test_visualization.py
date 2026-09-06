"""Offline tests for the term-structure plot (headless Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from finmodels.curves.bootstrapping import bootstrap_par_curve
from finmodels.curves.data_models import CurveMarketData
from finmodels.visualization import plot_term_structure

PILLARS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
PAR = np.array([0.043, 0.044, 0.045, 0.046, 0.047, 0.049, 0.050, 0.051, 0.053, 0.052])


@pytest.fixture
def par_data():
    return CurveMarketData(
        as_of_date="2026-09-04",
        curve_name="US_TREASURY_PAR_YIELD",
        pillars=PILLARS,
        rates=PAR,
    )


@pytest.fixture
def zero_curve(par_data):
    return bootstrap_par_curve(par_data, compounding="semi-annual")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_returns_figure(par_data, zero_curve):
    fig = plot_term_structure(par_data, zero_curve)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2  # rates panel + discount-factor panel


def test_rate_panel_has_all_three_series(par_data, zero_curve):
    fig = plot_term_structure(par_data, zero_curve)
    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert "Zero (spot)" in labels
    assert "Par (market)" in labels
    assert any("forward" in lbl for lbl in labels)


def test_par_markers_match_input(par_data, zero_curve):
    fig = plot_term_structure(par_data, zero_curve)
    (par_line,) = [
        ln for ln in fig.axes[0].get_lines() if ln.get_label() == "Par (market)"
    ]
    np.testing.assert_allclose(par_line.get_xdata(), PILLARS)
    np.testing.assert_allclose(par_line.get_ydata(), PAR * 100.0)


def test_save_path_writes_file(par_data, zero_curve, tmp_path):
    out = tmp_path / "nested" / "term_structure.png"
    plot_term_structure(par_data, zero_curve, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_custom_forward_tenor_label(par_data, zero_curve):
    fig = plot_term_structure(par_data, zero_curve, forward_tenor=1.0, forward_compounding="continuous")
    labels = {ln.get_label() for ln in fig.axes[0].get_lines()}
    assert "1y forward (continuous)" in labels


def test_rejects_non_positive_forward_tenor(par_data, zero_curve):
    with pytest.raises(ValueError):
        plot_term_structure(par_data, zero_curve, forward_tenor=0.0)


def test_handles_forward_window_wider_than_curve():
    pillars = np.array([0.25, 0.5])
    par = np.array([0.043, 0.044])
    md = CurveMarketData("2026-09-04", "SHORT", pillars, par)
    zc = bootstrap_par_curve(md, compounding="semi-annual")
    fig = plot_term_structure(md, zc, forward_tenor=2.0)  # window > curve span
    labels = {ln.get_label() for ln in fig.axes[0].get_lines()}
    assert not any("forward" in lbl for lbl in labels)  # forward line omitted
    assert "Zero (spot)" in labels
