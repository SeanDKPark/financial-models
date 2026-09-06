"""Term-structure visualization: par yields, bootstrapped zeros, rolling forwards."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from finmodels.curves.base import BaseYieldCurve
from finmodels.curves.data_models import CurveMarketData

_GRID_POINTS = 400

# Institutional-neutral palette (colour-blind safe, prints legibly in greyscale).
_PAR_COLOR = "#1f2a44"
_ZERO_COLOR = "#2f6f9f"
_FWD_COLOR = "#c2570c"


def plot_term_structure(
    par_data: CurveMarketData,
    zero_curve: BaseYieldCurve,
    forward_tenor: float = 0.5,
    forward_compounding: str = "semi-annual",
    save_path: str | Path | None = None,
    show: bool = False,
) -> matplotlib.figure.Figure:
    """Plot the par, zero and forward term structures on a shared time axis.

    Parameters
    ----------
    par_data:
        Market par yield quotes; plotted as discrete markers.
    zero_curve:
        Bootstrapped zero curve; the spot line and the rolling-forward line are
        both evaluated from it.
    forward_tenor:
        Length in years of the rolling forward window (default 0.5 = 6M).
    forward_compounding:
        Compounding convention passed to
        :meth:`BaseYieldCurve.forward_rate_rolling`.
    save_path:
        If given, the figure is written here (format inferred from the suffix).
    show:
        If ``True``, call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (two stacked panels: rates, then discount factors). The
        caller owns it and is responsible for closing it.
    """
    if forward_tenor <= 0.0:
        raise ValueError(f"forward_tenor must be positive, got {forward_tenor}")

    pillars = np.asarray(par_data.pillars, dtype=float)
    par_rates = np.asarray(par_data.rates, dtype=float)
    t_min = float(pillars[0])
    t_max = float(pillars[-1])

    grid = np.linspace(t_min, t_max, _GRID_POINTS)
    zeros = np.asarray(zero_curve.zero_rate(grid), dtype=float)
    dfs = np.asarray(zero_curve.discount_factor(grid), dtype=float)

    fwd_end = t_max - forward_tenor
    if fwd_end > t_min:
        fwd_grid = np.linspace(t_min, fwd_end, _GRID_POINTS)
        forwards = np.asarray(
            zero_curve.forward_rate_rolling(
                fwd_grid, tenor=forward_tenor, compounding=forward_compounding
            ),
            dtype=float,
        )
    else:  # window wider than the curve; nothing meaningful to draw
        fwd_grid = np.array([])
        forwards = np.array([])

    fig, (ax_rate, ax_df) = plt.subplots(
        2, 1, figsize=(9, 7), height_ratios=(3, 1), sharex=True
    )

    ax_rate.plot(grid, zeros * 100.0, color=_ZERO_COLOR, lw=1.8, label="Zero (spot)")
    if fwd_grid.size:
        ax_rate.plot(
            fwd_grid,
            forwards * 100.0,
            color=_FWD_COLOR,
            lw=1.6,
            ls="--",
            label=f"{forward_tenor:g}y forward ({forward_compounding})",
        )
    ax_rate.plot(
        pillars,
        par_rates * 100.0,
        color=_PAR_COLOR,
        marker="o",
        ms=5,
        ls="none",
        label="Par (market)",
        zorder=5,
    )

    as_of = getattr(par_data, "as_of_date", "")
    ax_rate.set_title(
        f"{par_data.curve_name} — term structure"
        + (f"  ({as_of})" if as_of else ""),
        fontsize=12,
        fontweight="bold",
        color=_PAR_COLOR,
    )
    ax_rate.set_ylabel("Rate (%)")
    ax_rate.legend(frameon=False, fontsize=9)
    ax_rate.grid(True, alpha=0.25)

    ax_df.plot(grid, dfs, color=_ZERO_COLOR, lw=1.5)
    ax_df.set_ylabel("P(0, t)")
    ax_df.set_xlabel("Tenor (years)")
    ax_df.set_ylim(0.0, 1.02)
    ax_df.grid(True, alpha=0.25)

    for ax in (ax_rate, ax_df):
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.01)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
