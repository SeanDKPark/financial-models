"""Compare five zero curves fitted to the same bootstrapped Treasury pillars.

    python scripts/compare_interpolators.py

Fetches the latest Treasury par curve, bootstraps it to zero rates, then fits
Linear / natural-Cubic / PCHIP interpolation, a calibrated Nelson-Siegel-Svensson
parametric curve, and a Hagan-West monotone-convex curve to the same market
pillars, and plots zero curves and 6M rolling forwards. The forward panel shows
how every method except Linear removes the 10Y-20Y "shark fin" discontinuity.

Writes reports/interpolation_comparison.png. Hits the network.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from finmodels.curves.bootstrapping import bootstrap_par_curve  # noqa: E402
from finmodels.curves.interpolation import LinearZeroInterpolator  # noqa: E402
from finmodels.curves.monotone_convex import HaganWestInterpolator  # noqa: E402
from finmodels.curves.parametric import SvenssonCurve  # noqa: E402
from finmodels.curves.splines import (  # noqa: E402
    CubicZeroInterpolator,
    PchipZeroInterpolator,
)
from finmodels.market_data.treasury_par import fetch_treasury_par_curve  # noqa: E402

OUT_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "interpolation_comparison.png"
)

FORWARD_TENOR = 0.5
STYLES = {
    "Linear": dict(color="#1f2a44", lw=1.4, ls="-"),
    "Cubic (natural)": dict(color="#2f6f9f", lw=1.4, ls="--"),
    "PCHIP": dict(color="#c2570c", lw=1.6, ls="-"),
    "Svensson (NSS)": dict(color="#2e7d32", lw=1.6, ls="--"),
    "Hagan-West (MC)": dict(color="#6a1b9a", lw=2.0, ls="-"),
}


def main() -> None:
    par_curve = fetch_treasury_par_curve()

    # Bootstrap once, then read the zero rate at each *market* pillar. Feeding the
    # sparse pillars (not the dense 0.5y bootstrap grid) is what makes the
    # interpolation choice visible: linear kinks between 10Y/20Y/30Y, the splines
    # round them off.
    dense = bootstrap_par_curve(par_curve, compounding="semi-annual")
    pillars = np.asarray(par_curve.pillars, dtype=float)
    zeros = np.asarray(dense.zero_rate(pillars), dtype=float)

    svensson = SvenssonCurve.calibrate(pillars, zeros)
    fitted = np.asarray(svensson.zero_rate(pillars), dtype=float)
    rmse_bp = float(np.sqrt(np.mean((fitted - zeros) ** 2)) * 1e4)

    curves = {
        "Linear": LinearZeroInterpolator(pillars, zeros),
        "Cubic (natural)": CubicZeroInterpolator(pillars, zeros),
        "PCHIP": PchipZeroInterpolator(pillars, zeros),
        "Svensson (NSS)": svensson,
        "Hagan-West (MC)": HaganWestInterpolator(pillars, zeros),
    }

    p = svensson.params
    print(f"Calibrated Nelson-Siegel-Svensson  (as of {par_curve.as_of_date})")
    print(f"  beta0 = {p['beta0']:+.6f}")
    print(f"  beta1 = {p['beta1']:+.6f}")
    print(f"  beta2 = {p['beta2']:+.6f}")
    print(f"  beta3 = {p['beta3']:+.6f}")
    print(f"  tau1  = {p['tau1']:.4f}")
    print(f"  tau2  = {p['tau2']:.4f}")
    print(f"  calibration RMSE = {rmse_bp:.2f} bp  ({pillars.size} pillars)")

    t_min, t_max = float(pillars[0]), float(pillars[-1])
    zero_grid = np.linspace(t_min, t_max, 500)
    fwd_grid = np.linspace(t_min, t_max - FORWARD_TENOR, 500)

    fig, (ax_zero, ax_fwd) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for name, curve in curves.items():
        ax_zero.plot(
            zero_grid, np.asarray(curve.zero_rate(zero_grid)) * 100.0,
            label=name, **STYLES[name],
        )
        ax_fwd.plot(
            fwd_grid,
            np.asarray(
                curve.forward_rate_rolling(
                    fwd_grid, tenor=FORWARD_TENOR, compounding="semi-annual"
                )
            ) * 100.0,
            label=name, **STYLES[name],
        )

    ax_zero.plot(
        pillars, zeros * 100.0, "o", ms=3, color="#666", label="Bootstrapped pillars",
        zorder=5,
    )

    ax_zero.set_title(
        f"{par_curve.curve_name} - zero-rate interpolation  ({par_curve.as_of_date})",
        fontsize=12, fontweight="bold", color="#1f2a44",
    )
    ax_zero.set_ylabel("Zero rate (%)")
    ax_fwd.set_title(f"{FORWARD_TENOR:g}y rolling forward  F(t, t+{FORWARD_TENOR:g})", fontsize=11)
    ax_fwd.set_ylabel("Forward rate (%)")
    ax_fwd.set_xlabel("Tenor (years)")

    for ax in (ax_zero, ax_fwd):
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.01)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    fig.clf()

    print(f"Wrote {OUT_PATH}  (as of {par_curve.as_of_date})")


if __name__ == "__main__":
    main()
