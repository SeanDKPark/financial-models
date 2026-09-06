"""Generate a combined term-structure report from the live Treasury curve.

    python scripts/term_structure_report.py

Fetches the latest Treasury par curve, bootstraps it to zeros (semi-annual),
and writes a par / zero / rolling-forward chart to
``reports/term_structure.png``.

Hits the network. Not a pytest test; nothing here is asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window

from finmodels.curves.bootstrapping import bootstrap_par_curve  # noqa: E402
from finmodels.market_data.treasury_par import fetch_treasury_par_curve  # noqa: E402
from finmodels.visualization import plot_term_structure  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "reports" / "term_structure.png"


def main() -> None:
    par_curve = fetch_treasury_par_curve()
    zero_curve = bootstrap_par_curve(par_curve, compounding="semi-annual")

    fig = plot_term_structure(
        par_curve,
        zero_curve,
        forward_tenor=0.5,
        forward_compounding="semi-annual",
        save_path=OUT_PATH,
    )
    fig.clf()

    print(f"Wrote {OUT_PATH}  (as of {par_curve.as_of_date})")


if __name__ == "__main__":
    main()
