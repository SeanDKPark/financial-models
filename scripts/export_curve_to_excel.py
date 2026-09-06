"""Rebuild the Treasury par-to-zero bootstrap workbook from scratch.

    python scripts/export_curve_to_excel.py

Produces ``reports/treasury_bootstrapping_static.xlsx`` with the same layout as
the live hand-maintained workbook:

* Sheet ``Treasury_pq`` holds a ``CMT_pq`` table of one year's daily CMT par
  yields (Date + the 14 CSV tenor columns, newest first, percentage points).
* Sheet ``Bootstrapped_Curve``:
    - an eval-date control block (``G1`` overridable, defaults to ``G2`` = last
      business day),
    - Section A: live ``INDEX``/``MATCH`` lookups of the eval-date row in
      ``CMT_pq`` (``... )%`` converts percentage points to decimals),
    - Section B: the dense semi-annual bootstrap ladder as native formulas,
    - Section C: par-bond repricing checks.

IMPORTANT — this is a STATIC SNAPSHOT. openpyxl cannot create the Power Query
web connection, so the ``CMT_pq`` table here is frozen at generation time and
``Data > Refresh All`` does nothing. The live workbook
(``reports/treasury_bootstrapping.xlsx``) keeps the real Power Query connection;
do not overwrite it with this file. Regenerate this one only as a reference or
a clean starting point.

Hits the network. Not a pytest test; nothing here is asserted.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.worksheet import Worksheet

from finmodels.market_data.treasury_par import (
    TreasuryParHistory,
    fetch_treasury_par_history,
)

REPORT_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "treasury_bootstrapping_static.xlsx"
)

FREQUENCY = 2  # semi-annual coupons
SHORT_END_CUTOFF = 1.0
LADDER_START_T = 0.5
LADDER_END_T = 30.0
LADDER_STEP = 0.5

PQ_SHEET = "Treasury_pq"
PQ_TABLE = "CMT_pq"

# Section A layout: (CSV header label, tenor in years). Labels must match the
# Treasury CSV headers exactly so MATCH(label, CMT_pq[#Headers]) resolves.
BENCHMARK_TENORS: list[tuple[str, float]] = [
    ("1 Mo", 1 / 12),
    ("1.5 Month", 1.5 / 12),
    ("2 Mo", 2 / 12),
    ("3 Mo", 3 / 12),
    ("4 Mo", 4 / 12),
    ("6 Mo", 6 / 12),
    ("1 Yr", 1.0),
    ("2 Yr", 2.0),
    ("3 Yr", 3.0),
    ("5 Yr", 5.0),
    ("7 Yr", 7.0),
    ("10 Yr", 10.0),
    ("20 Yr", 20.0),
    ("30 Yr", 30.0),
]

# ----------------------------------------------------------------------------- styling
NAVY = "1F2A44"
STEEL = "3B5BA5"
SLATE = "44546A"
INPUT_FILL = "FFF2CC"
WHITE = "FFFFFF"

HEADER_FONT = Font(bold=True, color=WHITE, size=11)
TITLE_FONT = Font(bold=True, color=NAVY, size=14)
SUBTITLE_FONT = Font(italic=True, color=SLATE, size=10)
BOLD = Font(bold=True)

THIN = Side(style="thin", color="B9C0CC")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

PCT_FMT = "0.000%"
DF_FMT = "0.000000"
DATE_FMT = "yyyy-mm-dd"


def _fill(color: str) -> PatternFill:
    return PatternFill(start_color=color, end_color=color, fill_type="solid")


def _section_header(ws: Worksheet, cell_range: str, text: str, color: str) -> None:
    ws.merge_cells(cell_range)
    c = ws[cell_range.split(":")[0]]
    c.value = text
    c.font = HEADER_FONT
    c.alignment = Alignment(horizontal="left", vertical="center")
    for row in ws[cell_range]:
        for cell in row:
            cell.fill = _fill(color)
            cell.border = BOX


def _column_headers(
    ws: Worksheet, row: int, col_start: int, labels: list[str], color: str
) -> None:
    for i, label in enumerate(labels):
        c = ws.cell(row=row, column=col_start + i, value=label)
        c.font = HEADER_FONT
        c.fill = _fill(color)
        c.border = BOX
        c.alignment = Alignment(horizontal="center", vertical="center")


def _build_pq_sheet(ws: Worksheet, history: TreasuryParHistory) -> None:
    """Write the static ``CMT_pq`` table (stand-in for the Power Query output)."""
    ws["A1"] = (
        f"STATIC SNAPSHOT generated {date.today().isoformat()} by "
        "scripts/export_curve_to_excel.py"
    )
    ws["A1"].font = BOLD
    ws["A2"] = (
        "The live workbook replaces this sheet with a Power Query web connection "
        "(Data > Refresh All). Do not paste this over the live file."
    )
    ws["A2"].font = SUBTITLE_FONT

    header_row = 4
    headers = ["Date", *history.tenor_labels]
    ws.append([])  # row 3 spacer
    ws.append(headers)  # row 4
    for obs, values in history.rows:
        ws.append([obs, *values])

    n_cols = len(headers)
    last_row = header_row + len(history.rows)
    ref = f"A{header_row}:{get_column_letter(n_cols)}{last_row}"

    table = Table(displayName=PQ_TABLE, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showRowStripes=True
    )
    ws.add_table(table)

    for r in range(header_row + 1, last_row + 1):
        ws.cell(row=r, column=1).number_format = DATE_FMT
        for c in range(2, n_cols + 1):
            ws.cell(row=r, column=c).number_format = "0.00"

    ws.column_dimensions["A"].width = 12
    for c in range(2, n_cols + 1):
        ws.column_dimensions[get_column_letter(c)].width = 9
    ws.freeze_panes = f"B{header_row + 1}"


def _build_curve_sheet(ws: Worksheet) -> None:
    ws.sheet_view.showGridLines = False
    for col, width in zip("ABCD", (14, 20, 22, 22)):
        ws.column_dimensions[col].width = width
    for col, width in zip("FGH", (11, 12, 14)):
        ws.column_dimensions[col].width = width

    # --------------------------------------------------------------- title block
    ws.merge_cells("A1:D1")
    ws["A1"] = "US_TREASURY_PAR_YIELD — Par-to-Zero Bootstrap"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A2:D2")
    ws["A2"] = (
        '="As of "&TEXT($G$1,"yyyy-mm-dd")'
        f'&"  |  frequency={FREQUENCY} (semi-annual)  |  '
        f'short-end cutoff={SHORT_END_CUTOFF:.1f}Y  |  '
        'Section A is a live lookup into '
        f'{PQ_TABLE}"'
    )
    ws["A2"].font = SUBTITLE_FONT

    # --------------------------------------------------- eval-date control block
    ws["F1"] = "Eval Dt."
    ws["F1"].font = BOLD
    ws["G1"] = "=G2"
    ws["G1"].number_format = DATE_FMT
    ws["G1"].fill = _fill(INPUT_FILL)  # override here; blank/formula falls back to G2
    ws["G1"].border = BOX
    ws["F2"] = "Last BD"
    ws["F2"].font = BOLD
    ws["G2"] = "=WORKDAY(TODAY()+1,-1)"
    ws["G2"].number_format = DATE_FMT
    ws["G2"].border = BOX

    # ============================================================= Section A
    a_first = 6
    a_last = a_first + len(BENCHMARK_TENORS) - 1

    _section_header(
        ws, "F4:H4", f"Section A — Market Inputs (lookup: {PQ_TABLE})", STEEL
    )
    _column_headers(ws, 5, 6, ["Label", "Tenor (Y)", "Par Yield"], SLATE)

    for i, (label, t) in enumerate(BENCHMARK_TENORS):
        r = a_first + i
        ws.cell(row=r, column=6, value=label).border = BOX
        c_t = ws.cell(row=r, column=7, value=t)
        c_t.border = BOX
        c_t.number_format = "0.0000"
        # Percentage points -> decimal via the trailing % operator.
        formula = (
            f"=INDEX({PQ_TABLE}[#Data], "
            f"MATCH($G$1, {PQ_TABLE}[Date], 0), "
            f"MATCH(F{r}, {PQ_TABLE}[#Headers], 0))%"
        )
        c_y = ws.cell(row=r, column=8, value=formula)
        c_y.border = BOX
        c_y.number_format = PCT_FMT

    mkt_t_range = f"$G${a_first}:$G${a_last}"
    mkt_y_range = f"$H${a_first}:$H${a_last}"

    # ============================================================= Section B
    b_first_row = 6
    _section_header(
        ws, "A4:D4",
        "Section B — Dense Bootstrap Ladder (Semi-Annual Grid, 0.5Y–30.0Y)",
        STEEL,
    )
    _column_headers(
        ws, 5, 1,
        ["Tenor (Y)", "Par Yield (interp.)", "Discount Factor P(0,t)", "Zero Rate (semi-ann.)"],
        SLATE,
    )

    n_steps = round((LADDER_END_T - LADDER_START_T) / LADDER_STEP) + 1
    b_last_row = b_first_row + n_steps - 1

    for i in range(n_steps):
        r = b_first_row + i
        t = round(LADDER_START_T + i * LADDER_STEP, 4)

        a_cell = ws.cell(row=r, column=1, value=t)
        a_cell.number_format = "0.00"
        a_cell.border = BOX

        # Col B: par yield linearly interpolated off the Section A table.
        idx = f"MATCH(A{r},{mkt_t_range},1)"
        lo_t = f"INDEX({mkt_t_range},{idx})"
        lo_y = f"INDEX({mkt_y_range},{idx})"
        hi_t = f"INDEX({mkt_t_range},{idx}+1)"
        hi_y = f"INDEX({mkt_y_range},{idx}+1)"
        interp = f"{lo_y}+({hi_y}-{lo_y})/({hi_t}-{lo_t})*(A{r}-{lo_t})"
        b_formula = f"=IFERROR(IF({lo_t}=A{r},{lo_y},{interp}),{lo_y})"
        b_cell = ws.cell(row=r, column=2, value=b_formula)
        b_cell.number_format = PCT_FMT
        b_cell.border = BOX

        # Col C: discount factor, short-end vs. long-end recursion.
        short_end_df = f"(1+B{r}/{FREQUENCY})^(-{FREQUENCY}*A{r})"
        if r == b_first_row:
            c_formula = f"=IF(A{r}<={SHORT_END_CUTOFF},{short_end_df},{short_end_df})"
        else:
            prev_sum = f"SUM($C${b_first_row}:C{r - 1})"
            long_end_df = f"(1-(B{r}/{FREQUENCY})*{prev_sum})/(1+B{r}/{FREQUENCY})"
            c_formula = f"=IF(A{r}<={SHORT_END_CUTOFF},{short_end_df},{long_end_df})"
        c_cell = ws.cell(row=r, column=3, value=c_formula)
        c_cell.number_format = DF_FMT
        c_cell.border = BOX

        # Col D: semi-annual zero rate implied by the discount factor.
        d_cell = ws.cell(row=r, column=4, value=f"=2*(C{r}^(-1/({FREQUENCY}*A{r}))-1)")
        d_cell.number_format = PCT_FMT
        d_cell.border = BOX

    ws.freeze_panes = f"A{b_first_row}"

    # ============================================================= Section C
    c_header_row = b_last_row + 2
    c_col_row = c_header_row + 1
    c_first_row = c_col_row + 1

    _section_header(
        ws, f"A{c_header_row}:D{c_header_row}",
        "Section C — Repricing Check (Par Bonds Must Reprice to 1.000000)",
        NAVY,
    )
    _column_headers(
        ws, c_col_row, 1,
        ["Maturity (Y)", "Par Yield", "PV (Coupons + Principal)", "Check"],
        SLATE,
    )

    for i, t in enumerate(m for m in (2.0, 5.0, 10.0, 30.0) if m <= LADDER_END_T):
        r = c_first_row + i
        ladder_row = b_first_row + round((t - LADDER_START_T) / LADDER_STEP)

        a_cell = ws.cell(row=r, column=1, value=t)
        a_cell.number_format = "0.0"
        a_cell.border = BOX
        a_cell.font = BOLD

        y_cell = ws.cell(row=r, column=2, value=f"=B{ladder_row}")
        y_cell.number_format = PCT_FMT
        y_cell.border = BOX

        pv_cell = ws.cell(
            row=r, column=3,
            value=(
                f"=(B{ladder_row}/{FREQUENCY})*SUM($C${b_first_row}:C{ladder_row})"
                f"+C{ladder_row}"
            ),
        )
        pv_cell.number_format = DF_FMT
        pv_cell.border = BOX
        pv_cell.font = BOLD

        check_cell = ws.cell(
            row=r, column=4, value=f'=IF(ABS(C{r}-1)<0.000001,"OK","FAIL")'
        )
        check_cell.border = BOX
        check_cell.font = BOLD
        check_cell.alignment = Alignment(horizontal="center")


def build_workbook(history: TreasuryParHistory) -> Workbook:
    wb = Workbook()
    curve = wb.active
    curve.title = "Bootstrapped_Curve"
    _build_curve_sheet(curve)
    _build_pq_sheet(wb.create_sheet(PQ_SHEET), history)
    return wb


def main() -> None:
    history = fetch_treasury_par_history()
    wb = build_workbook(history)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(REPORT_PATH)

    latest = history.rows[0][0]
    print(
        f"Wrote {REPORT_PATH}  "
        f"({len(history.rows)} history rows, latest {latest})"
    )
    print("STATIC snapshot - no Power Query connection. The live workbook is "
          "reports/treasury_bootstrapping.xlsx.")


if __name__ == "__main__":
    main()
