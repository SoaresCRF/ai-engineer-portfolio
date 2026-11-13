"""Blind labeling round-trip: export a spreadsheet for a human, import it back validated.

Usage:
    uv run python -m review_triage.labeling export   # data/raw -> sheet + sample metadata
    uv run python -m review_triage.labeling import   # filled sheet -> gold_dev.csv / gold_test.csv
"""

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from review_triage.data import load_reviews, sample_gold, validate_labels
from review_triage.schema import LABEL_FIELDS

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
LABELS_DIR = PROJECT_DIR / "data" / "labels"
SHEET_PATH = LABELS_DIR / "gold_labeling.xlsx"
# Score and delivery facts stay out of the sheet so labels are blind to them.
SAMPLE_PATH = LABELS_DIR / "gold_sample.csv"

SHEET_COLUMNS = ["review_id", "text", *LABEL_FIELDS]
TEXT_COLUMN_WIDTH = 90
LABEL_COLUMN_WIDTH = 30


def export_sheet(sample: pd.DataFrame, path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    ws.append(SHEET_COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in sample[["review_id", "text"]].itertuples(index=False):
        ws.append([row.review_id, row.text, None, None, None])

    last_row = len(sample) + 1
    for col_idx, enum in enumerate(LABEL_FIELDS.values(), start=3):
        letter = ws.cell(row=1, column=col_idx).column_letter
        options = ",".join(member.value for member in enum)
        validation = DataValidation(
            type="list", formula1=f'"{options}"', showErrorMessage=True, allow_blank=True
        )
        validation.add(f"{letter}2:{letter}{last_row}")
        ws.add_data_validation(validation)
        ws.column_dimensions[letter].width = LABEL_COLUMN_WIDTH

    ws.column_dimensions["A"].hidden = True
    ws.column_dimensions["B"].width = TEXT_COLUMN_WIDTH
    for (cell,) in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    wb.save(path)


def read_sheet(path: Path) -> pd.DataFrame:
    ws = load_workbook(path, read_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    return pd.DataFrame(rows[1:], columns=rows[0])[SHEET_COLUMNS]


def import_sheet(sheet_path: Path, sample_path: Path) -> pd.DataFrame:
    labels = read_sheet(sheet_path).drop(columns="text")
    sample = pd.read_csv(sample_path)
    gold = sample.merge(labels, on="review_id", how="inner", validate="one_to_one")
    if len(gold) != len(sample):
        raise ValueError(f"sheet has {len(gold)} of {len(sample)} sampled reviews")
    unlabeled = gold[list(LABEL_FIELDS)].isna().any(axis=1).sum()
    if unlabeled:
        raise ValueError(f"{unlabeled} rows still have empty labels")
    return validate_labels(gold)


def main(command: str) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if command == "export":
        if SHEET_PATH.exists():
            raise SystemExit(f"{SHEET_PATH} exists; refusing to overwrite labeling work")
        sample = sample_gold(load_reviews(RAW_DIR))
        sample.to_csv(SAMPLE_PATH, index=False)
        export_sheet(sample, SHEET_PATH)
        print(f"wrote {len(sample)} reviews to {SHEET_PATH}")
    elif command == "import":
        gold = import_sheet(SHEET_PATH, SAMPLE_PATH)
        for split, part in gold.groupby("split"):
            out = LABELS_DIR / f"gold_{split}.csv"
            part.drop(columns="split").to_csv(out, index=False)
            print(f"wrote {len(part)} rows to {out}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
