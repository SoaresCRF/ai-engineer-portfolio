import pandas as pd
import pytest
from openpyxl import load_workbook

from review_triage.labeling import export_sheet, import_sheet


@pytest.fixture
def sample(tmp_path):
    df = pd.DataFrame(
        {
            "review_id": ["a", "b"],
            "review_score": [1, 5],
            "text": ["não recebi", "ótimo"],
            "split": ["dev", "test"],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return df, path


def fill(sheet_path, labels):
    wb = load_workbook(sheet_path)
    ws = wb.active
    for row_idx, values in enumerate(labels, start=2):
        for col_idx, value in enumerate(values, start=3):
            ws.cell(row=row_idx, column=col_idx, value=value)
    wb.save(sheet_path)


def test_sheet_hides_score_and_round_trips(sample, tmp_path):
    df, sample_path = sample
    sheet = tmp_path / "sheet.xlsx"
    export_sheet(df, sheet)

    header = [c.value for c in load_workbook(sheet).active[1]]
    assert "review_score" not in header

    fill(
        sheet, [("nao_recebido", "negativo", "alta"), ("elogio_sem_problema", "positivo", "baixa")]
    )
    gold = import_sheet(sheet, sample_path).set_index("review_id")

    assert gold.loc["a", "urgencia"] == "alta"
    assert gold.loc["b", "review_score"] == 5  # metadata merged back from the sample


def test_import_accepts_csv_export(sample, tmp_path):
    df, sample_path = sample
    csv = tmp_path / "export.csv"
    df[["review_id", "text"]].assign(
        categoria=["nao_recebido", "elogio_sem_problema"],
        sentimento=["negativo", "positivo"],
        urgencia=["alta", "baixa"],
    ).to_csv(csv, index=False)

    gold = import_sheet(csv, sample_path)

    assert list(gold["categoria"]) == ["nao_recebido", "elogio_sem_problema"]


def test_import_rejects_unlabeled_rows(sample, tmp_path):
    df, sample_path = sample
    sheet = tmp_path / "sheet.xlsx"
    export_sheet(df, sheet)
    fill(sheet, [("nao_recebido", "negativo", "alta")])

    with pytest.raises(ValueError, match="1 rows still have empty labels"):
        import_sheet(sheet, sample_path)
