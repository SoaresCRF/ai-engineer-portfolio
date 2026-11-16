from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = "results/readme_tables_test.md"
MARKER = f"<!-- from {TABLES} -->"


def _marked_tables(readme: str) -> list[str]:
    """Each table block (consecutive `|` lines) that follows a MARKER line."""
    lines = readme.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        if line.strip() != MARKER:
            continue
        rest = lines[i + 1 :]
        while rest and not rest[0].strip():
            rest = rest[1:]
        table = []
        for row in rest:
            if not row.startswith("|"):
                break
            table.append(row)
        blocks.append("\n".join(table))
    return blocks


def test_readme_tables_match_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source = (ROOT / TABLES).read_text(encoding="utf-8")
    tables = _marked_tables(readme)
    assert tables, f"no table marked with {MARKER}"
    for table in tables:
        assert table, "a marker is not followed by a table"
        assert table in source, f"table not found verbatim in {TABLES}:\n{table}"
