from pathlib import Path

import pytest

from support_agent.corpus import decode_html, law_html_to_markdown, load_corpus, parse_markdown

FIXTURE = Path(__file__).parent / "fixtures" / "planalto_sample.html"


@pytest.fixture
def markdown() -> str:
    return law_html_to_markdown(FIXTURE.read_text(encoding="utf-8"), "Sample Law")


def test_law_html_to_markdown_splits_articles(markdown):
    headings = [line.split()[1] for line in markdown.splitlines() if line.startswith("## ")]
    assert headings == ["art1", "art10", "art49", "art54a", "art60"]
    assert markdown.startswith("# Sample Law\n")
    assert "Presidência" not in markdown  # preamble dropped


def test_struck_text_is_dropped(markdown):
    assert "revogado que nao" not in markdown
    assert "Trecho riscado" not in markdown


def test_paragraphs_stay_with_article(markdown):
    art10 = markdown.split("## art10")[1].split("## art49")[0]
    assert "§ 1º" in art10
    assert "Parágrafo único" in art10


def test_vetoed_and_revoked_articles_are_skipped(markdown):
    assert "art55" not in markdown
    assert "art56" not in markdown
    assert "art50" not in markdown


def test_parse_markdown_ids():
    passages = parse_markdown("# T\n## art49 Arrependimento\nx", "cdc")
    assert passages[0].passage_id == "cdc:art49"
    assert passages[0].doc_title == "T"
    assert passages[0].text == "x"


def test_parse_markdown_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        parse_markdown("# T\n## art1 A\nx\n## art1 B\ny", "cdc")


def test_load_corpus_reads_laws_and_policies(tmp_path):
    (tmp_path / "laws").mkdir()
    (tmp_path / "policies").mkdir()
    (tmp_path / "laws" / "cdc.md").write_text("# CDC\n## art1 A\nx\n", encoding="utf-8")
    (tmp_path / "laws" / "SOURCES.md").write_text("# Sources\n", encoding="utf-8")
    (tmp_path / "policies" / "returns.md").write_text("# R\n## 3.2 T\ny\n", encoding="utf-8")
    ids = [p.passage_id for p in load_corpus(tmp_path)]
    assert ids == ["cdc:art1", "returns:3.2"]


def test_decode_html_uses_meta_charset():
    html = "<meta charset=windows-1252><p>Art. 1º ação</p>".encode("cp1252")
    assert "Art. 1º ação" in decode_html(html, None)
    assert "ação" in decode_html("ação".encode("latin-1"), "iso-8859-1")


def test_footer_after_signature_is_dropped(markdown):
    assert "FOOTER SIGNATURE" not in markdown
    assert "Brasília" not in markdown


def test_decode_html_without_declaration_falls_back_to_cp1252():
    assert decode_html("ação".encode("cp1252"), None) == "ação"
    assert decode_html("ação".encode(), None) == "ação"


def test_text_inside_s_tag_is_kept(markdown):
    assert "§ 1º mantido" in markdown
