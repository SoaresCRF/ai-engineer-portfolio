from pathlib import Path

import pytest

from support_agent.corpus import (
    Passage,
    chunk_passages,
    decode_html,
    law_html_to_markdown,
    load_corpus,
    parse_markdown,
)

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


def _squash(text: str) -> str:
    return "".join(text.split())


def _passage(text: str, passage_id: str = "cdc:art49") -> Passage:
    return Passage(passage_id, "cdc", "CDC", "Art. 49", text)


def test_short_passage_one_chunk():
    chunks = chunk_passages([_passage("Short text.")])
    assert [c.chunk_id for c in chunks] == ["cdc:art49#0"]
    assert chunks[0].text == "CDC > Art. 49\nShort text."


def test_long_passage_splits_on_blank_lines():
    chunks = chunk_passages([_passage("\n\n".join(c * 700 for c in "abc"))])
    assert len(chunks) >= 2
    assert {c.passage_id for c in chunks} == {"cdc:art49"}
    assert all(c.text.startswith("CDC > Art. 49\n") for c in chunks)


def test_single_paragraph_over_limit_is_hard_split():
    prefix = "CDC > Art. 49\n"
    chunks = chunk_passages([_passage("x" * 4000)], max_chars=1500)
    assert len(chunks) == 3
    assert all(len(c.text) - len(prefix) <= 1500 for c in chunks)
    assert "".join(c.text[len(prefix) :] for c in chunks) == "x" * 4000


def test_oversized_paragraph_splits_on_sentence_boundary():
    sentences = [f"Sentence number {i:02d} is here" for i in range(10)]
    text = ". ".join(sentences)
    chunks = chunk_passages([_passage(text)], max_chars=100)
    assert len(chunks) > 1
    assert _squash("".join(c.text.split("\n", 1)[1] for c in chunks)) == _squash(text)
    assert all(c.text.split("\n", 1)[1].endswith(("here", "here.")) for c in chunks)


def test_real_corpus_chunks_within_limit():
    max_chars = 1500
    passages = load_corpus(Path("corpus"))
    chunks = chunk_passages(passages, max_chars=max_chars)
    for c in chunks:
        assert len(c.text.split("\n", 1)[1]) <= max_chars
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    by_passage = {p.passage_id: [] for p in passages}
    for c in chunks:
        by_passage[c.passage_id].append(c)
    assert all(by_passage.values())
    for passage in passages:
        assert _squash(
            "".join(c.text.split("\n", 1)[1] for c in by_passage[passage.passage_id])
        ) == _squash(passage.text)


STRUCTURE_HTML = """<html><body>
<p>TÍTULO I</p><p>Dos Direitos</p>
<p>CAPÍTULO I</p><p>Disposições Gerais</p>
<p>Art. 1° Primeiro artigo.</p>
<p>CAPÍTULO II</p><p>Da Política</p>
<p>SEÇÃO I</p><p>Da Proteção</p>
<p>Art. 2° Segundo artigo.</p>
<p>CAPÍTULO III</p><p>Dos Direitos</p>
<p>Art. 3° Terceiro artigo.</p>
<p>Brasília, 1 de janeiro de 2000.</p></body></html>"""


def test_structure_headings_move_into_article_titles():
    md = law_html_to_markdown(STRUCTURE_HTML, "Sample")
    art1 = md.split("## art1 ")[1].split("## art2 ")[0]
    assert "CAPÍTULO" not in art1
    assert "Da Política" not in art1
    assert "## art1 Art. 1° — Capítulo I: Disposições Gerais\n" in md
    assert "## art2 Art. 2° — Capítulo II: Da Política > Seção I: Da Proteção\n" in md
    assert "## art3 Art. 3° — Capítulo III: Dos Direitos\n" in md  # new capítulo resets seção
