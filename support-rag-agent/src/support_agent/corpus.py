"""Corpus loading: Planalto law HTML -> Markdown passages, and Markdown -> Passage objects."""

import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from support_agent import ROOT

LAW_TITLES: dict[str, str] = {
    "cdc": "Código de Defesa do Consumidor (Lei nº 8.078/1990)",
    "decreto_7962": "Decreto nº 7.962/2013 (Comércio Eletrônico)",
}

SOURCES_FILE = "SOURCES.md"
META_SNIFF_BYTES = 4096

_SKIPPED_TAGS = {"strike", "script", "style"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
# "Art. 1º", "Art. 10.", "Art. 54-A." -> number plus optional letter suffix.
_ARTICLE_RE = re.compile(r"^Art\.\s*(\d+)\s*[ºo°]?(?:\s*-\s*([A-Z]))?(?!\w)\.?", re.IGNORECASE)
_EMPTY_BODY_RE = re.compile(r"^[\s.\-–—]*(\(?\s*(vetado|revogado)\s*\)?[\s.]*)?$", re.IGNORECASE)
# Signature line ("Brasília, 11 de setembro de 1990; ...") ends the law text; footer follows.
_SIGNATURE_RE = re.compile(r"^Bras[ií]lia,\s*\d")
# "CAPÍTULO III", "SEÇÃO I", "CAPÍTULO VI-A": the next line holds the structure name.
_STRUCTURE_RE = re.compile(r"^(TÍTULO|CAPÍTULO|SEÇÃO)\s+([IVXLCDM]+(?:-[A-Z])?|ÚNICO)\s*$")
_STRUCTURE_LEVELS = {"TÍTULO": 0, "CAPÍTULO": 1, "SEÇÃO": 2}
_META_CHARSET_RE = re.compile(rb"charset\s*=\s*[\"']?([\w-]+)", re.IGNORECASE)
_HEADING_RE = re.compile(r"^## (\S+)\s*(.*)$")


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str


class _LineCollector(HTMLParser):
    """Collects visible text as lines, dropping struck-through (repealed) text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._current: list[str] = []
        self._skip_depth = 0

    def _flush(self) -> None:
        line = " ".join("".join(self._current).replace("\xa0", " ").split())
        if line:
            self.lines.append(line)
        self._current = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._current.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def decode_html(data: bytes, header_charset: str | None) -> str:
    """Decode with the HTTP charset, else the <meta> charset, else UTF-8 (Planalto: cp1252)."""
    charset = header_charset
    if not charset:
        match = _META_CHARSET_RE.search(data[:META_SNIFF_BYTES])
        if match:
            charset = match.group(1).decode("ascii")
        else:
            try:  # Planalto pages often declare nothing and are cp1252
                return data.decode("utf-8")
            except UnicodeDecodeError:
                charset = "windows-1252"
    if charset.lower() in {"iso-8859-1", "latin-1", "latin1"}:
        charset = "windows-1252"  # superset; browsers treat them the same
    return data.decode(charset, errors="replace")


def _local_id(match: re.Match) -> str:
    return f"art{match.group(1)}{(match.group(2) or '').lower()}"


def _is_empty_article(first_line: str, rest: list[str], prefix_end: int) -> bool:
    body = " ".join([first_line[prefix_end:], *rest])
    return bool(_EMPTY_BODY_RE.match(body))


class _Structure:
    """Tracks the current título/capítulo/seção while scanning law lines."""

    def __init__(self) -> None:
        self._labels = ["", "", ""]
        self._pending: int | None = None  # level whose name is on the next line

    def consume(self, line: str) -> bool:
        """Return True if the line is a structural heading or its name (not article text)."""
        if self._pending is not None:
            level, self._pending = self._pending, None
            if not _ARTICLE_RE.match(line):
                self._labels[level] += f": {line}"
                return True
        match = _STRUCTURE_RE.match(line)
        if not match:
            return False
        level = _STRUCTURE_LEVELS[match.group(1)]
        # A deeper level is reset whenever its parent changes.
        label = f"{match.group(1).capitalize()} {match.group(2)}"
        self._labels[level:] = [label] + [""] * (2 - level)
        self._pending = level
        return True

    def title(self) -> str:
        """Capítulo > seção; título only when the article sits outside any capítulo."""
        return " > ".join(label for label in self._labels[1:] if label) or self._labels[0]


def law_html_to_markdown(html: str, doc_title: str) -> str:
    collector = _LineCollector()
    collector.feed(html)
    collector.close()

    structure = _Structure()
    articles: list[tuple[re.Match, list[str], str]] = []
    for line in collector.lines:
        if _SIGNATURE_RE.match(line):
            break
        if structure.consume(line):
            continue
        match = _ARTICLE_RE.match(line)
        if match:
            articles.append((match, [line], structure.title()))
        elif articles:  # lines before the first article (preamble) are ignored
            articles[-1][1].append(line)

    parts = [f"# {doc_title}"]
    for match, lines, context in articles:
        if _is_empty_article(lines[0], lines[1:], match.end()):
            continue
        label = match.group(0).strip().rstrip(".")
        heading = f"{label} — {context}" if context else label
        parts.append(f"## {_local_id(match)} {heading}\n\n" + "\n\n".join(lines))
    return "\n\n".join(parts) + "\n"


def parse_markdown(md: str, doc_id: str) -> list[Passage]:
    lines = md.splitlines()
    doc_title = lines[0][2:].strip() if lines and lines[0].startswith("# ") else doc_id
    passages: list[Passage] = []
    seen: set[str] = set()
    current: tuple[str, str] | None = None
    body: list[str] = []

    def close() -> None:
        if current:
            local_id, title = current
            passages.append(
                Passage(f"{doc_id}:{local_id}", doc_id, doc_title, title, "\n".join(body).strip())
            )

    for line in lines[1:]:
        match = _HEADING_RE.match(line)
        if not match:
            body.append(line)
            continue
        close()
        local_id = match.group(1)
        if local_id in seen:
            raise ValueError(f"duplicate passage id {doc_id}:{local_id}")
        seen.add(local_id)
        current, body = (local_id, match.group(2).strip()), []
    close()
    return passages


def load_corpus(corpus_dir: Path) -> list[Passage]:
    passages: list[Passage] = []
    for sub in ("laws", "policies"):
        for path in sorted((corpus_dir / sub).glob("*.md")):
            if path.name == SOURCES_FILE:
                continue
            passages += parse_markdown(path.read_text(encoding="utf-8"), path.stem)
    return passages


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    passage_id: str
    text: str


DEFAULT_MAX_CHARS = 1500
_PARAGRAPH_SEP = "\n\n"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=\. )")  # keeps ". " at the end of each sentence


def _pack(parts: list[str], sep: str, max_chars: int) -> list[str]:
    """Greedily join parts with sep into groups of at most max_chars (parts must fit)."""
    groups: list[str] = []
    for part in parts:
        if groups and len(groups[-1]) + len(sep) + len(part) <= max_chars:
            groups[-1] += sep + part
        else:
            groups.append(part)
    return groups


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries; hard cut only as a last resort."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
        if len(sentence) > max_chars:  # no sentence boundary within the limit
            pieces += [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
        else:
            pieces.append(sentence)
    return [group.strip() for group in _pack(pieces, "", max_chars)]


def chunk_passages(passages: list[Passage], max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Pack paragraphs into chunks of at most max_chars, excluding the title prefix."""
    chunks: list[Chunk] = []
    for passage in passages:
        parts: list[str] = []
        for paragraph in passage.text.split(_PARAGRAPH_SEP):
            if len(paragraph) > max_chars:
                parts += _split_paragraph(paragraph, max_chars)
            else:
                parts.append(paragraph)
        prefix = f"{passage.doc_title} > {passage.heading}\n"
        for i, body in enumerate(_pack(parts, _PARAGRAPH_SEP, max_chars)):
            chunks.append(Chunk(f"{passage.passage_id}#{i}", passage.passage_id, prefix + body))
    return chunks


# Raw file name (saved manually: the site rejects non-browser clients) -> doc_id.
RAW_FILES: dict[str, str] = {"cdc.html": "cdc", "d7962.html": "decreto_7962"}


def build_laws(corpus_dir: Path) -> None:
    """Convert corpus/raw/*.html into corpus/laws/*.md. SOURCES.md is maintained by hand."""
    for raw_name, doc_id in RAW_FILES.items():
        data = (corpus_dir / "raw" / raw_name).read_bytes()
        markdown = law_html_to_markdown(decode_html(data, None), LAW_TITLES[doc_id])
        (corpus_dir / "laws" / f"{doc_id}.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] != ["build-laws"]:
        sys.exit("usage: python -m support_agent.corpus build-laws")
    build_laws(ROOT / "corpus")
