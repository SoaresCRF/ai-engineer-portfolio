"""Corpus loading: Planalto law HTML -> Markdown passages, and Markdown -> Passage objects."""

import hashlib
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

LAW_SOURCES: dict[str, str] = {
    "cdc": "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
    "decreto_7962": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2013/decreto/d7962.htm",
}
LAW_TITLES: dict[str, str] = {
    "cdc": "Código de Defesa do Consumidor (Lei nº 8.078/1990)",
    "decreto_7962": "Decreto nº 7.962/2013 (Comércio Eletrônico)",
}

USER_AGENT = "support-rag-agent/0.1"
SOURCES_FILE = "SOURCES.md"
META_SNIFF_BYTES = 4096

_SKIPPED_TAGS = {"strike", "s", "script", "style"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
# "Art. 1º", "Art. 10.", "Art. 54-A." -> number plus optional letter suffix.
_ARTICLE_RE = re.compile(r"^Art\.\s*(\d+)\s*[ºo°]?(?:\s*-\s*([A-Z]))?\b\.?", re.IGNORECASE)
_EMPTY_BODY_RE = re.compile(r"^[\s.\-–—]*(\(?\s*(vetado|revogado)\s*\)?[\s.]*)?$", re.IGNORECASE)
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
        charset = match.group(1).decode("ascii") if match else "utf-8"
    if charset.lower() in {"iso-8859-1", "latin-1", "latin1"}:
        charset = "windows-1252"  # superset; browsers treat them the same
    return data.decode(charset, errors="replace")


def _local_id(match: re.Match) -> str:
    return f"art{match.group(1)}{(match.group(2) or '').lower()}"


def _is_empty_article(first_line: str, rest: list[str], prefix_end: int) -> bool:
    body = " ".join([first_line[prefix_end:], *rest])
    return bool(_EMPTY_BODY_RE.match(body))


def law_html_to_markdown(html: str, doc_title: str) -> str:
    collector = _LineCollector()
    collector.feed(html)
    collector.close()

    articles: list[tuple[re.Match, list[str]]] = []
    for line in collector.lines:
        match = _ARTICLE_RE.match(line)
        if match:
            articles.append((match, [line]))
        elif articles:  # lines before the first article (preamble) are ignored
            articles[-1][1].append(line)

    parts = [f"# {doc_title}"]
    for match, lines in articles:
        if _is_empty_article(lines[0], lines[1:], match.end()):
            continue
        label = match.group(0).strip().rstrip(".")
        parts.append(f"## {_local_id(match)} {label}\n\n" + "\n\n".join(lines))
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


def _download(url: str) -> tuple[bytes, str | None, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get_content_charset(), response.geturl()


def build_laws(corpus_dir: Path) -> None:
    """Download each law, store raw HTML, write Markdown and record date + SHA-256 in SOURCES.md."""
    raw_dir, laws_dir = corpus_dir / "raw", corpus_dir / "laws"
    raw_dir.mkdir(parents=True, exist_ok=True)
    laws_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Sources",
        "",
        "| doc_id | URL | downloaded (UTC) | SHA-256 of raw HTML |",
        "|---|---|---|---|",
    ]
    for doc_id, url in LAW_SOURCES.items():
        data, charset, final_url = _download(url)
        (raw_dir / f"{doc_id}.html").write_bytes(data)
        markdown = law_html_to_markdown(decode_html(data, charset), LAW_TITLES[doc_id])
        (laws_dir / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        rows.append(f"| {doc_id} | {final_url} | {stamp} | {hashlib.sha256(data).hexdigest()} |")
    (laws_dir / SOURCES_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] != ["build-laws"]:
        sys.exit("usage: python -m support_agent.corpus build-laws")
    build_laws(Path(__file__).resolve().parents[2] / "corpus")
