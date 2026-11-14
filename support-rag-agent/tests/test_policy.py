import unicodedata
from pathlib import Path

from support_agent import policy
from support_agent.corpus import load_corpus

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
POLICY_DOC_COUNT = 8


def _expected_text(name: str) -> str:
    """Render the substring a constant must appear as in the policy text."""
    base, _, key = name.partition(".")
    value = getattr(policy, base)
    if key:
        value = value[key]
    if base == "LATE_COUPON_CENTS":
        return f"R$ {value // 100},{value % 100:02d}"
    if base == "LATE_COUPON_MINUTES":
        return f"{value} minutos"
    if base == "REGRET_STORE_TYPES":
        return " e ".join(value)
    return f"{value} dias"


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def test_policies_parse():
    passages = load_corpus(CORPUS)
    policy_docs = {p.doc_id for p in passages if (CORPUS / "policies" / f"{p.doc_id}.md").exists()}
    assert len(policy_docs) == POLICY_DOC_COUNT
    ids = [p.passage_id for p in passages]
    assert len(ids) == len(set(ids))


def test_constants_appear_in_policy_text():
    by_id = {p.passage_id: p.text for p in load_corpus(CORPUS)}
    required = {f"COMPLAINT_WINDOW_DAYS.{t}" for t in policy.STORE_TYPES} | {
        "REGRET_WINDOW_DAYS",
        "LATE_COUPON_MINUTES",
        "LATE_COUPON_CENTS",
    }
    assert required <= policy.POLICY_SOURCES.keys()
    for name, passage_id in policy.POLICY_SOURCES.items():
        assert _expected_text(name) in by_id[passage_id], (name, passage_id)


def test_no_third_party_brand():
    files = [p for p in CORPUS.rglob("*") if p.is_file() and (CORPUS / "raw") not in p.parents]
    assert files
    for path in files:
        assert "ifood" not in _normalize(path.read_text(encoding="utf-8")), path
