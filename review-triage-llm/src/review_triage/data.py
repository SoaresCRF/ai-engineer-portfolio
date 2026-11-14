"""Load and clean Olist reviews, and build the human-labeled (gold) sample."""

import re
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from review_triage.schema import LABEL_FIELDS, Triage

DATASET_HANDLE = "olistbr/brazilian-ecommerce"
REVIEWS_FILE = "olist_order_reviews_dataset.csv"
ORDERS_FILE = "olist_orders_dataset.csv"

SEED = 42
SCORES = (1, 2, 3, 4, 5)
# Equal allocation per star score: proportional sampling would make ~half the gold set
# 5-star praise and leave too few complaints to measure the rarer categories.
GOLD_DEV_PER_SCORE = 20
GOLD_TEST_PER_SCORE = 40
# 5 scores x 1,000 = 5,000 silver reviews; fits one Kaggle T4 session (see PLAN.md 3.3).
SILVER_PER_SCORE = 1_000

# Order matters: CPF must be masked before the looser phone pattern eats it.
PII_PATTERNS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b"), "[TELEFONE]"),
)

_WHITESPACE = re.compile(r"\s+")


def download_raw(raw_dir: Path) -> Path:
    """Download the public Olist dataset from Kaggle (no credentials needed)."""
    import kagglehub  # imported lazily: only needed for this one-off step

    return Path(kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(raw_dir)))


def mask_pii(text: str) -> str:
    for pattern, token in PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def build_text(title: str | float, message: str | float) -> str:
    """Join review title and message; either may be missing (NaN)."""
    parts = [normalize_text(p) for p in (title, message) if isinstance(p, str)]
    return ". ".join(p for p in parts if p)


def load_reviews(raw_dir: Path) -> pd.DataFrame:
    """One row per unique review text, with delivery facts used for sanity checks."""
    reviews = pd.read_csv(raw_dir / REVIEWS_FILE)
    orders = pd.read_csv(
        raw_dir / ORDERS_FILE,
        parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"],
    )

    reviews = reviews[reviews["review_comment_message"].notna()].copy()
    reviews["text"] = [
        mask_pii(build_text(t, m))
        for t, m in zip(
            reviews["review_comment_title"], reviews["review_comment_message"], strict=True
        )
    ]
    reviews = reviews[reviews["text"] != ""]
    reviews = reviews.drop_duplicates(subset="text").drop_duplicates(subset="review_id")

    orders["delivered"] = orders["order_delivered_customer_date"].notna()
    orders["late"] = orders["delivered"] & (
        orders["order_delivered_customer_date"].dt.normalize()
        > orders["order_estimated_delivery_date"]
    )
    df = reviews.merge(orders[["order_id", "delivered", "late"]], on="order_id", how="left")
    return df[["review_id", "order_id", "review_score", "text", "delivered", "late"]].reset_index(
        drop=True
    )


def sample_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified-by-score sample for blind human labeling, with a fixed dev/test split."""
    per_score = GOLD_DEV_PER_SCORE + GOLD_TEST_PER_SCORE
    parts = []
    for score in SCORES:
        picked = df[df["review_score"] == score].sample(n=per_score, random_state=SEED)
        picked = picked.assign(split=["dev"] * GOLD_DEV_PER_SCORE + ["test"] * GOLD_TEST_PER_SCORE)
        parts.append(picked)
    # Shuffle so the labeler can't infer the star score from position.
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def sample_silver(
    df: pd.DataFrame, exclude_ids: set[str], per_score: int = SILVER_PER_SCORE
) -> pd.DataFrame:
    """Training pool for LLM (silver) labels, balanced by score like the gold set."""
    pool = df[~df["review_id"].isin(exclude_ids)]
    parts = [
        pool[pool["review_score"] == score].sample(n=per_score, random_state=SEED)
        for score in SCORES
    ]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate labeled rows; raise listing every invalid row."""
    df = df.copy()
    for field in LABEL_FIELDS:
        df[field] = df[field].astype("string").str.strip().str.lower()

    errors = []
    for row in df.itertuples():
        try:
            Triage(**{f: getattr(row, f) for f in LABEL_FIELDS})
        except ValidationError as exc:
            errors.append(f"review_id={row.review_id}: {exc.errors()[0]['msg']}")
    if errors:
        raise ValueError(f"{len(errors)} invalid labeled rows:\n" + "\n".join(errors))
    return df
