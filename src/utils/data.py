import math
import os
import pickle
import re

import pandas as pd

DATA_PATH = "data/mobile_reviews.csv"

SENTENCE_EMBEDDING_CACHE_PATH = "data/sentence_embeddings.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers – model metadata
# ─────────────────────────────────────────────────────────────────────────────

def _extract_model_year_hint(model: str):
    if not isinstance(model, str):
        return None
    m = re.search(r"\b(20\d{2})\b", model)
    if not m:
        return None
    year = int(m.group(1))
    return year if 2000 <= year <= 2035 else None


def _extract_model_gen_hint(model: str):
    if not isinstance(model, str):
        return None
    nums = [int(x) for x in re.findall(r"\b(\d{1,3})[a-zA-Z]?\b", model)]
    if not nums:
        return None
    non_years = [n for n in nums if n < 100]
    if non_years:
        return max(non_years)
    years = [n for n in nums if 2000 <= n <= 2035]
    if years:
        return max(years) - 2000
    return max(nums)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers – text formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_score(x):
    try:
        return "unknown" if pd.isna(x) else f"{float(x):.1f}/5"
    except Exception:
        return "unknown"


def _fmt_price(x):
    try:
        return "unknown" if pd.isna(x) else f"${float(x):.0f}"
    except Exception:
        return "unknown"


def _build_product_description_text(row) -> str:
    brand = str(row.get("brand") or "").strip()
    model = str(row.get("model") or "").strip()
    bits = [
        f"{brand} {model}".strip(),
        f"price {_fmt_price(row.get('price_usd'))}",
        f"overall rating {_fmt_score(row.get('rating'))}",
        f"battery {_fmt_score(row.get('battery_life_rating'))}",
        f"camera {_fmt_score(row.get('camera_rating'))}",
        f"performance {_fmt_score(row.get('performance_rating'))}",
        f"design {_fmt_score(row.get('design_rating'))}",
        f"display {_fmt_score(row.get('display_rating'))}",
    ]
    gen = row.get("model_gen_hint")
    if pd.notna(gen):
        bits.append(f"generation {int(gen)}")
    latest_review_date = row.get("latest_review_date")
    if pd.notna(latest_review_date):
        try:
            bits.append(
                f"latest review {pd.to_datetime(latest_review_date).date().isoformat()}"
            )
        except Exception:
            pass
    reviews = row.get("review_count")
    if pd.notna(reviews):
        try:
            bits.append(f"{int(reviews)} reviews")
        except Exception:
            pass
    return ". ".join([b for b in bits if b and b != "unknown"])


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    """Split a review text into individual sentences."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 10]


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity (no extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors. Returns value in [-1, 1]."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level embedding precomputation & cache
# ─────────────────────────────────────────────────────────────────────────────

def _load_sentence_cache() -> dict:
    """Load { product_id: [{"sentence": str, "embedding": list}] } from disk."""
    if os.path.exists(SENTENCE_EMBEDDING_CACHE_PATH):
        try:
            with open(SENTENCE_EMBEDDING_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_sentence_cache(cache: dict):
    os.makedirs(os.path.dirname(SENTENCE_EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(SENTENCE_EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def precompute_sentence_embeddings(
    product_df: pd.DataFrame,
    providers: str = "openai",
    force_recompute: bool = False,
    progress_callback=None,
) -> dict:
    from .edenai import eden_embed

    def _log(msg):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    cache = {} if force_recompute else _load_sentence_cache()

    all_ids = list(product_df["product_id"].astype(str))
    missing_ids = [pid for pid in all_ids if pid not in cache]

    if not missing_ids:
        _log(f"Cache up to date ({len(cache)} products).")
        return cache

    _log(f"Computing embeddings for {len(missing_ids)} products...")

    id_to_row = {
        str(row["product_id"]): row
        for _, row in product_df[
            product_df["product_id"].astype(str).isin(missing_ids)
        ].iterrows()
    }

    SAVE_EVERY = 20

    for i, pid in enumerate(missing_ids):
        row = id_to_row[pid]
        reviews = row.get("review_text")
        if not isinstance(reviews, list):
            reviews = [str(reviews)] if reviews is not None else []

        pid_sentences = []
        for review in reviews[:5]:
            for sentence in split_sentences(str(review)):
                pid_sentences.append(sentence)

        if not pid_sentences:
            cache[pid] = []
            continue

        try:
            embeddings = eden_embed(pid_sentences, providers=providers)
            cache[pid] = [
                {"sentence": s, "embedding": e}
                for s, e in zip(pid_sentences, embeddings)
            ]
        except Exception as e:
            _log(f"Error on {pid}: {e}")
            continue

        if i % SAVE_EVERY == 0:
            _save_sentence_cache(cache)
            _log(f"Progress: {i}/{len(missing_ids)} products done")

    _save_sentence_cache(cache)
    _log(f"Done. {len(cache)} products cached.")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Main data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_product_data():
    """
    Load and aggregate the raw reviews CSV into a per-product DataFrame.

    Returns
    -------
    product_df : pd.DataFrame  one row per product, aggregated stats +
                               list of review texts
    raw_df     : pd.DataFrame  original row-per-review DataFrame
    """
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["brand", "model", "price_usd", "review_text"])
    df["review_date"] = pd.to_datetime(df.get("review_date"), errors="coerce")
    df["model_year_hint"] = df["model"].map(_extract_model_year_hint)
    df["model_gen_hint"] = df["model"].map(_extract_model_gen_hint)
    df["product_id"] = df["brand"].astype(str) + "_" + df["model"].astype(str)

    product_df = (
        df.groupby("product_id")
        .agg(
            {
                "brand": "first",
                "model": "first",
                "price_usd": "mean",
                "rating": "mean",
                "battery_life_rating": "mean",
                "camera_rating": "mean",
                "performance_rating": "mean",
                "design_rating": "mean",
                "display_rating": "mean",
                "review_text": list,
                "review_date": "max",
                "model_year_hint": "max",
                "model_gen_hint": "max",
                "review_id": "count",
            }
        )
        .reset_index()
    )

    product_df = product_df.rename(
        columns={
            "review_date": "latest_review_date",
            "review_id": "review_count",
        }
    )
    product_df["product_description_text"] = product_df.apply(
        _build_product_description_text, axis=1
    )

    return product_df, df