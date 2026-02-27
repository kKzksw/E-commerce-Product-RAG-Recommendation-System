import math
import re
from typing import Dict, List, Tuple

import pandas as pd

from .structured import structured_retrieval
from .review_insights import summarize_review_insights
from ..utils.data import cosine_similarity, split_sentences


STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "for", "to", "of", "on", "in", "with", "and", "or",
    "is", "are", "am", "be", "need", "want", "looking", "phone", "smartphone", "mobile",
    "that", "this", "it", "good", "best", "recommend", "show", "find", "please", "under",
    "around", "about",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _tokens(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    toks = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)]
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def _token_set(text: str) -> set:
    return set(_tokens(text))


def _query_token_set(query: str, routing: Dict) -> set:
    """Build keyword token set used for description scoring and fallback."""
    q = set(_tokens(query))
    for key in ("brands", "models", "spec_focus"):
        for item in _safe_list((routing or {}).get(key)):
            q.update(_tokens(str(item)))

    spec_focus = [str(x).lower() for x in _safe_list((routing or {}).get("spec_focus"))]
    synonym_map = {
        "battery": {"battery", "charging", "backup"},
        "camera": {"camera", "photo", "photos", "video", "portrait"},
        "performance": {"performance", "gaming", "speed", "lag", "processor"},
        "display": {"display", "screen", "brightness", "refresh"},
        "design": {"design", "build", "weight", "feel", "premium"},
    }
    for s in spec_focus:
        q.update(synonym_map.get(s, {s}))
    return q


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level semantic review scoring (primary method)
# ─────────────────────────────────────────────────────────────────────────────

def _score_reviews_semantic(
    query_embedding: list,
    product_id: str,
    sentence_cache: dict,
    top_snippets: int = 3,
    similarity_threshold: float = 0.5,
) -> Dict:
    """
    Score a product's review relevance using sentence-level embeddings.

    For each candidate product, we retrieve its pre-computed sentence
    embeddings from the cache and compute cosine similarity against the
    query embedding. The top-scoring sentences become the evidence snippets
    shown in the UI.

    Parameters
    ----------
    query_embedding      : embedding vector for the user query (1 API call)
    product_id           : key into sentence_cache
    sentence_cache       : { product_id: [{"sentence": str, "embedding": list}] }
    top_snippets         : how many top sentences to keep as evidence
    similarity_threshold : minimum cosine similarity to count as a match

    Returns None if the product is not in the cache (triggers keyword fallback).
    """
    entries = sentence_cache.get(str(product_id))
    if not entries or not query_embedding:
        return None  # triggers fallback

    scored = []
    for entry in entries:
        sim = cosine_similarity(query_embedding, entry["embedding"])
        if sim >= similarity_threshold:
            scored.append((sim, entry["sentence"]))

    if not scored:
        return {
            "review_relevance_score": 0.0,
            "review_match_count": 0,
            "review_retrieval_snippets": [],
            "review_matched_terms": [],
            "review_sentences_scanned": len(entries),
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    top_scores = [s for s, _ in scored[:top_snippets]]
    top_evidence = [s for _, s in scored[:top_snippets]]

    avg_sim = sum(top_scores) / len(top_scores)
    volume_bonus = min(1.0, len(scored) / 8.0)
    review_score = float(min(1.0, 0.7 * avg_sim + 0.3 * volume_bonus))

    return {
        "review_relevance_score": review_score,
        "review_match_count": len(scored),
        "review_retrieval_snippets": top_evidence,
        "review_matched_terms": [],   # not applicable for semantic scoring
        "review_sentences_scanned": len(entries),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Keyword review scoring (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _overlap_score(text: str, q_tokens: set) -> Tuple[float, List[str]]:
    if not q_tokens:
        return 0.0, []
    tset = _token_set(text)
    if not tset:
        return 0.0, []
    overlap = sorted(list(tset & q_tokens))
    if not overlap:
        return 0.0, []
    coverage = len(overlap) / max(1, len(q_tokens))
    density = len(overlap) / max(4.0, math.sqrt(len(tset)))
    score = 0.7 * coverage + 0.3 * min(1.0, density)
    return float(score), overlap


def _sample_reviews(reviews: List[str], max_reviews: int = 200) -> List[str]:
    if len(reviews) <= max_reviews:
        return reviews
    idxs = sorted(
        {int(i * (len(reviews) - 1) / (max_reviews - 1)) for i in range(max_reviews)}
    )
    return [reviews[i] for i in idxs]


def _score_reviews_keyword(
    query_tokens: set,
    reviews,
    max_reviews_scan: int = 200,
    top_snippets: int = 3,
) -> Dict:
    """Keyword-overlap fallback scorer (used when sentence cache is unavailable)."""
    reviews_list = [str(r) for r in reviews if isinstance(r, str) and r.strip()]
    if not reviews_list:
        return {
            "review_relevance_score": 0.0,
            "review_match_count": 0,
            "review_retrieval_snippets": [],
            "review_matched_terms": [],
            "review_sentences_scanned": 0,
        }

    sampled = _sample_reviews(reviews_list, max_reviews=max_reviews_scan)
    scored = []
    matched_terms = set()
    sentence_count = 0
    for r in sampled:
        for s in split_sentences(r):
            sentence_count += 1
            score, terms = _overlap_score(s, query_tokens)
            if score <= 0:
                continue
            matched_terms.update(terms)
            scored.append((score, s))

    if not scored:
        return {
            "review_relevance_score": 0.0,
            "review_match_count": 0,
            "review_retrieval_snippets": [],
            "review_matched_terms": [],
            "review_sentences_scanned": sentence_count,
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    top_scores = [s for s, _ in scored[:top_snippets]]
    top_evidence = [s for _, s in scored[:top_snippets]]

    avg_top = sum(top_scores) / max(1, len(top_scores))
    top1 = top_scores[0]
    volume_bonus = min(1.0, len(scored) / 8.0)
    review_score = float(min(1.0, 0.55 * top1 + 0.30 * avg_top + 0.15 * volume_bonus))

    return {
        "review_relevance_score": review_score,
        "review_match_count": int(len(scored)),
        "review_retrieval_snippets": [
            s if len(s) <= 220 else s[:217].rstrip() + "..." for s in top_evidence
        ],
        "review_matched_terms": sorted(list(matched_terms))[:12],
        "review_sentences_scanned": int(sentence_count),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Description scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_description(query_tokens: set, description_text: str) -> Dict:
    score, terms = _overlap_score(description_text or "", query_tokens)
    return {
        "description_relevance_score": float(min(1.0, score)),
        "description_matched_terms": terms[:12],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Structured signal
# ─────────────────────────────────────────────────────────────────────────────

def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(0.0, index=series.index, dtype="float64")
    mask = s.notna()
    if mask.sum() == 0:
        return out
    out.loc[mask] = s.loc[mask].rank(method="average", pct=True, ascending=ascending)
    return out


def _compute_structured_signal(df: pd.DataFrame, routing: Dict) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if n == 0:
        out["structured_signal_score"] = pd.Series(dtype="float64")
        out["structured_rank_score"] = pd.Series(dtype="float64")
        return out

    out["structured_rank_score"] = [1.0 - (i / max(1, n - 1)) for i in range(n)]
    out["rating_rank_score"] = _rank_pct(
        out.get("rating", pd.Series(index=out.index, dtype="float64")), ascending=True
    )
    out["price_value_score"] = _rank_pct(
        out.get("price_usd", pd.Series(index=out.index, dtype="float64")), ascending=False
    )
    out["spec_rank_score"] = _rank_pct(
        out.get("spec_score", pd.Series(index=out.index, dtype="float64")), ascending=True
    )
    out["freshness_rank_score"] = _rank_pct(
        out.get("freshness_score", pd.Series(index=out.index, dtype="float64")), ascending=True
    )

    spec_focus = [str(x).lower() for x in _safe_list((routing or {}).get("spec_focus"))]
    if spec_focus:
        out["structured_signal_score"] = (
            0.40 * out["spec_rank_score"]
            + 0.25 * out["rating_rank_score"]
            + 0.15 * out["price_value_score"]
            + 0.10 * out["freshness_rank_score"]
            + 0.10 * out["structured_rank_score"]
        )
    else:
        out["structured_signal_score"] = (
            0.40 * out["rating_rank_score"]
            + 0.20 * out["price_value_score"]
            + 0.20 * out["freshness_rank_score"]
            + 0.20 * out["structured_rank_score"]
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main retrieval function
# ─────────────────────────────────────────────────────────────────────────────

def multi_source_retrieval(
    product_df: pd.DataFrame,
    routing: Dict,
    query: str,
    top_k: int = 5,
    candidate_k: int = 40,
    providers: str = "openai",
    sentence_cache: dict = None,
) -> pd.DataFrame:
    """
    Multi-source retrieval combining:
      1. Structured filtering/ranking   (hard constraints: budget, brand, specs)
      2. Sentence-level semantic search (embedding cosine similarity — primary)
      3. Description relevance          (keyword overlap — lightweight fallback)

    Parameters
    ----------
    product_df     : aggregated product DataFrame from load_product_data()
    routing        : parsed query intent dict from llm_route()
    query          : raw user query string
    top_k          : number of final products to return
    candidate_k    : size of candidate pool after structured filtering
    providers      : EdenAI provider string (for query embedding)
    sentence_cache : { product_id: [{"sentence": str, "embedding": list}] }
                     from precompute_sentence_embeddings().
                     If None/empty, falls back to keyword matching.

    Returns
    -------
    pd.DataFrame with fused score columns and review evidence snippets.
    """
    if product_df is None or len(product_df) == 0:
        return pd.DataFrame()

    routing = routing or {}
    sentence_cache = sentence_cache or {}
    candidate_k = min(max(candidate_k, top_k), len(product_df))

    # structured retrieval narrows the search space
    candidates = structured_retrieval(product_df, routing, top_k=candidate_k)
    if candidates.empty:
        return candidates

    candidates = _compute_structured_signal(candidates, routing)

    # embed the query 
    query_embedding = None
    use_semantic = bool(sentence_cache)

    if use_semantic:
        try:
            from ..utils.edenai import eden_embed
            query_embedding = eden_embed([query], providers=providers)[0]
        except Exception as e:
            print(f"Warning: query embedding failed, using keyword fallback. Error: {e}")
            use_semantic = False

    # score each candidate
    q_tokens = _query_token_set(query, routing)

    review_scores = []
    desc_scores = []

    for _, row in candidates.iterrows():
        pid = str(row.get("product_id", ""))

        if use_semantic and query_embedding:
            review_info = _score_reviews_semantic(
                query_embedding, pid, sentence_cache
            )
            if review_info is None:
                reviews = row.get("review_text")
                if not isinstance(reviews, list):
                    reviews = [str(reviews)] if reviews is not None else []
                review_info = _score_reviews_keyword(q_tokens, reviews)
        else:
            reviews = row.get("review_text")
            if not isinstance(reviews, list):
                reviews = [str(reviews)] if reviews is not None else []
            review_info = _score_reviews_keyword(q_tokens, reviews)

        review_scores.append(review_info)

        desc_text = row.get("product_description_text") or ""
        if not desc_text:
            desc_text = f"{row.get('brand', '')} {row.get('model', '')}"
        desc_scores.append(_score_description(q_tokens, desc_text))

    review_df = pd.DataFrame(review_scores, index=candidates.index)
    desc_df = pd.DataFrame(desc_scores, index=candidates.index)
    fused = pd.concat([candidates, review_df, desc_df], axis=1)

    # dynamic weight fusion
    has_review = float(fused["review_relevance_score"].sum()) > 0.0
    has_desc = float(fused["description_relevance_score"].sum()) > 0.0

    if has_review and has_desc:
        w_struct, w_review, w_desc = 0.55, 0.30, 0.15
    elif has_review and not has_desc:
        w_struct, w_review, w_desc = 0.65, 0.35, 0.00
    elif not has_review and has_desc:
        w_struct, w_review, w_desc = 0.70, 0.00, 0.30
    else:
        w_struct, w_review, w_desc = 1.00, 0.00, 0.00

    fused["multi_source_score"] = (
        w_struct * fused["structured_signal_score"].fillna(0.0)
        + w_review * fused["review_relevance_score"].fillna(0.0)
        + w_desc * fused["description_relevance_score"].fillna(0.0)
    )
    fused["retrieval_weight_structured"] = w_struct
    fused["retrieval_weight_reviews"] = w_review
    fused["retrieval_weight_description"] = w_desc

    fused["retrieval_debug"] = fused.apply(
        lambda r: (
            f"ms={float(r.get('multi_source_score', 0)):.3f} | "
            f"struct={float(r.get('structured_signal_score', 0)):.3f} | "
            f"reviews={float(r.get('review_relevance_score', 0)):.3f} | "
            f"desc={float(r.get('description_relevance_score', 0)):.3f} | "
            f"method={'semantic' if use_semantic else 'keyword'}"
        ),
        axis=1,
    )

    sort_cols = ["multi_source_score", "rating", "price_usd"]
    fused = fused.sort_values(sort_cols, ascending=[False, False, True])
    top = fused.head(top_k).copy()

    # LLM-generated review insights for final results
    insight_rows = []
    for _, row in top.iterrows():
        reviews = row.get("review_text")
        if not isinstance(reviews, list):
            reviews = [str(reviews)] if reviews is not None else []
        insight_rows.append(
            summarize_review_insights(
                reviews,
                product_name=str(row.get("model", "")),
                providers=providers,
            )
        )

    if insight_rows:
        insight_df = pd.DataFrame(insight_rows, index=top.index)
        top = pd.concat([top, insight_df], axis=1)

    return top
