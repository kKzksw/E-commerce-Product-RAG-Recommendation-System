import math
import re
from typing import Dict, List, Optional

import pandas as pd

from ..retriever.structured import SPEC_MAP


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _product_label(row) -> str:
    brand = str(row.get("brand") or "").strip()
    model = str(row.get("model") or "").strip()
    return f"{brand} {model}".strip()


def _to_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _rank_pct(values: pd.Series, higher_better: bool = True) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    out = pd.Series(0.0, index=values.index, dtype="float64")
    mask = s.notna()
    if mask.sum() == 0:
        return out
    # We want larger normalized score = better score contribution.
    # Pandas rank(pct=True) yields larger pct for larger values when ascending=True.
    out.loc[mask] = s.loc[mask].rank(method="average", pct=True, ascending=higher_better)
    return out


def _parse_model_targets(routing: Dict) -> List[str]:
    models = []
    for m in _safe_list((routing or {}).get("models")):
        if not isinstance(m, str):
            continue
        m = m.strip()
        if not m:
            continue
        ml = m.lower()
        if ml in {"new model", "latest model", "new phone", "latest phone"}:
            continue
        models.append(m)
    return models


def _match_compare_candidates(top_df: pd.DataFrame, routing: Dict, max_items: int = 3) -> pd.DataFrame:
    if top_df is None or top_df.empty:
        return top_df
    cand = top_df.copy()

    model_targets = _parse_model_targets(routing)
    if model_targets:
        pattern = "|".join(re.escape(m) for m in model_targets if m)
        matched = cand[cand["model"].astype(str).str.contains(pattern, case=False, regex=True)]
        if len(matched) >= 2:
            # Preserve original ranking order.
            return matched.head(max_items)
        # Fallback token matching for noisy compare strings like "sumsung s22".
        mtoks = []
        for m in model_targets:
            for t in re.findall(r"[a-zA-Z0-9]+", str(m or "").lower()):
                if any(ch.isdigit() for ch in t) or len(t) >= 4:
                    mtoks.append(t)
        mtoks = list(dict.fromkeys(mtoks))
        if mtoks:
            token_pattern = "|".join(re.escape(t) for t in mtoks)
            token_matched = cand[cand["model"].astype(str).str.contains(token_pattern, case=False, regex=True)]
            merged = pd.concat([matched, token_matched], axis=0)
            merged = merged.loc[~merged.index.duplicated(keep="first")]
            if len(merged) >= 2:
                return merged.head(max_items)

    # If no clear model targets, choose top items but prefer diversity of brands.
    chosen_rows = []
    seen_brands = set()
    for idx, row in cand.iterrows():
        brand = str(row.get("brand") or "").lower()
        if brand and brand not in seen_brands:
            chosen_rows.append(idx)
            seen_brands.add(brand)
        if len(chosen_rows) >= max_items:
            break

    if len(chosen_rows) >= 2:
        return cand.loc[chosen_rows]
    return cand.head(max_items)


def _complaint_mentions_total(row) -> int:
    items = row.get("common_complaints") or []
    if not isinstance(items, list):
        return 0
    total = 0
    for item in items:
        if isinstance(item, dict):
            try:
                total += int(item.get("mentions") or 0)
            except Exception:
                pass
    return total


def _top_focus_aspects(routing: Dict) -> List[str]:
    aspects = [str(x).lower() for x in _safe_list((routing or {}).get("spec_focus"))]
    aspects = [a for a in aspects if a in SPEC_MAP]
    return aspects


def _query_signals(query: str, routing: Dict) -> Dict[str, bool]:
    q = (query or "").lower()
    spec_focus = _top_focus_aspects(routing)
    return {
        "has_spec_focus": bool(spec_focus),
        "price_sensitive": any(k in q for k in ["budget", "under ", "cheap", "cheaper", "price", "cost", "value", "worth"]),
        "review_or_reliability_sensitive": any(
            k in q for k in ["review", "reviews", "reliable", "reliability", "issue", "issues", "problem", "problems", "complaint", "complaints"]
        ),
        "thermal_sensitive": any(k in q for k in ["overheat", "heating", "hot"]),
        "freshness_sensitive": str((routing or {}).get("freshness_pref") or "").lower() == "latest"
        or any(k in q for k in ["latest", "newest", "new model", "recent", "current gen"]),
        "camera_sensitive": ("camera" in spec_focus) or ("camera" in q),
        "performance_sensitive": ("performance" in spec_focus) or any(k in q for k in ["performance", "gaming", "speed", "lag"]),
        "battery_sensitive": ("battery" in spec_focus) or ("battery" in q),
    }


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clamped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clamped.values())
    if total <= 0:
        # Safe fallback
        n = max(1, len(clamped))
        return {k: 1.0 / n for k in clamped}
    return {k: v / total for k, v in clamped.items()}


def _dynamic_compare_weights(routing: Dict, query: str) -> Dict:
    spec_focus = _top_focus_aspects(routing)
    signals = _query_signals(query, routing)

    weights = {
        "spec_focus": 0.18 if spec_focus else 0.05,
        "general_specs": 0.10 if spec_focus else 0.18,
        "rating": 0.18,
        "price": 0.16,
        "review_relevance": 0.14,
        "description": 0.06,
        "retrieval_match": 0.08,
        "complaint_safety": 0.07,
        "freshness": 0.03,
    }
    adjustments = []

    if spec_focus:
        # Stronger emphasis as more explicit specs are requested.
        delta = min(0.22, 0.08 + 0.06 * len(spec_focus))
        weights["spec_focus"] += delta
        weights["general_specs"] -= min(0.06, delta * 0.30)
        weights["retrieval_match"] -= min(0.04, delta * 0.18)
        weights["description"] -= min(0.03, delta * 0.14)
        weights["rating"] -= min(0.03, delta * 0.14)
        adjustments.append(f"spec_focus(+{delta:.2f}) from explicit aspects {spec_focus}")

    if signals["price_sensitive"]:
        delta = 0.10
        weights["price"] += delta
        weights["retrieval_match"] -= 0.03
        weights["description"] -= 0.02
        weights["rating"] -= 0.02
        weights["general_specs"] -= 0.03
        adjustments.append("price(+0.10) due to budget/value wording")

    if signals["review_or_reliability_sensitive"]:
        delta_r, delta_c = 0.08, 0.05
        weights["review_relevance"] += delta_r
        weights["complaint_safety"] += delta_c
        weights["retrieval_match"] -= 0.03
        weights["description"] -= 0.02
        weights["price"] -= 0.03
        weights["rating"] -= 0.03
        adjustments.append("reviews/complaint weights increased due to reliability/review wording")

    if signals["thermal_sensitive"]:
        weights["complaint_safety"] += 0.05
        weights["review_relevance"] += 0.03
        weights["description"] -= 0.02
        weights["retrieval_match"] -= 0.02
        weights["rating"] -= 0.02
        weights["general_specs"] -= 0.02
        adjustments.append("complaint_safety boosted for thermal concern")

    if signals["freshness_sensitive"]:
        delta = 0.10
        weights["freshness"] += delta
        weights["retrieval_match"] -= 0.03
        weights["description"] -= 0.02
        weights["rating"] -= 0.02
        weights["price"] -= 0.03
        adjustments.append("freshness(+0.10) due to latest/new request")

    # If query explicitly compares camera/performance, bias to focused specs a bit more.
    if signals["camera_sensitive"] and signals["performance_sensitive"]:
        weights["spec_focus"] += 0.05
        weights["retrieval_match"] -= 0.02
        weights["description"] -= 0.01
        weights["rating"] -= 0.02
        adjustments.append("spec_focus(+0.05) due to camera+performance comparison")

    weights = _normalize_weights(weights)
    return {
        "weights": weights,
        "spec_focus": spec_focus,
        "signals": signals,
        "adjustments": adjustments,
    }


def _compute_compare_component_scores(df: pd.DataFrame, routing: Dict) -> Dict[str, pd.Series]:
    empty = pd.Series(index=df.index, dtype="float64")
    rating_rank = _rank_pct(df.get("rating", empty), higher_better=True)
    price_rank = _rank_pct(df.get("price_usd", empty), higher_better=False)
    review_rank = _rank_pct(df.get("review_relevance_score", empty), higher_better=True)
    desc_rank = _rank_pct(df.get("description_relevance_score", empty), higher_better=True)
    ms_rank = _rank_pct(df.get("multi_source_score", empty), higher_better=True)
    complaint_rank = _rank_pct(df.get("complaint_mentions_total", empty), higher_better=False)
    freshness_rank = _rank_pct(df.get("freshness_score", empty), higher_better=True)

    all_spec_cols = [c for c in SPEC_MAP.values() if c in df.columns]
    if all_spec_cols:
        general_spec_mean = df[all_spec_cols].mean(axis=1)
        general_spec_rank = _rank_pct(general_spec_mean, higher_better=True)
    else:
        general_spec_rank = pd.Series(0.0, index=df.index, dtype="float64")

    spec_focus = _top_focus_aspects(routing)
    focus_cols = [SPEC_MAP[a] for a in spec_focus if SPEC_MAP.get(a) in df.columns]
    if focus_cols:
        focus_spec_mean = df[focus_cols].mean(axis=1)
        spec_focus_rank = _rank_pct(focus_spec_mean, higher_better=True)
    else:
        spec_focus_rank = pd.Series(0.0, index=df.index, dtype="float64")

    return {
        "spec_focus": spec_focus_rank,
        "general_specs": general_spec_rank,
        "rating": rating_rank,
        "price": price_rank,
        "review_relevance": review_rank,
        "description": desc_rank,
        "retrieval_match": ms_rank,
        "complaint_safety": complaint_rank,
        "freshness": freshness_rank,
    }


def _value_score(df: pd.DataFrame, routing: Dict, query: str = ""):
    weight_cfg = _dynamic_compare_weights(routing, query)
    weights = weight_cfg["weights"]
    components = _compute_compare_component_scores(df, routing)

    total = pd.Series(0.0, index=df.index, dtype="float64")
    for k, series in components.items():
        total = total + weights.get(k, 0.0) * series.fillna(0.0)

    # Flatten component and contribution columns for UI/debug.
    component_cols = {}
    for k, series in components.items():
        component_cols[f"compare_component_{k}"] = series.fillna(0.0)
        component_cols[f"compare_contrib_{k}"] = series.fillna(0.0) * float(weights.get(k, 0.0))

    return total, weight_cfg, component_cols


def _winner_for_metric(df: pd.DataFrame, column: str, higher_better: bool = True) -> Optional[str]:
    vals = pd.to_numeric(df[column], errors="coerce")
    vals = vals.dropna()
    if len(vals) == 0:
        return None
    best_val = vals.max() if higher_better else vals.min()
    winners = vals[vals == best_val].index.tolist()
    if len(winners) != 1:
        return "tie"
    return str(df.loc[winners[0], "compare_label"])


def _metric_row(df: pd.DataFrame, column: str, label: str, higher_better: bool = True) -> Dict:
    row = {
        "metric": label,
        "field": column,
        "direction": "higher_better" if higher_better else "lower_better",
        "winner": _winner_for_metric(df, column, higher_better=higher_better),
    }
    for idx, r in df.iterrows():
        row[str(r["compare_label"])] = r.get(column)
    return row


def build_structured_comparison(top_df: pd.DataFrame, routing: Dict, query: str = "", max_items: int = 3) -> Dict:
    """
    Deterministic comparison tool output for COMPARE intent.
    Consumes retrieved candidates (already enriched with review summaries/scores).
    """
    if top_df is None or top_df.empty:
        return {"selected_products": [], "comparison_rows": [], "score_breakdown": [], "final_pick": None}

    selected = _match_compare_candidates(top_df, routing, max_items=max_items).copy()
    if selected.empty:
        return {"selected_products": [], "comparison_rows": [], "score_breakdown": [], "final_pick": None}

    selected = selected.reset_index(drop=True)
    selected["compare_label"] = selected.apply(_product_label, axis=1)
    selected["complaint_mentions_total"] = selected.apply(_complaint_mentions_total, axis=1)

    # Deterministic compare score (dynamic weights, distinct from retrieval score).
    compare_scores, weight_cfg, component_cols = _value_score(selected, routing, query=query)
    selected["compare_total_score"] = compare_scores.fillna(0.0)
    selected["compare_total_score"] = selected["compare_total_score"].astype(float)
    for col, series in component_cols.items():
        selected[col] = series.astype(float)

    # Metrics to show.
    focus_aspects = _top_focus_aspects(routing)
    metric_defs = [
        ("price_usd", "Price (USD)", False),
        ("rating", "Overall Rating", True),
    ]
    for aspect in focus_aspects:
        col = SPEC_MAP.get(aspect)
        if col and col in selected.columns:
            metric_defs.append((col, f"{aspect.title()} Rating", True))

    # If no focus aspects, show all core specs.
    if not focus_aspects:
        for aspect, col in SPEC_MAP.items():
            if col in selected.columns:
                metric_defs.append((col, f"{aspect.title()} Rating", True))

    extra_metric_defs = [
        ("multi_source_score", "Retrieved Match Score", True),
        ("review_relevance_score", "Review Relevance Score", True),
        ("description_relevance_score", "Description Relevance Score", True),
        ("complaint_mentions_total", "Complaint Mentions (Top Issues)", False),
    ]
    if "freshness_score" in selected.columns and str((routing or {}).get("freshness_pref") or "").lower() == "latest":
        extra_metric_defs.append(("freshness_score", "Freshness Score", True))

    for col, lbl, hb in extra_metric_defs:
        if col in selected.columns:
            metric_defs.append((col, lbl, hb))

    comparison_rows = []
    for col, label, higher_better in metric_defs:
        comparison_rows.append(_metric_row(selected, col, label, higher_better))

    # Score breakdown and pick.
    score_breakdown = []
    for _, r in selected.sort_values(["compare_total_score", "rating", "price_usd"], ascending=[False, False, True]).iterrows():
        score_breakdown.append(
            {
                "product": str(r["compare_label"]),
                "compare_total_score": float(r.get("compare_total_score") or 0.0),
                "price_usd": _to_float(r.get("price_usd")),
                "rating": _to_float(r.get("rating")),
                "multi_source_score": _to_float(r.get("multi_source_score")),
                "review_relevance_score": _to_float(r.get("review_relevance_score")),
                "description_relevance_score": _to_float(r.get("description_relevance_score")),
                "complaint_mentions_total": int(r.get("complaint_mentions_total") or 0),
                "compare_component_spec_focus": _to_float(r.get("compare_component_spec_focus")),
                "compare_component_price": _to_float(r.get("compare_component_price")),
                "compare_component_rating": _to_float(r.get("compare_component_rating")),
                "compare_component_review_relevance": _to_float(r.get("compare_component_review_relevance")),
                "compare_component_complaint_safety": _to_float(r.get("compare_component_complaint_safety")),
                "compare_component_freshness": _to_float(r.get("compare_component_freshness")),
            }
        )

    final_pick = score_breakdown[0]["product"] if score_breakdown else None

    # Reasons for the winner based on metric wins.
    metric_wins = {}
    for row in comparison_rows:
        w = row.get("winner")
        if not w or w == "tie":
            continue
        metric_wins[w] = metric_wins.get(w, 0) + 1

    winner_reasons = []
    if final_pick:
        metric_reason_candidates = []
        for row in comparison_rows:
            if row.get("winner") == final_pick:
                metric_reason_candidates.append(str(row.get("metric")))
        # Also surface top weighted score contributors for transparency.
        winner_row = selected[selected["compare_label"] == final_pick]
        contribution_reasons = []
        if not winner_row.empty:
            wr = winner_row.iloc[0]
            contribs = []
            for k, w in (weight_cfg.get("weights") or {}).items():
                contribs.append((float(wr.get(f"compare_contrib_{k}", 0.0) or 0.0), k, float(w)))
            contribs.sort(key=lambda x: x[0], reverse=True)
            for _, k, w in contribs[:3]:
                contribution_reasons.append(f"{k} (weight {w:.2f})")
        winner_reasons = (metric_reason_candidates + contribution_reasons)[:6]

    selected_products = []
    for _, r in selected.iterrows():
        selected_products.append(
            {
                "product": str(r["compare_label"]),
                "brand": r.get("brand"),
                "model": r.get("model"),
                "price_usd": _to_float(r.get("price_usd")),
                "rating": _to_float(r.get("rating")),
                "review_pros": r.get("review_pros") if isinstance(r.get("review_pros"), list) else [],
                "review_cons": r.get("review_cons") if isinstance(r.get("review_cons"), list) else [],
                "common_complaints": r.get("common_complaints") if isinstance(r.get("common_complaints"), list) else [],
                "aspect_sentiment_summary": r.get("aspect_sentiment_summary") if isinstance(r.get("aspect_sentiment_summary"), dict) else {},
            }
        )

    return {
        "query": query,
        "selected_products": selected_products,
        "comparison_rows": comparison_rows,
        "score_breakdown": score_breakdown,
        "weight_config": weight_cfg,
        "final_pick": final_pick,
        "final_pick_reasons": winner_reasons,
        "metric_win_counts": metric_wins,
    }
