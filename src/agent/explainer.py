import json
from typing import List, Dict, Optional
from ..utils.edenai import eden_generate
from ..retriever.evidence import extract_evidence_simple


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _round_or_none(value, ndigits: int = 3):
    v = _safe_float(value)
    return None if v is None else round(v, ndigits)


def _short_text(text, max_chars: int = 180) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _compact_complaints(items, max_items: int = 2):
    out = []
    for item in _safe_list(items)[:max_items]:
        if isinstance(item, dict):
            issue = str(item.get("issue") or "").strip()
            mentions = item.get("mentions")
            if issue:
                if mentions is not None:
                    out.append(f"{issue} ({mentions})")
                else:
                    out.append(issue)
        else:
            s = str(item).strip()
            if s:
                out.append(_short_text(s, 80))
    return out


def _compact_aspect_summary(summary: Dict, max_items: int = 4) -> Dict:
    if not isinstance(summary, dict):
        return {}
    compact = {}
    # Keep only aspects with signals and round values for prompt size.
    for k, v in summary.items():
        if not isinstance(v, dict):
            continue
        mentions = int(v.get("mentions") or 0)
        if mentions <= 0:
            continue
        compact[k] = {
            "mentions": mentions,
            "sentiment": _round_or_none(v.get("sentiment_score"), 2),
        }
    if len(compact) <= max_items:
        return compact
    ordered = sorted(compact.items(), key=lambda kv: (kv[1].get("mentions") or 0), reverse=True)
    return dict(ordered[:max_items])


def _compact_candidate_rows(query: str, routing: Dict, top_df, max_items: int = 3, minimal: bool = False) -> List[Dict]:
    products = []
    rows = top_df.head(max_items) if hasattr(top_df, "head") else top_df
    for _, row in rows.iterrows():
        prod = {
            "brand": row.get("brand"),
            "model": row.get("model"),
            "price_usd": _round_or_none(row.get("price_usd"), 2),
            "rating": _round_or_none(row.get("rating"), 2),
        }

        # Include requested spec fields first; keep all five only in non-minimal mode.
        spec_keys = [
            "battery_life_rating",
            "camera_rating",
            "performance_rating",
            "design_rating",
            "display_rating",
        ]
        if minimal:
            requested = [str(x).lower() for x in _safe_list((routing or {}).get("spec_focus"))]
            map_ = {
                "battery": "battery_life_rating",
                "camera": "camera_rating",
                "performance": "performance_rating",
                "design": "design_rating",
                "display": "display_rating",
            }
            spec_keys = [map_[s] for s in requested if s in map_]
            if not spec_keys:
                spec_keys = ["battery_life_rating", "camera_rating", "performance_rating"]
        for k in spec_keys:
            prod[k] = _round_or_none(row.get(k), 2)

        reviews = row.get("review_text")
        if reviews is None:
            reviews = []
        elif not isinstance(reviews, list):
            reviews = [str(reviews)]
        retrieved_evidence = row.get("review_retrieval_snippets")
        if isinstance(retrieved_evidence, list) and retrieved_evidence:
            evidence = [ _short_text(str(x), 140 if minimal else 180) for x in retrieved_evidence[: (1 if minimal else 2)] ]
        else:
            evidence = [_short_text(x, 140 if minimal else 180) for x in extract_evidence_simple(query, reviews, k=(1 if minimal else 2))]

        review_summary = {
            "pros": [_short_text(x, 100) for x in _safe_list(row.get("review_pros"))[: (1 if minimal else 2)]],
            "cons": [_short_text(x, 100) for x in _safe_list(row.get("review_cons"))[: (1 if minimal else 2)]],
            "common_complaints": _compact_complaints(row.get("common_complaints"), max_items=(1 if minimal else 2)),
        }
        if not minimal:
            review_summary["aspect_sentiment_summary"] = _compact_aspect_summary(row.get("aspect_sentiment_summary"), max_items=4)

        item = {
            "product": prod,
            "evidence": evidence,
            "review_summary": review_summary,
        }
        if not minimal:
            item["retrieval_scores"] = {
                "multi_source_score": _round_or_none(row.get("multi_source_score"), 3),
                "structured_signal_score": _round_or_none(row.get("structured_signal_score"), 3),
                "review_relevance_score": _round_or_none(row.get("review_relevance_score"), 3),
                "description_relevance_score": _round_or_none(row.get("description_relevance_score"), 3),
            }
        products.append(item)
    return products


def _compact_compare_result(compare_result: Optional[Dict], minimal: bool = False) -> Dict:
    if not isinstance(compare_result, dict) or not compare_result:
        return {}

    out = {
        "final_pick": compare_result.get("final_pick"),
        "final_pick_reasons": [str(x) for x in _safe_list(compare_result.get("final_pick_reasons"))[: (2 if minimal else 4)]],
    }

    rows = []
    for r in _safe_list(compare_result.get("comparison_rows")):
        if not isinstance(r, dict):
            continue
        metric = r.get("metric")
        winner = r.get("winner")
        if not metric:
            continue
        rr = {"metric": metric, "winner": winner}
        for k, v in r.items():
            if k in {"metric", "winner", "field", "direction"}:
                continue
            if isinstance(v, (int, float)):
                rr[k] = round(float(v), 2)
            elif isinstance(v, str):
                rr[k] = _short_text(v, 40)
        rows.append(rr)

    preferred = [
        "Price (USD)",
        "Camera Rating",
        "Performance Rating",
        "Battery Rating",
        "Overall Rating",
        "Complaint Mentions (Top Issues)",
        "Freshness Score",
    ]
    row_map = {str(r.get("metric")): r for r in rows if r.get("metric")}
    selected = []
    for m in preferred:
        if m in row_map:
            selected.append(row_map[m])
    if len(selected) < (4 if minimal else 7):
        for r in rows:
            if r in selected:
                continue
            if r.get("winner") and r.get("winner") != "tie":
                selected.append(r)
            if len(selected) >= (4 if minimal else 7):
                break
    out["comparison_rows"] = selected[: (4 if minimal else 7)]

    # Keep a tiny review snapshot for compared products; omit bulky score tables/weights.
    selected_products = []
    for p in _safe_list(compare_result.get("selected_products"))[:3]:
        if not isinstance(p, dict):
            continue
        selected_products.append(
            {
                "product": p.get("product"),
                "review_pros": [_short_text(x, 90) for x in _safe_list(p.get("review_pros"))[: (1 if minimal else 2)]],
                "review_cons": [_short_text(x, 90) for x in _safe_list(p.get("review_cons"))[: (1 if minimal else 2)]],
                "common_complaints": _compact_complaints(p.get("common_complaints"), max_items=(1 if minimal else 2)),
            }
        )
    if selected_products:
        out["selected_products"] = selected_products

    return out


def _build_prompt(query: str, routing: Dict, products: List[Dict], compare_payload: Dict) -> str:
    return f"""
You are an assistant that explains product recommendations using ONLY the provided candidates and evidence.
Do NOT invent facts. If evidence is weak or missing, explicitly say so.

User query: {query}
Parsed constraints: {json.dumps(routing, ensure_ascii=False)}

Candidates (only these):
{json.dumps(products, ensure_ascii=False)}

Structured comparison tool output (if provided):
{json.dumps(compare_payload or {}, ensure_ascii=False)}

Task:
- Recommend best 1-3 options and justify with specs, price, and review sentiment.
- If intent=COMPARE, provide a concise comparison and a final pick.
- Use compare tool output as primary facts when present.
- Summarize customer experience using review pros/cons/complaints.
- Keep the answer short and user-friendly.
"""


def _is_context_length_error(msg: str) -> bool:
    m = (msg or "").lower()
    return (
        "maximum context length" in m
        or "providerinvalidinputtextlengtherror" in m
        or "reduce your prompt" in m
        or "requested" in m and "tokens" in m and "context length" in m
    )


def llm_explain(query: str, routing: Dict, top_df, providers: str = "openai", compare_result: Optional[Dict] = None) -> str:
    is_compare = str((routing or {}).get("intent") or "").upper() == "COMPARE"
    candidate_limit = 3 if is_compare else 4
    products = _compact_candidate_rows(query, routing, top_df, max_items=candidate_limit, minimal=False)
    compare_payload = _compact_compare_result(compare_result, minimal=False)
    prompt = _build_prompt(query, routing, products, compare_payload)

    # Preemptively shrink prompt for small-context provider backends.
    approx_prompt_tokens = max(1, len(prompt) // 4)
    max_completion_tokens = 320 if is_compare else 280
    if approx_prompt_tokens + max_completion_tokens > 3300:
        products = _compact_candidate_rows(query, routing, top_df, max_items=(2 if is_compare else 3), minimal=True)
        compare_payload = _compact_compare_result(compare_result, minimal=True)
        prompt = _build_prompt(query, routing, products, compare_payload)
        max_completion_tokens = 220

    try:
        return eden_generate(prompt, providers=providers, max_tokens=max_completion_tokens, temperature=0.2)
    except Exception as e:
        msg = str(e)
        if not _is_context_length_error(msg):
            raise

        # One final ultra-compact retry.
        products = _compact_candidate_rows(query, routing, top_df, max_items=2, minimal=True)
        compare_payload = _compact_compare_result(compare_result, minimal=True)
        prompt = _build_prompt(query, routing, products, compare_payload)
        return eden_generate(prompt, providers=providers, max_tokens=180, temperature=0.2)
