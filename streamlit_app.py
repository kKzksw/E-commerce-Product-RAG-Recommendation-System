import math
import re
import streamlit as st
import pandas as pd

from src.agent.compare_tool import build_structured_comparison
from src.agent.explainer import llm_explain
from src.agent.router import llm_route
from src.retriever.multi_source import multi_source_retrieval
from src.retriever.structured import SPEC_MAP
from src.utils.data import load_product_data, precompute_sentence_embeddings


st.set_page_config(page_title="Mobile Recommendation Agent", layout="wide")
st.title("📱 Mobile Phone Recommendation Agent")
st.caption(
    "Two modes: Recommendation Mode and Compare Mode. "
    "Ask naturally, and the agent chooses the right mode."
)


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


def _product_label(row) -> str:
    return f"{str(row.get('brand') or '').strip()} {str(row.get('model') or '').strip()}".strip()


def _fmt_price(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "-"


def _fmt_score(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def _top_spec_bullets(row, routing, max_items: int = 2):
    bullets = []
    spec_focus = [
        str(x).lower()
        for x in _safe_list((routing or {}).get("spec_focus"))
        if str(x).strip()
    ]
    for spec in spec_focus:
        col = SPEC_MAP.get(spec)
        if not col:
            continue
        v = row.get(col)
        try:
            bullets.append(f"{spec.title()} rating is {_fmt_score(v)} (requested).")
        except Exception:
            continue
        if len(bullets) >= max_items:
            return bullets

    scored = []
    for spec, col in SPEC_MAP.items():
        v = row.get(col)
        try:
            scored.append((float(v), spec))
        except Exception:
            continue
    scored.sort(reverse=True)
    for v, spec in scored[:max_items]:
        bullets.append(f"Strong {spec} score ({v:.2f}).")
    return bullets[:max_items]


def _build_fit_bullets(row, routing, rank: int, max_items: int = 3):
    bullets = []
    budget = (routing or {}).get("budget_usd")
    price = row.get("price_usd")
    try:
        if budget is not None and price is not None:
            p, b = float(price), float(budget)
            if p <= b:
                bullets.append(f"Fits your budget ({_fmt_price(p)} within {_fmt_price(b)}).")
            else:
                bullets.append(f"Slightly above budget ({_fmt_price(p)} vs {_fmt_price(b)}).")
    except Exception:
        pass

    bullets.extend(_top_spec_bullets(row, routing, max_items=2))

    try:
        ms = row.get("multi_source_score")
        if ms is not None:
            bullets.append(f"High match score ({float(ms):.2f}) based on specs + reviews.")
    except Exception:
        pass

    if (
        str((routing or {}).get("freshness_pref") or "").lower() == "latest"
        and row.get("latest_review_date") is not None
    ):
        bullets.append(
            f"Recent model supported by latest review date ({row.get('latest_review_date')})."
        )

    if rank > 1:
        try:
            rating = row.get("rating")
            if rating is not None:
                bullets.append(f"Alternative option with overall rating {_fmt_score(rating)}.")
        except Exception:
            pass

    out, seen = [], set()
    for b in bullets:
        key = b.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(b)
        if len(out) >= max_items:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UI rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_top3_recommendation_cards(top_df, routing):
    top3 = top_df.iloc[:3] if len(top_df) >= 1 else top_df.iloc[0:0]
    if top3.empty:
        return
    st.subheader("Top 3 Recommendations")
    cols = st.columns(len(top3))
    rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}
    for rank, (col, (_, row)) in enumerate(zip(cols, top3.iterrows()), start=1):
        with col:
            if rank == 1:
                st.markdown("### 🎯 Top pick")
            st.markdown(f"**{rank_icons.get(rank, '•')} #{rank} {_product_label(row)}**")
            st.caption(
                f"{_fmt_price(row.get('price_usd'))} | Rating {_fmt_score(row.get('rating'))}"
            )
            if row.get("multi_source_score") is not None:
                st.caption(f"Match score: {_fmt_score(row.get('multi_source_score'))}")
            if (
                rank == 1
                and str((routing or {}).get("freshness_pref") or "").lower() == "latest"
                and row.get("latest_review_date") is not None
            ):
                st.caption(f"Latest review: {row.get('latest_review_date')}")

            pros = _safe_list(row.get("review_pros"))
            cons = _safe_list(row.get("review_cons"))
            evidence = _safe_list(row.get("review_retrieval_snippets"))

            fit_bullets = _build_fit_bullets(row, routing, rank=rank, max_items=3)
            st.write("Why it fits your needs")
            for b in fit_bullets or ["Good overall match for your query."]:
                st.write(f"• {b}")

            if pros:
                st.write("Review pros")
                for p in pros[:2]:
                    st.write(f"+ {p}")
            elif evidence:
                st.write("Review evidence")
                st.write(f"• {evidence[0]}")
            if cons:
                st.write("Review cons")
                for c in cons[:2]:
                    st.write(f"- {c}")


def _technical_table_columns(top_df, routing):
    cols = ["brand", "model", "price_usd", "rating"] + list(SPEC_MAP.values())
    if str((routing or {}).get("freshness_pref") or "").lower() == "latest":
        for c in ["latest_review_date", "model_gen_hint", "freshness_score"]:
            if c in top_df.columns and c not in cols:
                cols.append(c)
    for c in [
        "multi_source_score",
        "structured_signal_score",
        "review_relevance_score",
        "description_relevance_score",
        "retrieval_weight_structured",
        "retrieval_weight_reviews",
    ]:
        if c in top_df.columns and c not in cols:
            cols.append(c)
    return cols


def _render_technical_details(top_df, routing):
    with st.expander("Technical details", expanded=False):
        st.write("Router output")
        st.json(routing)
        st.write("Candidate table")
        cols = _technical_table_columns(top_df, routing)
        st.dataframe(top_df[cols].reset_index(drop=True), use_container_width=True)
        for _, row in top_df.iterrows():
            label = _product_label(row)
            st.markdown(f"---\n**{label}**")
            if row.get("retrieval_debug"):
                st.code(str(row.get("retrieval_debug")), language="text")
            review_snips = _safe_list(row.get("review_retrieval_snippets"))
            if review_snips:
                st.write("Review evidence (semantic matches)")
                for i, s in enumerate(review_snips[:3], start=1):
                    st.write(f"{i}. {s}")
            desc_terms = _safe_list(row.get("description_matched_terms"))
            if desc_terms:
                st.write("Description matched terms:", ", ".join([str(t) for t in desc_terms]))


def _extract_compare_labels(compare_result):
    rows = _safe_list((compare_result or {}).get("comparison_rows"))
    if not rows:
        return []
    row = rows[0]
    if isinstance(row, dict):
        return [k for k in row.keys() if k not in {"metric", "field", "direction", "winner"}]
    return []


def _to_num(v):
    try:
        if v is None:
            return None
        x = float(v)
        return None if math.isnan(x) else x
    except Exception:
        return None


def _format_compare_bullet(row, labels, final_pick):
    if not isinstance(row, dict):
        return None
    metric = str(row.get("metric") or "")
    winner = row.get("winner")
    if not metric or not winner or winner == "tie":
        return None
    if labels and len(labels) >= 2:
        a, b = labels[0], labels[1]
        va, vb = _to_num(row.get(a)), _to_num(row.get(b))
        if va is not None and vb is not None:
            if "Price" in metric:
                cheaper = a if va < vb else b
                return f"{cheaper} is cheaper by ${abs(va - vb):,.0f} ({_fmt_price(va)} vs {_fmt_price(vb)})."
            diff = abs(va - vb)
            if diff > 0:
                return f"{winner} leads on {metric} by {diff:.2f} ({a}: {va:.2f}, {b}: {vb:.2f})."
    return f"{winner} wins on {metric}." if metric and winner else None


def _key_difference_bullets(compare_result, max_items: int = 3):
    rows = _safe_list((compare_result or {}).get("comparison_rows"))
    labels = _extract_compare_labels(compare_result)
    final_pick = (compare_result or {}).get("final_pick")
    preferred_order = [
        "Price (USD)", "Camera Rating", "Performance Rating",
        "Battery Rating", "Overall Rating",
        "Complaint Mentions (Top Issues)", "Freshness Score",
    ]
    row_map = {str(r.get("metric") or ""): r for r in rows if isinstance(r, dict)}
    selected = []
    for m in preferred_order:
        if m in row_map:
            selected.append(row_map[m])
        if len(selected) >= max_items:
            break
    for r in rows:
        if r not in selected and isinstance(r, dict) and r.get("winner") and r.get("winner") != "tie":
            selected.append(r)
        if len(selected) >= max_items:
            break
    return [
        text for r in selected[:max_items]
        if (text := _format_compare_bullet(r, labels, final_pick))
    ]


def _render_recommend_mode(top_df, routing, explanation: str):
    st.subheader("Recommendation Mode")
    _render_top3_recommendation_cards(top_df, routing)
    st.markdown("### Personalized recommendation rationale")
    st.markdown(explanation) if explanation else st.write("No explanation available.")
    _render_technical_details(top_df, routing)


def _render_complaint_analysis(compare_result):
    selected_products = _safe_list((compare_result or {}).get("selected_products"))
    if not selected_products:
        st.write("No complaint analysis available.")
        return
    cols = st.columns(min(3, len(selected_products)))
    for col, p in zip(cols, selected_products[:3]):
        with col:
            st.markdown(f"**{p.get('product')}**")
            for item in _safe_list(p.get("common_complaints"))[:3]:
                if isinstance(item, dict):
                    st.write(f"- {item.get('issue')}: {item.get('mentions')} mentions")
                else:
                    st.write(f"- {item}")
            for s in _safe_list(p.get("review_pros"))[:2]:
                st.write(f"+ {s}")
            for s in _safe_list(p.get("review_cons"))[:2]:
                st.write(f"- {s}")


def _render_compare_mode(top_df, routing, compare_result, explanation: str):
    st.subheader("Compare Mode")
    final_pick = (compare_result or {}).get("final_pick")
    reasons = _safe_list((compare_result or {}).get("final_pick_reasons"))
    st.markdown("### 🏆 Winner")
    if final_pick:
        st.markdown(f"**{final_pick}**")
        if reasons:
            st.caption("Why: " + ", ".join([str(r) for r in reasons[:4]]))
    else:
        st.write("No clear winner could be determined.")
    if explanation:
        st.markdown(explanation)
    st.write("3 key differences")
    diff_bullets = _key_difference_bullets(compare_result, max_items=3)
    for b in diff_bullets or ["• The compared products are close; check expanded metrics."]:
        st.write(f"• {b}")
    with st.expander("Expand for metrics", expanded=False):
        comparison_rows = _safe_list((compare_result or {}).get("comparison_rows"))
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        score_breakdown = _safe_list((compare_result or {}).get("score_breakdown"))
        if score_breakdown:
            st.dataframe(pd.DataFrame(score_breakdown), use_container_width=True, hide_index=True)
        weight_cfg = (compare_result or {}).get("weight_config") or {}
        if isinstance(weight_cfg, dict) and weight_cfg:
            weights = weight_cfg.get("weights") or {}
            if isinstance(weights, dict) and weights:
                rows_w = sorted(
                    [{"component": k, "weight": float(v)} for k, v in weights.items()],
                    key=lambda x: x["weight"], reverse=True,
                )
                st.dataframe(pd.DataFrame(rows_w), use_container_width=True, hide_index=True)
    with st.expander("Expand for complaint analysis", expanded=False):
        _render_complaint_analysis(compare_result)


def _render_compare_mode_unmatched(top_df, routing, explanation: str):
    st.subheader("Compare Mode")
    st.warning(
        "Comparison query detected, but could not match two products. "
        "Check spelling or choose from the closest matches below."
    )
    if explanation:
        st.markdown(explanation)
    _render_top3_recommendation_cards(top_df, routing)
    _render_technical_details(top_df, routing)


# ─────────────────────────────────────────────────────────────────────────────
# App startup: load data + precompute sentence embeddings (once per session)
# ─────────────────────────────────────────────────────────────────────────────

if "product_df" not in st.session_state:
    with st.spinner("Loading product data..."):
        st.session_state.product_df, _ = load_product_data()
    st.success(f"Product data loaded! ({len(st.session_state.product_df)} products)")

if "sentence_cache" not in st.session_state:
    with st.spinner(
        "Precomputing sentence embeddings for semantic search... "
        "(first run only — saved to data/sentence_embeddings.pkl)"
    ):
        st.session_state.sentence_cache = precompute_sentence_embeddings(
            st.session_state.product_df,
            providers="openai",
        )
    n = len(st.session_state.sentence_cache)
    if n > 0:
        total_sentences = sum(
            len(v) for v in st.session_state.sentence_cache.values()
        )
        st.success(
            f"Semantic search ready — {total_sentences:,} sentences indexed "
            f"across {n} products."
        )
    else:
        st.warning(
            "Could not precompute embeddings (check EDENAI_API_KEY). "
            "Falling back to keyword search."
        )

product_df = st.session_state.product_df
sentence_cache = st.session_state.sentence_cache

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")
    st.markdown(
        f"**Products loaded:** {len(product_df)}  \n"
        f"**Sentences cached:** "
        f"{sum(len(v) for v in sentence_cache.values()):,}"
        if sentence_cache else "**Sentences cached:** 0 (keyword fallback active)"
    )
    if st.button("🔄 Recompute embeddings"):
        with st.spinner("Recomputing all sentence embeddings..."):
            st.session_state.sentence_cache = precompute_sentence_embeddings(
                product_df,
                providers="openai",
                force_recompute=True,
            )
        sentence_cache = st.session_state.sentence_cache
        st.success(f"Done — {len(sentence_cache)} products cached.")

# ─────────────────────────────────────────────────────────────────────────────
# User input
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Find your perfect phone")
query = st.text_input(
    "Enter your search query:",
    value="Recommend a phone under $600 with good battery life",
    help="Examples: 'Best camera phone under $800' or 'Compare iPhone 14 vs Galaxy S24'",
)
providers = st.selectbox("LLM Provider:", ["openai", "mistral"], index=0)

# ─────────────────────────────────────────────────────────────────────────────
# Main query flow
# ─────────────────────────────────────────────────────────────────────────────

if st.button("🔍 Analyze", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            with st.spinner("Analysing your query..."):
                routing = llm_route(query, providers=providers)

            with st.spinner("Finding matching products..."):
                top = multi_source_retrieval(
                    product_df,
                    routing,
                    query,
                    top_k=5,
                    candidate_k=40,
                    providers=providers,
                    sentence_cache=sentence_cache,
                )

            if top.empty:
                st.warning(
                    "❌ No products match your constraints. "
                    "Try relaxing model/brand constraints or budget."
                )
            else:
                compare_result = None
                is_compare = str((routing or {}).get("intent") or "").upper() == "COMPARE"

                if is_compare and len(top) >= 2:
                    compare_result = build_structured_comparison(
                        top, routing, query=query, max_items=3
                    )

                with st.spinner("Generating recommendation reason..."):
                    explanation = llm_explain(
                        query, routing, top,
                        providers=providers,
                        compare_result=compare_result,
                    )

                if is_compare:
                    if compare_result and _safe_list(compare_result.get("selected_products")):
                        _render_compare_mode(top, routing, compare_result, explanation)
                    else:
                        _render_compare_mode_unmatched(top, routing, explanation)
                else:
                    _render_recommend_mode(top, routing, explanation)

        except Exception as e:
            msg = str(e)
            st.error(f"⚠️ Error: {msg}")
            msg_lower = msg.lower()
            if "maximum context length" in msg_lower or "reduce your prompt" in msg_lower:
                st.info("Prompt too long. Try a shorter query or switch provider.")
            elif "edenai_api_key" in msg_lower or "401" in msg:
                st.info("Check that EDENAI_API_KEY is set in .env.")
            else:
                st.info("App logic error — check the traceback above.")