"""
evaluation.py
=============
Evaluation module for the Agentic RAG E-commerce Recommendation System.

Two main capabilities:
1. Multi-model comparison  – run the same query through several EdenAI
   providers and score each response automatically.
2. System evaluation       – run a test suite and save a structured report.

Usage (standalone, from project root):
    python -m src.utils.evaluation

Or import in Streamlit:
    from src.utils.evaluation import run_multi_model_eval, run_test_suite
"""

import json
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd


AVAILABLE_PROVIDERS = [
    "openai",     
]


DEFAULT_TEST_QUERIES = [
    {
        "id": "q1",
        "query": "I need a phone with great camera under $600",
        "expected_intent": "RECOMMEND",
        "expected_spec_focus": ["camera"],
        "expected_budget": 600,
    },
    {
        "id": "q2",
        "query": "Compare Samsung Galaxy S23 vs iPhone 14",
        "expected_intent": "COMPARE",
        "expected_brands": ["Samsung", "Apple"],
        "expected_budget": None,
    },
    {
        "id": "q3",
        "query": "Best battery life phone for long trips, budget around $400",
        "expected_intent": "RECOMMEND",
        "expected_spec_focus": ["battery"],
        "expected_budget": 400,
    },
    {
        "id": "q4",
        "query": "Latest flagship from Google or OnePlus",
        "expected_intent": "RECOMMEND",
        "expected_brands": ["Google", "OnePlus"],
        "expected_freshness": "latest",
    },
    {
        "id": "q5",
        "query": "Gaming phone with good performance and display under $800",
        "expected_intent": "RECOMMEND",
        "expected_spec_focus": ["performance", "display"],
        "expected_budget": 800,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouterEvalResult:
    query_id: str
    query: str
    provider: str
    latency_s: float
    routing: Dict

    intent_correct: bool
    budget_correct: bool
    spec_focus_recall: float  
    brand_recall: float     
    freshness_correct: bool
    overall_score: float       
    error: Optional[str] = None


@dataclass
class ExplainerEvalResult:
    query_id: str
    query: str
    provider: str
    latency_s: float
    explanation: str

    length_score: float        
    mentions_product: bool    
    mentions_price: bool      
    mentions_spec: bool      
    no_hallucination_flag: bool  
    overall_score: float
    error: Optional[str] = None


@dataclass
class MultiModelReport:
    timestamp: str
    providers_tested: List[str]
    router_results: List[RouterEvalResult] = field(default_factory=list)
    explainer_results: List[ExplainerEvalResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Router evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_router(routing: Dict, expected: Dict) -> Tuple[bool, bool, float, float, bool, float]:
    """Return (intent_ok, budget_ok, spec_recall, brand_recall, freshness_ok, overall)."""

    intent_ok = routing.get("intent", "").upper() == expected.get("expected_intent", "RECOMMEND").upper()

    exp_budget = expected.get("expected_budget")
    got_budget = routing.get("budget_usd")
    if exp_budget is None:
        budget_ok = got_budget is None or got_budget == 0
    else:
        try:
            budget_ok = abs(float(got_budget or 0) - float(exp_budget)) / float(exp_budget) <= 0.05
        except Exception:
            budget_ok = False

    exp_specs = [s.lower() for s in (expected.get("expected_spec_focus") or [])]
    got_specs = [s.lower() for s in (routing.get("spec_focus") or [])]
    if exp_specs:
        spec_recall = len(set(exp_specs) & set(got_specs)) / len(exp_specs)
    else:
        spec_recall = 1.0 if not got_specs else 0.8  


    exp_brands = [b.lower() for b in (expected.get("expected_brands") or [])]
    got_brands = [b.lower() for b in (routing.get("brands") or [])]
    if exp_brands:
        brand_recall = len(set(exp_brands) & set(got_brands)) / len(exp_brands)
    else:
        brand_recall = 1.0


    exp_fresh = expected.get("expected_freshness", "none")
    freshness_ok = routing.get("freshness_pref", "none") == exp_fresh


    overall = (
        0.30 * float(intent_ok) +
        0.25 * float(budget_ok) +
        0.20 * spec_recall +
        0.15 * brand_recall +
        0.10 * float(freshness_ok)
    )
    return intent_ok, budget_ok, spec_recall, brand_recall, freshness_ok, round(overall, 3)


def evaluate_router_single(
    query_id: str,
    query: str,
    expected: Dict,
    provider: str,
) -> RouterEvalResult:
    """Run router for one query/provider and return scored result."""
    from .edenai import eden_generate 


    try:
        from ..agent.router import llm_route
        t0 = time.time()
        routing = llm_route(query, providers=provider)
        latency = round(time.time() - t0, 3)
        intent_ok, budget_ok, spec_recall, brand_recall, fresh_ok, overall = _score_router(routing, expected)
        return RouterEvalResult(
            query_id=query_id,
            query=query,
            provider=provider,
            latency_s=latency,
            routing=routing,
            intent_correct=intent_ok,
            budget_correct=budget_ok,
            spec_focus_recall=round(spec_recall, 3),
            brand_recall=round(brand_recall, 3),
            freshness_correct=fresh_ok,
            overall_score=overall,
        )
    except Exception as e:
        return RouterEvalResult(
            query_id=query_id, query=query, provider=provider,
            latency_s=0.0, routing={},
            intent_correct=False, budget_correct=False,
            spec_focus_recall=0.0, brand_recall=0.0,
            freshness_correct=False, overall_score=0.0,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Explainer evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_explanation(
    explanation: str,
    query: str,
    candidate_products: List[str],  
    expected: Dict,
) -> Tuple[float, bool, bool, bool, bool, float]:
    """Return (length_score, mentions_product, mentions_price, mentions_spec, no_hallucination, overall)."""
    text = (explanation or "").lower()
    words = len(text.split())


    if words < 20:
        length_score = 0.2
    elif words < 50:
        length_score = 0.6
    elif words <= 300:
        length_score = 1.0
    else:
        length_score = max(0.5, 1.0 - (words - 300) / 500)


    mentions_product = any(p.lower() in text for p in candidate_products) if candidate_products else True

    mentions_price = bool(re.search(r"\$\d+|usd|\bprice\b|\bcost\b|\bbudget\b", text))

    spec_keywords = {
        "camera": ["camera", "photo", "image quality"],
        "battery": ["battery", "charge", "backup"],
        "performance": ["performance", "speed", "fast", "processor"],
        "display": ["display", "screen"],
        "design": ["design", "build", "premium"],
    }
    focus_specs = [s.lower() for s in (expected.get("expected_spec_focus") or [])]
    if focus_specs:
        mentions_spec = any(
            any(kw in text for kw in spec_keywords.get(s, [s]))
            for s in focus_specs
        )
    else:
        mentions_spec = True   

    known_brands = {"samsung", "apple", "google", "xiaomi", "oneplus", "motorola",
                    "realme", "oppo", "vivo", "nokia", "sony", "huawei"}
    mentioned_brands = {b for b in known_brands if b in text}
    candidate_brands = {p.lower().split()[0] for p in candidate_products} if candidate_products else set()
    invented = mentioned_brands - candidate_brands
    no_hallucination = len(invented) == 0

    overall = round(
        0.20 * length_score +
        0.25 * float(mentions_product) +
        0.20 * float(mentions_price) +
        0.20 * float(mentions_spec) +
        0.15 * float(no_hallucination),
        3,
    )
    return length_score, mentions_product, mentions_price, mentions_spec, no_hallucination, overall


def evaluate_explainer_single(
    query_id: str,
    query: str,
    routing: Dict,
    top_df: pd.DataFrame,
    expected: Dict,
    provider: str,
) -> ExplainerEvalResult:
    """Run explainer for one query/provider and return scored result."""
    try:
        from ..agent.explainer import llm_explain
        candidate_products = list(top_df["model"].dropna().astype(str)) if "model" in top_df.columns else []
        t0 = time.time()
        explanation = llm_explain(query, routing, top_df, providers=provider)
        latency = round(time.time() - t0, 3)
        length_score, mp, mpr, ms, nh, overall = _score_explanation(
            explanation, query, candidate_products, expected
        )
        return ExplainerEvalResult(
            query_id=query_id, query=query, provider=provider,
            latency_s=latency, explanation=explanation,
            length_score=round(length_score, 3),
            mentions_product=mp, mentions_price=mpr,
            mentions_spec=ms, no_hallucination_flag=nh,
            overall_score=overall,
        )
    except Exception as e:
        return ExplainerEvalResult(
            query_id=query_id, query=query, provider=provider,
            latency_s=0.0, explanation="",
            length_score=0.0, mentions_product=False,
            mentions_price=False, mentions_spec=False,
            no_hallucination_flag=True, overall_score=0.0,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_model_eval(
    product_df: pd.DataFrame,
    providers: Optional[List[str]] = None,
    test_queries: Optional[List[Dict]] = None,
    eval_router: bool = True,
    eval_explainer: bool = True,
    progress_callback=None,   
) -> MultiModelReport:
    """
    Run all test queries across all providers.

    Parameters
    ----------
    product_df      : the loaded product DataFrame (for retrieval during explainer eval)
    providers       : list of EdenAI provider strings, defaults to ["openai", "google", "mistral"]
    test_queries    : list of query dicts; defaults to DEFAULT_TEST_QUERIES
    eval_router     : whether to evaluate the router
    eval_explainer  : whether to evaluate the explainer (costs more tokens)
    progress_callback : optional function(str) to stream progress messages

    Returns
    -------
    MultiModelReport with all results and a summary dict
    """
    from ..retriever.multi_source import multi_source_retrieval

    if providers is None:
        providers = ["openai"]
    if test_queries is None:
        test_queries = DEFAULT_TEST_QUERIES

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    report = MultiModelReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        providers_tested=providers,
    )

    for tq in test_queries:
        qid = tq["id"]
        query = tq["query"]
        _log(f"▶ Query [{qid}]: {query}")

        for provider in providers:
            _log(f"  → Provider: {provider}")

            router_result = None
            if eval_router:
                router_result = evaluate_router_single(qid, query, tq, provider)
                report.router_results.append(router_result)
                _log(
                    f"    Router score: {router_result.overall_score:.2f} | "
                    f"intent={'✓' if router_result.intent_correct else '✗'} | "
                    f"latency={router_result.latency_s}s"
                    + (f" | ERROR: {router_result.error}" if router_result.error else "")
                )

            if eval_explainer:
                routing = (router_result.routing if router_result else {}) or {}

                try:
                    top_df = multi_source_retrieval(product_df, routing, query, top_k=3)
                except Exception as e:
                    _log(f"    Retrieval error: {e}")
                    top_df = pd.DataFrame()

                exp_result = evaluate_explainer_single(qid, query, routing, top_df, tq, provider)
                report.explainer_results.append(exp_result)
                _log(
                    f"    Explainer score: {exp_result.overall_score:.2f} | "
                    f"latency={exp_result.latency_s}s"
                    + (f" | ERROR: {exp_result.error}" if exp_result.error else "")
                )


    report.summary = _build_summary(report, providers, test_queries)
    return report


def _build_summary(report: MultiModelReport, providers: List[str], test_queries: List[Dict]) -> Dict:
    """Aggregate scores per provider."""
    summary = {"per_provider": {}, "best_router": None, "best_explainer": None}

    for provider in providers:
        r_scores = [r.overall_score for r in report.router_results if r.provider == provider and not r.error]
        e_scores = [r.overall_score for r in report.explainer_results if r.provider == provider and not r.error]
        r_latencies = [r.latency_s for r in report.router_results if r.provider == provider and not r.error]
        e_latencies = [r.latency_s for r in report.explainer_results if r.provider == provider and not r.error]

        summary["per_provider"][provider] = {
            "router_avg_score": round(sum(r_scores) / len(r_scores), 3) if r_scores else None,
            "router_avg_latency_s": round(sum(r_latencies) / len(r_latencies), 3) if r_latencies else None,
            "explainer_avg_score": round(sum(e_scores) / len(e_scores), 3) if e_scores else None,
            "explainer_avg_latency_s": round(sum(e_latencies) / len(e_latencies), 3) if e_latencies else None,
            "queries_failed_router": sum(1 for r in report.router_results if r.provider == provider and r.error),
            "queries_failed_explainer": sum(1 for r in report.explainer_results if r.provider == provider and r.error),
        }


    valid_router = {p: v["router_avg_score"] for p, v in summary["per_provider"].items() if v["router_avg_score"] is not None}
    valid_explainer = {p: v["explainer_avg_score"] for p, v in summary["per_provider"].items() if v["explainer_avg_score"] is not None}

    if valid_router:
        summary["best_router"] = max(valid_router, key=valid_router.get)
    if valid_explainer:
        summary["best_explainer"] = max(valid_explainer, key=valid_explainer.get)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def report_to_dataframes(report: MultiModelReport) -> Dict[str, pd.DataFrame]:
    """Convert a MultiModelReport into DataFrames for display or CSV export."""
    router_df = pd.DataFrame([asdict(r) for r in report.router_results]) if report.router_results else pd.DataFrame()
    explainer_df = pd.DataFrame([asdict(r) for r in report.explainer_results]) if report.explainer_results else pd.DataFrame()

    summary_rows = []
    for provider, stats in report.summary.get("per_provider", {}).items():
        row = {"provider": provider}
        row.update(stats)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    return {
        "router": router_df,
        "explainer": explainer_df,
        "summary": summary_df,
    }


def report_to_json(report: MultiModelReport) -> str:
    """Serialize the full report to a JSON string."""
    return json.dumps(asdict(report), indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit display helper 
# ─────────────────────────────────────────────────────────────────────────────

def display_eval_report_streamlit(report: MultiModelReport):
    """
    Call this from your Streamlit app to render the evaluation report.

    Example usage in streamlit_app.py:
        from src.utils.evaluation import run_multi_model_eval, display_eval_report_streamlit
        if st.button("Run Evaluation"):
            report = run_multi_model_eval(product_df, providers=["openai","google","mistral"])
            display_eval_report_streamlit(report)
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not available")
        return

    dfs = report_to_dataframes(report)

    st.subheader("📊 Multi-Model Evaluation Summary")
    if not dfs["summary"].empty:
        st.dataframe(dfs["summary"], use_container_width=True)

    best_r = report.summary.get("best_router")
    best_e = report.summary.get("best_explainer")
    if best_r:
        st.success(f"🏆 Best Router Provider: **{best_r}**")
    if best_e:
        st.success(f"🏆 Best Explainer Provider: **{best_e}**")

    with st.expander("Router Results (per query/provider)"):
        if not dfs["router"].empty:
            display_cols = ["query_id", "provider", "overall_score", "intent_correct",
                            "budget_correct", "spec_focus_recall", "brand_recall", "latency_s", "error"]
            display_cols = [c for c in display_cols if c in dfs["router"].columns]
            st.dataframe(dfs["router"][display_cols], use_container_width=True)
        else:
            st.info("No router results.")

    with st.expander("Explainer Results (per query/provider)"):
        if not dfs["explainer"].empty:
            display_cols = ["query_id", "provider", "overall_score", "length_score",
                            "mentions_product", "mentions_price", "mentions_spec",
                            "no_hallucination_flag", "latency_s", "error"]
            display_cols = [c for c in display_cols if c in dfs["explainer"].columns]
            st.dataframe(dfs["explainer"][display_cols], use_container_width=True)
            # Show explanations
            st.markdown("**Sample Explanations**")
            for _, row in dfs["explainer"].iterrows():
                if row.get("explanation"):
                    st.markdown(f"**[{row['provider']}] {row['query_id']}:** {row['explanation']}")
        else:
            st.info("No explainer results.")

    with st.expander("Export"):
        json_str = report_to_json(report)
        st.download_button(
            "⬇ Download Full Report (JSON)",
            data=json_str,
            file_name=f"eval_report_{report.timestamp.replace(':','-')}.json",
            mime="application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.utils.data import load_product_df  

    print("Loading data...")
    df = load_product_df()

    print("Running evaluation with openai and mistral...")
    report = run_multi_model_eval(
        df,
        providers=["openai", "mistral"],
        eval_router=True,
        eval_explainer=True,
        progress_callback=print,
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(report.summary, indent=2))

    out_path = "eval_report.json"
    with open(out_path, "w") as f:
        f.write(report_to_json(report))
    print(f"\nFull report saved to {out_path}")