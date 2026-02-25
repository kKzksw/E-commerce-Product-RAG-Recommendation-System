"""
review_insights.py
==================
AI-powered semantic review analysis via EdenAI.
Replaces keyword-based approach with LLM understanding.
"""

import json
import re
from typing import Dict, List, Optional


def _sample_reviews(reviews: List[str], max_chars: int = 3000) -> str:
    """Pick a spread of reviews up to max_chars to stay within LLM context."""
    if not reviews:
        return ""
    step = max(1, len(reviews) // 20)
    sampled = []
    total = 0
    for i in range(0, len(reviews), step):
        r = str(reviews[i]).strip()
        if not r:
            continue
        if total + len(r) > max_chars:
            break
        sampled.append(r)
        total += len(r)
    return "\n---\n".join(sampled)


def summarize_review_insights(
    reviews: List[str],
    product_name: str = "",
    focus_aspects: Optional[List[str]] = None,
    providers: str = "openai",
) -> Dict:
    """
    Use EdenAI LLM to semantically analyze customer reviews.

    Parameters
    ----------
    reviews        : list of review text strings
    product_name   : product name for context (e.g. "Samsung Galaxy S23")
    focus_aspects  : aspects to pay attention to (e.g. ["camera", "battery"])
    providers      : EdenAI provider string (e.g. "openai", "google", "mistral")

    Returns
    -------
    dict with keys:
        review_pros         - list of strengths from customer reviews
        review_cons         - list of weaknesses from customer reviews
        common_complaints   - list of {"issue": str, "mentions": int}
        aspect_sentiment_summary - dict mapping aspect -> {"sentiment": str, "note": str}
        ai_summary          - one-paragraph overall sentiment summary
        provider            - which provider was used
    """
    from ..utils.edenai import eden_generate

    reviews_list = [str(r) for r in reviews if isinstance(r, str) and r.strip()]
    if not reviews_list:
        return _empty_result()

    review_text = _sample_reviews(reviews_list, max_chars=3000)
    product_hint = f" for {product_name}" if product_name else ""
    aspects_hint = f"\nPay special attention to: {', '.join(focus_aspects)}." if focus_aspects else ""

    prompt = f"""You are analyzing customer reviews{product_hint}.{aspects_hint}

Reviews (separated by "---"):
{review_text}

Return ONLY valid JSON with exactly this structure:
{{
  "pros": ["strength 1", "strength 2", "strength 3"],
  "cons": ["weakness 1", "weakness 2", "weakness 3"],
  "common_complaints": [
    {{"issue": "overheating", "mentions": 12}},
    {{"issue": "battery drain", "mentions": 8}}
  ],
  "aspect_sentiment_summary": {{
    "camera": {{"sentiment": "positive", "note": "customers praise night mode"}},
    "battery": {{"sentiment": "negative", "note": "drains fast under heavy use"}},
    "performance": {{"sentiment": "positive", "note": "smooth and lag-free"}},
    "display": {{"sentiment": "neutral", "note": "decent but not outstanding"}},
    "design": {{"sentiment": "positive", "note": "premium build quality"}}
  }},
  "summary": "One paragraph summarizing overall customer sentiment."
}}

Rules:
- pros/cons: 2-4 items each, use real customer language
- common_complaints: top 2-3 issues with estimated mention counts
- aspect_sentiment_summary: sentiment must be "positive", "negative", or "neutral"; set note to null if not mentioned
- summary: max 3 sentences
- Do NOT invent facts not present in the reviews
"""

    try:
        raw = eden_generate(prompt, providers=providers, max_tokens=700, temperature=0.1)
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise ValueError("No JSON in response")
        data = json.loads(m.group(0))

        return {
            "review_pros": data.get("pros") or [],
            "review_cons": data.get("cons") or [],
            "common_complaints": data.get("common_complaints") or [],
            "aspect_sentiment_summary": {
                k: v for k, v in (data.get("aspect_sentiment_summary") or {}).items()
                if v and v.get("note")
            },
            "ai_summary": data.get("summary") or "",
            "provider": providers,
        }

    except Exception as e:
        return {**_empty_result(), "error": str(e)}


def _empty_result() -> Dict:
    return {
        "review_pros": [],
        "review_cons": [],
        "common_complaints": [],
        "aspect_sentiment_summary": {},
        "ai_summary": "",
        "provider": "",
    }