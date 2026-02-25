import json
import re
from ..utils.edenai import eden_generate


BRAND_ALIASES = {
    "iphone": "Apple",
    "ios": "Apple",
    "galaxy": "Samsung",
    "sumsung": "Samsung",
    "samsng": "Samsung",
    "samgsung": "Samsung",
    "samasung": "Samsung",
    "pixel": "Google",
    "redmi": "Xiaomi",
    "poco": "Xiaomi",
    "mi": "Xiaomi",
    "moto": "Motorola",
    "razr": "Motorola",
    "edge": "Motorola",
    "oneplus nord": "OnePlus",
    "nord": "OnePlus",
    "narzo": "Realme",
}


def _normalize_brand_name(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    sl = s.lower()
    # longest alias first to avoid short-token collisions
    for alias in sorted(BRAND_ALIASES.keys(), key=len, reverse=True):
        if sl == alias or sl.startswith(alias + " ") or (" " + alias + " ") in (" " + sl + " "):
            return BRAND_ALIASES[alias]
    # Title-case fallback for consistency
    return s


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for x in items:
        key = str(x).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def llm_route(query: str, providers: str = "openai") -> dict:
    schema = {
        "intent": "RECOMMEND | COMPARE",
        "budget_usd": "number or null",
        "spec_focus": "list of any of: battery,camera,performance,design,display",
        "brands": "list of brand strings (optional)",
        "models": "list of model strings (optional)",
        "freshness_pref": "latest | none",
    }

    prompt = f"""
You are a router for a mobile phone recommendation agent.
Extract constraints from the user's query and return ONLY valid JSON.
Schema: {json.dumps(schema)}

Rules:
- intent=COMPARE if user asks to compare models or uses 'vs'
- budget_usd: if user mentions a budget like '$600' or 'under 600'
- spec_focus: infer which specs matter (battery/camera/performance/design/display)
- brands/models: include if explicitly mentioned
- Do NOT put generic phrases like "new model", "latest phone", "good phone" into models.
- models should only contain actual product names (e.g., "iPhone 15", "Galaxy S24").
- freshness_pref=latest when user asks for newest/latest/new model/recent model/current generation.
- freshness_pref=none otherwise.

User query: {query}
"""

    raw = eden_generate(prompt, providers=providers, max_tokens=250, temperature=0.0)

    # Robust JSON extraction
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"Router did not return JSON. Got:\n{raw}")
    data = json.loads(m.group(0))

    # Normalize common nullable fields so downstream code can iterate safely.
    for key in ("spec_focus", "brands", "models"):
        v = data.get(key)
        if v is None:
            data[key] = []
        elif isinstance(v, str):
            data[key] = [v]
        elif not isinstance(v, list):
            data[key] = []

    # Normalize brand aliases (e.g., iPhone -> Apple, Galaxy -> Samsung).
    data["brands"] = _dedupe_keep_order(
        [_normalize_brand_name(b) for b in data.get("brands", []) if isinstance(b, str) and b.strip()]
    )

    # Normalize freshness preference with a query-keyword fallback.
    fp = data.get("freshness_pref")
    if isinstance(fp, str):
        fp_norm = fp.strip().lower()
        if fp_norm in {"latest", "new", "newest", "recent", "current"}:
            data["freshness_pref"] = "latest"
        else:
            data["freshness_pref"] = "none"
    else:
        data["freshness_pref"] = "none"

    q = query.lower()
    # Hard heuristic fallback for compare intent (LLM may miss it on typos/noisy text).
    if any(
        marker in q
        for marker in ["compare ", " vs ", " versus ", "comparison", "difference between"]
    ):
        data["intent"] = "COMPARE"
    else:
        intent = str(data.get("intent") or "").strip().upper()
        data["intent"] = "COMPARE" if intent == "COMPARE" else "RECOMMEND"

    if any(k in q for k in ["latest", "newest", "new model", "new phone", "recent model", "current gen", "current generation"]):
        data["freshness_pref"] = "latest"

    # If models imply a brand and router missed/under-normalized brands, infer brands from model strings.
    inferred_brands = []
    for model_str in data.get("models", []):
        if not isinstance(model_str, str):
            continue
        brand_guess = _normalize_brand_name(model_str)
        if brand_guess and brand_guess.lower() != model_str.strip().lower():
            inferred_brands.append(brand_guess)
    if inferred_brands:
        data["brands"] = _dedupe_keep_order(list(data.get("brands", [])) + inferred_brands)

    return data
