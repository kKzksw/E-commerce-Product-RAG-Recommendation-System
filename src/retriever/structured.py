import pandas as pd
from typing import Dict

SPEC_MAP = {
    "battery": "battery_life_rating",
    "camera": "camera_rating",
    "performance": "performance_rating",
    "design": "design_rating",
    "display": "display_rating",
}


def _safe_list(value):
    """Normalize router outputs where list fields may be null/non-list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


GENERIC_MODEL_PHRASES = {
    "new model",
    "latest model",
    "new phone",
    "latest phone",
    "newest model",
    "recent model",
}


BRAND_ALIASES = {
    "iphone": "Apple",
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
    "realme narzo": "Realme",
    "narzo": "Realme",
}


def _rank_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(0.0, index=series.index, dtype="float64")
    mask = s.notna()
    if mask.sum() == 0:
        return out
    out.loc[mask] = s.loc[mask].rank(method="average", pct=True)
    return out


def _date_rank_pct(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    out = pd.Series(0.0, index=series.index, dtype="float64")
    mask = s.notna()
    if mask.sum() == 0:
        return out
    ord_vals = pd.Series(index=series.index, dtype="float64")
    ord_vals.loc[mask] = s.loc[mask].astype("int64").astype("float64")
    out.loc[mask] = ord_vals.loc[mask].rank(method="average", pct=True)
    return out


def _group_rank_pct(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Rank numeric values within each group (e.g., model generation within brand)."""
    s = pd.to_numeric(values, errors="coerce")
    g = groups.astype("string")
    out = pd.Series(0.0, index=values.index, dtype="float64")
    valid = s.notna() & g.notna()
    if valid.sum() == 0:
        return out

    valid_groups = g.loc[valid]
    for grp in valid_groups.unique():
        idx = valid_groups[valid_groups == grp].index
        out.loc[idx] = s.loc[idx].rank(method="average", pct=True)
    return out


def _normalize_brand_token(s: str) -> str:
    t = str(s or "").strip().lower()
    if not t:
        return ""
    
    for alias in sorted(BRAND_ALIASES.keys(), key=len, reverse=True):
        if t == alias or t.startswith(alias + " ") or (" " + alias + " ") in (" " + t + " "):
            return BRAND_ALIASES[alias]
    return t


def _model_match_tokens(model_targets):
    import re

    tokens = []
    brand_like = set(BRAND_ALIASES.keys()) | set(BRAND_ALIASES.values())
    generic = {"phone", "model", "new", "latest", "pro", "plus", "ultra"}

    for m in model_targets:
        for t in re.findall(r"[a-zA-Z0-9]+", str(m or "").lower()):
            if not t:
                continue
            if t in generic:
                continue
            if t in brand_like and len(t) > 3:
                continue
            if any(ch.isdigit() for ch in t) or len(t) >= 3:
                tokens.append(t)

    return list(dict.fromkeys(tokens))


def structured_retrieval(product_df: pd.DataFrame, routing: Dict, top_k: int = 5) -> pd.DataFrame:
    cand = product_df.copy()
    routing = routing or {}

    # Budget filter
    b = routing.get("budget_usd")
    if b is not None:
        try:
            cand = cand[cand["price_usd"] <= float(b)]
        except Exception:
            pass

    brands = [x.strip() for x in _safe_list(routing.get("brands")) if isinstance(x, str) and x and x.strip()]
    if brands:
        brand_targets = []
        for b in brands:
            norm = _normalize_brand_token(b)
            if norm:
                brand_targets.append(norm)
        brand_targets = list(dict.fromkeys([b for b in brand_targets if b]))
        if brand_targets:
            brand_filtered = cand[cand["brand"].astype(str).isin(brand_targets)]
            if not brand_filtered.empty:
                cand = brand_filtered

    models = [
        x.strip()
        for x in _safe_list(routing.get("models"))
        if isinstance(x, str) and x and x.strip()
    ]

    models = [m for m in models if m.lower() not in GENERIC_MODEL_PHRASES]
    if models:
        import re

        pattern = "|".join([re.escape(m) for m in models if m])
        if pattern:
            model_filtered = cand[cand["model"].str.contains(pattern, case=False, regex=True)]
 
            token_filtered = pd.DataFrame()
            mtoks = _model_match_tokens(models)
            if mtoks:
                token_pattern = "|".join([re.escape(t) for t in mtoks if t])
                if token_pattern:
                    token_filtered = cand[cand["model"].astype(str).str.contains(token_pattern, case=False, regex=True)]

            if not model_filtered.empty:
                if len(models) >= 2 and len(model_filtered) < 2 and not token_filtered.empty:
                    merged = pd.concat([model_filtered, token_filtered], axis=0)
                    cand = merged.loc[~merged.index.duplicated(keep="first")]
                else:
                    cand = model_filtered
            elif not token_filtered.empty:
                cand = token_filtered

    spec_focus = [s for s in _safe_list(routing.get("spec_focus")) if isinstance(s, str)]
    cols = [SPEC_MAP[s] for s in spec_focus if s in SPEC_MAP]

    if "battery" in spec_focus:
        if len(cand) > 0 and "battery_life_rating" in cand.columns:
            thr = cand["battery_life_rating"].quantile(0.70)
            filtered = cand[cand["battery_life_rating"] >= thr]
            if len(filtered) >= 3:
                cand = filtered

    freshness_pref = str(routing.get("freshness_pref") or "none").lower()
    freshness_enabled = freshness_pref == "latest"

    if freshness_enabled:
        cand = cand.copy()

        empty_num = pd.Series(index=cand.index, dtype="float64")
        empty_date = pd.Series(index=cand.index, dtype="datetime64[ns]")
        date_rank = _date_rank_pct(cand.get("latest_review_date", empty_date))
        brand_series = cand.get("brand", pd.Series(index=cand.index, dtype="object"))
        gen_rank = _group_rank_pct(cand.get("model_gen_hint", empty_num), brand_series)
        year_rank = _rank_pct(cand.get("model_year_hint", empty_num))
        cand["freshness_score"] = 0.65 * date_rank + 0.25 * gen_rank + 0.10 * year_rank

    if cols:
        cand = cand.copy()
        cand["spec_score"] = cand[cols].mean(axis=1)
        sort_cols = ["spec_score"]
        asc = [False]
        if freshness_enabled and "freshness_score" in cand.columns:
            sort_cols.append("freshness_score")
            asc.append(False)
        sort_cols += ["rating", "price_usd"]
        asc += [False, True]
        cand = cand.sort_values(sort_cols, ascending=asc)
    else:
        sort_cols = []
        asc = []
        if freshness_enabled and "freshness_score" in cand.columns:
            sort_cols.append("freshness_score")
            asc.append(False)
        sort_cols += ["rating", "price_usd"]
        asc += [False, True]
        cand = cand.sort_values(sort_cols, ascending=asc)

    return cand.head(top_k)
