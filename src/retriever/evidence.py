from typing import List
import re


def _sent_tokenize(text: str):
    # very small sentence splitter
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_evidence_simple(query: str, reviews: List[str], k: int = 5):
    """Select up to k evidence snippets from reviews based on lexical overlap with query.

    Returns list of truncated snippets (~220 chars).
    """
    q_tokens = set([t.lower() for t in re.findall(r"\w+", query)])
    scored = []
    for r in reviews:
        for s in _sent_tokenize(r):
            tokens = set([t.lower() for t in re.findall(r"\w+", s)])
            score = len(q_tokens & tokens)
            if score > 0:
                scored.append((score, s))

    if not scored:
        # fallback to first k review sentences
        fallbacks = []
        for r in reviews[:k]:
            sents = _sent_tokenize(r)
            if sents:
                fallbacks.append(sents[0])
        snippets = fallbacks[:k]
    else:
        scored.sort(key=lambda x: x[0], reverse=True)
        snippets = [s for _, s in scored[:k]]

    # truncate
    out = []
    for s in snippets:
        if len(s) > 220:
            out.append(s[:217].rstrip() + "...")
        else:
            out.append(s)
    return out
