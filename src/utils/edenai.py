import os
import requests
from dotenv import load_dotenv

load_dotenv()

EDENAI_API_KEY = os.environ.get("EDENAI_API_KEY")
EDEN_GENERATION_URL = "https://api.edenai.run/v2/text/generation"
EDEN_EMBEDDING_URL = "https://api.edenai.run/v2/text/embeddings"

if not EDENAI_API_KEY:
    EDENAI_API_KEY = None


def eden_generate(
    text: str,
    providers: str = "openai",
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> str:
    """
    Call EdenAI text generation.
    Raises RuntimeError if the request fails or provider status != 'success'.
    """
    if not EDENAI_API_KEY:
        raise RuntimeError("Missing EDENAI_API_KEY in environment (.env)")

    headers = {
        "Authorization": f"Bearer {EDENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "providers": providers,
        "text": text,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(EDEN_GENERATION_URL, json=payload, headers=headers, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"EdenAI HTTP error: {e} - {resp.text}")

    data = resp.json()
    provider = providers.split(",")[0].strip()
    provider_out = data.get(provider) or {}

    if provider_out.get("status") != "success":
        raise RuntimeError(f"EdenAI provider returned error: {provider_out}")

    generated = provider_out.get("generated_text")
    if generated is None:
        for v in provider_out.values():
            if isinstance(v, str) and v.strip():
                generated = v
                break
    return (generated or "").strip()


def eden_embed(
    texts: list,
    providers: str = "openai",
) -> list:
    """
    Convert a list of texts into embedding vectors via EdenAI.

    Parameters
    ----------
    texts     : list of strings to embed
    providers : EdenAI provider string, default "openai"

    Returns
    -------
    List of embedding vectors (list of floats), same order as input texts.
    Raises RuntimeError on API failure.
    """
    if not EDENAI_API_KEY:
        raise RuntimeError("Missing EDENAI_API_KEY in environment (.env)")

    if not texts:
        return []

    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        headers = {
            "Authorization": f"Bearer {EDENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "providers": providers,
            "texts": batch,
        }

        resp = requests.post(
            EDEN_EMBEDDING_URL, json=payload, headers=headers, timeout=120
        )
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"EdenAI Embedding HTTP error: {e} - {resp.text}")

        data = resp.json()
        provider = providers.split(",")[0].strip()
        provider_out = data.get(provider) or {}

        if provider_out.get("status") != "success":
            raise RuntimeError(f"EdenAI embedding provider error: {provider_out}")

        # EdenAI embedding response:
        # {"items": [{"embedding": [0.12, -0.34, ...]}, ...]}
        items = provider_out.get("items") or []
        if len(items) != len(batch):
            raise RuntimeError(
                f"EdenAI returned {len(items)} embeddings for {len(batch)} texts"
            )

        all_embeddings.extend([item["embedding"] for item in items])

    return all_embeddings