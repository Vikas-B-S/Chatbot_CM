"""
db/embedder.py — Shared embedding utility

Single instance used across the whole app (Redis summaries,
future vector search elsewhere). Uses the same OpenRouter-compatible
endpoint already configured for Graphiti.

embed_text(text)       → list[float]   (1536 dims for text-embedding-3-small)
embed_batch(texts)     → list[list[float]]
cosine_similarity(a,b) → float in [-1, 1]
"""
from __future__ import annotations

import math
from typing import Optional

from openai import AsyncOpenAI
from config import get_settings

settings  = get_settings()
_client: Optional[AsyncOpenAI] = None

_MODEL = "openai/text-embedding-3-small"
_DIM   = 1536


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    return _client


async def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns zero vector on failure."""
    try:
        client = _get_client()
        resp   = await client.embeddings.create(model=_MODEL, input=text[:8000])
        return resp.data[0].embedding
    except Exception as e:
        print(f"⚠ Embedding failed: {e}")
        return [0.0] * _DIM


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in one API call."""
    if not texts:
        return []
    try:
        client = _get_client()
        resp   = await client.embeddings.create(
            model=_MODEL,
            input=[t[:8000] for t in texts]
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        print(f"⚠ Batch embedding failed: {e}")
        return [[0.0] * _DIM for _ in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on zero vectors."""
    dot  = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)