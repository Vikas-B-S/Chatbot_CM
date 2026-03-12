"""
db/embedder.py — Local embeddings via sentence-transformers

WHY LOCAL:
  OpenRouter embedding API = ~300ms per call (network round trip)
  Local sentence-transformers = ~5-20ms per call (CPU inference, no network)

  This single change saves 300ms of latency BEFORE the LLM even starts,
  which is the main reason streaming felt slow despite being implemented.

MODEL: all-MiniLM-L6-v2
  - 22MB download, cached after first run
  - 384 dimensions (vs 1536 for OpenAI — smaller = faster cosine math)
  - Good enough semantic quality for summary/episode retrieval
  - Runs entirely on CPU, no GPU needed

COST:
  OpenAI embeddings = $0.02 per 1M tokens
  Local embeddings  = $0.00 forever

  For a chatbot with 100 turns/day:
    Old: ~100 API calls/day → cost + latency
    New: 0 API calls/day    → free + fast
"""
from __future__ import annotations

import math
import threading
from typing import Optional

_model = None
_lock  = threading.Lock()
_DIM   = 384   # all-MiniLM-L6-v2 output dimension


def _get_model():
    """Lazy load — model loads once on first embed call, reused forever."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                print("Loading local embedding model (one-time ~2s)...")
                _model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✓ Local embedding model ready")
    return _model


async def embed_text(text: str) -> list[float]:
    """
    Embed a single string locally. ~5-20ms, no API call.
    Returns zero vector on failure.
    """
    try:
        import asyncio
        model = _get_model()
        # Run CPU inference in thread pool so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        vec  = await loop.run_in_executor(
            None,
            lambda: model.encode(text[:512], normalize_embeddings=True).tolist()
        )
        return vec
    except Exception as e:
        print(f"⚠ Local embed failed: {e}")
        return [0.0] * _DIM


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in one local inference call."""
    if not texts:
        return []
    try:
        import asyncio
        model  = _get_model()
        loop   = asyncio.get_event_loop()
        vecs   = await loop.run_in_executor(
            None,
            lambda: model.encode(
                [t[:512] for t in texts],
                normalize_embeddings=True
            ).tolist()
        )
        return vecs
    except Exception as e:
        print(f"⚠ Local batch embed failed: {e}")
        return [[0.0] * _DIM for _ in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity. Since all-MiniLM-L6-v2 uses normalize_embeddings=True,
    vectors are already unit-length so this is just a dot product.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp to [-1, 1] to handle float precision
    return max(-1.0, min(1.0, dot))