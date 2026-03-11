"""
db/mongo_manager.py
MongoDB — episodic memory storage with hybrid relevance+recency retrieval.

Retrieval strategy:
  When a query string is provided (the user's current message):
    final_score = (0.7 × text_relevance) + (0.3 × recency_decay)
    - text_relevance: MongoDB $text search score, normalised to [0, 1]
    - recency_decay:  exp(-days_old / 30) — full score today, ~0.37 at 30 days,
                      ~0.05 at 90 days — so old irrelevant episodes fade out
                      but old RELEVANT ones still surface strongly
  When no query:
    Falls back to recency-only (e.g. context preview, cold start)

Text index covers: title, content, key_entities (stringified), tags
"""
import uuid
import math
from datetime import datetime, timezone
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import get_settings

settings = get_settings()

_client: Optional[AsyncIOMotorClient] = None
_RECENCY_HALF_LIFE_DAYS = 30   # tune this — lower = prefer recent more strongly
_RELEVANCE_WEIGHT       = 0.7
_RECENCY_WEIGHT         = 0.3


def get_db() -> AsyncIOMotorDatabase:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client[settings.mongo_db]


async def init_mongo():
    db = get_db()
    col = db[settings.mongo_episodic_collection]

    # Operational indexes
    await col.create_index([("user_id", 1), ("created_at", -1)])
    await col.create_index([("session_id", 1)])
    await col.create_index([("user_id", 1), ("tags", 1)])
    await col.create_index("memory_id", unique=True)

    # Text index for hybrid retrieval
    # Weights: title is most signal-dense, content has the full narrative,
    # tags/key_entities provide topical anchors
    try:
        await col.create_index(
            [
                ("title",        "text"),
                ("content",      "text"),
                ("tags",         "text"),
                ("key_entities", "text"),
            ],
            weights={
                "title":        10,
                "tags":          6,
                "key_entities":  4,
                "content":       1,
            },
            name="episodic_text_search"
        )
    except Exception:
        # Index may already exist with different weights — safe to ignore
        pass


async def close_mongo():
    global _client
    if _client:
        _client.close()
        _client = None


async def store_episodic_memory(
    user_id: str,
    session_id: str,
    title: str,
    content: str,
    outcome: str,
    turn_start: int,
    turn_end: int,
    key_entities: list = None,
    emotional_tone: str = "neutral",
    tags: list = None
) -> str:
    memory_id = f"ep_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)
    doc = {
        "memory_id":     memory_id,
        "user_id":       user_id,
        "session_id":    session_id,
        "title":         title,
        "content":       content,
        "outcome":       outcome,
        "turn_start":    turn_start,
        "turn_end":      turn_end,
        "key_entities":  key_entities or [],
        "emotional_tone": emotional_tone,
        "tags":          tags or [],
        "created_at":    now,
        "updated_at":    now,
    }
    db = get_db()
    await db[settings.mongo_episodic_collection].insert_one(doc)
    return memory_id


def _recency_score(created_at: datetime) -> float:
    """
    Exponential decay based on age.
    Score = exp(-days_old / half_life)
    Today → 1.0 | 30 days → 0.37 | 60 days → 0.14 | 90 days → 0.05
    """
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days_old = max((now - created_at).total_seconds() / 86400, 0)
    return math.exp(-days_old / _RECENCY_HALF_LIFE_DAYS)


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


async def get_user_episodic_memories(
    user_id: str,
    limit: int = 5,
    session_id: str = None,
    query: str = None
) -> list:
    """
    Retrieve episodic memories using hybrid relevance + recency scoring.

    With query (normal chat turns):
      Fetches up to 4× limit candidates via MongoDB $text search,
      scores each by (0.7 × text_relevance) + (0.3 × recency_decay),
      returns top `limit` sorted by combined score.

    Without query (cold-start / preview endpoints):
      Falls back to pure recency ordering.
    """
    db = get_db()
    col = db[settings.mongo_episodic_collection]

    base_filter: dict = {"user_id": user_id}
    # Note: intentionally NOT filtering by session_id during retrieval —
    # memories from OTHER sessions of the same user are valuable context.
    # session_id filter is only applied when you want session-scoped views.

    if not query:
        # ── Recency-only fallback ──────────────────────────────
        cursor = (
            col.find(base_filter, {"_id": 0})
               .sort("created_at", -1)
               .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ── Hybrid retrieval ──────────────────────────────────────
    # Fetch a larger candidate pool so scoring has material to work with
    candidate_limit = limit * 4

    text_filter = {**base_filter, "$text": {"$search": query}}
    cursor = col.find(
        text_filter,
        {"_id": 0, "score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(candidate_limit)

    candidates = await cursor.to_list(length=candidate_limit)

    if not candidates:
        # Text search found nothing — fall back to recency
        cursor = (
            col.find(base_filter, {"_id": 0})
               .sort("created_at", -1)
               .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # Compute recency scores
    recency_scores = [_recency_score(c["created_at"]) for c in candidates]

    # Normalise text scores to [0, 1]
    raw_text_scores = [c.get("score", 0.0) for c in candidates]
    norm_text_scores = _normalise(raw_text_scores)

    # Combine
    for i, doc in enumerate(candidates):
        doc["_hybrid_score"] = (
            _RELEVANCE_WEIGHT * norm_text_scores[i] +
            _RECENCY_WEIGHT   * recency_scores[i]
        )
        doc.pop("score", None)   # remove raw text score before returning

    # Sort by combined score, return top limit
    candidates.sort(key=lambda x: x["_hybrid_score"], reverse=True)
    results = candidates[:limit]

    # Strip internal scoring field
    for doc in results:
        doc.pop("_hybrid_score", None)

    return results


async def get_episodic_by_tags(user_id: str, tags: list, limit: int = 3) -> list:
    db = get_db()
    cursor = (
        get_db()[settings.mongo_episodic_collection]
        .find({"user_id": user_id, "tags": {"$in": tags}}, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)