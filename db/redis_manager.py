"""
db/redis_manager.py
Redis — conversation summary storage with hierarchical compression.

Key schema:
  summary:{session_id}  → Sorted Set (score = batch_start)
                           members = JSON-encoded summary documents

Summary document fields:
  summary_id, session_id, user_id, batch_start, batch_end,
  level (0=raw batch, 1=meta-compressed), summary_text,
  key_topics, source_summary_ids, created_at

Hierarchy:
  Level 0 = raw 3-turn batch summary
  Level 1 = LLM-compressed merge of two level-0 summaries (covers 6 turns)
"""
import json
import uuid
from datetime import datetime
from typing import Optional
import redis.asyncio as aioredis
from config import get_settings

settings = get_settings()

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True
        )
    return _redis


async def close_redis():
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


async def save_summary(
    session_id: str,
    user_id: str,
    batch_start: int,
    batch_end: int,
    summary_text: str,
    key_topics: list = None,
    level: int = 0,
    source_summary_ids: list = None
) -> str:
    """Store a summary in Redis sorted set (score = batch_start)."""
    summary_id = f"sum_{uuid.uuid4().hex[:12]}"
    doc = {
        "summary_id": summary_id,
        "session_id": session_id,
        "user_id": user_id,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "level": level,
        "summary_text": summary_text,
        "key_topics": key_topics or [],
        "source_summary_ids": source_summary_ids or [],
        "created_at": datetime.utcnow().isoformat()
    }
    r = await get_redis()
    await r.zadd(f"summary:{session_id}", {json.dumps(doc): batch_start})
    return summary_id


async def get_session_summaries(session_id: str, min_level: int = 0) -> list:
    """All summaries for a session, sorted by batch_start."""
    r = await get_redis()
    members = await r.zrange(f"summary:{session_id}", 0, -1)
    summaries = [json.loads(m) for m in members]
    if min_level > 0:
        summaries = [s for s in summaries if s.get("level", 0) >= min_level]
    return summaries


async def get_latest_summaries_for_context(
    session_id: str, max_summaries: int = 6
) -> list:
    """
    Efficient summary chain for context window.
    Prefers level-1 meta-summaries (they cover 6 turns in ~the space of 1).
    Fills remainder with uncovered level-0 summaries.
    """
    all_s = await get_session_summaries(session_id)
    if not all_s:
        return []

    metas = [s for s in all_s if s.get("level", 0) >= 1]
    raws  = [s for s in all_s if s.get("level", 0) == 0]

    if not metas:
        return raws[-max_summaries:]

    used_in_meta = set()
    for m in metas:
        used_in_meta.update(m.get("source_summary_ids", []))

    uncovered = [s for s in raws if s["summary_id"] not in used_in_meta]
    result = sorted(metas + uncovered, key=lambda s: s["batch_start"])
    return result[-max_summaries:]


async def get_oldest_two_uncompressed_level0(session_id: str) -> list:
    """
    Return the two oldest level-0 summaries not yet used as
    source in any level-1 meta-summary. Used to trigger meta-compression.
    """
    all_s = await get_session_summaries(session_id)
    used = set()
    for s in all_s:
        if s.get("level", 0) >= 1:
            used.update(s.get("source_summary_ids", []))
    uncompressed = [
        s for s in all_s
        if s.get("level", 0) == 0 and s["summary_id"] not in used
    ]
    return uncompressed[:2]
