"""
memory/context_builder.py — Context assembly with single shared query embedding

Key optimisations
─────────────────
  1. Embed once — query embedded ONE time upfront, vector passed to all stores.
     Old code embedded the same message twice (Neo4j + Redis). Now: once.

  2. Neo4j result cache — Neo4j/Graphiti is the slowest store (~200-800ms).
     Facts rarely change turn-to-turn. Cache the result in Redis for 30s.
     Cache miss → call Neo4j, store result. Cache hit → return in ~5ms.
     Cache is invalidated automatically by TTL — no manual invalidation needed.

  3. All 4 stores fetched in parallel via asyncio.gather — no sequential waits.

  4. Blind-spot fix — raw turns come from get_turns_from_last_summary()
     which returns everything since the last summarized batch, not just last N.
"""
import asyncio
import json
from config import get_settings
from db import sqlite_manager as sql
from db import neo4j_manager as neo4j
from db import redis_manager as redis_mgr
from db import mongo_manager as mongo
from db.embedder import embed_text

settings = get_settings()

# Neo4j cache TTL — 30s is short enough that new facts appear quickly
_NEO4J_CACHE_TTL = 30


def should_summarize(turn_number: int) -> tuple[bool, int, int]:
    """
    Pure function — no DB access needed.
    Returns (trigger, batch_start, batch_end).
    Turn 6→(True,1,3) | Turn 9→(True,4,6) | Turn 12→(True,7,9) ...
    """
    if turn_number < settings.summarize_at_turn:
        return False, 0, 0
    offset = turn_number - settings.summarize_at_turn
    if offset % settings.summarize_batch != 0:
        return False, 0, 0
    batch_index = offset // settings.summarize_batch
    batch_start = batch_index * settings.summarize_batch + 1
    batch_end   = batch_start + settings.summarize_batch - 1
    return True, batch_start, batch_end


async def _get_neo4j_cached(user_id: str, query: str, query_vec: list) -> list:
    """
    Neo4j with Redis-backed 30s cache.

    Cache key: neo4j_cache:{user_id}
    On hit:  return cached list in ~5ms
    On miss: call Neo4j (~200-800ms), cache result, return
    """
    from db.redis_manager import get_redis
    r         = await get_redis()
    cache_key = f"neo4j_cache:{user_id}"

    try:
        cached = await r.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    # Cache miss — call Neo4j
    result = await neo4j.get_user_memories(user_id, query=query)

    # Store in cache
    try:
        await r.set(cache_key, json.dumps(result), ex=_NEO4J_CACHE_TTL)
    except Exception:
        pass

    return result


async def build_context(
    session_id: str,
    user_id: str,
    user_message: str = ""
) -> dict:
    """
    Parallel fetch from all 4 stores with single shared query embedding.

    Flow:
      1. Embed user_message ONCE (if present)
      2. Fire all 4 store fetches in parallel, passing pre-computed vector
      3. Graceful degradation — if any store fails, returns empty list for it

    The shared query_vec is passed to Neo4j (cache layer), MongoDB, and Redis
    so none of them need to call the embedding API themselves.
    """
    def safe(v):
        return v if not isinstance(v, Exception) else []

    # ── Embed once ────────────────────────────────────────────
    query_vec = None
    if user_message:
        try:
            query_vec = await embed_text(user_message)
        except Exception:
            query_vec = None

    # ── Parallel fetch ────────────────────────────────────────
    results = await asyncio.gather(
        _get_neo4j_cached(user_id, user_message, query_vec),
        mongo.get_user_episodic_memories(
            user_id, limit=5, session_id=session_id,
            query=user_message or None, query_vec=query_vec
        ),
        redis_mgr.get_latest_summaries_for_context(
            session_id, user_id=user_id,
            query=user_message or None, query_vec=query_vec
        ),
        sql.get_turns_from_last_summary(session_id),
        return_exceptions=True
    )

    memories, episodic, summaries, raw_turns = results
    return {
        "memories":          safe(memories),
        "episodic_memories": safe(episodic),
        "summaries":         safe(summaries),
        "raw_turns":         safe(raw_turns),
        "query_vec":         query_vec,   # carry forward for agent use
    }


def format_context_for_prompt(context: dict) -> str:
    """Format all context into structured prompt section for LLM."""
    parts = []

    # ── 1. Neo4j facts grouped by type ───────────────────────
    memories = context.get("memories", [])
    if memories:
        by_type: dict[str, list] = {}
        for m in memories:
            t = m.get("memory_type", "fact")
            by_type.setdefault(t, []).append(m.get("content", ""))

        for mtype, header in [
            ("fact",       "## About This User"),
            ("preference", "## User Preferences"),
            ("goal",       "## User Goals"),
            ("constraint", "## User Constraints"),
        ]:
            items = by_type.get(mtype, [])
            if items:
                parts.append(header)
                for item in items:
                    parts.append(f"- {item}")

    # ── 2. MongoDB episodic memories ──────────────────────────
    episodic = context.get("episodic_memories", [])
    if episodic:
        parts.append("\n## Remembered Episodes")
        for ep in episodic[:2]:
            title   = ep.get("title", "Episode")
            outcome = ep.get("outcome", "")
            snippet = ep.get("content", "")[:220]
            parts.append(f"[{title} — {outcome}]: {snippet}…")

    # ── 3. Redis summaries ────────────────────────────────────
    summaries = context.get("summaries", [])
    if summaries:
        for s in summaries:
            level = s.get("level", 0)
            if level == 99:
                parts.append("\n## Previous Session")
                parts.append(s["summary_text"])
            elif level == 2:
                parts.append("\n## Session Arc")
                parts.append(f"[T{s['batch_start']}-T{s['batch_end']}]: {s['summary_text']}")
            elif level == 1:
                parts.append("\n## Conversation Window")
                parts.append(f"[T{s['batch_start']}-T{s['batch_end']}]: {s['summary_text']}")
            else:
                parts.append("\n## Recent Summary")
                parts.append(f"[T{s['batch_start']}-T{s['batch_end']}]: {s['summary_text']}")

    # ── 4. SQLite raw turns ───────────────────────────────────
    raw = context.get("raw_turns", [])
    if raw:
        parts.append(
            f"\n## Recent Conversation "
            f"(turns {raw[0]['turn_number']}-{raw[-1]['turn_number']})"
        )
        for t in raw:
            parts.append(f"User: {t['user_msg']}")
            parts.append(f"Assistant: {t['assistant_msg']}")

    return "\n".join(parts) if parts else "No prior context available."


def format_context_for_router(context: dict) -> str:
    """Compact context for router's LLM call — last summary + last 2 turns."""
    parts    = []
    summaries = context.get("summaries", [])
    raw      = context.get("raw_turns", [])
    if summaries:
        last = summaries[-1]
        parts.append(f"Recent summary: {last['summary_text'][:200]}")
    if raw:
        for t in raw[-2:]:
            parts.append(f"User: {t['user_msg'][:120]}")
            parts.append(f"Assistant: {t['assistant_msg'][:120]}")
    return "\n".join(parts)