"""
memory/context_builder.py — Context assembly with single shared query embedding

Key optimisations
─────────────────
  1. Embed once — query embedded ONE time upfront, vector passed to all stores.
  2. Neo4j result cache — cached in Redis for 30s.
  3. All 4 stores fetched in parallel via asyncio.gather.
  4. Blind-spot fix — raw turns come from get_turns_from_last_summary().

Diagnostics
───────────
  Per-step timing is printed on every context build so you can see
  exactly which store is slow. Look for lines like:
    ⏱ [ctx] embed: 12ms
    ⏱ [ctx] neo4j: 45ms   ← or 13000ms if Neo4j is the bottleneck
    ⏱ [ctx] mongo: 8ms
    ⏱ [ctx] redis: 6ms
    ⏱ [ctx] sqlite: 3ms
"""
import asyncio
import json
import time
from config import get_settings
from db import sqlite_manager as sql
from db import neo4j_manager as neo4j
from db import redis_manager as redis_mgr
from db import mongo_manager as mongo
from db.embedder import embed_text

settings = get_settings()

_NEO4J_CACHE_TTL = 30


def should_summarize(turn_number: int) -> tuple[bool, int, int]:
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
    from db.redis_manager import get_redis
    r         = await get_redis()
    cache_key = f"neo4j_cache:{user_id}"

    try:
        cached = await r.get(cache_key)
        if cached:
            records = json.loads(cached)
            if query_vec and records:
                from db.neo4j_manager import _local_relevance_score
                scored = sorted(
                    records,
                    key=lambda rec: _local_relevance_score(rec.get("content", ""), query_vec),
                    reverse=True
                )
                return scored
            return records
    except Exception:
        pass

    result = await neo4j.get_user_memories(user_id, query=query, query_vec=query_vec)

    try:
        await r.set(cache_key, json.dumps(result), ex=_NEO4J_CACHE_TTL)
    except Exception:
        pass

    return result


# ── Timed wrappers — each prints its own duration ─────────────────────────────

async def _timed_neo4j(user_id, query, query_vec):
    t = time.monotonic()
    try:
        result = await _get_neo4j_cached(user_id, query, query_vec)
        print(f"  ⏱ [ctx] neo4j:  {(time.monotonic()-t)*1000:.0f}ms → {len(result)} memories")
        return result
    except Exception as e:
        print(f"  ⏱ [ctx] neo4j:  {(time.monotonic()-t)*1000:.0f}ms → FAILED: {e}")
        return []


async def _timed_mongo(user_id, session_id, query, query_vec):
    t = time.monotonic()
    try:
        result = await mongo.get_user_episodic_memories(
            user_id, limit=5, session_id=session_id,
            query=query or None, query_vec=query_vec
        )
        print(f"  ⏱ [ctx] mongo:  {(time.monotonic()-t)*1000:.0f}ms → {len(result)} episodes")
        return result
    except Exception as e:
        print(f"  ⏱ [ctx] mongo:  {(time.monotonic()-t)*1000:.0f}ms → FAILED: {e}")
        return []


async def _timed_redis(session_id, user_id, query, query_vec):
    t = time.monotonic()
    try:
        result = await redis_mgr.get_latest_summaries_for_context(
            session_id, user_id=user_id,
            query=query or None, query_vec=query_vec
        )
        print(f"  ⏱ [ctx] redis:  {(time.monotonic()-t)*1000:.0f}ms → {len(result)} summaries")
        return result
    except Exception as e:
        print(f"  ⏱ [ctx] redis:  {(time.monotonic()-t)*1000:.0f}ms → FAILED: {e}")
        return []


async def _timed_sqlite(session_id):
    t = time.monotonic()
    try:
        result = await sql.get_turns_from_last_summary(session_id)
        print(f"  ⏱ [ctx] sqlite: {(time.monotonic()-t)*1000:.0f}ms → {len(result)} turns")
        return result
    except Exception as e:
        print(f"  ⏱ [ctx] sqlite: {(time.monotonic()-t)*1000:.0f}ms → FAILED: {e}")
        return []


# ── Main context builder ───────────────────────────────────────────────────────

async def build_context(
    session_id: str,
    user_id: str,
    user_message: str = ""
) -> dict:
    """
    Parallel fetch from all 4 stores with single shared query embedding.
    Embed once, pass vector to all stores — no duplicate API calls.
    Per-step timing printed to console so slow stores are immediately visible.
    """
    t_total = time.monotonic()

    # ── Embed query once ──────────────────────────────────────
    query_vec = None
    if user_message:
        t_emb = time.monotonic()
        try:
            query_vec = await embed_text(user_message)
            print(f"  ⏱ [ctx] embed:  {(time.monotonic()-t_emb)*1000:.0f}ms")
        except Exception as e:
            print(f"  ⏱ [ctx] embed:  FAILED: {e}")
            query_vec = None

    # ── Parallel fetch — each store timed independently ───────
    memories, episodic, summaries, raw_turns = await asyncio.gather(
        _timed_neo4j(user_id, user_message, query_vec),
        _timed_mongo(user_id, session_id, user_message, query_vec),
        _timed_redis(session_id, user_id, user_message, query_vec),
        _timed_sqlite(session_id),
    )

    print(f"  ⏱ [ctx] total:  {(time.monotonic()-t_total)*1000:.0f}ms")

    return {
        "memories":          memories,
        "episodic_memories": episodic,
        "summaries":         summaries,
        "raw_turns":         raw_turns,
        "query_vec":         query_vec,
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
    parts     = []
    summaries = context.get("summaries", [])
    raw       = context.get("raw_turns", [])
    if summaries:
        last = summaries[-1]
        parts.append(f"Recent summary: {last['summary_text'][:200]}")
    if raw:
        for t in raw[-2:]:
            parts.append(f"User: {t['user_msg'][:120]}")
            parts.append(f"Assistant: {t['assistant_msg'][:120]}")
    return "\n".join(parts)