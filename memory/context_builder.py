"""
memory/context_builder.py — Context assembly with session-scoped retrieval

Session-scoping behaviour
──────────────────────────
  Neo4j facts        → user-scoped  (universal, no session filter)
  Neo4j prefs/goals/constraints → user-scoped (persists across sessions, decays if unused)
  MongoDB episodic   → session-scoped (only episodes from current session_id)
  Redis summaries    → session-scoped (already was, unchanged)
  SQLite raw turns   → session-scoped (already was, unchanged)

Key optimisations
─────────────────
  1. Embed once — query embedded ONE time upfront, vector passed to all stores.
  2. Neo4j result cache — cached in Redis for 30s (keyed by user+session).
  3. All 4 stores fetched in parallel via asyncio.gather.
  4. Blind-spot fix — raw turns come from get_turns_from_last_summary().

Token budget enforcement (v3.5)
────────────────────────────────
  Three targeted limits to prevent context window overflow:

  1. Neo4j:  LIMIT 20 in Cypher query (fixed in neo4j_manager.py)
             20 memories × ~15 tokens = ~300 tokens max

  2. SQLite: Each raw turn truncated in format_context_for_prompt():
             user_msg     → max _TURN_MSG_MAX_CHARS   (800 chars  ≈ 200 tokens)
             assistant_msg → max _TURN_REPLY_MAX_CHARS (1200 chars ≈ 300 tokens)
             Keeps the most recent/relevant part of each message.

  3. Measurement: estimate_tokens(text) counts approximate tokens before
             the prompt is sent to the LLM. Logged as a warning if over
             _TOKEN_WARN_THRESHOLD. Used in agent.py stream_tokens().

  Budget estimate at worst case after fixes:
    Neo4j    20 memories  × ~15t = 300
    MongoDB   5 episodes  × ~80t = 400
    Redis     3 summaries × ~90t = 270
    SQLite    3 turns      × ~125t= 375  ← was unbounded, now capped
    Headers/labels               = 150
    ─────────────────────────────────────
    Context section total        ~ 1495 tokens  (was 4200+ unbounded)
    + system prompt base         =  120
    + user message               =  500 (typical)
    + max_tokens reserved        =  800
    ─────────────────────────────────────
    Grand total                  ~ 2915 tokens  ✓ safe for all major models
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

# ── Token budget constants ─────────────────────────────────────────────────────
# SQLite raw turn truncation limits
_TURN_MSG_MAX_CHARS   = 800    # user message  — ~200 tokens
_TURN_REPLY_MAX_CHARS = 1200   # assistant msg — ~300 tokens

# Warning threshold — log if context section exceeds this
_TOKEN_WARN_THRESHOLD = 2500   # tokens in context_section alone


def estimate_tokens(text: str) -> int:
    """
    Fast approximate token count. 1 token ≈ 4 characters for English text.
    Used for budget monitoring — not passed to the API.
    Accurate enough to catch overflow before it happens.
    """
    return max(1, len(text) // 4)


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


async def _get_neo4j_cached(
    user_id: str,
    session_id: str,
    query: str,
    query_vec: list
) -> list:
    """
    Fetch Neo4j memories with 30s Redis cache.
    Cache key includes session_id — switching sessions gets fresh results.
    """
    from db.redis_manager import get_redis
    r         = await get_redis()
    cache_key = f"neo4j_cache:{user_id}:{session_id}"

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

    result = await neo4j.get_user_memories(
        user_id,
        session_id=session_id,
        query=query,
        query_vec=query_vec
    )

    try:
        await r.set(cache_key, json.dumps(result), ex=_NEO4J_CACHE_TTL)
    except Exception:
        pass

    return result


# ── Timed wrappers ─────────────────────────────────────────────────────────────

async def _timed_neo4j(user_id: str, session_id: str, query: str, query_vec: list):
    t = time.monotonic()
    try:
        result = await _get_neo4j_cached(user_id, session_id, query, query_vec)
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] neo4j:  {ms}ms → {len(result)} memories")
        return result, ms
    except Exception as e:
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] neo4j:  {ms}ms → FAILED: {e}")
        return [], ms


async def _timed_mongo(user_id: str, session_id: str, query: str, query_vec: list):
    t = time.monotonic()

    # Gate: skip MongoDB entirely for trivial messages.
    # The router heuristic pre-filter already classifies messages like
    # "thank you", "ok", "yes", "hello", "it is going good" as skip.
    # These messages have no episodic relevance — fetching MongoDB for them
    # wastes ~100ms+ (especially after the keyword-miss fallback was added)
    # and contributes nothing to the LLM's response quality.
    # Cost: one cheap regex check (~0ms) saves a full MongoDB round-trip.
    if query:
        from core.router import _should_skip_router
        skip, reason = _should_skip_router(query)
        if skip:
            ms = round((time.monotonic() - t) * 1000)
            print(f"  ⏱ [ctx] mongo:  {ms}ms → skipped (trivial: {reason})")
            return [], ms

    try:
        result = await mongo.get_user_episodic_memories(
            user_id,
            limit=5,
            session_id=session_id,
            query=query or None,
            query_vec=query_vec
        )
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] mongo:  {ms}ms → {len(result)} episodes")
        return result, ms
    except Exception as e:
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] mongo:  {ms}ms → FAILED: {e}")
        return [], ms


async def _timed_redis(session_id: str, user_id: str, query: str, query_vec: list):
    t = time.monotonic()
    try:
        result = await redis_mgr.get_latest_summaries_for_context(
            session_id, user_id=user_id,
            query=query or None, query_vec=query_vec
        )
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] redis:  {ms}ms → {len(result)} summaries")
        return result, ms
    except Exception as e:
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] redis:  {ms}ms → FAILED: {e}")
        return [], ms


async def _timed_sqlite(session_id: str):
    t = time.monotonic()
    try:
        result = await sql.get_turns_from_last_summary(session_id)
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] sqlite: {ms}ms → {len(result)} turns")
        return result, ms
    except Exception as e:
        ms = round((time.monotonic() - t) * 1000)
        print(f"  ⏱ [ctx] sqlite: {ms}ms → FAILED: {e}")
        return [], ms


# ── Main context builder ───────────────────────────────────────────────────────

async def build_context(
    session_id: str,
    user_id: str,
    user_message: str = ""
) -> dict:
    """
    Parallel fetch from all 4 stores with single shared query embedding.
    Returns per-store timing and hit status alongside results for UI display.
    """
    t_total = time.monotonic()

    query_vec = None
    if user_message:
        t_emb = time.monotonic()
        try:
            query_vec = await embed_text(user_message)
            print(f"  ⏱ [ctx] embed:  {round((time.monotonic()-t_emb)*1000)}ms")
        except Exception as e:
            print(f"  ⏱ [ctx] embed:  FAILED: {e}")
            query_vec = None

    (memories, neo4j_ms), (episodic, mongo_ms), (summaries, redis_ms), (raw_turns, sqlite_ms) = \
        await asyncio.gather(
            _timed_neo4j(user_id, session_id, user_message, query_vec),
            _timed_mongo(user_id, session_id, user_message, query_vec),
            _timed_redis(session_id, user_id, user_message, query_vec),
            _timed_sqlite(session_id),
        )

    total_ms = round((time.monotonic() - t_total) * 1000)
    print(f"  ⏱ [ctx] total:  {total_ms}ms")

    return {
        "memories":          memories,
        "episodic_memories": episodic,
        "summaries":         summaries,
        "raw_turns":         raw_turns,
        "query_vec":         query_vec,
        "session_id":        session_id,
        "user_id":           user_id,
        # Per-store timing + hit status — used by agent.py for the done chunk
        "store_timings": {
            "neo4j":  {"ms": neo4j_ms,  "count": len(memories),  "hit": len(memories) > 0},
            "mongo":  {"ms": mongo_ms,  "count": len(episodic),  "hit": len(episodic) > 0},
            "redis":  {"ms": redis_ms,  "count": len(summaries), "hit": len(summaries) > 0},
            "sqlite": {"ms": sqlite_ms, "count": len(raw_turns), "hit": len(raw_turns) > 0},
            "total_ms": total_ms,
        },
    }


def format_context_for_prompt(context: dict) -> str:
    """Format all context into structured prompt section for LLM."""
    parts = []

    memories = context.get("memories", [])
    if memories:
        by_type: dict[str, list] = {}
        for m in memories:
            t = m.get("memory_type", "fact")
            by_type.setdefault(t, []).append(m.get("content", ""))

        for mtype, header in [
            ("fact",       "## About This User"),
            ("preference", "## User Preferences (this session)"),
            ("goal",       "## User Goals (this session)"),
            ("constraint", "## User Constraints (this session)"),
        ]:
            items = by_type.get(mtype, [])
            if items:
                parts.append(header)
                for item in items:
                    parts.append(f"- {item}")

    episodic = context.get("episodic_memories", [])
    if episodic:
        parts.append("\n## Remembered Episodes (this session)")
        for ep in episodic[:2]:
            title   = ep.get("title", "Episode")
            outcome = ep.get("outcome", "")
            snippet = ep.get("content", "")[:220]
            parts.append(f"[{title} — {outcome}]: {snippet}…")

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

    raw = context.get("raw_turns", [])
    if raw:
        parts.append(
            f"\n## Recent Conversation "
            f"(turns {raw[0]['turn_number']}-{raw[-1]['turn_number']})"
        )
        for t in raw:
            # Truncate long messages to prevent context overflow.
            # Keep the tail (most recent part) — more relevant than the start.
            user_msg      = t['user_msg']
            assistant_msg = t['assistant_msg']

            if len(user_msg) > _TURN_MSG_MAX_CHARS:
                user_msg = "…" + user_msg[-_TURN_MSG_MAX_CHARS:]

            if len(assistant_msg) > _TURN_REPLY_MAX_CHARS:
                assistant_msg = "…" + assistant_msg[-_TURN_REPLY_MAX_CHARS:]

            parts.append(f"User: {user_msg}")
            parts.append(f"Assistant: {assistant_msg}")

    result = "\n".join(parts) if parts else "No prior context available."

    # ── Token budget warning ──────────────────────────────────
    token_estimate = estimate_tokens(result)
    if token_estimate > _TOKEN_WARN_THRESHOLD:
        print(f"  ⚠ [ctx] context section is large: "
              f"~{token_estimate} tokens "
              f"(threshold={_TOKEN_WARN_THRESHOLD}). "
              f"Consider reducing limits.")
    else:
        print(f"  · [ctx] context section: ~{token_estimate} tokens")

    return result


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