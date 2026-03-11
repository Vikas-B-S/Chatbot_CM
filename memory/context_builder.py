"""
memory/context_builder.py
Builds the full context window for each chat turn.

Context order (most persistent → most recent):
  1. Graphiti (Neo4j) — hybrid search scoped to user_message for relevance
  2. Episodic memories (MongoDB) — recent meaningful narrative episodes
  3. Summaries from Redis — compressed history (prefers L1 meta-summaries)
  4. Last 6 raw turns (SQLite) — the immediate sliding window

Key change from v1: get_user_memories() now receives the user_message so
Graphiti does a TARGETED semantic search instead of a broad profile dump.
This means context is always relevant to what the user just asked.
"""
import asyncio
from config import get_settings
from db import sqlite_manager as sql
from db import neo4j_manager as neo4j
from db import redis_manager as redis_mgr
from db import mongo_manager as mongo

settings = get_settings()


def should_summarize(turn_number: int) -> tuple[bool, int, int]:
    """
    Pure function — no DB access needed.
    Returns (trigger, batch_start, batch_end).

    Turn 6  → (True, 1, 3)
    Turn 9  → (True, 4, 6)
    Turn 12 → (True, 7, 9)  ...etc.
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


async def build_context(
    session_id: str,
    user_id: str,
    user_message: str = ""     # ← NEW: passed through for targeted Graphiti search
) -> dict:
    """
    Parallel fetch from all stores. Graceful degradation if any store is down.

    user_message is forwarded to get_user_memories() so Graphiti performs
    a targeted hybrid search (semantic + BM25 + graph) against what the
    user just said — returning the most RELEVANT memories, not just all of them.

    If user_message is empty (e.g. context preview endpoint) falls back
    to a broad profile sweep.
    """
    def safe(v):
        return v if not isinstance(v, Exception) else []

    results = await asyncio.gather(
        neo4j.get_user_memories(user_id, query=user_message or None),
        mongo.get_user_episodic_memories(user_id, limit=3, session_id=session_id, query=user_message or None),
        redis_mgr.get_latest_summaries_for_context(session_id),
        sql.get_last_n_turns(session_id, n=settings.max_raw_turns),
        return_exceptions=True
    )

    memories, episodic, summaries, raw_turns = results
    return {
        "memories":          safe(memories),
        "episodic_memories": safe(episodic),
        "summaries":         safe(summaries),
        "raw_turns":         safe(raw_turns),
    }


def format_context_for_prompt(context: dict) -> str:
    """
    Format all context into a clean structured prompt section.
    Ordered: most persistent (Graphiti facts) → most recent (raw turns).

    Graphiti returns flat facts with inferred memory_type — we group them
    by type for clean presentation to the LLM.
    """
    parts = []

    # ── 1. Graphiti memory — grouped by type ──────────────────
    memories = context.get("memories", [])
    if memories:
        # Group by memory_type
        by_type: dict[str, list] = {}
        for m in memories:
            t = m.get("memory_type", "fact")
            by_type.setdefault(t, []).append(m.get("content", ""))

        sections = [
            ("fact",       "## About This User"),
            ("preference", "## User Preferences"),
            ("goal",       "## User Goals"),
            ("constraint", "## User Constraints"),
        ]
        for mtype, header in sections:
            items = by_type.get(mtype, [])
            if items:
                parts.append(header)
                for item in items:
                    parts.append(f"- {item}")

    # ── 2. Episodic memories from MongoDB ─────────────────────
    episodic = context.get("episodic_memories", [])
    if episodic:
        parts.append("\n## Remembered Episodes")
        for ep in episodic[:2]:
            title   = ep.get("title", "Episode")
            outcome = ep.get("outcome", "")
            snippet = ep.get("content", "")[:220]
            parts.append(f"[{title} — {outcome}]: {snippet}…")

    # ── 3. Conversation history summaries from Redis ───────────
    summaries = context.get("summaries", [])
    if summaries:
        parts.append("\n## Conversation History")
        for s in summaries:
            label = "Meta" if s.get("level", 0) >= 1 else "Summary"
            parts.append(
                f"[{label} T{s['batch_start']}-T{s['batch_end']}]: {s['summary_text']}"
            )

    # ── 4. Last N raw turns from SQLite ───────────────────────
    raw = context.get("raw_turns", [])
    if raw:
        parts.append(f"\n## Recent Conversation (last {len(raw)} turns)")
        for t in raw:
            parts.append(f"User: {t['user_msg']}")
            parts.append(f"Assistant: {t['assistant_msg']}")

    return "\n".join(parts) if parts else "No prior context available."


def format_context_for_router(context: dict) -> str:
    """
    Compact context string for the router's LLM call.
    Only needs recent summaries + last 2 turns.
    """
    parts = []
    summaries = context.get("summaries", [])
    raw = context.get("raw_turns", [])

    if summaries:
        last = summaries[-1]
        parts.append(f"Recent summary: {last['summary_text'][:200]}")
    if raw:
        for t in raw[-2:]:
            parts.append(f"User: {t['user_msg'][:120]}")
            parts.append(f"Assistant: {t['assistant_msg'][:120]}")
    return "\n".join(parts)