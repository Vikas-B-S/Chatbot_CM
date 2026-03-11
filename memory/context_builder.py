"""
memory/context_builder.py
Builds the full context window for each chat turn.

Context order (most persistent → most recent):
  1. User memories from Neo4j   — facts, preferences, goals, constraints
  2. Episodic memories (MongoDB)— recent meaningful narrative episodes
  3. Summaries from Redis       — compressed history (prefers L1 meta-summaries)
  4. Last 6 raw turns (SQLite)  — the immediate sliding window

All fetches run concurrently. Graceful degradation if any store is down.
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
    user_message: str = ""    # add this line
) -> dict:
    """Parallel fetch from all stores, graceful degradation on errors."""
    def safe(v):
        return v if not isinstance(v, Exception) else []

    results = await asyncio.gather(
        neo4j.get_user_memories(user_id),
        mongo.get_user_episodic_memories(user_id, limit=3, session_id=session_id),
        redis_mgr.get_latest_summaries_for_context(session_id),
        sql.get_last_n_turns(session_id, n=settings.max_raw_turns),
        return_exceptions=True
    )

    memories, episodic, summaries, raw_turns = results
    return {
        "memories":         safe(memories),
        "episodic_memories": safe(episodic),
        "summaries":        safe(summaries),
        "raw_turns":        safe(raw_turns),
    }


def format_context_for_prompt(context: dict) -> str:
    """
    Format all context into a clean, structured prompt section.
    Ordered: most persistent (facts) → most recent (raw turns).
    """
    parts = []

    # ── 1. User Memory from Neo4j (all types) ─────────────────
    memories = context.get("memories", [])
    if memories:
        by_type: dict[str, list] = {}
        for m in memories:
            by_type.setdefault(m.get("memory_type", "fact"), []).append(m["content"])

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
    Only needs recent summaries + last user message for good routing decisions.
    """
    parts = []
    summaries = context.get("summaries", [])
    raw = context.get("raw_turns", [])

    if summaries:
        last = summaries[-1]
        parts.append(f"Recent summary: {last['summary_text'][:200]}")
    if raw:
        for t in raw[-2:]:   # last 2 turns for context
            parts.append(f"User: {t['user_msg'][:120]}")
            parts.append(f"Assistant: {t['assistant_msg'][:120]}")
    return "\n".join(parts)
