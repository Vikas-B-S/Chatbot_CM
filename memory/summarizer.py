"""
memory/summarizer.py
Hierarchical summarization pipeline — triggered after every turn.

Trigger logic:
  Turn 6  → summarize turns 1-3   → Redis L0 (S1)
  Turn 9  → summarize turns 4-6   → Redis L0 (S2)
              compress S1 + S2     → Redis L1 (M1, covers T1-T6)
  Turn 12 → summarize turns 7-9   → Redis L0 (S3)
  Turn 15 → summarize turns 10-12 → Redis L0 (S4)
              compress S3 + S4     → Redis L1 (M2, covers T7-T12)

Note: should_summarize() is a plain (non-async) function in context_builder.
      Do NOT await it — it does no I/O.
"""
from db import sqlite_manager as sql
from db import redis_manager as redis_mgr
from db import mongo_manager as mongo
from memory.extractor import summarize_turns, compress_summaries, create_episodic_narrative
from memory.context_builder import should_summarize


async def check_and_run_summarization(
    session_id: str,
    user_id: str,
    turn_number: int,
    episodic_decision: dict = None
) -> dict | None:
    """
    Called after every turn. Runs summarization + meta-compression if triggered.
    Returns result dict or None if not a summarization turn.

    should_summarize() is a pure function — call it directly, no await.
    """
    # Pure function — no await needed
    do_summarize, batch_start, batch_end = should_summarize(turn_number)
    if not do_summarize:
        return None

    turns = await sql.get_turns_in_range(session_id, batch_start, batch_end)
    if not turns:
        return None

    actual_start = turns[0]["turn_number"]
    actual_end   = turns[-1]["turn_number"]

    # ── A: Summarise batch → Redis L0 ────────────────────────
    result       = await summarize_turns(turns)
    summary_text = result["summary"]
    key_topics   = result["key_topics"]

    new_id = await redis_mgr.save_summary(
        session_id=session_id,
        user_id=user_id,
        batch_start=actual_start,
        batch_end=actual_end,
        summary_text=summary_text,
        key_topics=key_topics,
        level=0
    )
    await sql.mark_turns_summarized(session_id, actual_start, actual_end)

    # ── B: Meta-compression → Redis L1 ───────────────────────
    meta = await _maybe_compress_meta(session_id, user_id)

    # ── C: Episodic narrative for this batch → MongoDB ────────
    episodic_stored = None
    if episodic_decision and episodic_decision.get("should_store"):
        narrative = await create_episodic_narrative(turns, episodic_decision)
        ep_id = await mongo.store_episodic_memory(
            user_id=user_id,
            session_id=session_id,
            title=narrative["title"],
            content=narrative["content"],
            outcome=narrative["outcome"],
            turn_start=actual_start,
            turn_end=actual_end,
            key_entities=episodic_decision.get("key_entities", []),
            emotional_tone=episodic_decision.get("emotional_tone", "neutral"),
            tags=episodic_decision.get("tags", [])
        )
        episodic_stored = {"memory_id": ep_id, "title": narrative["title"]}

    return {
        "summarized":    True,
        "batch_start":   actual_start,
        "batch_end":     actual_end,
        "summary_id":    new_id,
        "summary_text":  summary_text,
        "key_topics":    key_topics,
        "meta_compression": meta,
        "episodic_stored":  episodic_stored
    }


async def _maybe_compress_meta(session_id: str, user_id: str) -> dict | None:
    """
    If there are exactly 2 uncompressed L0 summaries → merge into L1.
    """
    pair = await redis_mgr.get_oldest_two_uncompressed_level0(session_id)
    if len(pair) < 2:
        return None

    s_a, s_b = pair[0], pair[1]
    meta = await compress_summaries(s_a, s_b)
    if not meta.get("meta_summary"):
        return None

    meta_text = meta["meta_summary"]
    if meta.get("evolution_note"):
        meta_text += f" {meta['evolution_note']}"

    meta_id = await redis_mgr.save_summary(
        session_id=session_id,
        user_id=user_id,
        batch_start=s_a["batch_start"],
        batch_end=s_b["batch_end"],
        summary_text=meta_text,
        key_topics=meta.get("key_topics", []),
        level=1,
        source_summary_ids=[s_a["summary_id"], s_b["summary_id"]]
    )

    return {
        "meta_id":          meta_id,
        "covers":           f"T{s_a['batch_start']}-T{s_b['batch_end']}",
        "compressed_from":  [s_a["summary_id"], s_b["summary_id"]],
        "meta_text":        meta_text
    }
