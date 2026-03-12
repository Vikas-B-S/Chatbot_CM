"""
memory/summarizer.py — 3-level hierarchical summarization pipeline

Trigger schedule (with default batch=3, summarize_at=6):
  Turn 6  → summarize T1-T3  → L0 S1
  Turn 9  → summarize T4-T6  → L0 S2
  Turn 12 → summarize T7-T9  → L0 S3
              3 L0s uncompressed → compress S1+S2+S3 → L1 M1 (covers T1-T9)
  Turn 15 → summarize T10-T12 → L0 S4
  Turn 18 → summarize T13-T15 → L0 S5
  Turn 21 → summarize T16-T18 → L0 S6
              3 L0s (S4+S5+S6) uncompressed → compress → L1 M2 (covers T10-T18)
  Turn 24 → summarize T19-T21 → L0 S7
  Turn 27 → summarize T22-T24 → L0 S8
  Turn 30 → summarize T25-T27 → L0 S9
              3 L0s (S7+S8+S9) uncompressed → compress → L1 M3 (covers T19-T27)
              3 L1s (M1+M2+M3) uncompressed → compress → L2 A1 (covers T1-T27)
              L2 A1 → write handoff summary for next session

  Pattern: L0 every 3 turns | L1 every 9 turns | L2 every 27 turns | Handoff with L2
"""
from db import sqlite_manager as sql
from db import redis_manager as redis_mgr
from db import mongo_manager as mongo
from memory.extractor import (
    summarize_turns, compress_summaries, compress_to_arc,
    create_episodic_narrative, create_handoff_summary
)
from memory.context_builder import should_summarize


async def check_and_run_summarization(
    session_id: str,
    user_id: str,
    turn_number: int,
    episodic_decision: dict = None,
) -> dict | None:
    """
    Called after every turn. Full pipeline:
      A. L0 batch summary (if batch boundary)
      B. L1 compression (if 3 uncompressed L0s exist)
      C. L2 arc compression (if 3 uncompressed L1s exist)
      D. Handoff summary (if L2 was just created)
      E. Episodic narrative (if router flagged this batch)
    """
    do_summarize, batch_start, batch_end = should_summarize(turn_number)
    if not do_summarize:
        return None

    turns = await sql.get_turns_in_range(session_id, batch_start, batch_end)
    if not turns:
        return None

    actual_start = turns[0]["turn_number"]
    actual_end   = turns[-1]["turn_number"]

    # ── A: L0 batch summary ───────────────────────────────────
    result     = await summarize_turns(turns)
    l0_id      = await redis_mgr.save_summary(
        session_id=session_id, user_id=user_id,
        batch_start=actual_start, batch_end=actual_end,
        summary_text=result["summary"], key_topics=result["key_topics"],
        level=0
    )
    await sql.mark_turns_summarized(session_id, actual_start, actual_end)

    # ── B: L1 compression ────────────────────────────────────
    l1_result = await _maybe_compress(session_id, user_id, level=0, target_level=1)

    # ── C: L2 arc compression ─────────────────────────────────
    l2_result = None
    if l1_result:
        l2_result = await _maybe_compress(session_id, user_id, level=1, target_level=2)

    # ── D: Handoff summary (written when L2 is created) ───────
    handoff_result = None
    if l2_result:
        handoff_result = await _write_handoff(session_id, user_id)

    # Episodic storage is handled by agent.py _background_store() directly.
    # episodic_decision param kept for API compatibility but not used here.
    episodic_stored = None

    return {
        "summarized":    True,
        "batch_start":   actual_start,
        "batch_end":     actual_end,
        "l0_id":         l0_id,
        "l1":            l1_result,
        "l2":            l2_result,
        "handoff":       handoff_result,
        "episodic":      episodic_stored,
    }


async def _maybe_compress(
    session_id: str,
    user_id: str,
    level: int,
    target_level: int
) -> dict | None:
    """
    Compress `_COMPRESS_AT` summaries at `level` into one summary at `target_level`.
    Returns result dict or None if not enough uncompressed summaries exist.
    """
    candidates = await redis_mgr.get_uncompressed_at_level(session_id, level)
    if len(candidates) < redis_mgr._COMPRESS_AT:
        return None

    # Take the oldest _COMPRESS_AT uncompressed summaries
    batch = sorted(candidates, key=lambda s: s["batch_start"])[:redis_mgr._COMPRESS_AT]

    if target_level == 1:
        # L0→L1: standard 3-batch compression
        compressed = await compress_summaries(batch)
    else:
        # L1→L2: arc-level compression (broader narrative)
        compressed = await compress_to_arc(batch)

    if not compressed.get("summary"):
        return None

    new_id = await redis_mgr.save_summary(
        session_id=session_id, user_id=user_id,
        batch_start=batch[0]["batch_start"],
        batch_end=batch[-1]["batch_end"],
        summary_text=compressed["summary"],
        key_topics=compressed.get("key_topics", []),
        level=target_level,
        source_summary_ids=[s["summary_id"] for s in batch],
    )

    label = f"L{target_level}"
    print(f"  ↳ {label} compressed: T{batch[0]['batch_start']}-T{batch[-1]['batch_end']}")
    return {
        "summary_id":  new_id,
        "level":       target_level,
        "covers":      f"T{batch[0]['batch_start']}-T{batch[-1]['batch_end']}",
        "from_count":  len(batch),
        "summary":     compressed["summary"][:120] + "...",
    }


async def _write_handoff(session_id: str, user_id: str) -> dict | None:
    """
    Generate and store the cross-session handoff summary.
    Called whenever an L2 is created — gives the next session a solid starting point.
    """
    # Get the best summary to base the handoff on
    best = await redis_mgr.get_best_session_summary(session_id)
    if not best:
        return None

    handoff = await create_handoff_summary(best)
    if not handoff.get("summary"):
        return None

    await redis_mgr.save_handoff_summary(
        user_id=user_id,
        summary_text=handoff["summary"],
        key_topics=handoff.get("key_topics", []),
        session_id=session_id,
    )

    print(f"  ↳ Handoff written for user {user_id[:12]}...")
    return {"written": True, "summary": handoff["summary"][:100] + "..."}