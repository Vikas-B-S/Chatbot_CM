"""
db/redis_manager.py — 3-level hierarchical summary store with semantic retrieval

Key schema
──────────
  summary:{session_id}          Sorted Set   full summary docs, score=batch_start
  emb:{session_id}              Hash         summary_id → JSON embedding vector
  handoff:{user_id}             String       cross-session handoff doc (JSON)
  handoff_emb:{user_id}         String       handoff embedding vector (JSON)

Why separate embedding hashes
──────────────────────────────
  Semantic search needs to score EVERY summary against the query vector.
  If embeddings live inside the full JSON docs (sorted set members), we must
  deserialize every full document just to get the float array — wasteful.

  Instead:
    Write time:  embed → store full doc in sorted set + store ONLY the
                          embedding in emb:{session_id} hash (tiny)
    Query time:  1. HGETALL emb:{session_id}          → all embeddings, tiny
                 2. Cosine score all in Python         → get top-k IDs
                 3. Fetch only those k full docs       → minimal deserialization

  This means scoring is cheap regardless of how large the full docs are.

Summary levels
──────────────
  L0  3-turn batch   TTL 7d
  L1  9-turn window  TTL 30d   (3 L0s compressed)
  L2  27-turn arc    TTL 90d   (3 L1s compressed)
  HS  cross-session  TTL 180d  (written when L2 is created)

Context assembly (max 8 summaries, always)
──────────────────────────────────────────
  With query (semantic mode):
    1. Handoff always included
    2. Best 1 high-level summary guaranteed (L2 or L1)
    3. Remaining slots filled by cosine similarity to query
    4. Most recent L0 always included (last batch needs no filtering)

  Without query (fallback):
    Recency-based selection (same as old behaviour)
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

from config import get_settings
from db.embedder import embed_text, cosine_similarity

settings = get_settings()
_redis: Optional[aioredis.Redis] = None

# TTLs
_TTL_L0      = 7   * 86400
_TTL_L1      = 30  * 86400
_TTL_L2      = 90  * 86400
_TTL_HANDOFF = 180 * 86400

# How many of level N trigger compression to level N+1
_COMPRESS_AT = 3

# Level score boosts (ensures high-level summaries win ties in semantic scoring)
_LEVEL_BOOST = {0: 0.0, 1: 0.08, 2: 0.15}


# ─── Connection ───────────────────────────────────────────────────────────────

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Write ────────────────────────────────────────────────────────────────────

async def save_summary(
    session_id: str,
    user_id: str,
    batch_start: int,
    batch_end: int,
    summary_text: str,
    key_topics: list = None,
    level: int = 0,
    source_summary_ids: list = None,
) -> str:
    """
    Store a summary document + its embedding separately.

    Two writes per summary:
      1. Full doc → sorted set  (score = batch_start for chronological ordering)
      2. Embedding only → hash  (for fast cosine scoring without full deserialization)
    """
    summary_id = f"sum_{uuid.uuid4().hex[:12]}"

    # Embed: text + topics together give richer semantic signal
    topics_str  = " ".join(key_topics or [])
    embed_input = f"{summary_text} {topics_str}".strip()
    embedding   = await embed_text(embed_input)

    doc = {
        "summary_id":         summary_id,
        "session_id":         session_id,
        "user_id":            user_id,
        "batch_start":        batch_start,
        "batch_end":          batch_end,
        "level":              level,
        "summary_text":       summary_text,
        "key_topics":         key_topics or [],
        "source_summary_ids": source_summary_ids or [],
        "token_estimate":     _token_estimate(summary_text),
        "created_at":         _now_iso(),
        # NOTE: embedding NOT stored inside the doc — lives in emb: hash instead
    }

    r = await get_redis()
    ttl = {0: _TTL_L0, 1: _TTL_L1, 2: _TTL_L2}.get(level, _TTL_L0)

    # Write 1: full doc in sorted set
    await r.zadd(f"summary:{session_id}", {json.dumps(doc): batch_start})
    await r.expire(f"summary:{session_id}", ttl)

    # Write 2: embedding only in hash (summary_id → JSON float array)
    await r.hset(f"emb:{session_id}", summary_id, json.dumps(embedding))
    await r.expire(f"emb:{session_id}", ttl)

    return summary_id


async def save_handoff_summary(
    user_id: str,
    summary_text: str,
    key_topics: list,
    session_id: str
):
    """
    Store cross-session handoff + its embedding.
    Overwritten on each session end — always the most recent full picture.
    """
    doc = {
        "summary_text": summary_text,
        "key_topics":   key_topics,
        "session_id":   session_id,
        "created_at":   _now_iso(),
    }

    # Embed handoff too so it participates in semantic scoring
    embedding = await embed_text(f"{summary_text} {' '.join(key_topics)}")

    r = await get_redis()
    await r.set(f"handoff:{user_id}",     json.dumps(doc),      ex=_TTL_HANDOFF)
    await r.set(f"handoff_emb:{user_id}", json.dumps(embedding), ex=_TTL_HANDOFF)


async def get_handoff_summary(user_id: str) -> dict | None:
    r   = await get_redis()
    raw = await r.get(f"handoff:{user_id}")
    if not raw:
        return None
    await r.expire(f"handoff:{user_id}",     _TTL_HANDOFF)
    await r.expire(f"handoff_emb:{user_id}", _TTL_HANDOFF)
    return json.loads(raw)


# ─── Read ─────────────────────────────────────────────────────────────────────

async def get_session_summaries(session_id: str, min_level: int = 0) -> list:
    """All summaries for a session, sorted by batch_start ascending."""
    r       = await get_redis()
    members = await r.zrange(f"summary:{session_id}", 0, -1)
    summaries = [json.loads(m) for m in members]
    if min_level > 0:
        summaries = [s for s in summaries if s.get("level", 0) >= min_level]
    return summaries


async def get_latest_summaries_for_context(
    session_id: str,
    user_id: str = None,
    query: str = None,
    query_vec: list = None,   # pre-computed embedding — skip embed call if provided
    max_total: int = 8,
) -> list:
    """
    Assemble the optimal summary chain for context injection.

    SEMANTIC MODE (query provided):
      Step 1 — fetch all embeddings from emb:{session_id} hash (tiny, fast)
      Step 2 — embed the query (1 API call)
      Step 3 — cosine score all summaries in Python (cheap float math)
      Step 4 — pick top-k by score + structural guarantees
      Step 5 — fetch ONLY those k full docs from sorted set

    FALLBACK MODE (no query):
      Recency-based: handoff + latest L2s + uncovered L1s + uncovered L0s
    """
    all_s = await get_session_summaries(session_id)

    # Always include handoff first
    result = []
    handoff_emb = None
    if user_id:
        handoff = await get_handoff_summary(user_id)
        if handoff:
            result.append(_handoff_to_summary(handoff))
            # Load handoff embedding for scoring
            r   = await get_redis()
            raw = await r.get(f"handoff_emb:{user_id}")
            if raw:
                handoff_emb = json.loads(raw)

    if not all_s:
        return result

    # Build coverage sets (which summaries are already consumed by higher levels)
    consumed_by_l2: set[str] = set()
    consumed_by_l1: set[str] = set()
    l2s = [s for s in all_s if s.get("level") == 2]
    l1s = [s for s in all_s if s.get("level") == 1]
    l0s = [s for s in all_s if s.get("level") == 0]
    for s in l2s: consumed_by_l2.update(s.get("source_summary_ids", []))
    for s in l1s: consumed_by_l1.update(s.get("source_summary_ids", []))

    uncovered_l1s = [s for s in l1s if s["summary_id"] not in consumed_by_l2]
    uncovered_l0s = [s for s in l0s if s["summary_id"] not in consumed_by_l1]

    # Most recent L0 — always include regardless of semantic score
    most_recent_l0 = sorted(uncovered_l0s, key=lambda s: s["batch_start"])[-1:] if uncovered_l0s else []
    remaining_l0s  = [s for s in uncovered_l0s if s not in most_recent_l0]

    # Candidates for semantic scoring
    candidates = l2s + uncovered_l1s + remaining_l0s
    slots      = max_total - len(result) - len(most_recent_l0)

    if not query or not candidates:
        # ── Recency fallback ──────────────────────────────────
        result.extend(sorted(l2s,          key=lambda s: s["batch_start"])[-2:])
        result.extend(sorted(uncovered_l1s, key=lambda s: s["batch_start"])[-3:])
        result.extend(most_recent_l0)
    else:
        # ── Semantic scoring — fast path ──────────────────────
        # Step 1: fetch all embeddings from hash (tiny payload)
        r         = await get_redis()
        emb_hash  = await r.hgetall(f"emb:{session_id}")
        query_vec = query_vec or await embed_text(query)

        # Step 2: score all candidates — only float math
        scored = []
        for s in candidates:
            sid = s["summary_id"]
            raw_emb = emb_hash.get(sid)
            if raw_emb:
                sim = cosine_similarity(query_vec, json.loads(raw_emb))
            else:
                sim = 0.0  # no embedding (old record) — will be outscored
            boost = _LEVEL_BOOST.get(s.get("level", 0), 0.0)
            scored.append((s, sim + boost))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 3: guarantee at least 1 high-level summary
        has_high = any(s.get("level", 0) >= 1 for s, _ in scored[:slots])
        if not has_high:
            high = next((s for s, _ in scored if s.get("level", 0) >= 1), None)
            if high:
                result.append(high)
                slots -= 1
                scored = [(s, sc) for s, sc in scored if s["summary_id"] != high["summary_id"]]

        # Step 4: fill remaining slots with top semantic matches
        for s, sim in scored[:slots]:
            result.append(s)
            if sim > 0:
                print(f"  ↳ Redis [{s['summary_id'][:8]}] "
                      f"L{s.get('level',0)} "
                      f"T{s['batch_start']}-T{s['batch_end']} "
                      f"sim={sim:.3f}")

        result.extend(most_recent_l0)

    # Deduplicate and sort chronologically
    seen: set[str] = set()
    deduped = []
    for s in result:
        sid = s.get("summary_id", "")
        if sid not in seen:
            seen.add(sid)
            deduped.append(s)

    return sorted(deduped[:max_total], key=lambda s: s.get("batch_start", 0))


def _handoff_to_summary(handoff: dict) -> dict:
    """Convert handoff doc into a pseudo-summary entry for context rendering."""
    return {
        "summary_id":   "handoff",
        "level":        99,       # sentinel — rendered as "Previous Session" in context
        "batch_start":  0,
        "batch_end":    0,
        "summary_text": handoff["summary_text"],
        "key_topics":   handoff.get("key_topics", []),
        "session_id":   handoff.get("session_id", ""),
        "created_at":   handoff.get("created_at", ""),
    }


# ─── Compression triggers ─────────────────────────────────────────────────────

async def get_uncompressed_at_level(session_id: str, level: int) -> list:
    """
    Returns summaries at `level` not yet used as source for level+1.
    When count >= _COMPRESS_AT, summarizer triggers compression.
    """
    all_s = await get_session_summaries(session_id)
    used: set[str] = set()
    for s in all_s:
        if s.get("level", 0) == level + 1:
            used.update(s.get("source_summary_ids", []))
    return [s for s in all_s if s.get("level", 0) == level and s["summary_id"] not in used]


async def get_best_session_summary(session_id: str) -> dict | None:
    """Best available summary for handoff generation. Priority: L2 > L1 > L0."""
    for level in [2, 1, 0]:
        summaries = [s for s in await get_session_summaries(session_id)
                     if s.get("level") == level]
        if summaries:
            return max(summaries, key=lambda s: s["batch_end"])
    return None


# ─── Delete ───────────────────────────────────────────────────────────────────

async def delete_user_summaries(session_ids: list[str], user_id: str = None) -> int:
    """Delete all summary + embedding keys for sessions, plus user handoff."""
    r       = await get_redis()
    deleted = 0
    for sid in (session_ids or []):
        for key in [f"summary:{sid}", f"emb:{sid}"]:
            if await r.exists(key):
                await r.delete(key)
                deleted += 1
    if user_id:
        for key in [f"handoff:{user_id}", f"handoff_emb:{user_id}"]:
            if await r.exists(key):
                await r.delete(key)
                deleted += 1
    return deleted
