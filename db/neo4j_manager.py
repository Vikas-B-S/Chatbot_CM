"""
db/neo4j_manager.py — Hybrid Graphiti + Direct Cypher memory manager

Architecture
────────────
Two parallel systems work together:

  1. MemoryRecord nodes (Direct Cypher)
     Owns ALL state management: active/inactive, deduplication, reactivation.
     Written and read via raw Cypher queries on the Neo4j driver.
     This is the source of truth for "what is currently true about the user".

  2. Graphiti episodes
     Purely for semantic search indexing. Every state change writes a new
     episode so Graphiti's hybrid search can find relevant memories during
     context building. Graphiti does NOT own state — we do.

Session-scoping rules
─────────────────────
  All memory types (fact, preference, goal, constraint) are USER-SCOPED.
  They persist across sessions. Forgetting is handled by the confidence
  decay mechanism, not by session boundaries.

MemoryRecord node schema
────────────────────────
  (:MemoryRecord {
    user_id:           string   — scopes to one user
    session_id:        string   — session that last activated this record
    canonical_key:     string   — stable snake_case identifier e.g. "coding_language"
    memory_type:       string   — fact | preference | goal | constraint
    content:           string   — the actual memory text
    content_hash:      string   — MD5 of normalised content (dedup key)
    status:            string   — "active" | "inactive"
    version:           int      — increments on every state change for this key
    confidence:        float    — 0.0-1.0, decays when not accessed (default 1.0)
    last_accessed_at:  datetime — updated every time this memory is retrieved
    access_count:      int      — total retrieval count (never decremented)
    is_temporary:      bool     — True for short-horizon goals (expire in 3d not 30d)
    created_at:        datetime — when this record was first created
    activated_at:      datetime — when it last became active
    deactivated_at:    datetime — when it was last deactivated (null if active)
  })

Forget mechanism (confidence decay)
────────────────────────────────────
  Memories that are never retrieved lose confidence over time:
    Day 0:   confidence = 1.0  (just created)
    Day 30:  confidence = 0.70 (unused for 30 days)
    Day 60:  confidence = 0.49
    Day 90:  confidence = 0.34 (filtered from context at < 0.4)
    Day 120: confidence = 0.24 (pruned from DB at < 0.25)

  Temporary memories (is_temporary=True) decay on a 3-day cycle instead.

  Decay is triggered:
    - On session creation (passive maintenance, no latency hit)
    - Via POST /users/{user_id}/memory/maintenance (manual)

State transition rules
──────────────────────
  NEW VALUE, NO HISTORY:
    CREATE  MemoryRecord(status=active, version=1, session_id=current)

  EXACT DUPLICATE — same content_hash, already active:
    SKIP but UPDATE session_id to current (claims it for this session)

  EXACT DUPLICATE — same content_hash, currently inactive:
    REACTIVATE  deactivate current active for this key, reactivate this node
                version += 1, session_id = current, confidence reset to 1.0

  NEW VALUE — different content_hash, key exists:
    TRANSITION  deactivate all active for this key
                CREATE new MemoryRecord(status=active, version=prev+1)

Cache invalidation fix (v3.3)
─────────────────────────────
  invalidate_neo4j_cache() accepts session_id and deletes the correct key:
  neo4j_cache:{user_id}:{session_id}
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

from config import get_settings

settings = get_settings()
_graphiti: Optional[Graphiti] = None
_username_cache: dict[str, str] = {}

# Confidence decay parameters
_DECAY_INTERVAL_DAYS  = 30    # normal memories: decay every 30 days
_DECAY_TEMP_DAYS      = 3     # temporary memories: decay every 3 days
_DECAY_MULTIPLIER     = 0.7   # per interval: 1.0 → 0.70 → 0.49 → 0.34 → 0.24
_CONFIDENCE_MIN_SHOW  = 0.4   # below this: filtered from context
_CONFIDENCE_MIN_KEEP  = 0.25  # below this: pruned from DB

# Semantic conflict detection threshold
# If a new memory has cosine similarity >= this against any existing active memory
# it is treated as a conflict regardless of canonical_key name.
# This catches cases where the router uses different keys for the same concept
# e.g. "employer" vs "workplace" vs "company" for the same job fact.
# 0.85 = high enough to avoid false positives on related-but-different concepts.
_CONFLICT_SIM_THRESHOLD = 0.85


# ─── Graphiti lifecycle ───────────────────────────────────────────────────────

def _build_graphiti() -> Graphiti:
    llm_config = LLMConfig(
        api_key=settings.openrouter_api_key,
        model=settings.claude_model,
        base_url=settings.openrouter_base_url,
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        embedding_model="openai/text-embedding-3-small"
    )
    return Graphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_username,
        password=settings.neo4j_password,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config)
    )


async def _get_graphiti() -> Graphiti:
    global _graphiti
    if _graphiti is None:
        _graphiti = _build_graphiti()
    return _graphiti


async def _get_driver():
    g = await _get_graphiti()
    return g.driver


async def init_neo4j():
    g = await _get_graphiti()
    await g.build_indices_and_constraints()
    driver = g.driver
    async with driver.session() as s:
        await s.run(
            "CREATE INDEX memory_record_lookup IF NOT EXISTS "
            "FOR (m:MemoryRecord) ON (m.user_id, m.canonical_key)"
        )
        await s.run(
            "CREATE INDEX memory_record_hash IF NOT EXISTS "
            "FOR (m:MemoryRecord) ON (m.user_id, m.content_hash)"
        )
        await s.run(
            "CREATE INDEX memory_record_type IF NOT EXISTS "
            "FOR (m:MemoryRecord) ON (m.user_id, m.memory_type, m.status)"
        )
        # Index for decay queries — scanning stale memories by access time
        await s.run(
            "CREATE INDEX memory_record_accessed IF NOT EXISTS "
            "FOR (m:MemoryRecord) ON (m.user_id, m.last_accessed_at)"
        )
        # Index for confidence filtering
        await s.run(
            "CREATE INDEX memory_record_confidence IF NOT EXISTS "
            "FOR (m:MemoryRecord) ON (m.user_id, m.confidence, m.status)"
        )
    print("✓ Graphiti (Neo4j) ready")


async def close_driver():
    global _graphiti
    if _graphiti:
        await _graphiti.close()
        _graphiti = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _content_hash(content: str) -> str:
    """MD5 of normalised content — deduplication key."""
    return hashlib.md5(_normalise(content).encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dedup_batch(memories: list[dict]) -> list[dict]:
    """Per canonical_key keep highest-confidence entry within one batch."""
    best: dict[str, dict] = {}
    for m in memories:
        key = m.get("canonical_key", "unknown")
        ex  = best.get(key)
        if ex is None:
            best[key] = m
        else:
            if m.get("confidence", 1.0) > ex.get("confidence", 1.0):
                best[key] = m
            elif (m.get("confidence", 1.0) == ex.get("confidence", 1.0)
                  and len(m.get("content", "")) > len(ex.get("content", ""))):
                best[key] = m
    return list(best.values())


# ─── User node seeding ────────────────────────────────────────────────────────

async def ensure_user_node(user_id: str, username: str):
    _username_cache[user_id] = username
    g      = await _get_graphiti()
    driver = g.driver

    async with driver.session() as s:
        await s.run(
            """
            MERGE (u:UserProfile {user_id: $uid})
            ON CREATE SET u.username = $username, u.created_at = $now
            ON MATCH  SET u.username = $username
            """,
            uid=user_id, username=username, now=_now().isoformat()
        )

    identity_body = (
        f"{username} is a user of Synapse, a personal AI assistant with persistent memory. "
        f"Their unique identifier is {user_id}. "
        f"All memories, facts, preferences, goals, and constraints stored here belong to {username}."
    )
    try:
        await g.add_episode(
            name=f"identity_{user_id}",
            episode_body=identity_body,
            source=EpisodeType.text,
            source_description="User identity — created on signup",
            reference_time=_now(),
            group_id=user_id
        )
    except Exception as e:
        print(f"⚠ Graphiti identity episode failed (non-fatal): {e}")


async def ensure_session_node(session_id: str, user_id: str):
    pass


# ─── Core write — state machine ───────────────────────────────────────────────

async def store_memories_batch(
    user_id: str,
    session_id: str,
    memories: list[dict],
    source_turn: int = None
) -> list[str]:
    """
    Write memories through the state machine. Each memory is processed
    sequentially (not concurrent) to avoid race conditions on the same key
    within one batch.
    """
    if not memories:
        return []

    username = _username_cache.get(user_id, "the user")
    valid    = [m for m in memories if m.get("content") and m.get("memory_type")]
    deduped  = _dedup_batch(valid)

    stored_ids: list[str] = []
    for i, m in enumerate(deduped):
        try:
            result = await _process_one_memory(
                user_id, username, session_id, m, source_turn, i
            )
            if result:
                stored_ids.append(result)
        except Exception as e:
            print(f"⚠ Memory write error [{m.get('canonical_key')}]: {e}")

    return stored_ids


async def _find_semantic_conflict(
    driver,
    user_id: str,
    new_content: str,
    new_ckey: str,
    new_mtype: str,
) -> dict | None:
    """
    Layer 2 conflict detection — catches inconsistent canonical_keys.

    Two-pass approach:
      Pass 1 — cosine similarity >= 0.85 (works well for full sentences)
      Pass 2 — entity/keyword overlap (catches short content like "Microsoft"
                vs "works at a startup in Bangalore" which embed differently
                but clearly represent the same concept when one is a substring
                of the other or shares a key entity)

    Only runs when Layer 1 (same canonical_key) found nothing to deactivate.
    """
    try:
        from db.embedder import _get_model, cosine_similarity
        model = _get_model()

        async with driver.session() as s:
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, status:'active', memory_type:$mtype}) "
                "WHERE m.canonical_key <> $ckey "
                "RETURN m.canonical_key AS key, m.content AS content "
                "LIMIT 50",
                uid=user_id, mtype=new_mtype, ckey=new_ckey
            )
            records = await res.data()

        if not records:
            return None

        new_vec      = model.encode(new_content[:256], normalize_embeddings=True).tolist()
        new_lower    = new_content.lower()
        # Extract meaningful words from new content (skip stop words)
        _STOP = {"a","an","the","is","are","was","were","i","my","me","at","in",
                 "to","for","of","and","or","on","with","this","that","it","its"}
        new_words = {w for w in re.findall(r'\b\w+\b', new_lower) if w not in _STOP and len(w) > 2}

        best_sim    = 0.0
        best_record = None

        for r in records:
            existing_lower = r["content"].lower()
            existing_vec   = model.encode(
                r["content"][:256], normalize_embeddings=True
            ).tolist()
            sim = cosine_similarity(new_vec, existing_vec)

            # Pass 1: cosine similarity threshold
            if sim >= _CONFLICT_SIM_THRESHOLD:
                if sim > best_sim:
                    best_sim    = sim
                    best_record = {**r, "sim": sim, "method": "cosine"}
                continue

            # Pass 2: entity/keyword overlap for short content
            # Catches: new="works at a startup in Bangalore" existing="Microsoft"
            # or new="moved to Bangalore" existing="lives in Hyderabad"
            existing_words = {w for w in re.findall(r'\b\w+\b', existing_lower)
                              if w not in _STOP and len(w) > 2}

            # Check if old content appears as substring in new content
            # e.g. "microsoft" in "leaving microsoft to join a startup"
            old_in_new = r["content"].lower() in new_lower
            # Check if new content shares significant words with old
            overlap = new_words & existing_words
            overlap_ratio = len(overlap) / max(len(new_words), 1)

            # Short existing content (single word / short phrase) embedded in new content
            # is a very strong conflict signal — e.g. old="Microsoft" new mentions Microsoft
            if old_in_new and len(r["content"].split()) <= 4:
                score = 0.80   # treat as high-confidence conflict
                if score > best_sim:
                    best_sim    = score
                    best_record = {**r, "sim": score, "method": "substring"}
            # Significant word overlap between same-type memories
            elif overlap_ratio >= 0.5 and len(overlap) >= 2:
                score = 0.70 + overlap_ratio * 0.15
                if score > best_sim:
                    best_sim    = score
                    best_record = {**r, "sim": score, "method": "word_overlap"}

        if best_sim >= 0.70 and best_record:
            return best_record

        return None

    except Exception as e:
        print(f"  ⚠ _find_semantic_conflict failed (non-fatal): {e}")
        return None


async def _process_one_memory(
    user_id: str,
    username: str,
    session_id: str,
    m: dict,
    source_turn: int,
    idx: int
) -> str | None:
    """
    State machine for a single memory.
    All memory types are USER-SCOPED — session_id stored for provenance only.
    confidence, last_accessed_at, access_count, is_temporary added for decay.

    Returns: episode_name (str) if written, None if skipped.
    """
    mtype        = m["memory_type"]
    content      = m["content"]
    ckey         = m.get("canonical_key", "unknown")
    conf         = m.get("confidence", 1.0)
    is_temporary = m.get("is_temporary", False)

    # Detect temporary goals from content
    # Router can set is_temporary=True, or we infer from keywords
    if not is_temporary and mtype == "goal":
        temp_signals = ["today", "this week", "by friday", "tonight", "this morning",
                        "by tomorrow", "this sprint", "this deadline"]
        if any(sig in content.lower() for sig in temp_signals):
            is_temporary = True

    if conf < 0.5:
        return None

    chash  = _content_hash(content)
    now    = _now()
    driver = await _get_driver()
    action = None

    async with driver.session() as s:

        # ── State 1: exact content already active? ────────────
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, content_hash:$h, status:'active'}) "
            "RETURN m.canonical_key AS k, m.session_id AS sid LIMIT 1",
            uid=user_id, h=chash
        )
        existing_row = await res.single()
        if existing_row:
            # Already active — update session_id and reset confidence to 1.0
            await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, content_hash:$h, status:'active'}) "
                "SET m.session_id = $sid, m.activated_at = $now, "
                "    m.confidence = 1.0, m.last_accessed_at = $now",
                uid=user_id, h=chash, sid=session_id, now=now.isoformat()
            )
            if existing_row["sid"] != session_id:
                print(f"  ↳ CLAIM [{ckey}]: re-stated in new session, confidence reset")
            else:
                print(f"  ↳ REFRESH [{ckey}]: re-stated, confidence reset to 1.0")
            return None

        # ── State 2: same content exists but inactive → REACTIVATE ──
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, content_hash:$h, status:'inactive'}) "
            "RETURN m.canonical_key AS k LIMIT 1",
            uid=user_id, h=chash
        )
        if await res.single():
            await _deactivate_key(s, user_id, ckey, now)
            await s.run(
                """
                MATCH (m:MemoryRecord {user_id:$uid, content_hash:$h})
                SET m.status           = 'active',
                    m.session_id       = $sid,
                    m.activated_at     = $now,
                    m.deactivated_at   = null,
                    m.version          = m.version + 1,
                    m.confidence       = 1.0,
                    m.last_accessed_at = $now,
                    m.access_count     = 0,
                    m.is_temporary     = $is_temp
                """,
                uid=user_id, h=chash, sid=session_id, now=now.isoformat(),
                is_temp=is_temporary
            )
            action = "reactivated"
            print(f"  ↳ REACTIVATE [{ckey}]: previously known value restored, confidence=1.0")

        else:
            # ── State 3/4: new content → TRANSITION or CREATE ────
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, canonical_key:$key}) "
                "RETURN max(m.version) AS v",
                uid=user_id, key=ckey
            )
            row   = await res.single()
            max_v = (row["v"] or 0) if row else 0

            # Layer 1: deactivate same canonical_key (already existing logic)
            deactivated = await _deactivate_key(s, user_id, ckey, now)

            # Layer 2: semantic conflict detection
            # Only runs when Layer 1 found nothing to deactivate (deactivated == 0)
            # meaning no existing record shares this canonical_key.
            # Catches cross-key conflicts e.g. "employer" vs "workplace" for same job.
            if deactivated == 0:
                conflict = await _find_semantic_conflict(
                    driver, user_id, content, ckey, mtype
                )
                if conflict:
                    # Different key, same concept — deactivate the stale record
                    await _deactivate_key(s, user_id, conflict["key"], now)
                    deactivated = 1
                    print(f"  ↳ CONFLICT-RESOLVED [{conflict['key']} → {ckey}] "
                          f"sim={conflict['sim']:.2f}: "
                          f"deactivated stale '{conflict['content'][:50]}'")

            await s.run(
                """
                CREATE (m:MemoryRecord {
                    user_id:           $uid,
                    session_id:        $sid,
                    canonical_key:     $key,
                    memory_type:       $mtype,
                    content:           $content,
                    content_hash:      $chash,
                    status:            'active',
                    version:           $ver,
                    confidence:        1.0,
                    last_accessed_at:  $now,
                    access_count:      0,
                    is_temporary:      $is_temp,
                    created_at:        $now,
                    activated_at:      $now,
                    deactivated_at:    null
                })
                """,
                uid=user_id, sid=session_id, key=ckey, mtype=mtype,
                content=content, chash=chash, ver=max_v + 1, now=now.isoformat(),
                is_temp=is_temporary
            )
            action = "updated" if deactivated else "created"
            print(f"  ↳ {action.upper()} [{ckey}] v{max_v+1}: {content[:70]}")

    # ── Write Graphiti episode for semantic search ────────────
    frames = {
        "fact":       f"{username}'s personal fact: {{}}",
        "preference": f"{username}'s preference: {{}}",
        "goal":       f"{username}'s goal: {{}}",
        "constraint": f"{username}'s constraint: {{}}",
    }
    ep_name = f"mem_{user_id}_{ckey}_t{source_turn}_{idx}_{action[:3]}"
    ep_body = frames.get(mtype, "{}").format(content)
    ep_desc = (
        f"type:{mtype} | key:{ckey} | action:{action} | "
        f"turn:{source_turn} | conf:{conf} | session:{session_id} | "
        f"temporary:{is_temporary}"
    )
    try:
        g = await _get_graphiti()
        await g.add_episode(
            name=ep_name, episode_body=ep_body,
            source=EpisodeType.text, source_description=ep_desc,
            reference_time=now, group_id=user_id
        )
    except Exception as e:
        print(f"  ⚠ Graphiti episode failed [{ckey}] (non-fatal): {e}")

    return ep_name


async def _deactivate_key(session, user_id: str, key: str, now: datetime) -> int:
    """Deactivate all active records for canonical_key. Returns count."""
    res = await session.run(
        """
        MATCH (m:MemoryRecord {user_id:$uid, canonical_key:$key, status:'active'})
        SET m.status='inactive', m.deactivated_at=$now
        RETURN count(m) AS cnt
        """,
        uid=user_id, key=key, now=now.isoformat()
    )
    row = await res.single()
    return row["cnt"] if row else 0


# ─── Core read ────────────────────────────────────────────────────────────────

async def get_user_memories(
    user_id: str,
    session_id: str = None,
    memory_type: str = None,
    query: str = None,
    query_vec: list = None,
) -> list[dict]:
    """
    Returns ACTIVE memories with confidence >= _CONFIDENCE_MIN_SHOW.
    All types are now user-scoped — session_id is kept for provenance but
    does NOT filter results.

    Ranked by local cosine similarity when query_vec provided.
    Memories below _MIN_RELEVANCE are filtered out to reduce prompt noise —
    e.g. "dark theme" preference is irrelevant to "Should I take the job?"
    A minimum of _MIN_GUARANTEED_MEMORIES are always returned regardless
    of score (name, employer, location should always be in context).
    """
    # Minimum relevance score to be included in context
    _MIN_RELEVANCE           = 0.25
    # Always keep at least this many top memories regardless of score
    _MIN_GUARANTEED_MEMORIES = 4

    records = await _get_memories_direct(user_id, memory_type=memory_type)

    if not records or not query_vec:
        return records

    # Local cosine ranking — no API call
    scored = []
    for r in records:
        score = _local_relevance_score(r.get("content", ""), query_vec)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter below threshold but always guarantee minimum memories
    above_threshold = [(s, r) for s, r in scored if s >= _MIN_RELEVANCE]
    if len(above_threshold) >= _MIN_GUARANTEED_MEMORIES:
        result = [r for _, r in above_threshold]
    else:
        # Not enough above threshold — take top N guaranteed
        result = [r for _, r in scored[:_MIN_GUARANTEED_MEMORIES]]

    print(f"  · neo4j relevance filter: {len(scored)} total → {len(result)} above threshold")
    return result


def _local_relevance_score(content: str, query_vec: list) -> float:
    try:
        from db.embedder import _get_model, cosine_similarity
        model       = _get_model()
        content_vec = model.encode(content[:256], normalize_embeddings=True).tolist()
        return cosine_similarity(query_vec, content_vec)
    except Exception:
        return 0.5


async def _get_memories_direct(
    user_id: str,
    memory_type: str = None,
    limit: int = 20,
) -> list[dict]:
    """
    Direct Cypher read. All types are user-scoped.
    Filters out memories with confidence < _CONFIDENCE_MIN_SHOW (half-decayed).

    LIMIT 20 (token budget fix):
      Ordered by confidence DESC so the highest-quality memories are always
      kept when a power user has more than 20 active records.
      20 memories × ~15 tokens = ~300 tokens max — safe regardless of history length.
    """
    driver = await _get_driver()

    async with driver.session() as s:
        if memory_type:
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, status:'active', memory_type:$mtype}) "
                "WHERE m.confidence >= $min_conf "
                "RETURN m ORDER BY m.confidence DESC, m.activated_at DESC "
                "LIMIT $limit",
                uid=user_id, mtype=memory_type,
                min_conf=_CONFIDENCE_MIN_SHOW, limit=limit
            )
        else:
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, status:'active'}) "
                "WHERE m.confidence >= $min_conf "
                "RETURN m ORDER BY m.confidence DESC, m.memory_type, m.activated_at DESC "
                "LIMIT $limit",
                uid=user_id, min_conf=_CONFIDENCE_MIN_SHOW, limit=limit
            )
        records = await res.data()

    return [_row_to_dict(r["m"]) for r in records]


# ─── Forget mechanism — access tracking ───────────────────────────────────────

async def touch_memories(user_id: str, canonical_keys: list[str]):
    """
    Called after every response that retrieved memories from Neo4j.
    Updates last_accessed_at and increments access_count.
    Resets confidence to 1.0 if memory was actively used.

    This is the signal that prevents decay: a memory that keeps being
    retrieved will never decay below 1.0.

    Called from agent.py _background_store() — runs in background,
    zero latency impact on the response.
    """
    if not canonical_keys:
        return

    driver = await _get_driver()
    now    = _now().isoformat()

    try:
        async with driver.session() as s:
            await s.run(
                """
                UNWIND $keys AS key
                MATCH (m:MemoryRecord {user_id: $uid, canonical_key: key, status: 'active'})
                SET m.last_accessed_at = $now,
                    m.access_count     = coalesce(m.access_count, 0) + 1,
                    m.confidence       = 1.0
                """,
                uid=user_id, keys=canonical_keys, now=now
            )
        print(f"  · touch_memories: refreshed {len(canonical_keys)} memories for {user_id[:8]}")
    except Exception as e:
        print(f"  ⚠ touch_memories failed (non-fatal): {e}")


# ─── Forget mechanism — decay ─────────────────────────────────────────────────

async def decay_stale_memories(user_id: str) -> dict:
    """
    Applies confidence decay to memories not accessed within their decay window.

    Normal memories:    decay every 30 days if not accessed
    Temporary memories: decay every 3 days if not accessed

    Decay multiplier per interval: 0.7
      After 1 interval: 1.0 → 0.70
      After 2 intervals: 0.70 → 0.49
      After 3 intervals: 0.49 → 0.34 (filtered from context)
      After 4 intervals: 0.34 → 0.24 (pruned)

    Safe to call on every session start — only touches stale records.
    Returns counts of memories decayed.
    """
    driver   = await _get_driver()
    now_iso  = _now().isoformat()

    try:
        async with driver.session() as s:
            # Decay normal (non-temporary) stale memories
            res = await s.run(
                """
                MATCH (m:MemoryRecord {user_id: $uid, status: 'active'})
                WHERE (m.is_temporary = false OR m.is_temporary IS NULL)
                  AND m.last_accessed_at < datetime($now) - duration({days: $interval})
                  AND m.confidence > $min_keep
                SET m.confidence = m.confidence * $multiplier
                RETURN count(m) AS cnt
                """,
                uid=user_id,
                now=now_iso,
                interval=_DECAY_INTERVAL_DAYS,
                multiplier=_DECAY_MULTIPLIER,
                min_keep=_CONFIDENCE_MIN_KEEP
            )
            row    = await res.single()
            normal = row["cnt"] if row else 0

            # Decay temporary memories on shorter cycle
            res = await s.run(
                """
                MATCH (m:MemoryRecord {user_id: $uid, status: 'active', is_temporary: true})
                WHERE m.last_accessed_at < datetime($now) - duration({days: $interval})
                  AND m.confidence > $min_keep
                SET m.confidence = m.confidence * $multiplier
                RETURN count(m) AS cnt
                """,
                uid=user_id,
                now=now_iso,
                interval=_DECAY_TEMP_DAYS,
                multiplier=_DECAY_MULTIPLIER,
                min_keep=_CONFIDENCE_MIN_KEEP
            )
            row  = await res.single()
            temp = row["cnt"] if row else 0

        total = normal + temp
        if total > 0:
            print(f"  · decay: {normal} normal + {temp} temporary memories decayed for {user_id[:8]}")
        return {"decayed_normal": normal, "decayed_temporary": temp, "total_decayed": total}

    except Exception as e:
        print(f"  ⚠ decay_stale_memories failed (non-fatal): {e}")
        return {"decayed_normal": 0, "decayed_temporary": 0, "total_decayed": 0, "error": str(e)}


async def prune_dead_memories(user_id: str) -> dict:
    """
    Hard-deletes (deactivates) memories whose confidence has fallen below
    _CONFIDENCE_MIN_KEEP. Called after decay_stale_memories.

    We mark as 'inactive' rather than DELETE so the history is preserved
    in case the user asks "what did you used to know about me?".
    Actual node deletion only happens in delete_user_graph().

    Returns count of pruned memories.
    """
    driver  = await _get_driver()
    now_iso = _now().isoformat()

    try:
        async with driver.session() as s:
            res = await s.run(
                """
                MATCH (m:MemoryRecord {user_id: $uid, status: 'active'})
                WHERE m.confidence < $threshold
                SET m.status         = 'inactive',
                    m.deactivated_at = $now
                RETURN count(m) AS cnt, collect(m.canonical_key) AS keys
                """,
                uid=user_id, threshold=_CONFIDENCE_MIN_KEEP, now=now_iso
            )
            row   = await res.single()
            count = row["cnt"] if row else 0
            keys  = row["keys"] if row else []

        if count > 0:
            print(f"  · prune: {count} dead memories deactivated for {user_id[:8]}: {keys[:5]}")
        return {"pruned": count, "keys": keys}

    except Exception as e:
        print(f"  ⚠ prune_dead_memories failed (non-fatal): {e}")
        return {"pruned": 0, "keys": [], "error": str(e)}


async def run_memory_maintenance(user_id: str) -> dict:
    """
    Convenience wrapper: decay then prune in sequence.
    Called on session start and via the /memory/maintenance endpoint.
    Safe to call frequently — only touches stale/dead records.
    """
    decay_result = await decay_stale_memories(user_id)
    prune_result = await prune_dead_memories(user_id)
    return {
        "user_id": user_id,
        "decay":   decay_result,
        "prune":   prune_result,
    }


# ─── Timeline APIs ────────────────────────────────────────────────────────────

async def get_fact_history(user_id: str, topic: str) -> list[dict]:
    driver = await _get_driver()
    async with driver.session() as s:
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, canonical_key:$t}) "
            "RETURN m ORDER BY m.version DESC",
            uid=user_id, t=topic
        )
        records = await res.data()

        if not records:
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid}) "
                "WHERE toLower(m.content) CONTAINS toLower($t) "
                "   OR toLower(m.canonical_key) CONTAINS toLower($t) "
                "RETURN m ORDER BY m.canonical_key, m.version DESC",
                uid=user_id, t=topic
            )
            records = await res.data()

    return [{
        "content":         r["m"].get("content", ""),
        "memory_type":     r["m"].get("memory_type", "fact"),
        "canonical_key":   r["m"].get("canonical_key", ""),
        "session_id":      r["m"].get("session_id", ""),
        "status":          r["m"].get("status", "inactive"),
        "version":         r["m"].get("version", 1),
        "confidence":      r["m"].get("confidence", 1.0),
        "last_accessed_at": r["m"].get("last_accessed_at"),
        "activated_at":    r["m"].get("activated_at"),
        "deactivated_at":  r["m"].get("deactivated_at"),
        "is_reactivation": r["m"].get("version", 1) > 1 and r["m"].get("status") == "active",
    } for r in records]


async def get_memory_timeline(user_id: str, canonical_key: str) -> dict:
    driver = await _get_driver()
    async with driver.session() as s:
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, canonical_key:$key}) "
            "RETURN m ORDER BY m.version ASC",
            uid=user_id, key=canonical_key
        )
        records = await res.data()

    if not records:
        return {"canonical_key": canonical_key, "found": False}

    versions = [
        {
            "version":          r["m"]["version"],
            "content":          r["m"]["content"],
            "status":           r["m"]["status"],
            "confidence":       r["m"].get("confidence", 1.0),
            "access_count":     r["m"].get("access_count", 0),
            "last_accessed_at": r["m"].get("last_accessed_at"),
            "is_temporary":     r["m"].get("is_temporary", False),
            "session_id":       r["m"].get("session_id", ""),
            "activated_at":     r["m"]["activated_at"],
            "deactivated_at":   r["m"]["deactivated_at"],
        }
        for r in records
    ]
    active = next((v for v in versions if v["status"] == "active"), None)

    return {
        "canonical_key":  canonical_key,
        "memory_type":    records[0]["m"].get("memory_type"),
        "found":          True,
        "current_value":  active["content"] if active else None,
        "is_active":      active is not None,
        "confidence":     active["confidence"] if active else 0.0,
        "last_accessed_at": active["last_accessed_at"] if active else None,
        "total_versions": len(versions),
        "versions":       versions,
    }


# ─── Full profile ─────────────────────────────────────────────────────────────

async def get_full_memory_graph(user_id: str, session_id: str = None) -> dict:
    """
    Full memory profile. All types are user-scoped.
    Includes confidence scores for observability.
    """
    driver = await _get_driver()
    async with driver.session() as s:
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, status:'active'}) "
            "WHERE m.confidence >= $min_conf "
            "RETURN m ORDER BY m.memory_type, m.activated_at DESC",
            uid=user_id, min_conf=_CONFIDENCE_MIN_SHOW
        )
        active_records = await res.data()

        # Low-confidence memories (visible but filtered from context)
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, status:'active'}) "
            "WHERE m.confidence < $min_conf "
            "RETURN m ORDER BY m.confidence ASC",
            uid=user_id, min_conf=_CONFIDENCE_MIN_SHOW
        )
        fading_records = await res.data()

        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, status:'inactive'}) "
            "RETURN m.memory_type AS mtype, count(m) AS cnt",
            uid=user_id
        )
        inactive_raw = await res.data()

    inactive_counts = {r["mtype"]: r["cnt"] for r in inactive_raw}
    by_type: dict[str, list] = {"fact": [], "preference": [], "goal": [], "constraint": []}

    for r in active_records:
        t = r["m"].get("memory_type", "fact")
        by_type.setdefault(t, []).append(_row_to_dict(r["m"]))

    return {
        "user_id":           user_id,
        "memories_by_type":  by_type,
        "fading_memories":   [_row_to_dict(r["m"]) for r in fading_records],
        "active_totals":     {t: len(v) for t, v in by_type.items()},
        "inactive_totals":   inactive_counts,
        "total_active":      sum(len(v) for v in by_type.values()),
        "total_fading":      len(fading_records),
        "total_inactive":    sum(inactive_counts.values()),
        "scope_note":        "all types are user-scoped | decay removes unused memories over time",
        "engine":            "Hybrid: Direct Cypher state machine + Graphiti semantic search",
    }


# ─── Delete ───────────────────────────────────────────────────────────────────

async def delete_user_graph(user_id: str):
    g      = await _get_graphiti()
    driver = g.driver
    async with driver.session() as s:
        await s.run("MATCH (m:MemoryRecord {user_id:$uid}) DELETE m",      uid=user_id)
        await s.run("MATCH (u:UserProfile {user_id:$uid}) DELETE u",       uid=user_id)
        await s.run("MATCH (e:Episodic {group_id:$gid}) DETACH DELETE e",  gid=user_id)
        await s.run("MATCH ()-[r:RELATES_TO {group_id:$gid}]-() DELETE r", gid=user_id)
        await s.run(
            "MATCH (n:Entity {group_id:$gid}) "
            "WHERE NOT (n)-[:RELATES_TO]-() AND NOT ()-[:RELATES_TO]->(n) DELETE n",
            gid=user_id
        )
    _username_cache.pop(user_id, None)


# ─── Cache invalidation ───────────────────────────────────────────────────────

async def invalidate_neo4j_cache(user_id: str, session_id: str):
    """
    Delete the Neo4j cache entry for this user+session.
    Cache key format: neo4j_cache:{user_id}:{session_id}
    """
    try:
        from db.redis_manager import get_redis
        r = await get_redis()
        deleted = await r.delete(f"neo4j_cache:{user_id}:{session_id}")
        if deleted:
            print(f"  ✓ Neo4j cache invalidated for {user_id[:8]}:{session_id[:8]}")
        else:
            print(f"  · Neo4j cache: no entry found to invalidate (already expired or never cached)")
    except Exception as e:
        print(f"  ⚠ Neo4j cache invalidation failed (non-fatal): {e}")


# ─── Row serialisation ────────────────────────────────────────────────────────

def _row_to_dict(m: dict) -> dict:
    return {
        "content":           m.get("content", ""),
        "memory_type":       m.get("memory_type", "fact"),
        "canonical_key":     m.get("canonical_key", ""),
        "session_id":        m.get("session_id", ""),
        "status":            m.get("status", "active"),
        "version":           m.get("version", 1),
        "confidence":        m.get("confidence", 1.0),
        "last_accessed_at":  m.get("last_accessed_at"),
        "access_count":      m.get("access_count", 0),
        "is_temporary":      m.get("is_temporary", False),
        "activated_at":      m.get("activated_at"),
        "deactivated_at":    m.get("deactivated_at"),
        "is_current":        m.get("status") == "active",
    }


def _infer_type(fact: str) -> str:
    f = fact.lower()
    if any(w in f for w in ["prefer", "like", "love", "enjoy", "dislike", "hate"]):
        return "preference"
    if any(w in f for w in ["want", "goal", "achieve", "learn", "build"]):
        return "goal"
    if any(w in f for w in ["cannot", "must", "budget", "limit", "avoid", "constraint"]):
        return "constraint"
    return "fact"