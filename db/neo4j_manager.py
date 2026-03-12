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

Why this hybrid approach
────────────────────────
Graphiti's automatic contradiction resolution is probabilistic — it uses an
LLM to detect contradictions, which can miss things or create duplicates over
long usage. For state that must be perfectly reliable (active/inactive,
no duplicates, reactivation), direct Cypher is deterministic and guaranteed.

MemoryRecord node schema
────────────────────────
  (:MemoryRecord {
    user_id:          string   — scopes to one user
    canonical_key:    string   — stable snake_case identifier e.g. "coding_language"
    memory_type:      string   — fact | preference | goal | constraint
    content:          string   — the actual memory text
    content_hash:     string   — MD5 of normalised content (dedup key)
    status:           string   — "active" | "inactive"
    version:          int      — increments on every state change for this key
    created_at:       datetime — when this record was first created
    activated_at:     datetime — when it last became active
    deactivated_at:   datetime — when it was last deactivated (null if active)
  })

State transition rules
──────────────────────
  NEW VALUE, NO HISTORY:
    CREATE  MemoryRecord(status=active, version=1)

  EXACT DUPLICATE — same content_hash, already active:
    SKIP    no write, no change

  EXACT DUPLICATE — same content_hash, currently inactive:
    REACTIVATE  deactivate current active for this key, reactivate this node
                version += 1, deactivated_at = null

  NEW VALUE — different content_hash, key exists:
    TRANSITION  deactivate all active for this key
                CREATE new MemoryRecord(status=active, version=prev+1)

  Examples:
    T1: "I love Python"   → Python: active   (v1)
    T2: "switched to C++" → Python: inactive, C++: active (v2)
    T3: "back to Python"  → C++: inactive, Python: REACTIVATED (same node, v+=1)
    T4: "now using Rust"  → Python: inactive, Rust: active (v3)
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
    g  = await _get_graphiti()
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

    Returns: episode_name (str) if written, None if skipped.
    """
    mtype   = m["memory_type"]
    content = m["content"]
    ckey    = m.get("canonical_key", "unknown")
    conf    = m.get("confidence", 1.0)

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
            "RETURN m.canonical_key AS k LIMIT 1",
            uid=user_id, h=chash
        )
        if await res.single():
            print(f"  ↳ SKIP [{ckey}]: identical content already active")
            return None  # Pure duplicate — nothing to do

        # ── State 2: same content exists but inactive → REACTIVATE ──
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, content_hash:$h, status:'inactive'}) "
            "RETURN m.canonical_key AS k LIMIT 1",
            uid=user_id, h=chash
        )
        if await res.single():
            # Deactivate whatever is currently active for this key
            await _deactivate_key(s, user_id, ckey, now)
            # Reactivate this node
            await s.run(
                """
                MATCH (m:MemoryRecord {user_id:$uid, content_hash:$h})
                SET m.status         = 'active',
                    m.activated_at   = $now,
                    m.deactivated_at = null,
                    m.version        = m.version + 1
                """,
                uid=user_id, h=chash, now=now.isoformat()
            )
            action = "reactivated"
            print(f"  ↳ REACTIVATE [{ckey}]: previously known value restored")

        else:
            # ── State 3/4: new content → TRANSITION or CREATE ────
            # Get current max version for this key
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, canonical_key:$key}) "
                "RETURN max(m.version) AS v",
                uid=user_id, key=ckey
            )
            row = await res.single()
            max_v = (row["v"] or 0) if row else 0

            deactivated = await _deactivate_key(s, user_id, ckey, now)

            await s.run(
                """
                CREATE (m:MemoryRecord {
                    user_id:        $uid,
                    canonical_key:  $key,
                    memory_type:    $mtype,
                    content:        $content,
                    content_hash:   $chash,
                    status:         'active',
                    version:        $ver,
                    created_at:     $now,
                    activated_at:   $now,
                    deactivated_at: null
                })
                """,
                uid=user_id, key=ckey, mtype=mtype, content=content,
                chash=chash, ver=max_v + 1, now=now.isoformat()
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
        f"turn:{source_turn} | conf:{conf} | session:{session_id}"
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
    memory_type: str = None,
    query: str = None,
    query_vec: list = None,   # pre-computed local embedding — avoids all remote API calls
) -> list[dict]:
    """
    Returns ACTIVE memories, ranked by local cosine similarity when query provided.

    IMPORTANT: We do NOT call Graphiti's g.search() here for context retrieval.
    Graphiti uses OpenAIEmbedder → remote API call (~300ms) on every invocation.
    Instead: fetch all active records via direct Cypher (~20ms), then rank
    them locally using the pre-computed query_vec (~1ms cosine math).

    Graphiti search is only used when WRITING memories (contradiction detection),
    not when READING them for context.
    """
    records = await _get_memories_direct(user_id, memory_type)

    if not records or not query_vec:
        return records

    # Local cosine ranking — no API call, ~1ms
    from db.embedder import cosine_similarity
    scored = []
    for r in records:
        # Embed the content string for scoring
        content = r.get("content", "")
        # Use cached embedding if available, else score by text overlap heuristic
        score = _local_relevance_score(content, query_vec)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _local_relevance_score(content: str, query_vec: list) -> float:
    """
    Fast local relevance scoring using pre-computed query_vec.
    Embeds content synchronously (model already loaded) and computes cosine.
    Falls back to 0.5 if embedding fails.
    """
    try:
        from db.embedder import _get_model, cosine_similarity
        model = _get_model()
        content_vec = model.encode(content[:256], normalize_embeddings=True).tolist()
        return cosine_similarity(query_vec, content_vec)
    except Exception:
        return 0.5


async def _get_memories_direct(user_id: str, memory_type: str = None) -> list[dict]:
    driver = await _get_driver()
    async with driver.session() as s:
        if memory_type:
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, status:'active', memory_type:$mtype}) "
                "RETURN m ORDER BY m.activated_at DESC",
                uid=user_id, mtype=memory_type
            )
        else:
            res = await s.run(
                "MATCH (m:MemoryRecord {user_id:$uid, status:'active'}) "
                "RETURN m ORDER BY m.memory_type, m.activated_at DESC",
                uid=user_id
            )
        records = await res.data()
    return [_row_to_dict(r["m"]) for r in records]


async def _get_memories_semantic(
    user_id: str, query: str, memory_type: str = None
) -> list[dict]:
    """Graphiti search → intersect with active MemoryRecord hashes."""
    g = await _get_graphiti()
    try:
        edges = await g.search(query=query, group_ids=[user_id], num_results=15)
    except Exception:
        edges = []

    # Get all active records for this user
    driver = await _get_driver()
    async with driver.session() as s:
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, status:'active'}) RETURN m",
            uid=user_id
        )
        active = await res.data()

    active_by_hash = {r["m"]["content_hash"]: r["m"] for r in active}

    seen: set[str] = set()
    results: list[dict] = []

    for edge in edges:
        fact = getattr(edge, "fact", "") or ""
        if not fact or "unique identifier is" in fact:
            continue
        h = _content_hash(fact)
        if h in active_by_hash and h not in seen:
            seen.add(h)
            rec = active_by_hash[h]
            if memory_type and rec.get("memory_type") != memory_type:
                continue
            results.append(_row_to_dict(rec))

    # Fallback if Graphiti matched nothing
    if not results:
        filtered = [r for r in active
                    if not memory_type or r["m"].get("memory_type") == memory_type]
        return [_row_to_dict(r["m"]) for r in filtered]

    return results


def _row_to_dict(m: dict) -> dict:
    return {
        "content":        m.get("content", ""),
        "memory_type":    m.get("memory_type", "fact"),
        "canonical_key":  m.get("canonical_key", ""),
        "status":         m.get("status", "active"),
        "version":        m.get("version", 1),
        "activated_at":   m.get("activated_at"),
        "deactivated_at": m.get("deactivated_at"),
        "is_current":     m.get("status") == "active",
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


# ─── Timeline APIs ────────────────────────────────────────────────────────────

async def get_fact_history(user_id: str, topic: str) -> list[dict]:
    """
    Full timeline for a topic — all versions, active and inactive, in order.

    Shows Python→C++→Python (reactivation) as:
      [ { content: "Python", status: active,   version: 3 }   ← reactivated
        { content: "C++",    status: inactive, version: 2 }
        { content: "Python", status: inactive, version: 1 } ]  ← original
    """
    driver = await _get_driver()
    async with driver.session() as s:
        # Try exact key first
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, canonical_key:$t}) "
            "RETURN m ORDER BY m.version DESC",
            uid=user_id, t=topic
        )
        records = await res.data()

        if not records:
            # Fuzzy content/key search
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
        "status":          r["m"].get("status", "inactive"),
        "version":         r["m"].get("version", 1),
        "activated_at":    r["m"].get("activated_at"),
        "deactivated_at":  r["m"].get("deactivated_at"),
        "is_reactivation": r["m"].get("version", 1) > 1 and r["m"].get("status") == "active",
    } for r in records]


async def get_memory_timeline(user_id: str, canonical_key: str) -> dict:
    """Structured timeline for one canonical_key with current value highlighted."""
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

    versions = [{"version": r["m"]["version"], "content": r["m"]["content"],
                 "status": r["m"]["status"], "activated_at": r["m"]["activated_at"],
                 "deactivated_at": r["m"]["deactivated_at"]} for r in records]
    active = next((v for v in versions if v["status"] == "active"), None)

    return {
        "canonical_key":  canonical_key,
        "memory_type":    records[0]["m"].get("memory_type"),
        "found":          True,
        "current_value":  active["content"] if active else None,
        "is_active":      active is not None,
        "total_versions": len(versions),
        "versions":       versions,
    }


# ─── Full profile ─────────────────────────────────────────────────────────────

async def get_full_memory_graph(user_id: str) -> dict:
    """Full memory profile: active memories + inactive counts per type."""
    driver = await _get_driver()
    async with driver.session() as s:
        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, status:'active'}) "
            "RETURN m ORDER BY m.memory_type, m.activated_at DESC",
            uid=user_id
        )
        active_records = await res.data()

        res = await s.run(
            "MATCH (m:MemoryRecord {user_id:$uid, status:'inactive'}) "
            "RETURN m.memory_type AS mtype, count(m) AS cnt",
            uid=user_id
        )
        inactive_raw = await res.data()

    inactive_counts = {r["mtype"]: r["cnt"] for r in inactive_raw}
    by_type: dict[str, list] = {
        "fact": [], "preference": [], "goal": [], "constraint": []
    }
    for r in active_records:
        t = r["m"].get("memory_type", "fact")
        by_type.setdefault(t, []).append(_row_to_dict(r["m"]))

    return {
        "user_id":          user_id,
        "memories_by_type": by_type,
        "active_totals":    {t: len(v) for t, v in by_type.items()},
        "inactive_totals":  inactive_counts,
        "total_active":     sum(len(v) for v in by_type.values()),
        "total_inactive":   sum(inactive_counts.values()),
        "engine":           "Hybrid: Direct Cypher state machine + Graphiti semantic search",
    }


# ─── Delete ───────────────────────────────────────────────────────────────────

async def delete_user_graph(user_id: str):
    g      = await _get_graphiti()
    driver = g.driver
    async with driver.session() as s:
        await s.run("MATCH (m:MemoryRecord {user_id:$uid}) DELETE m",     uid=user_id)
        await s.run("MATCH (u:UserProfile {user_id:$uid}) DELETE u",      uid=user_id)
        await s.run("MATCH (e:Episodic {group_id:$gid}) DETACH DELETE e", gid=user_id)
        await s.run("MATCH ()-[r:RELATES_TO {group_id:$gid}]-() DELETE r", gid=user_id)
        await s.run(
            "MATCH (n:Entity {group_id:$gid}) "
            "WHERE NOT (n)-[:RELATES_TO]-() AND NOT ()-[:RELATES_TO]->(n) DELETE n",
            gid=user_id
        )
    _username_cache.pop(user_id, None)