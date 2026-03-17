"""
db/mongo_manager.py — Enhanced episodic memory store

Schema
──────
Each episodic memory document:
  memory_id        string    — unique ep_<hex16>
  user_id          string    — owner
  session_id       string    — session it was created in
  title            string    — short evocative title (8 words max)
  content          string    — rich 4-6 sentence narrative
  outcome          string    — resolved | ongoing | abandoned | unclear
  outcome_note     string    — optional resolution note (filled later)
  turn_start       int       — first turn of this episode
  turn_end         int       — last turn of this episode
  key_entities     list[str] — people, tools, concepts mentioned
  emotional_tone   string    — neutral | curious | frustrated | excited | anxious | satisfied
  emotional_intensity int    — 1 (neutral) to 5 (very strong)
  tags             list[str] — topical tags for clustering
  topic_cluster    string    — stable cluster key (e.g. "career", "health", "project_x")
  importance_score float     — 1.0-10.0, LLM-assigned at write time
  content_hash     string    — MD5(normalised title + key_entities) for dedup
  access_count     int       — how many times this episode was retrieved
  last_accessed_at datetime  — last retrieval timestamp
  related_ids      list[str] — manually linked related episode IDs
  is_summary       bool      — True if this is a consolidation summary
  source_ids       list[str] — for summary episodes: which episodes it covers
  superseded       bool      — True if forgotten/consolidated
  superseded_at    datetime  — when it was marked superseded (null if active)
  supersede_reason string    — why it was superseded (stale/pruned/consolidated/abandoned)
  created_at       datetime
  updated_at       datetime

Session-scoping + cross-session fallback (v3.3)
───────────────────────────────────────────────
  get_user_episodic_memories(session_id=...) first tries to return
  episodes from the current session only.

  BUT — if fewer than _CROSS_SESSION_THRESHOLD episodes are found
  (e.g. a brand new session with no episodes yet), it falls back to
  returning the most important episodes across ALL sessions.

4-Component Retrieval Scoring
──────────────────────────────
  final = (0.50 × relevance) + (0.20 × recency) + (0.20 × importance) + (0.10 × access_freq)

Forget strategy (v3.4)
──────────────────────
  Three-stage forgetting for MongoDB episodic memories:

  Stage 1 — Stale detection (forget_stale_episodes):
    Mark as superseded if ANY condition met:
      A) age > 60d + importance < 5.0 + access_count < 2   (low-value, ignored)
      B) age > 180d + access_count == 0                     (never retrieved, ever)
      C) outcome == "abandoned" + age > 30d + importance < 6 (user gave up, old)
      D) outcome == "resolved" + age > 120d + access_count == 0 (done, forgotten)

  Stage 2 — Hard prune (prune_superseded_episodes):
    Delete superseded docs older than 180 days (not summary episodes).
    Summaries are kept longer (365 days) because they cover many episodes.

  Stage 3 — run_mongo_maintenance: calls stage 1 then stage 2.
    Triggered on session creation (background) — same as Neo4j decay.
"""

import hashlib
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import get_settings

settings = get_settings()
_client: Optional[AsyncIOMotorClient] = None

_W_RELEVANCE  = 0.50
_W_RECENCY    = 0.20
_W_IMPORTANCE = 0.20
_W_ACCESS     = 0.10
_RECENCY_HALF_LIFE = 30   # days

_CROSS_SESSION_THRESHOLD = 2

# ── Forget thresholds ────────────────────────────────────────────────────────
# Stage 1: stale detection
_FORGET_LOW_VALUE_DAYS       = 60    # A: old low-importance low-access episodes
_FORGET_LOW_VALUE_MAX_IMP    = 5.0   # A: importance threshold
_FORGET_LOW_VALUE_MAX_ACCESS = 2     # A: access count threshold
_FORGET_NEVER_READ_DAYS      = 180   # B: never retrieved at all
_FORGET_ABANDONED_DAYS       = 30    # C: abandoned episodes
_FORGET_ABANDONED_MAX_IMP    = 6.0   # C: importance threshold for abandoned
_FORGET_RESOLVED_DAYS        = 120   # D: resolved, never re-read

# Stage 2: hard prune
_PRUNE_SUPERSEDED_DAYS       = 180   # delete superseded episodes older than this
_PRUNE_SUMMARY_DAYS          = 365   # summaries kept longer


# ─── Connection ───────────────────────────────────────────────────────────────

def get_db() -> AsyncIOMotorDatabase:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client[settings.mongo_db]


def _col():
    return get_db()[settings.mongo_episodic_collection]


async def _get_collection():
    return _col()


async def init_mongo():
    col = _col()

    await col.create_index([("user_id", 1), ("created_at", -1)])
    await col.create_index([("user_id", 1), ("session_id", 1), ("created_at", -1)])
    await col.create_index([("user_id", 1), ("topic_cluster", 1), ("created_at", -1)])
    await col.create_index(
        [("user_id", 1), ("importance_score", -1), ("created_at", -1)],
        name="idx_user_importance_recency"
    )
    await col.create_index([("user_id", 1), ("tags", 1)])
    await col.create_index([("user_id", 1), ("outcome", 1)])
    await col.create_index([("user_id", 1), ("content_hash", 1)])
    await col.create_index("memory_id", unique=True)
    await col.create_index([("session_id", 1)])

    # Indexes for forget mechanism queries
    await col.create_index(
        [("user_id", 1), ("superseded", 1), ("created_at", -1)],
        name="idx_user_superseded_age"
    )
    await col.create_index(
        [("user_id", 1), ("last_accessed_at", 1), ("importance_score", 1)],
        name="idx_user_access_importance"
    )
    await col.create_index(
        [("user_id", 1), ("outcome", 1), ("created_at", -1)],
        name="idx_user_outcome_age"
    )

    try:
        await col.create_index(
            [
                ("title",         "text"),
                ("content",       "text"),
                ("tags",          "text"),
                ("key_entities",  "text"),
                ("topic_cluster", "text"),
            ],
            weights={
                "title":         10,
                "tags":           6,
                "key_entities":   5,
                "topic_cluster":  4,
                "content":        1,
            },
            name="episodic_text_search"
        )
    except Exception:
        pass

    print("✓ MongoDB ready")


async def close_mongo():
    global _client
    if _client:
        _client.close()
        _client = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _make_content_hash(title: str, key_entities: list) -> str:
    raw = _normalise_text(title) + "|" + "|".join(sorted([e.lower() for e in key_entities]))
    return hashlib.md5(raw.encode()).hexdigest()


def _recency_score(created_at: datetime) -> float:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = max((datetime.now(timezone.utc) - created_at).total_seconds() / 86400, 0)
    return math.exp(-days / _RECENCY_HALF_LIFE)


def _normalise_list(values: list[float]) -> list[float]:
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(doc: dict) -> dict:
    doc.pop("_id", None)
    for k, v in doc.items():
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


# ─── Write ────────────────────────────────────────────────────────────────────

async def store_episodic_memory(
    user_id: str,
    session_id: str,
    title: str,
    content: str,
    outcome: str,
    turn_start: int,
    turn_end: int,
    key_entities: list = None,
    emotional_tone: str = "neutral",
    emotional_intensity: int = 2,
    tags: list = None,
    topic_cluster: str = "",
    importance_score: float = 5.0,
    embedding: list = None,
) -> str:
    """
    Write an episodic memory with full deduplication logic.

    Dedup behaviour:
      SAME hash, created < 24h ago  → SKIP
      SAME hash, created >= 24h ago → UPDATE access metadata
      NEW hash                       → INSERT new document
    """
    key_entities = key_entities or []
    tags         = tags or []
    now          = _now()
    chash        = _make_content_hash(title, key_entities)
    col          = _col()

    existing = await col.find_one(
        {"user_id": user_id, "content_hash": chash},
        {"memory_id": 1, "created_at": 1}
    )

    if existing:
        age_hours = (now - existing["created_at"].replace(tzinfo=timezone.utc)).total_seconds() / 3600
        if age_hours < 24:
            print(f"  ↳ SKIP episodic [{title[:40]}]: written < 24h ago")
            return existing["memory_id"]
        else:
            await col.update_one(
                {"memory_id": existing["memory_id"]},
                {"$inc": {"access_count": 1},
                 "$set": {"last_accessed_at": now, "updated_at": now}}
            )
            print(f"  ↳ TOUCH episodic [{title[:40]}]: exists, refreshed access")
            return existing["memory_id"]

    memory_id = f"ep_{uuid.uuid4().hex[:16]}"
    doc = {
        "memory_id":           memory_id,
        "user_id":             user_id,
        "session_id":          session_id,
        "title":               title,
        "content":             content,
        "outcome":             outcome,
        "outcome_note":        "",
        "turn_start":          turn_start,
        "turn_end":            turn_end,
        "key_entities":        key_entities,
        "emotional_tone":      emotional_tone,
        "emotional_intensity": max(1, min(5, emotional_intensity)),
        "tags":                tags,
        "topic_cluster":       topic_cluster or _infer_cluster(tags, title),
        "importance_score":    max(1.0, min(10.0, importance_score)),
        "content_hash":        chash,
        "access_count":        0,
        "last_accessed_at":    now,
        "related_ids":         [],
        "is_summary":          False,
        "source_ids":          [],
        "superseded":          False,
        "superseded_at":       None,
        "supersede_reason":    None,
        "created_at":          now,
        "updated_at":          now,
    }
    await col.insert_one(doc)
    doc.pop("_id", None)
    print(f"  ↳ CREATE episodic [{title[:40]}] importance={importance_score}")
    return memory_id


def _infer_cluster(tags: list, title: str) -> str:
    combined = " ".join(tags).lower() + " " + title.lower()
    clusters = {
        "career":       ["job", "work", "career", "salary", "interview", "promotion", "startup", "company"],
        "health":       ["health", "medical", "doctor", "diet", "exercise", "mental", "sleep", "therapy"],
        "learning":     ["learn", "study", "course", "skill", "book", "programming", "language", "tutorial"],
        "finance":      ["money", "budget", "invest", "savings", "expense", "cost", "finance", "bank"],
        "project":      ["project", "build", "launch", "product", "app", "deploy", "feature", "mvp"],
        "relationship": ["family", "friend", "partner", "colleague", "team", "social", "relationship"],
        "travel":       ["travel", "trip", "visit", "country", "city", "flight", "hotel"],
        "personal":     ["goal", "habit", "routine", "life", "personal", "growth", "motivation"],
    }
    for cluster, keywords in clusters.items():
        if any(k in combined for k in keywords):
            return cluster
    return "general"


# ─── Outcome update ───────────────────────────────────────────────────────────

async def update_episode_outcome(
    memory_id: str,
    outcome: str,
    outcome_note: str = "",
    importance_boost: float = 0.0
) -> bool:
    col    = _col()
    update = {
        "$set": {
            "outcome":      outcome,
            "outcome_note": outcome_note,
            "updated_at":   _now(),
        }
    }
    if importance_boost:
        update["$inc"] = {"importance_score": importance_boost}

    result = await col.update_one({"memory_id": memory_id}, update)
    return result.modified_count > 0


async def link_related_episodes(memory_id_a: str, memory_id_b: str):
    col = _col()
    await col.update_one({"memory_id": memory_id_a}, {"$addToSet": {"related_ids": memory_id_b}})
    await col.update_one({"memory_id": memory_id_b}, {"$addToSet": {"related_ids": memory_id_a}})


# ─── Read ─────────────────────────────────────────────────────────────────────

async def get_user_episodic_memories(
    user_id: str,
    limit: int = 5,
    session_id: str = None,
    query: str = None,
    query_vec: list = None,
) -> list:
    """
    Retrieve episodic memories for a user.
    Only returns non-superseded episodes.

    Three-stage retrieval (v3.5):
      Stage 1 — text search within current session (keyword match)
      Stage 2 — keyword-miss fallback: if Stage 1 returned 0 results but
                 a query was provided, retry WITHOUT the text filter using
                 importance + recency ranking. This handles short follow-up
                 questions like "Which is best?", "Tell me more", "What do
                 you think?" that have no keyword overlap with stored episodes
                 but are clearly continuations of the current context.
      Stage 3 — cross-session fallback: if still below threshold, pull top
                 episodes from other sessions by importance + recency.
    """
    col = _col()
    base_filter: dict = {"user_id": user_id, "superseded": {"$ne": True}}

    session_docs = []
    if session_id:
        session_filter = {**base_filter, "session_id": session_id}

        if not query:
            # No query — pure importance + recency ranking
            cursor = (
                col.find(session_filter, {"_id": 0})
                   .sort([("importance_score", -1), ("created_at", -1)])
                   .limit(limit)
            )
            session_docs = await cursor.to_list(length=limit)
        else:
            # Stage 1 — text search
            text_filter     = {**session_filter, "$text": {"$search": query}}
            candidate_limit = limit * 5
            try:
                cursor = (
                    col.find(text_filter, {"_id": 0, "score": {"$meta": "textScore"}})
                       .sort([("score", {"$meta": "textScore"})])
                       .limit(candidate_limit)
                )
                candidates = await cursor.to_list(length=candidate_limit)
                session_docs = _score_and_rank(candidates, limit)
            except Exception:
                session_docs = []

            # Stage 2 — keyword-miss fallback
            # Fires when text search returned nothing — short follow-up questions
            # like "Which is best?" have no keyword overlap with stored episodes
            # but the user clearly expects episodic context. Fall back to the
            # most important recent episodes from this session without filtering.
            if len(session_docs) == 0:
                cursor = (
                    col.find(session_filter, {"_id": 0})
                       .sort([("importance_score", -1), ("created_at", -1)])
                       .limit(limit)
                )
                session_docs = await cursor.to_list(length=limit)
                if session_docs:
                    print(f"  ↳ Episodic keyword-miss fallback: "
                          f"query '{query[:40]}' had no text matches → "
                          f"returning top {len(session_docs)} by importance")

    # Stage 3 — cross-session fallback
    if len(session_docs) < _CROSS_SESSION_THRESHOLD:
        needed       = limit - len(session_docs)
        existing_ids = {d["memory_id"] for d in session_docs}

        fallback_filter = {
            **base_filter,
            "memory_id": {"$nin": list(existing_ids)},
        }
        if session_id:
            fallback_filter["session_id"] = {"$ne": session_id}

        if not query:
            cursor = (
                col.find(fallback_filter, {"_id": 0})
                   .sort([("importance_score", -1), ("created_at", -1)])
                   .limit(needed)
            )
            fallback_docs = await cursor.to_list(length=needed)
        else:
            # Try text search first for cross-session too
            text_filter     = {**fallback_filter, "$text": {"$search": query}}
            candidate_limit = needed * 5
            try:
                cursor = (
                    col.find(text_filter, {"_id": 0, "score": {"$meta": "textScore"}})
                       .sort([("score", {"$meta": "textScore"})])
                       .limit(candidate_limit)
                )
                candidates    = await cursor.to_list(length=candidate_limit)
                fallback_docs = _score_and_rank(candidates, needed)
            except Exception:
                fallback_docs = []

            # Keyword-miss fallback for cross-session too
            if len(fallback_docs) == 0:
                cursor = (
                    col.find(fallback_filter, {"_id": 0})
                       .sort([("importance_score", -1), ("created_at", -1)])
                       .limit(needed)
                )
                fallback_docs = await cursor.to_list(length=needed)

        if fallback_docs:
            print(f"  ↳ Episodic cross-session fallback: "
                  f"{len(session_docs)} this session → "
                  f"+{len(fallback_docs)} from previous sessions")

        docs = session_docs + fallback_docs
    else:
        docs = session_docs

    await _bump_access(col, [d["memory_id"] for d in docs])
    return [_clean(d) for d in docs]


def _score_and_rank(candidates: list, limit: int) -> list:
    if not candidates:
        return []

    raw_text   = [c.get("score", 0.0) for c in candidates]
    recency    = [_recency_score(c["created_at"]) for c in candidates]
    importance = [(c.get("importance_score", 5.0) - 1) / 9 for c in candidates]
    access_raw = [math.log(1 + c.get("access_count", 0)) for c in candidates]

    norm_text   = _normalise_list(raw_text)
    norm_access = _normalise_list(access_raw)

    for i, doc in enumerate(candidates):
        doc["_score"] = (
            _W_RELEVANCE  * norm_text[i]   +
            _W_RECENCY    * recency[i]      +
            _W_IMPORTANCE * importance[i]   +
            _W_ACCESS     * norm_access[i]
        )
        doc.pop("score", None)

    candidates.sort(key=lambda x: x["_score"], reverse=True)
    return candidates[:limit]


async def _bump_access(col, memory_ids: list):
    if not memory_ids:
        return
    await col.update_many(
        {"memory_id": {"$in": memory_ids}},
        {"$inc": {"access_count": 1},
         "$set": {"last_accessed_at": _now()}}
    )


def _clean(doc: dict) -> dict:
    doc.pop("_score", None)
    doc.pop("_id", None)
    return doc


# ─── Specialised queries ──────────────────────────────────────────────────────

async def get_episodes_by_cluster(
    user_id: str,
    cluster: str,
    limit: int = 5,
    include_inactive: bool = False
) -> list:
    col  = _col()
    filt = {"user_id": user_id, "topic_cluster": cluster, "superseded": {"$ne": True}}
    if not include_inactive:
        filt["is_summary"] = {"$ne": True}
    cursor = (
        col.find(filt, {"_id": 0})
           .sort([("importance_score", -1), ("created_at", -1)])
           .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def get_ongoing_episodes(user_id: str) -> list:
    col = _col()
    cursor = (
        col.find(
            {"user_id": user_id, "outcome": "ongoing", "superseded": {"$ne": True}},
            {"_id": 0}
        )
        .sort([("importance_score", -1), ("created_at", -1)])
        .limit(20)
    )
    return await cursor.to_list(length=20)


async def get_high_importance_episodes(
    user_id: str, min_importance: float = 7.0, limit: int = 5
) -> list:
    col = _col()
    cursor = (
        col.find(
            {"user_id": user_id, "importance_score": {"$gte": min_importance},
             "superseded": {"$ne": True}},
            {"_id": 0}
        )
        .sort([("importance_score", -1), ("created_at", -1)])
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def get_episodes_by_tags(user_id: str, tags: list, limit: int = 3) -> list:
    col = _col()
    cursor = (
        col.find(
            {"user_id": user_id, "tags": {"$in": tags}, "superseded": {"$ne": True}},
            {"_id": 0}
        )
        .sort([("importance_score", -1), ("created_at", -1)])
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


# ─── Forget mechanism ─────────────────────────────────────────────────────────

async def forget_stale_episodes(user_id: str) -> dict:
    """
    Stage 1 of MongoDB forget mechanism.
    Marks old, unused, or resolved episodes as superseded.

    Four conditions — episode is superseded if ANY is met:
      A) age > 60d  AND importance < 5.0  AND access_count < 2
         → low-value and ignored by retrieval
      B) age > 180d AND access_count == 0
         → never retrieved in 6 months, clearly irrelevant
      C) outcome == "abandoned" AND age > 30d AND importance < 6.0
         → user gave up and hasn't thought about it since
      D) outcome == "resolved"  AND age > 120d AND access_count == 0
         → fully done, never revisited

    Summary episodes (is_summary=True) are never superseded here —
    they cover many other episodes and should outlive their sources.

    High-importance episodes (>= 8.0) are never superseded by this
    function regardless of other conditions — they're definitionally
    memorable (major life events, job changes, etc).
    """
    col = _col()
    now = _now()

    def cutoff(days: int) -> datetime:
        return now - timedelta(days=days)

    # Shared safety conditions applied to all rules
    base_safety = {
        "user_id":        user_id,
        "superseded":     {"$ne": True},
        "is_summary":     {"$ne": True},
        "importance_score": {"$lt": 8.0},   # never auto-forget high-importance episodes
    }

    results = {"a": 0, "b": 0, "c": 0, "d": 0}
    now_iso = now

    # ── Rule A: old + low importance + rarely accessed ────────
    filter_a = {
        **base_safety,
        "created_at":       {"$lt": cutoff(_FORGET_LOW_VALUE_DAYS)},
        "importance_score": {"$lt": _FORGET_LOW_VALUE_MAX_IMP},
        "access_count":     {"$lt": _FORGET_LOW_VALUE_MAX_ACCESS},
    }
    result = await col.update_many(
        filter_a,
        {"$set": {
            "superseded":       True,
            "superseded_at":    now_iso,
            "supersede_reason": "stale:low_value",
            "updated_at":       now_iso,
        }}
    )
    results["a"] = result.modified_count

    # ── Rule B: very old + never accessed at all ──────────────
    filter_b = {
        **base_safety,
        "created_at":   {"$lt": cutoff(_FORGET_NEVER_READ_DAYS)},
        "access_count": 0,
    }
    result = await col.update_many(
        filter_b,
        {"$set": {
            "superseded":       True,
            "superseded_at":    now_iso,
            "supersede_reason": "stale:never_read",
            "updated_at":       now_iso,
        }}
    )
    results["b"] = result.modified_count

    # ── Rule C: abandoned + old + low importance ──────────────
    filter_c = {
        **base_safety,
        "outcome":          "abandoned",
        "created_at":       {"$lt": cutoff(_FORGET_ABANDONED_DAYS)},
        "importance_score": {"$lt": _FORGET_ABANDONED_MAX_IMP},
    }
    result = await col.update_many(
        filter_c,
        {"$set": {
            "superseded":       True,
            "superseded_at":    now_iso,
            "supersede_reason": "stale:abandoned",
            "updated_at":       now_iso,
        }}
    )
    results["c"] = result.modified_count

    # ── Rule D: resolved + old + never re-read ────────────────
    filter_d = {
        **base_safety,
        "outcome":      "resolved",
        "created_at":   {"$lt": cutoff(_FORGET_RESOLVED_DAYS)},
        "access_count": 0,
    }
    result = await col.update_many(
        filter_d,
        {"$set": {
            "superseded":       True,
            "superseded_at":    now_iso,
            "supersede_reason": "stale:resolved_forgotten",
            "updated_at":       now_iso,
        }}
    )
    results["d"] = result.modified_count

    total = sum(results.values())
    if total > 0:
        print(f"  · forget_episodes [{user_id[:8]}]: "
              f"{results['a']} low-value, {results['b']} never-read, "
              f"{results['c']} abandoned, {results['d']} resolved → {total} superseded")

    return {
        "superseded_low_value":     results["a"],
        "superseded_never_read":    results["b"],
        "superseded_abandoned":     results["c"],
        "superseded_resolved":      results["d"],
        "total_superseded":         total,
    }


async def prune_superseded_episodes(user_id: str) -> dict:
    """
    Stage 2 of MongoDB forget mechanism.
    Hard-deletes superseded documents after they've sat long enough.

    Normal superseded episodes → delete after 180 days of being superseded.
    Summary episodes            → delete after 365 days (longer — they cover many sources).

    We keep superseded docs for a grace period so:
      - The user can still query "what did you forget?" via stats
      - Ongoing summaries that reference source_ids still resolve correctly
      - Nothing is deleted instantly (safe to roll back if decay was too aggressive)
    """
    col = _col()
    now = _now()

    def supersede_cutoff(days: int) -> datetime:
        return now - timedelta(days=days)

    # Delete non-summary superseded episodes past grace period
    result_normal = await col.delete_many({
        "user_id":       user_id,
        "superseded":    True,
        "is_summary":    {"$ne": True},
        "superseded_at": {"$lt": supersede_cutoff(_PRUNE_SUPERSEDED_DAYS)},
    })

    # Delete summary episodes past their longer grace period
    result_summary = await col.delete_many({
        "user_id":       user_id,
        "superseded":    True,
        "is_summary":    True,
        "superseded_at": {"$lt": supersede_cutoff(_PRUNE_SUMMARY_DAYS)},
    })

    total = result_normal.deleted_count + result_summary.deleted_count
    if total > 0:
        print(f"  · prune_episodes [{user_id[:8]}]: "
              f"{result_normal.deleted_count} normal + "
              f"{result_summary.deleted_count} summaries deleted")

    return {
        "pruned_normal":   result_normal.deleted_count,
        "pruned_summaries": result_summary.deleted_count,
        "total_pruned":    total,
    }


async def run_mongo_maintenance(user_id: str) -> dict:
    """
    Full MongoDB forget pipeline: forget stale episodes then prune old superseded ones.
    Safe to call frequently — only touches stale/superseded records.
    Triggered on session creation (background) alongside Neo4j decay.
    """
    forget_result = await forget_stale_episodes(user_id)
    prune_result  = await prune_superseded_episodes(user_id)
    return {
        "user_id": user_id,
        "forget":  forget_result,
        "prune":   prune_result,
    }


# ─── Consolidation ────────────────────────────────────────────────────────────

async def consolidate_old_episodes(
    user_id: str,
    older_than_days: int = 90,
    max_importance: float = 4.0,
    max_access_count: int = 2,
    dry_run: bool = False
) -> dict:
    col    = _col()
    cutoff = _now() - timedelta(days=older_than_days)

    candidates = await col.find(
        {
            "user_id":          user_id,
            "importance_score": {"$lt": max_importance},
            "access_count":     {"$lt": max_access_count},
            "created_at":       {"$lt": cutoff},
            "superseded":       {"$ne": True},
            "is_summary":       {"$ne": True},
        },
        {"_id": 0}
    ).to_list(length=500)

    if not candidates:
        return {"consolidated": 0, "summaries_created": 0, "clusters": []}

    by_cluster: dict[str, list] = {}
    for ep in candidates:
        c = ep.get("topic_cluster", "general")
        by_cluster.setdefault(c, []).append(ep)

    clusters_to_process = {c: eps for c, eps in by_cluster.items() if len(eps) >= 3}

    if not clusters_to_process:
        return {"consolidated": 0, "summaries_created": 0, "clusters": []}

    consolidated_count = 0
    summaries_created  = 0
    processed_clusters = []
    summary_id         = "dry_run"
    now                = _now()

    for cluster, episodes in clusters_to_process.items():
        episode_lines = [
            f"- {ep['title']} ({ep['outcome']}): {ep['content'][:120]}..."
            for ep in episodes
        ]
        summary_content = (
            f"Consolidated memory covering {len(episodes)} older episodes "
            f"in the '{cluster}' cluster from "
            f"{episodes[-1]['created_at'].strftime('%b %Y')} to "
            f"{episodes[0]['created_at'].strftime('%b %Y')}.\n\n"
            + "\n".join(episode_lines)
        )
        summary_title  = f"[Summary] {cluster.title()} — {len(episodes)} episodes"
        all_entities   = list(set(e for ep in episodes for e in ep.get("key_entities", [])))
        all_tags       = list(set(t for ep in episodes for t in ep.get("tags", [])))
        avg_importance = sum(ep.get("importance_score", 5) for ep in episodes) / len(episodes)
        source_ids     = [ep["memory_id"] for ep in episodes]

        if not dry_run:
            summary_id = f"ep_{uuid.uuid4().hex[:16]}"
            await col.insert_one({
                "memory_id":           summary_id,
                "user_id":             user_id,
                "session_id":          "consolidation",
                "title":               summary_title,
                "content":             summary_content,
                "outcome":             "resolved",
                "outcome_note":        f"Auto-consolidated {len(episodes)} episodes",
                "turn_start":          episodes[-1].get("turn_start", 0),
                "turn_end":            episodes[0].get("turn_end", 0),
                "key_entities":        all_entities[:20],
                "emotional_tone":      "neutral",
                "emotional_intensity": 1,
                "tags":                all_tags[:15],
                "topic_cluster":       cluster,
                "importance_score":    max(avg_importance, 4.0),
                "content_hash":        _make_content_hash(summary_title, all_entities),
                "access_count":        0,
                "last_accessed_at":    now,
                "related_ids":         [],
                "is_summary":          True,
                "source_ids":          source_ids,
                "superseded":          False,
                "superseded_at":       None,
                "supersede_reason":    None,
                "created_at":          now,
                "updated_at":          now,
            })
            await col.update_many(
                {"memory_id": {"$in": source_ids}},
                {"$set": {
                    "superseded":       True,
                    "superseded_at":    now,
                    "supersede_reason": "consolidated",
                    "updated_at":       now,
                }}
            )

        consolidated_count += len(episodes)
        summaries_created  += 1
        processed_clusters.append({
            "cluster":    cluster,
            "episodes":   len(episodes),
            "summary_id": summary_id,
        })

    return {
        "consolidated":      consolidated_count,
        "summaries_created": summaries_created,
        "clusters":          processed_clusters,
        "dry_run":           dry_run,
    }


# ─── Stats & admin ────────────────────────────────────────────────────────────

async def get_episodic_stats(user_id: str) -> dict:
    col = _col()

    total      = await col.count_documents({"user_id": user_id})
    active     = await col.count_documents({"user_id": user_id, "superseded": {"$ne": True}})
    superseded = await col.count_documents({"user_id": user_id, "superseded": True})
    ongoing    = await col.count_documents({"user_id": user_id, "outcome": "ongoing",    "superseded": {"$ne": True}})
    summaries  = await col.count_documents({"user_id": user_id, "is_summary": True})

    # Breakdown by supersede reason
    reason_pipeline = [
        {"$match": {"user_id": user_id, "superseded": True}},
        {"$group": {"_id": "$supersede_reason", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    reason_counts = await col.aggregate(reason_pipeline).to_list(length=20)

    pipeline = [
        {"$match": {"user_id": user_id, "superseded": {"$ne": True}}},
        {"$group": {"_id": "$topic_cluster", "count": {"$sum": 1},
                    "avg_importance": {"$avg": "$importance_score"}}},
        {"$sort": {"count": -1}}
    ]
    clusters = await col.aggregate(pipeline).to_list(length=20)

    top_accessed = await (
        col.find({"user_id": user_id, "superseded": {"$ne": True}},
                 {"_id": 0, "title": 1, "access_count": 1, "importance_score": 1})
           .sort("access_count", -1)
           .limit(5)
    ).to_list(length=5)

    return {
        "total_episodes":       total,
        "active_episodes":      active,
        "superseded_episodes":  superseded,
        "ongoing_episodes":     ongoing,
        "summary_episodes":     summaries,
        "supersede_reasons":    {r["_id"]: r["count"] for r in reason_counts},
        "cluster_distribution": [
            {"cluster": c["_id"], "count": c["count"],
             "avg_importance": round(c["avg_importance"], 1)}
            for c in clusters
        ],
        "most_accessed": top_accessed,
    }


# ─── Lifecycle helpers ────────────────────────────────────────────────────────

async def get_episodes_for_consolidation(
    user_id: str,
    older_than: datetime,
    max_importance: float = 7.9
) -> list:
    col    = _col()
    cursor = col.find({
        "user_id":          user_id,
        "created_at":       {"$lt": older_than},
        "importance_score": {"$lte": max_importance},
        "tags":             {"$nin": ["consolidated"]},
        "superseded":       {"$ne": True},
    }).sort("created_at", 1)
    docs = await cursor.to_list(length=500)
    return [_serialize(d) for d in docs]


async def delete_episodes_by_ids(user_id: str, memory_ids: list) -> int:
    from bson import ObjectId
    col       = _col()
    valid_ids = []
    for mid in memory_ids:
        try:
            valid_ids.append(ObjectId(mid))
        except Exception:
            continue
    if not valid_ids:
        return 0
    result = await col.delete_many({"_id": {"$in": valid_ids}, "user_id": user_id})
    return result.deleted_count


# ─── Delete ───────────────────────────────────────────────────────────────────

async def delete_user_episodic_memories(user_id: str) -> int:
    result = await _col().delete_many({"user_id": user_id})
    return result.deleted_count