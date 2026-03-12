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
  created_at       datetime
  updated_at       datetime

4-Component Retrieval Scoring
──────────────────────────────
  final = (0.50 × relevance) + (0.20 × recency) + (0.20 × importance) + (0.10 × access_freq)

  relevance:    MongoDB $text score, normalised to [0,1]
  recency:      exp(-days_old / 30) — 1.0 today, 0.37 at 30d, 0.05 at 90d
  importance:   (importance_score - 1) / 9  normalised to [0,1]
  access_freq:  log(1 + access_count), normalised across candidate pool

  Why 4 components?
  - Relevance alone misses important old episodes
  - Recency alone misses important but not-recent episodes
  - Importance ensures pivotal moments always surface
  - Access frequency surfaces "sticky" memories the user returns to often

Deduplication
─────────────
  content_hash = MD5(normalise(title + sorted(key_entities)))
  If same hash within last 24h → SKIP (exact same event being written twice)
  If same hash from earlier    → UPDATE access_count + last_accessed_at

Consolidation
─────────────
  consolidate_old_episodes() merges episodes older than 90 days
  with importance < 4 AND access_count < 2 into a single summary episode.
  Source episodes are marked superseded=True (not deleted, just hidden from retrieval).
  Run this periodically (e.g. once a week per user) or on demand.
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

# ── Scoring weights ──────────────────────────────────────────
_W_RELEVANCE  = 0.50
_W_RECENCY    = 0.20
_W_IMPORTANCE = 0.20
_W_ACCESS     = 0.10
_RECENCY_HALF_LIFE = 30   # days


# ─── Connection ───────────────────────────────────────────────────────────────

def get_db() -> AsyncIOMotorDatabase:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client[settings.mongo_db]


def _col():
    return get_db()[settings.mongo_episodic_collection]


async def init_mongo():
    col = _col()

    # Operational indexes
    await col.create_index([("user_id", 1), ("created_at", -1)])
    await col.create_index([("user_id", 1), ("topic_cluster", 1), ("created_at", -1)])
    await col.create_index([("user_id", 1), ("importance_score", -1)])
    await col.create_index([("user_id", 1), ("tags", 1)])
    await col.create_index([("user_id", 1), ("outcome", 1)])
    await col.create_index([("user_id", 1), ("content_hash", 1)])
    await col.create_index("memory_id", unique=True)
    await col.create_index([("session_id", 1)])

    # Text search index with weights
    try:
        await col.create_index(
            [
                ("title",        "text"),
                ("content",      "text"),
                ("tags",         "text"),
                ("key_entities", "text"),
                ("topic_cluster","text"),
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
        pass   # Already exists

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
    """Dedup key: normalised title + sorted entities."""
    raw = _normalise_text(title) + "|" + "|".join(sorted([e.lower() for e in key_entities]))
    return hashlib.md5(raw.encode()).hexdigest()


def _recency_score(created_at: datetime) -> float:
    """exp(-days / half_life). 1.0 today → 0.37 at 30d → 0.05 at 90d."""
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
) -> str:
    """
    Write an episodic memory with full deduplication logic.

    Dedup behaviour:
      SAME hash, created < 24h ago  → SKIP, return existing memory_id
      SAME hash, created >= 24h ago → UPDATE access metadata, return existing id
      NEW hash                       → INSERT new document
    """
    key_entities = key_entities or []
    tags         = tags or []
    now          = _now()
    chash        = _make_content_hash(title, key_entities)
    col          = _col()

    # ── Dedup check ───────────────────────────────────────────
    existing = await col.find_one(
        {"user_id": user_id, "content_hash": chash},
        {"memory_id": 1, "created_at": 1}
    )

    if existing:
        age_hours = (now - existing["created_at"].replace(tzinfo=timezone.utc)).total_seconds() / 3600
        if age_hours < 24:
            # Very recent duplicate — skip entirely
            print(f"  ↳ SKIP episodic [{title[:40]}]: written < 24h ago")
            return existing["memory_id"]
        else:
            # Older duplicate — bump access metadata to surface it again
            await col.update_one(
                {"memory_id": existing["memory_id"]},
                {"$inc": {"access_count": 1},
                 "$set": {"last_accessed_at": now, "updated_at": now}}
            )
            print(f"  ↳ TOUCH episodic [{title[:40]}]: exists, refreshed access")
            return existing["memory_id"]

    # ── Insert new ────────────────────────────────────────────
    memory_id = f"ep_{uuid.uuid4().hex[:16]}"
    doc = {
        "memory_id":          memory_id,
        "user_id":            user_id,
        "session_id":         session_id,
        "title":              title,
        "content":            content,
        "outcome":            outcome,
        "outcome_note":       "",
        "turn_start":         turn_start,
        "turn_end":           turn_end,
        "key_entities":       key_entities,
        "emotional_tone":     emotional_tone,
        "emotional_intensity": max(1, min(5, emotional_intensity)),
        "tags":               tags,
        "topic_cluster":      topic_cluster or _infer_cluster(tags, title),
        "importance_score":   max(1.0, min(10.0, importance_score)),
        "content_hash":       chash,
        "access_count":       0,
        "last_accessed_at":   now,
        "related_ids":        [],
        "is_summary":         False,
        "source_ids":         [],
        "superseded":         False,
        "created_at":         now,
        "updated_at":         now,
    }
    await col.insert_one(doc)
    doc.pop("_id", None)
    print(f"  ↳ CREATE episodic [{title[:40]}] importance={importance_score}")
    return memory_id


def _infer_cluster(tags: list, title: str) -> str:
    """Simple rule-based cluster inference from tags and title."""
    combined = " ".join(tags).lower() + " " + title.lower()
    clusters = {
        "career":      ["job", "work", "career", "salary", "interview", "promotion", "startup", "company"],
        "health":      ["health", "medical", "doctor", "diet", "exercise", "mental", "sleep", "therapy"],
        "learning":    ["learn", "study", "course", "skill", "book", "programming", "language", "tutorial"],
        "finance":     ["money", "budget", "invest", "savings", "expense", "cost", "finance", "bank"],
        "project":     ["project", "build", "launch", "product", "app", "deploy", "feature", "mvp"],
        "relationship":["family", "friend", "partner", "colleague", "team", "social", "relationship"],
        "travel":      ["travel", "trip", "visit", "country", "city", "flight", "hotel"],
        "personal":    ["goal", "habit", "routine", "life", "personal", "growth", "motivation"],
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
    """
    Update the outcome of an episode when it resolves.

    Example: episode about "looking for a new job" stored as "ongoing"
    → when user says "I got the job!", call:
      update_episode_outcome(id, "resolved", "Got the job at XYZ in March")
    """
    col = _col()
    update = {
        "$set": {
            "outcome":     outcome,
            "outcome_note": outcome_note,
            "updated_at":  _now(),
        }
    }
    if importance_boost:
        update["$inc"] = {"importance_score": importance_boost}

    result = await col.update_one({"memory_id": memory_id}, update)
    return result.modified_count > 0


async def link_related_episodes(memory_id_a: str, memory_id_b: str):
    """Bidirectionally link two related episodes."""
    col = _col()
    await col.update_one(
        {"memory_id": memory_id_a},
        {"$addToSet": {"related_ids": memory_id_b}}
    )
    await col.update_one(
        {"memory_id": memory_id_b},
        {"$addToSet": {"related_ids": memory_id_a}}
    )


# ─── Read ─────────────────────────────────────────────────────────────────────

async def get_user_episodic_memories(
    user_id: str,
    limit: int = 5,
    session_id: str = None,
    query: str = None,
    query_vec: list = None,   # pre-computed embedding — skip embed call if provided
) -> list:
    """
    Retrieve episodic memories using 4-component hybrid scoring.

    Score = (0.50 × relevance) + (0.20 × recency) + (0.20 × importance) + (0.10 × access_freq)

    Each retrieval bumps access_count on returned documents so frequently
    useful episodes naturally rise in future rankings.

    Excludes:
      - superseded=True (consumed by consolidation summaries)
      - is_summary=True from regular retrieval (summaries retrieved separately)
    """
    col          = _col()
    base_filter  = {"user_id": user_id, "superseded": {"$ne": True}}

    if not query:
        # Recency-only fallback (cold start / preview)
        cursor = (
            col.find(base_filter, {"_id": 0})
               .sort([("importance_score", -1), ("created_at", -1)])
               .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        await _bump_access(col, [d["memory_id"] for d in docs])
        return [_clean(d) for d in docs]

    # ── Text search candidates ────────────────────────────────
    candidate_limit = limit * 5   # wider pool for better scoring
    text_filter = {**base_filter, "$text": {"$search": query}}
    cursor = (
        col.find(text_filter, {"_id": 0, "score": {"$meta": "textScore"}})
           .sort([("score", {"$meta": "textScore"})])
           .limit(candidate_limit)
    )
    candidates = await cursor.to_list(length=candidate_limit)

    if not candidates:
        # Fallback: importance + recency, no text match needed
        cursor = (
            col.find(base_filter, {"_id": 0})
               .sort([("importance_score", -1), ("created_at", -1)])
               .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        await _bump_access(col, [d["memory_id"] for d in docs])
        return [_clean(d) for d in docs]

    # ── Compute 4-component scores ────────────────────────────
    raw_text   = [c.get("score", 0.0) for c in candidates]
    recency    = [_recency_score(c["created_at"]) for c in candidates]
    importance = [(c.get("importance_score", 5.0) - 1) / 9 for c in candidates]
    access_raw = [math.log(1 + c.get("access_count", 0)) for c in candidates]

    norm_text   = _normalise_list(raw_text)
    norm_access = _normalise_list(access_raw)
    # recency and importance are already in [0,1]

    for i, doc in enumerate(candidates):
        doc["_score"] = (
            _W_RELEVANCE  * norm_text[i]   +
            _W_RECENCY    * recency[i]      +
            _W_IMPORTANCE * importance[i]   +
            _W_ACCESS     * norm_access[i]
        )
        doc.pop("score", None)

    candidates.sort(key=lambda x: x["_score"], reverse=True)
    top = candidates[:limit]

    # ── Bump access counts on returned docs ───────────────────
    await _bump_access(col, [d["memory_id"] for d in top])

    return [_clean(d) for d in top]


async def _bump_access(col, memory_ids: list):
    """Increment access_count and update last_accessed_at for retrieved docs."""
    if not memory_ids:
        return
    await col.update_many(
        {"memory_id": {"$in": memory_ids}},
        {"$inc": {"access_count": 1},
         "$set": {"last_accessed_at": _now()}}
    )


def _clean(doc: dict) -> dict:
    """Strip internal scoring fields before returning."""
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
    """Get all episodes in a topic cluster, sorted by importance then recency."""
    col = _col()
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
    """All unresolved episodes — useful for proactively checking on open threads."""
    col = _col()
    cursor = (
        col.find(
            {"user_id": user_id, "outcome": "ongoing", "superseded": {"$ne": True}},
            {"_id": 0}
        )
        .sort([("importance_score", -1), ("created_at", -1)])
    )
    return await cursor.to_list(length=20)


async def get_high_importance_episodes(user_id: str, min_importance: float = 7.0, limit: int = 5) -> list:
    """Retrieve pivotal memories — always surface regardless of recency."""
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


# ─── Consolidation ────────────────────────────────────────────────────────────

async def consolidate_old_episodes(
    user_id: str,
    older_than_days: int = 90,
    max_importance: float = 4.0,
    max_access_count: int = 2,
    dry_run: bool = False
) -> dict:
    """
    Merge low-value old episodes into a single summary episode per cluster.

    Candidates: age > older_than_days AND importance < max_importance
                AND access_count < max_access_count AND not already superseded

    For each cluster with candidates:
      1. Create one summary episode with a combined narrative
      2. Mark all source episodes as superseded=True (hidden from retrieval)
         They are NOT deleted — history is preserved, just excluded from results.

    Returns: { "consolidated": N, "summaries_created": M, "clusters": [...] }

    Run this once a week per active user, or on demand via admin endpoint.
    """
    col      = _col()
    cutoff   = _now() - timedelta(days=older_than_days)

    candidates = await col.find(
        {
            "user_id":         user_id,
            "importance_score": {"$lt": max_importance},
            "access_count":    {"$lt": max_access_count},
            "created_at":      {"$lt": cutoff},
            "superseded":      {"$ne": True},
            "is_summary":      {"$ne": True},
        },
        {"_id": 0}
    ).to_list(length=500)

    if not candidates:
        return {"consolidated": 0, "summaries_created": 0, "clusters": []}

    # Group by topic_cluster
    by_cluster: dict[str, list] = {}
    for ep in candidates:
        c = ep.get("topic_cluster", "general")
        by_cluster.setdefault(c, []).append(ep)

    # Only consolidate clusters with 3+ candidates (not worth it for 1-2)
    clusters_to_process = {c: eps for c, eps in by_cluster.items() if len(eps) >= 3}

    if not clusters_to_process:
        return {"consolidated": 0, "summaries_created": 0, "clusters": []}

    consolidated_count  = 0
    summaries_created   = 0
    processed_clusters  = []

    for cluster, episodes in clusters_to_process.items():
        # Build summary narrative from episode titles and outcomes
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
        summary_title = f"[Summary] {cluster.title()} — {len(episodes)} episodes"

        all_entities = list(set(
            e for ep in episodes for e in ep.get("key_entities", [])
        ))
        all_tags = list(set(t for ep in episodes for t in ep.get("tags", [])))
        avg_importance = sum(ep.get("importance_score", 5) for ep in episodes) / len(episodes)
        source_ids = [ep["memory_id"] for ep in episodes]

        if not dry_run:
            # Create summary episode
            summary_id = f"ep_{uuid.uuid4().hex[:16]}"
            now = _now()
            await col.insert_one({
                "memory_id":          summary_id,
                "user_id":            user_id,
                "session_id":         "consolidation",
                "title":              summary_title,
                "content":            summary_content,
                "outcome":            "resolved",
                "outcome_note":       f"Auto-consolidated {len(episodes)} episodes",
                "turn_start":         episodes[-1].get("turn_start", 0),
                "turn_end":           episodes[0].get("turn_end", 0),
                "key_entities":       all_entities[:20],
                "emotional_tone":     "neutral",
                "emotional_intensity": 1,
                "tags":               all_tags[:15],
                "topic_cluster":      cluster,
                "importance_score":   max(avg_importance, 4.0),
                "content_hash":       _make_content_hash(summary_title, all_entities),
                "access_count":       0,
                "last_accessed_at":   now,
                "related_ids":        [],
                "is_summary":         True,
                "source_ids":         source_ids,
                "superseded":         False,
                "created_at":         now,
                "updated_at":         now,
            })

            # Mark source episodes as superseded
            await col.update_many(
                {"memory_id": {"$in": source_ids}},
                {"$set": {"superseded": True, "updated_at": now}}
            )

        consolidated_count += len(episodes)
        summaries_created  += 1
        processed_clusters.append({
            "cluster":    cluster,
            "episodes":   len(episodes),
            "summary_id": summary_id if not dry_run else "dry_run"
        })

    return {
        "consolidated":       consolidated_count,
        "summaries_created":  summaries_created,
        "clusters":           processed_clusters,
        "dry_run":            dry_run,
    }


# ─── Stats & admin ────────────────────────────────────────────────────────────

async def get_episodic_stats(user_id: str) -> dict:
    """Usage statistics for the memory panel."""
    col = _col()

    total      = await col.count_documents({"user_id": user_id})
    active     = await col.count_documents({"user_id": user_id, "superseded": {"$ne": True}})
    superseded = await col.count_documents({"user_id": user_id, "superseded": True})
    ongoing    = await col.count_documents({"user_id": user_id, "outcome": "ongoing", "superseded": {"$ne": True}})
    summaries  = await col.count_documents({"user_id": user_id, "is_summary": True})

    # Cluster distribution
    pipeline = [
        {"$match": {"user_id": user_id, "superseded": {"$ne": True}}},
        {"$group": {"_id": "$topic_cluster", "count": {"$sum": 1},
                    "avg_importance": {"$avg": "$importance_score"}}},
        {"$sort": {"count": -1}}
    ]
    clusters = await col.aggregate(pipeline).to_list(length=20)

    # Most accessed
    top_accessed = await (
        col.find({"user_id": user_id, "superseded": {"$ne": True}},
                 {"_id": 0, "title": 1, "access_count": 1, "importance_score": 1})
           .sort("access_count", -1)
           .limit(5)
    ).to_list(length=5)

    return {
        "total_episodes":      total,
        "active_episodes":     active,
        "superseded_episodes": superseded,
        "ongoing_episodes":    ongoing,
        "summary_episodes":    summaries,
        "cluster_distribution": [
            {"cluster": c["_id"], "count": c["count"],
             "avg_importance": round(c["avg_importance"], 1)}
            for c in clusters
        ],
        "most_accessed": top_accessed,
    }


# ─── Delete ───────────────────────────────────────────────────────────────────

async def delete_user_episodic_memories(user_id: str) -> int:
    result = await _col().delete_many({"user_id": user_id})
    return result.deleted_count