"""
db/mongo_manager.py
MongoDB — episodic memory storage.

Episodic memories are rich narrative records of meaningful multi-turn exchanges.
They capture the full arc: situation, challenge, approach, resolution, emotional tone.

Collection: chatbot.episodic_memories
Indexes: user_id+created_at, session_id, user_id+tags
"""
import uuid
from datetime import datetime
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import get_settings

settings = get_settings()

_client: Optional[AsyncIOMotorClient] = None


def get_db() -> AsyncIOMotorDatabase:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client[settings.mongo_db]


async def init_mongo():
    db = get_db()
    col = db[settings.mongo_episodic_collection]
    await col.create_index([("user_id", 1), ("created_at", -1)])
    await col.create_index([("session_id", 1)])
    await col.create_index([("user_id", 1), ("tags", 1)])
    await col.create_index("memory_id", unique=True)


async def close_mongo():
    global _client
    if _client:
        _client.close()
        _client = None


async def store_episodic_memory(
    user_id: str,
    session_id: str,
    title: str,
    content: str,
    outcome: str,                   # resolved | ongoing | abandoned | unclear
    turn_start: int,
    turn_end: int,
    key_entities: list = None,
    emotional_tone: str = "neutral",
    tags: list = None
) -> str:
    memory_id = f"ep_{uuid.uuid4().hex[:16]}"
    now = datetime.utcnow()
    doc = {
        "memory_id": memory_id,
        "user_id": user_id,
        "session_id": session_id,
        "title": title,
        "content": content,
        "outcome": outcome,
        "turn_start": turn_start,
        "turn_end": turn_end,
        "key_entities": key_entities or [],
        "emotional_tone": emotional_tone,
        "tags": tags or [],
        "created_at": now,
        "updated_at": now
    }
    db = get_db()
    await db[settings.mongo_episodic_collection].insert_one(doc)
    return memory_id


async def get_user_episodic_memories(
    user_id: str,
    limit: int = 5,
    session_id: str = None
) -> list:
    db = get_db()
    query: dict = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    cursor = (
        db[settings.mongo_episodic_collection]
        .find(query, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def get_episodic_by_tags(user_id: str, tags: list, limit: int = 3) -> list:
    db = get_db()
    cursor = (
        db[settings.mongo_episodic_collection]
        .find({"user_id": user_id, "tags": {"$in": tags}}, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)
