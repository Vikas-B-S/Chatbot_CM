"""
api/server.py — FastAPI REST server

KEY FIX (v3.2):
  Background storage is now handled by FastAPI BackgroundTasks instead of
  asyncio.create_task() inside the streaming generator.

  Root cause of the previous bug:
    The SSE client closes the connection as soon as it receives the final
    "done: true" chunk. FastAPI then cancels the generator. Since
    asyncio.create_task(_background_store()) was called at the END of the
    generator (after yielding all tokens), the client disconnect happened
    first — the task was never created, so nothing was ever stored.

  The fix:
    chat_stream() is now split into two functions:
      1. stream_tokens()  — pure generator, yields tokens + done chunk
      2. background_store_wrapper() — called via BackgroundTasks AFTER
         the response is fully sent, guaranteed to run even if client
         disconnects early.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import json as _json
from pydantic import BaseModel
from typing import Optional

from config import get_settings
from db import sqlite_manager as sql
from db import neo4j_manager as neo4j
from db import mongo_manager as mongo
from db import redis_manager as redis_mgr
from core.agent import stream_tokens, background_store_wrapper, chat
from memory.context_builder import build_context, format_context_for_prompt

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await sql.init_db()

    try:
        from db.embedder import embed_text
        await embed_text("warmup")
        print("✓ Embedding model ready")
    except Exception as e:
        print(f"⚠ Embedding warmup: {e}")

    try:
        await neo4j.init_neo4j()
        print("✓ Graphiti/Neo4j ready")
    except Exception as e:
        print(f"⚠ Graphiti/Neo4j: {e}")

    try:
        driver = await neo4j._get_driver()
        async with driver.session() as s:
            await s.run("RETURN 1")
        print("✓ Neo4j connection pre-warmed")
    except Exception as e:
        print(f"⚠ Neo4j pre-warm failed: {e}")

    try:
        await mongo.init_mongo()
        print("✓ MongoDB ready")
    except Exception as e:
        print(f"⚠ MongoDB: {e}")

    try:
        r = await redis_mgr.get_redis()
        await r.ping()
        print("✓ Redis ready")
    except Exception as e:
        print(f"⚠ Redis failed: {e}")
        print(f"  → AUTH error? Set REDIS_PASSWORD= (empty) in .env")

    yield

    await neo4j.close_driver()
    await mongo.close_mongo()
    await redis_mgr.close_redis()


app = FastAPI(
    title="MNEMO — Contextual Chatbot API",
    version="3.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir), name="ui")

@app.get("/", include_in_schema=False)
async def root():
    index = os.path.join(_ui_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "MNEMO API — visit /docs"}


# ─── Models ───────────────────────────────────────────────────

class SignupReq(BaseModel):
    email:    str
    username: str
    password: str
    metadata: dict = {}

class LoginReq(BaseModel):
    email:    str
    password: str

class CreateSessionReq(BaseModel):
    user_id: str
    name:    Optional[str] = None

class ChatReq(BaseModel):
    user_id:    str
    session_id: str
    message:    str

class ChatResp(BaseModel):
    response:        str
    turn_number:     int
    routing:         dict
    memories_stored: list
    episodic_stored: Optional[dict]
    user_db_updated: bool
    summarization:   Optional[dict]
    context_used:    dict


# ─── Health ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.2.0"}


# ─── Auth ─────────────────────────────────────────────────────

@app.post("/auth/signup", status_code=201)
async def signup(req: SignupReq):
    pw = req.password or ""
    pw_errors = []
    if len(pw) < 8:                          pw_errors.append("at least 8 characters")
    if not re.search(r"[A-Z]", pw):          pw_errors.append("one uppercase letter")
    if not re.search(r"[0-9]", pw):          pw_errors.append("one number")
    if not re.search(r"[^A-Za-z0-9]", pw):  pw_errors.append("one special character")
    if pw_errors:
        raise HTTPException(400, "Password must contain: " + ", ".join(pw_errors) + ".")
    if await sql.get_user_by_email(req.email):
        raise HTTPException(400, f"Account with email '{req.email}' already exists.")
    if await sql.get_user_by_username(req.username):
        raise HTTPException(400, f"Username '{req.username}' is taken.")
    user = await sql.create_user(req.username, req.email, req.password, req.metadata)
    try:
        await neo4j.ensure_user_node(user["user_id"], req.username)
    except Exception as e:
        print(f"⚠ Graphiti seed: {e}")
    session = await sql.create_session(user["user_id"], name="Session 1")
    return {"user": user, "session": session, "is_new": True}


@app.post("/auth/login")
async def login(req: LoginReq):
    user = await sql.get_user_by_email(req.email)
    if not user:
        raise HTTPException(404, "No account found with that email.")
    if not user.get("password_hash") or not sql.verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect password.")
    user = {k: v for k, v in user.items() if k != "password_hash"}
    neo4j._username_cache[user["user_id"]] = user["username"]
    existing     = await sql.get_user_sessions(user["user_id"])
    session_name = f"Session {len(existing) + 1}"
    session      = await sql.create_session(user["user_id"], name=session_name)
    return {"user": user, "session": session, "is_new": False, "sessions": existing}


# ─── Users ────────────────────────────────────────────────────

@app.get("/users")
async def list_users():
    return await sql.list_users()

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    u = await sql.get_user(user_id)
    if not u: raise HTTPException(404, "User not found")
    return u

@app.get("/users/{user_id}/memory")
async def get_user_memory(user_id: str, type: Optional[str] = Query(None)):
    if not await sql.get_user(user_id): raise HTTPException(404, "User not found")
    try:
        if type:
            mems = await neo4j.get_user_memories(user_id, memory_type=type)
            return {"user_id": user_id, "type": type, "memories": mems, "count": len(mems)}
        return await neo4j.get_full_memory_graph(user_id)
    except Exception as e:
        raise HTTPException(503, f"Neo4j unavailable: {e}")

@app.get("/users/{user_id}/memory/history")
async def get_memory_history(user_id: str, topic: str = Query(...)):
    if not await sql.get_user(user_id): raise HTTPException(404, "User not found")
    try:
        history = await neo4j.get_fact_history(user_id, topic)
        return {
            "user_id":       user_id,
            "topic":         topic,
            "current_facts": [h for h in history if h["status"] == "current"],
            "expired_facts": [h for h in history if h["status"] == "expired"],
            "total":         len(history)
        }
    except Exception as e:
        raise HTTPException(503, f"Neo4j unavailable: {e}")

@app.get("/users/{user_id}/memory/timeline")
async def memory_timeline(user_id: str, key: str):
    return await neo4j.get_memory_timeline(user_id, key)

@app.get("/users/{user_id}/episodic")
async def get_user_episodic(user_id: str, limit: int = 10):
    if not await sql.get_user(user_id): raise HTTPException(404, "User not found")
    try:
        return await mongo.get_user_episodic_memories(user_id, limit=limit)
    except Exception as e:
        raise HTTPException(503, f"MongoDB unavailable: {e}")

@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    if not await sql.get_user(user_id): raise HTTPException(404, "User not found")
    return await sql.get_user_sessions(user_id)


# ─── Sessions ─────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def create_session(req: CreateSessionReq):
    if not await sql.get_user(req.user_id): raise HTTPException(404, "User not found")
    existing = await sql.get_user_sessions(req.user_id)
    name     = req.name or f"Session {len(existing) + 1}"
    return await sql.create_session(req.user_id, name=name)

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    s = await sql.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    return s

@app.get("/sessions/{session_id}/last_turn")
async def get_last_turn(session_id: str):
    s = await sql.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    turn = await sql.get_last_turn(session_id)
    if not turn: return {"turn_number": None}
    rd = turn.get("router_decision") or {}
    return {
        "turn_number":     turn["turn_number"],
        "routing":         rd,
        "memories_stored": rd.get("user_memories", []),
        "episodic_stored": {"title": "stored"} if rd.get("episodic_should_store") else None,
        "summarization":   None,
    }

@app.get("/sessions/{session_id}/summaries")
async def get_session_summaries(session_id: str):
    if not await sql.get_session(session_id): raise HTTPException(404, "Session not found")
    try:
        all_s = await redis_mgr.get_session_summaries(session_id)
        return {
            "session_id":      session_id,
            "total":           len(all_s),
            "level_0_batches": [s for s in all_s if s.get("level", 0) == 0],
            "level_1_meta":    [s for s in all_s if s.get("level", 0) >= 1],
        }
    except Exception as e:
        raise HTTPException(503, f"Redis unavailable: {e}")

@app.get("/sessions/{session_id}/context")
async def get_session_context(session_id: str):
    s = await sql.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    ctx = await build_context(session_id, s["user_id"])
    return {
        "session_id":        session_id,
        "component_counts":  {k: len(v) for k, v in ctx.items()},
        "formatted_context": format_context_for_prompt(ctx),
        "raw":               ctx
    }


# ─── Chat ─────────────────────────────────────────────────────
# KEY CHANGE: background_tasks.add_task() is used instead of
# asyncio.create_task() so storage runs AFTER the response is
# fully sent, even if the SSE client disconnects early.

@app.post("/chat")
async def chat_endpoint(req: ChatReq, background_tasks: BackgroundTasks):
    if not await sql.get_user(req.user_id):  raise HTTPException(404, "User not found")
    s = await sql.get_session(req.session_id)
    if not s:                                raise HTTPException(404, "Session not found")
    if s["user_id"] != req.user_id:          raise HTTPException(403, "Session does not belong to this user")
    if not req.message.strip():              raise HTTPException(400, "Message cannot be empty")

    # Collect the full response while streaming tokens to client
    # then schedule storage via BackgroundTasks (runs after response)
    collected = {"full_response": "", "context": None}

    async def event_stream():
        try:
            async for chunk in stream_tokens(
                user_id=req.user_id,
                session_id=req.session_id,
                user_message=req.message,
                collected=collected   # stream_tokens writes into this dict
            ):
                yield f"data: {_json.dumps(chunk)}\n\n"
        except Exception as e:
            print(f"  ✗ Stream error: {e}")
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"

    # Schedule background storage BEFORE returning the response
    # BackgroundTasks guarantees this runs after the response is sent
    background_tasks.add_task(
        background_store_wrapper,
        user_id=req.user_id,
        session_id=req.session_id,
        user_message=req.message,
        collected=collected
    )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/sync")
async def chat_sync_endpoint(req: ChatReq) -> ChatResp:
    if not await sql.get_user(req.user_id):  raise HTTPException(404, "User not found")
    s = await sql.get_session(req.session_id)
    if not s:                                raise HTTPException(404, "Session not found")
    if s["user_id"] != req.user_id:          raise HTTPException(403, "Session does not belong to this user")
    if not req.message.strip():              raise HTTPException(400, "Message cannot be empty")
    result = await chat(user_id=req.user_id, session_id=req.session_id, user_message=req.message)
    return ChatResp(**result)


# ─── Delete ───────────────────────────────────────────────────

@app.delete("/users/{user_id}/credentials", status_code=200)
async def delete_credentials(user_id: str):
    user = await sql.get_user(user_id)
    if not user: raise HTTPException(404, "User not found.")
    await sql.delete_user_credentials(user_id)
    return {"user_id": user_id, "username": user["username"], "status": "credentials removed"}

@app.delete("/users/{user_id}", status_code=200)
async def delete_user(user_id: str):
    user = await sql.get_user(user_id)
    if not user: raise HTTPException(404, "User not found.")
    errors = []; results = {}
    try:
        sessions    = await sql.get_user_sessions(user_id)
        session_ids = [s["session_id"] for s in sessions]
        results["sqlite"] = await sql.delete_user_data(user_id)
    except Exception as e:
        errors.append(f"SQLite: {e}"); session_ids = []
    try:
        await neo4j.delete_user_graph(user_id); results["neo4j"] = "deleted"
    except Exception as e:
        errors.append(f"Neo4j: {e}"); results["neo4j"] = f"error: {e}"
    try:
        count = await mongo.delete_user_episodic_memories(user_id)
        results["mongodb"] = f"{count} episodes deleted"
    except Exception as e:
        errors.append(f"MongoDB: {e}"); results["mongodb"] = f"error: {e}"
    try:
        count = await redis_mgr.delete_user_summaries(session_ids, user_id=user_id)
        results["redis"] = f"{count} keys deleted"
    except Exception as e:
        errors.append(f"Redis: {e}"); results["redis"] = f"error: {e}"
    neo4j._username_cache.pop(user_id, None)
    return {"deleted_user_id": user_id, "username": user["username"],
            "results": results, "errors": errors,
            "status": "complete" if not errors else "partial"}


# ─── Episodic extras ──────────────────────────────────────────

@app.get("/users/{user_id}/episodic/stats")
async def episodic_stats(user_id: str):
    return await mongo.get_episodic_stats(user_id)

@app.get("/users/{user_id}/episodic/ongoing")
async def ongoing_episodes(user_id: str):
    return await mongo.get_ongoing_episodes(user_id)

@app.get("/users/{user_id}/episodic/important")
async def important_episodes(user_id: str, min_importance: float = 7.0):
    return await mongo.get_high_importance_episodes(user_id, min_importance)

@app.get("/users/{user_id}/episodic/cluster/{cluster}")
async def episodes_by_cluster(user_id: str, cluster: str):
    return await mongo.get_episodes_by_cluster(user_id, cluster)

@app.post("/users/{user_id}/episodic/{memory_id}/outcome")
async def update_outcome(user_id: str, memory_id: str, outcome: str, note: str = "", boost: float = 0.0):
    updated = await mongo.update_episode_outcome(memory_id, outcome, note, boost)
    return {"updated": updated, "memory_id": memory_id, "new_outcome": outcome}

@app.post("/users/{user_id}/episodic/consolidate")
async def consolidate(user_id: str, older_than_days: int = 90, dry_run: bool = False):
    return await mongo.consolidate_old_episodes(user_id, older_than_days=older_than_days, dry_run=dry_run)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)