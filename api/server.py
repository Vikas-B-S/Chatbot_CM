"""
api/server.py — FastAPI REST server

Auth endpoints:
  POST /auth/signup   create account (email + preferred name)
  POST /auth/login    login by email → returns user + new session

User endpoints:
  GET  /users                          list users
  GET  /users/{id}                     get user
  GET  /users/{id}/memory              current memories (Graphiti/Neo4j)
  GET  /users/{id}/memory?type=fact    filtered by type
  GET  /users/{id}/memory/history      temporal history for a topic
  GET  /users/{id}/episodic            episodic memories
  GET  /users/{id}/sessions            all sessions for user

Session endpoints:
  POST /sessions                       create new session
  GET  /sessions/{id}                  get session info
  GET  /sessions/{id}/summaries        Redis summaries (L0 + L1)
  GET  /sessions/{id}/context          full context snapshot

Chat:
  POST /chat                           send message

Misc:
  GET  /health
  GET  /                               chat UI
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import json as _json
from pydantic import BaseModel, EmailStr
from typing import Optional

from config import get_settings
from db import sqlite_manager as sql
from db import neo4j_manager as neo4j
from db import mongo_manager as mongo
from db import redis_manager as redis_mgr
from core.agent import chat, chat_stream
from memory.context_builder import build_context, format_context_for_prompt

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await sql.init_db()
    for fn, label in [
        (neo4j.init_neo4j, "Graphiti/Neo4j"),
        (mongo.init_mongo,  "MongoDB"),
    ]:
        try:
            await fn()
            print(f"✓ {label} ready")
        except Exception as e:
            print(f"⚠ {label}: {e}")
    yield
    await neo4j.close_driver()
    await mongo.close_mongo()
    await redis_mgr.close_redis()


app = FastAPI(
    title="MNEMO — Contextual Chatbot API",
    description=(
        "Multi-user chatbot with persistent temporal memory.\n\n"
        "**Auth:** email + preferred name. Email is the login key.\n\n"
        "**Neo4j + Graphiti** → temporal knowledge graph (facts, preferences, goals, constraints)\n"
        "**MongoDB** → episodic memories\n"
        "**Redis** → conversation summaries\n"
        "**SQLite** → users, sessions, turn logs"
    ),
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Serve UI ──────────────────────────────────────────────────
_ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir), name="ui")

@app.get("/", include_in_schema=False)
async def root():
    index = os.path.join(_ui_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "MNEMO API — visit /docs"}


# ─── Request / Response models ────────────────────────────────

class SignupReq(BaseModel):
    email:    str
    username: str        # preferred display name
    password: str
    metadata: dict = {}

class LoginReq(BaseModel):
    email:    str
    password: str

class CreateSessionReq(BaseModel):
    user_id: str
    name:    Optional[str] = None   # optional label e.g. "Work stuff"

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
    return {
        "status":  "ok",
        "version": "3.1.0",
        "storage": {
            "neo4j":   "Graphiti temporal knowledge graph",
            "mongodb": "episodic memories",
            "redis":   "conversation summaries (L0 + L1 meta)",
            "sqlite":  "users, sessions, turn logs"
        }
    }


# ─── Auth ─────────────────────────────────────────────────────

@app.post("/auth/signup", status_code=201)
async def signup(req: SignupReq):
    """
    Create a new account. Email must be unique. Username is display name.
    Automatically creates the first session.
    """
    pw = req.password or ""
    pw_errors = []
    if len(pw) < 8:                          pw_errors.append("at least 8 characters")
    if not re.search(r"[A-Z]", pw):          pw_errors.append("one uppercase letter")
    if not re.search(r"[0-9]", pw):          pw_errors.append("one number")
    if not re.search(r"[^A-Za-z0-9]", pw):  pw_errors.append("one special character")
    if pw_errors:
        raise HTTPException(400, "Password must contain: " + ", ".join(pw_errors) + ".")
    if await sql.get_user_by_email(req.email):
        raise HTTPException(400, f"An account with email '{req.email}' already exists. Use /auth/login instead.")
    if await sql.get_user_by_username(req.username):
        raise HTTPException(400, f"Username '{req.username}' is taken. Choose another.")

    user = await sql.create_user(req.username, req.email, req.password, req.metadata)

    # Seed Graphiti with user identity
    try:
        await neo4j.ensure_user_node(user["user_id"], req.username)
    except Exception as e:
        print(f"⚠ Graphiti seed: {e}")

    # Auto-create first session
    session = await sql.create_session(user["user_id"], name="Session 1")

    return {
        "user":    user,
        "session": session,
        "is_new":  True
    }


@app.post("/auth/login")
async def login(req: LoginReq):
    """
    Login with email. Returns user + creates a new session.
    If email not found returns 404 — client should redirect to signup.
    """
    user = await sql.get_user_by_email(req.email)
    if not user:
        raise HTTPException(404, "No account found with that email. Please sign up.")
    if not user.get("password_hash") or not sql.verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect password.")
    # Strip password_hash before returning
    user = {k: v for k, v in user.items() if k != "password_hash"}

    # Warm username cache so Graphiti frames future episodes with real name
    neo4j._username_cache[user["user_id"]] = user["username"]

    # Create a new session for this login
    existing = await sql.get_user_sessions(user["user_id"])
    session_name = f"Session {len(existing) + 1}"
    session = await sql.create_session(user["user_id"], name=session_name)

    return {
        "user":     user,
        "session":  session,
        "is_new":   False,
        "sessions": existing   # all previous sessions for switcher
    }


# ─── Users ────────────────────────────────────────────────────

@app.get("/users")
async def list_users():
    return await sql.list_users()


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    u = await sql.get_user(user_id)
    if not u:
        raise HTTPException(404, "User not found")
    return u


@app.get("/users/{user_id}/memory")
async def get_user_memory(
    user_id: str,
    type: Optional[str] = Query(None, description="fact|preference|goal|constraint")
):
    if not await sql.get_user(user_id):
        raise HTTPException(404, "User not found")
    try:
        if type:
            mems = await neo4j.get_user_memories(user_id, memory_type=type)
            return {"user_id": user_id, "type": type, "memories": mems, "count": len(mems)}
        return await neo4j.get_full_memory_graph(user_id)
    except Exception as e:
        raise HTTPException(503, f"Graphiti/Neo4j unavailable: {e}")


@app.get("/users/{user_id}/memory/history")
async def get_memory_history(
    user_id: str,
    topic: str = Query(..., description="e.g. 'coding language preference'")
):
    """Full temporal history of a fact including expired versions."""
    if not await sql.get_user(user_id):
        raise HTTPException(404, "User not found")
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
        raise HTTPException(503, f"Graphiti/Neo4j unavailable: {e}")




@app.get("/users/{user_id}/memory/timeline")
async def memory_timeline(user_id: str, key: str):
    """
    Structured timeline for a single canonical_key.
    Shows every version: active/inactive, when it changed, reactivations.

    Example: GET /users/{id}/memory/timeline?key=coding_language
    Returns: Python(v1 inactive) → C++(v2 inactive) → Python(v3 active, reactivated)
    """
    timeline = await neo4j.get_memory_timeline(user_id, key)
    return timeline

@app.get("/users/{user_id}/episodic")
async def get_user_episodic(user_id: str, limit: int = 10):
    if not await sql.get_user(user_id):
        raise HTTPException(404, "User not found")
    try:
        return await mongo.get_user_episodic_memories(user_id, limit=limit)
    except Exception as e:
        raise HTTPException(503, f"MongoDB unavailable: {e}")


@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    if not await sql.get_user(user_id):
        raise HTTPException(404, "User not found")
    return await sql.get_user_sessions(user_id)


# ─── Sessions ─────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def create_session(req: CreateSessionReq):
    if not await sql.get_user(req.user_id):
        raise HTTPException(404, "User not found")
    existing = await sql.get_user_sessions(req.user_id)
    name = req.name or f"Session {len(existing) + 1}"
    session = await sql.create_session(req.user_id, name=name)
    return session


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    s = await sql.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@app.get("/sessions/{session_id}/summaries")
async def get_session_summaries(session_id: str):
    if not await sql.get_session(session_id):
        raise HTTPException(404, "Session not found")
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
    if not s:
        raise HTTPException(404, "Session not found")
    ctx = await build_context(session_id, s["user_id"])
    return {
        "session_id":        session_id,
        "component_counts":  {k: len(v) for k, v in ctx.items()},
        "formatted_context": format_context_for_prompt(ctx),
        "raw": ctx
    }


# ─── Chat ─────────────────────────────────────────────────────



# ─── Delete user ──────────────────────────────────────────────



@app.delete("/users/{user_id}/credentials", status_code=200)
async def delete_credentials(user_id: str):
    """
    Wipe login credentials (email + password) only.
    All memory, sessions, and conversation history are preserved.
    """
    user = await sql.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found.")
    await sql.delete_user_credentials(user_id)
    return {
        "user_id": user_id,
        "username": user["username"],
        "status": "credentials removed — memory and sessions preserved"
    }

@app.delete("/users/{user_id}", status_code=200)
async def delete_user(user_id: str):
    """
    Hard delete a user and ALL their data across every store:
      - SQLite : user row, all sessions, all turn logs
      - Neo4j  : entire Graphiti group (all episodes, entities, edges)
      - MongoDB: all episodic memories
      - Redis  : all summary keys for all sessions

    This is irreversible.
    """
    # Verify user exists first
    user = await sql.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found.")

    errors = []
    results = {}

    # ── 1. SQLite — sessions + turns + user row ───────────────
    try:
        sessions = await sql.get_user_sessions(user_id)
        session_ids = [s["session_id"] for s in sessions]
        deleted_counts = await sql.delete_user_data(user_id)
        results["sqlite"] = deleted_counts
    except Exception as e:
        errors.append(f"SQLite: {e}")
        session_ids = []

    # ── 2. Neo4j / Graphiti ───────────────────────────────────
    try:
        await neo4j.delete_user_graph(user_id)
        results["neo4j"] = "deleted"
    except Exception as e:
        errors.append(f"Neo4j: {e}")
        results["neo4j"] = f"error: {e}"

    # ── 3. MongoDB — episodic memories ───────────────────────
    try:
        count = await mongo.delete_user_episodic_memories(user_id)
        results["mongodb"] = f"{count} episodes deleted"
    except Exception as e:
        errors.append(f"MongoDB: {e}")
        results["mongodb"] = f"error: {e}"

    # ── 4. Redis — all session summaries ─────────────────────
    try:
        count = await redis_mgr.delete_user_summaries(session_ids, user_id=user_id)
        results["redis"] = f"{count} keys deleted"
    except Exception as e:
        errors.append(f"Redis: {e}")
        results["redis"] = f"error: {e}"

    # Clean username cache
    neo4j._username_cache.pop(user_id, None)

    return {
        "deleted_user_id": user_id,
        "username":        user["username"],
        "results":         results,
        "errors":          errors,
        "status":          "complete" if not errors else "partial"
    }


@app.get("/users/{user_id}/episodic/stats")
async def episodic_stats(user_id: str):
    """Usage stats: cluster distribution, access counts, ongoing episodes."""
    return await mongo.get_episodic_stats(user_id)


@app.get("/users/{user_id}/episodic/ongoing")
async def ongoing_episodes(user_id: str):
    """All unresolved ongoing episodes — open threads."""
    return await mongo.get_ongoing_episodes(user_id)


@app.get("/users/{user_id}/episodic/important")
async def important_episodes(user_id: str, min_importance: float = 7.0):
    """High-importance episodes — always surface regardless of recency."""
    return await mongo.get_high_importance_episodes(user_id, min_importance)


@app.get("/users/{user_id}/episodic/cluster/{cluster}")
async def episodes_by_cluster(user_id: str, cluster: str):
    """All episodes in a topic cluster (career, health, learning, etc.)."""
    return await mongo.get_episodes_by_cluster(user_id, cluster)


@app.post("/users/{user_id}/episodic/{memory_id}/outcome")
async def update_outcome(user_id: str, memory_id: str, outcome: str, note: str = "", boost: float = 0.0):
    """Update episode outcome when it resolves. outcome: resolved|ongoing|abandoned."""
    updated = await mongo.update_episode_outcome(memory_id, outcome, note, boost)
    return {"updated": updated, "memory_id": memory_id, "new_outcome": outcome}


@app.post("/users/{user_id}/episodic/consolidate")
async def consolidate(user_id: str, older_than_days: int = 90, dry_run: bool = False):
    """
    Merge low-value old episodes into cluster summaries.
    Use dry_run=true to preview what would be consolidated without writing.
    """
    result = await mongo.consolidate_old_episodes(
        user_id, older_than_days=older_than_days, dry_run=dry_run
    )
    return result

@app.post("/chat")
async def chat_endpoint(req: ChatReq):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Yields tokens as they arrive — user sees first token in ~300ms.
    Storage fires in background after stream completes.

    Response format (SSE):
      data: {"token": "Hello"}
      data: {"token": " there"}
      data: {"done": true, "context_used": {...}}
    """
    if not await sql.get_user(req.user_id):
        raise HTTPException(404, "User not found")
    s = await sql.get_session(req.session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s["user_id"] != req.user_id:
        raise HTTPException(403, "Session does not belong to this user")
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    async def event_stream():
        try:
            async for chunk in chat_stream(
                user_id=req.user_id,
                session_id=req.session_id,
                user_message=req.message
            ):
                yield f"data: {_json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/sync")
async def chat_sync_endpoint(req: ChatReq) -> ChatResp:
    """Non-streaming fallback — waits for full response before returning."""
    if not await sql.get_user(req.user_id):
        raise HTTPException(404, "User not found")
    s = await sql.get_session(req.session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s["user_id"] != req.user_id:
        raise HTTPException(403, "Session does not belong to this user")
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    result = await chat(
        user_id=req.user_id,
        session_id=req.session_id,
        user_message=req.message
    )
    return ChatResp(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)