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
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import Optional

from config import get_settings
from db import sqlite_manager as sql
from db import neo4j_manager as neo4j
from db import mongo_manager as mongo
from db import redis_manager as redis_mgr
from core.agent import chat
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
    metadata: dict = {}

class LoginReq(BaseModel):
    email: str

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
    if await sql.get_user_by_email(req.email):
        raise HTTPException(400, f"An account with email '{req.email}' already exists. Use /auth/login instead.")
    if await sql.get_user_by_username(req.username):
        raise HTTPException(400, f"Username '{req.username}' is taken. Choose another.")

    user = await sql.create_user(req.username, req.email, req.metadata)

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

@app.post("/chat")
async def chat_endpoint(req: ChatReq) -> ChatResp:
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