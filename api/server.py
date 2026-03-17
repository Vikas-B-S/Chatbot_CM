"""
api/server.py — FastAPI REST server

KEY FIX (v3.2):
  Background storage is now handled by FastAPI BackgroundTasks instead of
  asyncio.create_task() inside the streaming generator.

FORGET MECHANISM (v3.4):
  Memory maintenance (decay + prune) is triggered:
    1. On every session creation — passive background maintenance
    2. Via POST /users/{user_id}/memory/maintenance — manual trigger

  This is safe to call frequently — decay only touches memories that
  haven't been accessed within the decay window (30d normal / 3d temporary).
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
    version="3.5.0",
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
    return {"status": "ok", "version": "3.5.0"}


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
async def get_user_memory(
    user_id: str,
    type: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
):
    if not await sql.get_user(user_id): raise HTTPException(404, "User not found")
    try:
        graph = await neo4j.get_full_memory_graph(user_id, session_id=session_id)
        if type:
            return {
                "user_id":    user_id,
                "type":       type,
                "memories":   graph["memories_by_type"].get(type, []),
                "fading":     [m for m in graph.get("fading_memories", [])
                               if m.get("memory_type") == type],
            }
        return graph
    except Exception as e:
        raise HTTPException(503, f"Neo4j unavailable: {e}")


# ─── Memory maintenance (forget mechanism) ────────────────────

@app.post("/users/{user_id}/memory/maintenance")
async def run_maintenance(user_id: str, background_tasks: BackgroundTasks):
    """
    Manually trigger memory decay + prune for a user.
    Also called automatically in background on every session creation.

    Decay reduces confidence of memories not accessed in 30d (normal) or 3d (temporary).
    Prune deactivates memories whose confidence has fallen below 0.25.
    """
    if not await sql.get_user(user_id):
        raise HTTPException(404, "User not found")

    background_tasks.add_task(_run_maintenance_bg, user_id)
    return {
        "status":  "scheduled",
        "user_id": user_id,
        "message": "Memory maintenance scheduled in background"
    }


@app.post("/users/{user_id}/memory/maintenance/sync")
async def run_maintenance_sync(user_id: str):
    """
    Synchronous version of maintenance — waits for result.
    Runs both Neo4j decay+prune and MongoDB forget+prune.
    Useful for testing/debugging the forget mechanism.
    """
    if not await sql.get_user(user_id):
        raise HTTPException(404, "User not found")

    neo4j_result = await neo4j.run_memory_maintenance(user_id)
    mongo_result = await mongo.run_mongo_maintenance(user_id)

    return {
        "user_id": user_id,
        "neo4j":   neo4j_result,
        "mongodb": mongo_result,
    }


async def _run_maintenance_bg(user_id: str):
    """
    Background wrapper for full memory maintenance — Neo4j + MongoDB.
    Catches all exceptions independently so one failure doesn't block the other.
    """
    # Neo4j: confidence decay + prune dead memories
    try:
        neo4j_result = await neo4j.run_memory_maintenance(user_id)
        neo4j_total  = (neo4j_result["decay"]["total_decayed"] +
                        neo4j_result["prune"]["pruned"])
        if neo4j_total > 0:
            print(f"  ✓ Neo4j maintenance [{user_id[:8]}]: "
                  f"{neo4j_result['decay']['total_decayed']} decayed, "
                  f"{neo4j_result['prune']['pruned']} pruned")
    except Exception as e:
        print(f"  ⚠ Neo4j maintenance failed for {user_id[:8]}: {e}")

    # MongoDB: forget stale episodes + prune superseded docs
    try:
        mongo_result = await mongo.run_mongo_maintenance(user_id)
        mongo_total  = (mongo_result["forget"]["total_superseded"] +
                        mongo_result["prune"]["total_pruned"])
        if mongo_total > 0:
            print(f"  ✓ MongoDB maintenance [{user_id[:8]}]: "
                  f"{mongo_result['forget']['total_superseded']} forgotten, "
                  f"{mongo_result['prune']['total_pruned']} pruned")
    except Exception as e:
        print(f"  ⚠ MongoDB maintenance failed for {user_id[:8]}: {e}")


@app.get("/users/{user_id}/episodic")
async def get_user_episodic(
    user_id: str,
    limit: int = 10,
    session_id: Optional[str] = Query(None),
):
    if not await sql.get_user(user_id): raise HTTPException(404, "User not found")
    try:
        return await mongo.get_user_episodic_memories(user_id, limit=limit, session_id=session_id)
    except Exception as e:
        raise HTTPException(503, f"MongoDB unavailable: {e}")

@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    if not await sql.get_user(user_id): raise HTTPException(404, "User not found")
    return await sql.get_user_sessions(user_id)


# ─── Sessions ─────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def create_session(req: CreateSessionReq, background_tasks: BackgroundTasks):
    """
    Create a new session. Also schedules memory maintenance in background —
    this is the primary trigger for the forget mechanism. Since session
    creation happens infrequently (once per conversation), it's the ideal
    low-cost moment to run decay + prune without any latency impact.
    """
    if not await sql.get_user(req.user_id): raise HTTPException(404, "User not found")
    existing = await sql.get_user_sessions(req.user_id)
    name     = req.name or f"Session {len(existing) + 1}"
    session  = await sql.create_session(req.user_id, name=name)

    # Schedule maintenance in background — zero latency impact on session creation
    background_tasks.add_task(_run_maintenance_bg, req.user_id)

    return session


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


@app.get("/sessions/{session_id}/turns")
async def get_session_turns(session_id: str, limit: int = 50):
    s = await sql.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    turns = await sql.get_last_n_turns(session_id, n=limit)
    for t in turns:
        if t.get("router_decision") and isinstance(t["router_decision"], str):
            try:
                import json
                t["router_decision"] = json.loads(t["router_decision"])
            except Exception:
                pass
    return {"session_id": session_id, "turns": turns, "count": len(turns)}

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


class RenameSessionReq(BaseModel):
    name: str


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionReq):
    """Rename a session."""
    s = await sql.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    updated = await sql.rename_session(session_id, req.name)
    if not updated: raise HTTPException(500, "Rename failed")
    return {"session_id": session_id, "name": req.name.strip()}


@app.delete("/sessions/{session_id}", status_code=200)
async def delete_session(session_id: str):
    """
    Delete a session and clean up all associated data across every store:
      SQLite  — turn logs + session row
      Redis   — L0/L1/L2 summaries + embedding hashes for this session
      MongoDB — episodic memories scoped to this session

    Neo4j memories are NOT deleted — they are user-scoped and belong
    to the user, not the session. The forget mechanism handles their lifecycle.

    Returns summary of what was deleted across all stores.
    """
    s = await sql.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")

    user_id = s["user_id"]
    results = {}
    errors  = []

    # Check user still has other sessions — must keep at least one
    all_sessions = await sql.get_user_sessions(user_id)
    if len(all_sessions) <= 1:
        raise HTTPException(400, "Cannot delete the only session. Create a new session first.")

    # SQLite: turns + session row
    try:
        results["sqlite"] = await sql.delete_session(session_id)
    except Exception as e:
        errors.append(f"SQLite: {e}")
        results["sqlite"] = {"error": str(e)}

    # Redis: summaries + embeddings for this session
    try:
        deleted_keys = await redis_mgr.delete_user_summaries(
            session_ids=[session_id], user_id=None   # user_id=None → skip handoff deletion
        )
        results["redis"] = {"keys_deleted": deleted_keys}
    except Exception as e:
        errors.append(f"Redis: {e}")
        results["redis"] = {"error": str(e)}

    # MongoDB: episodes scoped to this session
    try:
        col = await mongo._get_collection()
        mongo_result = await col.delete_many({"user_id": user_id, "session_id": session_id})
        results["mongodb"] = {"episodes_deleted": mongo_result.deleted_count}
    except Exception as e:
        errors.append(f"MongoDB: {e}")
        results["mongodb"] = {"error": str(e)}

    return {
        "deleted_session_id": session_id,
        "user_id":            user_id,
        "results":            results,
        "errors":             errors,
        "status":             "complete" if not errors else "partial",
    }


# ─── Chat ─────────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(req: ChatReq, background_tasks: BackgroundTasks):
    if not await sql.get_user(req.user_id):  raise HTTPException(404, "User not found")
    s = await sql.get_session(req.session_id)
    if not s:                                raise HTTPException(404, "Session not found")
    if s["user_id"] != req.user_id:          raise HTTPException(403, "Session does not belong to this user")
    if not req.message.strip():              raise HTTPException(400, "Message cannot be empty")

    collected = {"full_response": "", "context": None}

    async def event_stream():
        try:
            async for chunk in stream_tokens(
                user_id=req.user_id,
                session_id=req.session_id,
                user_message=req.message,
                collected=collected
            ):
                yield f"data: {_json.dumps(chunk)}\n\n"
        except Exception as e:
            print(f"  ✗ Stream error: {e}")
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"

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