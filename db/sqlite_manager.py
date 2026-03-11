"""
db/sqlite_manager.py
SQLite — operational data only: users, sessions, raw turn logs.

All user memory (facts, preferences, goals, constraints) lives in Neo4j/Graphiti.
Episodic memories live in MongoDB.
Summaries live in Redis.
"""
import aiosqlite
import json
import uuid
from datetime import datetime
from typing import Optional
from config import get_settings

settings = get_settings()


def get_db():
    return aiosqlite.connect(settings.sqlite_db_path)


async def init_db():
    """Create all operational tables + run migrations for existing DBs."""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row

        # ── Create tables (new installs) ──────────────────────
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            username    TEXT UNIQUE NOT NULL,
            created_at  TEXT NOT NULL,
            metadata    TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL REFERENCES users(user_id),
            created_at   TEXT NOT NULL,
            last_active  TEXT NOT NULL,
            total_turns  INTEGER DEFAULT 0,
            is_active    INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS turn_logs (
            turn_id         TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(session_id),
            user_id         TEXT NOT NULL,
            turn_number     INTEGER NOT NULL,
            user_msg        TEXT NOT NULL,
            assistant_msg   TEXT NOT NULL,
            router_decision TEXT DEFAULT NULL,
            created_at      TEXT NOT NULL,
            is_summarized   INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session
            ON turn_logs(session_id, turn_number);
        CREATE INDEX IF NOT EXISTS idx_turns_user
            ON turn_logs(user_id);
        """)

        # ── Migrations — add new columns to existing DBs ──────
        # Each ALTER is wrapped in try/except so it silently skips
        # if the column already exists (SQLite has no IF NOT EXISTS for ALTER)
        migrations = [
            "ALTER TABLE users    ADD COLUMN email TEXT",
            "ALTER TABLE sessions ADD COLUMN name  TEXT DEFAULT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)",
        ]
        for sql in migrations:
            try:
                await db.execute(sql)
            except Exception:
                pass  # column/index already exists

        await db.commit()


# ─── User CRUD ────────────────────────────────────────────────

async def create_user(username: str, email: str, metadata: dict = None) -> dict:
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            "INSERT INTO users (user_id, username, email, created_at, metadata) VALUES (?,?,?,?,?)",
            (user_id, username, email.lower().strip(), now, json.dumps(metadata or {}))
        )
        await db.commit()
    return {"user_id": user_id, "username": username, "email": email, "created_at": now}


async def get_user(user_id: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_user_by_email(email: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),))
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_user_by_username(username: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users WHERE username=?", (username,))
        row = await cur.fetchone()
    return dict(row) if row else None


async def list_users() -> list:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users ORDER BY created_at DESC")
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ─── Session CRUD ─────────────────────────────────────────────

async def create_session(user_id: str, name: str = None) -> dict:
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            "INSERT INTO sessions (session_id, user_id, name, created_at, last_active) VALUES (?,?,?,?,?)",
            (session_id, user_id, name, now, now)
        )
        await db.commit()
    return {"session_id": session_id, "user_id": user_id, "name": name, "created_at": now, "total_turns": 0}


async def get_session(session_id: str) -> Optional[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_user_sessions(user_id: str) -> list:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY last_active DESC",
            (user_id,)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def increment_session_turn(session_id: str) -> int:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            "UPDATE sessions SET total_turns=total_turns+1, last_active=? WHERE session_id=?",
            (datetime.utcnow().isoformat(), session_id)
        )
        await db.commit()
        cur = await db.execute(
            "SELECT total_turns FROM sessions WHERE session_id=?", (session_id,)
        )
        row = await cur.fetchone()
    return row["total_turns"]


# ─── Turn log CRUD ────────────────────────────────────────────

async def save_turn(
    session_id: str,
    user_id: str,
    turn_number: int,
    user_msg: str,
    assistant_msg: str,
    router_decision: dict = None
) -> dict:
    turn_id = f"turn_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            """INSERT INTO turn_logs
               (turn_id, session_id, user_id, turn_number, user_msg, assistant_msg,
                router_decision, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (turn_id, session_id, user_id, turn_number, user_msg, assistant_msg,
             json.dumps(router_decision) if router_decision else None, now)
        )
        await db.commit()
    return {"turn_id": turn_id, "turn_number": turn_number}


async def get_last_n_turns(session_id: str, n: int = 6) -> list:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM turn_logs WHERE session_id=? ORDER BY turn_number DESC LIMIT ?",
            (session_id, n)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in reversed(rows)]


async def get_turns_in_range(session_id: str, start: int, end: int) -> list:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT * FROM turn_logs
               WHERE session_id=? AND turn_number>=? AND turn_number<=?
               ORDER BY turn_number ASC""",
            (session_id, start, end)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def mark_turns_summarized(session_id: str, start: int, end: int):
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            """UPDATE turn_logs SET is_summarized=1
               WHERE session_id=? AND turn_number>=? AND turn_number<=?""",
            (session_id, start, end)
        )
        await db.commit()