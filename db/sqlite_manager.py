"""
db/sqlite_manager.py — Operational store: users, sessions, raw turn logs

Tables
──────
  users       — credentials + profile
  sessions    — one per conversation, tracks turn count + last active
  turn_logs   — every raw message, permanent archive

Short-term memory strategy
──────────────────────────
  get_last_n_turns()         — last N turns verbatim (sliding window for LLM context)
  get_turns_from_last_summary() — turns since the last summarized batch
                                  fixes the blind-spot: turns between last
                                  summarized batch and the raw window are
                                  never lost
  get_unsummarized_turns()   — all turns not yet in any Redis summary
                                  used by summarizer to know exactly what needs compressing

Blind-spot fix
──────────────
  Old behaviour:
    Raw window = last 6 turns (T45-T50)
    Last summarized = T1-T42 (in Redis)
    T43, T44 = BLIND SPOT — not in raw window, not yet summarized

  New behaviour:
    get_turns_from_last_summary() returns T43-T50 (everything since last batch)
    context_builder merges this with the raw window
    blind spot eliminated regardless of batch size or timing
"""

import aiosqlite
import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from config import get_settings

settings = get_settings()


def get_db():
    return aiosqlite.connect(settings.sqlite_db_path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Password helpers ─────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """PBKDF2-SHA256 with random salt. Format: salt$hash"""
    salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return f"{salt}${h.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time comparison — prevents timing attacks."""
    try:
        salt, h = stored_hash.split("$", 1)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
        return hmac.compare_digest(candidate.hex(), h)
    except Exception:
        return False


# ─── Schema ───────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables + run migrations safely."""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row

        await db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       TEXT PRIMARY KEY,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT,
            password_hash TEXT,
            created_at    TEXT NOT NULL,
            metadata      TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id    TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL REFERENCES users(user_id),
            name          TEXT DEFAULT NULL,
            summary       TEXT DEFAULT NULL,
            created_at    TEXT NOT NULL,
            last_active   TEXT NOT NULL,
            total_turns   INTEGER DEFAULT 0,
            last_summarized_turn INTEGER DEFAULT 0
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
        CREATE INDEX IF NOT EXISTS idx_turns_unsummarized
            ON turn_logs(session_id, is_summarized, turn_number);
        CREATE INDEX IF NOT EXISTS idx_users_email
            ON users(email);
        """)

        # Safe migrations for existing databases
        migrations = [
            "ALTER TABLE users    ADD COLUMN email                TEXT",
            "ALTER TABLE users    ADD COLUMN password_hash        TEXT",
            "ALTER TABLE sessions ADD COLUMN name                 TEXT DEFAULT NULL",
            "ALTER TABLE sessions ADD COLUMN summary              TEXT DEFAULT NULL",
            "ALTER TABLE sessions ADD COLUMN last_summarized_turn INTEGER DEFAULT 0",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email)",
        ]
        for sql in migrations:
            try:
                await db.execute(sql)
            except Exception:
                pass

        await db.commit()


# ─── Users ────────────────────────────────────────────────────────────────────

async def create_user(username: str, email: str, password: str, metadata: dict = None) -> dict:
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    now     = _now()
    pw_hash = _hash_password(password)
    async with get_db() as db:
        await db.execute(
            "INSERT INTO users (user_id, username, email, password_hash, created_at, metadata) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, username, email.lower().strip(), pw_hash, now, json.dumps(metadata or {}))
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


async def delete_user_data(user_id: str) -> dict:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT COUNT(*) as c FROM turn_logs WHERE user_id=?", (user_id,))
        turns = (await cur.fetchone())["c"]
        cur = await db.execute("SELECT COUNT(*) as c FROM sessions WHERE user_id=?", (user_id,))
        sessions = (await cur.fetchone())["c"]
        await db.execute("DELETE FROM turn_logs WHERE user_id=?", (user_id,))
        await db.execute("DELETE FROM sessions  WHERE user_id=?", (user_id,))
        await db.execute("DELETE FROM users      WHERE user_id=?", (user_id,))
        await db.commit()
    return {"turns": turns, "sessions": sessions, "user": 1}


async def delete_user_credentials(user_id: str) -> bool:
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET email=NULL, password_hash=NULL WHERE user_id=?", (user_id,)
        )
        await db.commit()
    return True


# ─── Sessions ─────────────────────────────────────────────────────────────────

async def create_session(user_id: str, name: str = None) -> dict:
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    now        = _now()
    async with get_db() as db:
        await db.execute(
            "INSERT INTO sessions (session_id, user_id, name, created_at, last_active) "
            "VALUES (?,?,?,?,?)",
            (session_id, user_id, name, now, now)
        )
        await db.commit()
    return {"session_id": session_id, "user_id": user_id, "name": name,
            "created_at": now, "total_turns": 0}


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
            "SELECT * FROM sessions WHERE user_id=? ORDER BY last_active DESC", (user_id,)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def increment_session_turn(session_id: str) -> int:
    """Bump turn count + last_active. Returns new turn number."""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            "UPDATE sessions SET total_turns=total_turns+1, last_active=? WHERE session_id=?",
            (_now(), session_id)
        )
        await db.commit()
        cur = await db.execute(
            "SELECT total_turns FROM sessions WHERE session_id=?", (session_id,)
        )
        row = await cur.fetchone()
    return row["total_turns"]


async def update_session_summary(session_id: str, summary: str):
    """Store a short description of what this session was about (set by summarizer)."""
    async with get_db() as db:
        await db.execute(
            "UPDATE sessions SET summary=? WHERE session_id=?", (summary, session_id)
        )
        await db.commit()


async def get_last_summarized_turn(session_id: str) -> int:
    """Returns the highest turn number that has been summarized into Redis."""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT MAX(turn_number) as t FROM turn_logs "
            "WHERE session_id=? AND is_summarized=1",
            (session_id,)
        )
        row = await cur.fetchone()
    return row["t"] or 0


# ─── Turn logs ────────────────────────────────────────────────────────────────

async def save_turn(
    session_id: str,
    user_id: str,
    turn_number: int,
    user_msg: str,
    assistant_msg: str,
    router_decision: dict = None
) -> dict:
    turn_id = f"turn_{uuid.uuid4().hex[:12]}"
    now     = _now()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO turn_logs
               (turn_id, session_id, user_id, turn_number, user_msg,
                assistant_msg, router_decision, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (turn_id, session_id, user_id, turn_number,
             user_msg, assistant_msg,
             json.dumps(router_decision) if router_decision else None, now)
        )
        await db.commit()
    return {"turn_id": turn_id, "turn_number": turn_number}


async def get_last_n_turns(session_id: str, n: int = 6) -> list:
    """
    Last N turns verbatim — the immediate sliding window.
    Used as the raw short-term memory in every context window.
    """
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM turn_logs WHERE session_id=? "
            "ORDER BY turn_number DESC LIMIT ?",
            (session_id, n)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in reversed(rows)]


async def get_turns_from_last_summary(session_id: str) -> list:
    """
    Returns all turns AFTER the last summarized batch.

    This is the blind-spot fix. Instead of only showing last N turns,
    context_builder calls this to get everything since the last Redis summary.
    Ensures no turns fall into the gap between 'already summarized' and
    'inside raw window'.

    Example:
      Last summarized: turn 42
      Total turns: 50
      Returns: turns 43, 44, 45, 46, 47, 48, 49, 50
    """
    last_summarized = await get_last_summarized_turn(session_id)
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM turn_logs WHERE session_id=? AND turn_number > ? "
            "ORDER BY turn_number ASC",
            (session_id, last_summarized)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_unsummarized_turns(session_id: str) -> list:
    """All turns not yet summarized — used by summarizer to find pending batches."""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM turn_logs WHERE session_id=? AND is_summarized=0 "
            "ORDER BY turn_number ASC",
            (session_id,)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_turns_in_range(session_id: str, start: int, end: int) -> list:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM turn_logs WHERE session_id=? "
            "AND turn_number>=? AND turn_number<=? ORDER BY turn_number ASC",
            (session_id, start, end)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def mark_turns_summarized(session_id: str, start: int, end: int):
    """Mark a batch of turns as summarized + update session's last_summarized_turn."""
    async with get_db() as db:
        await db.execute(
            "UPDATE turn_logs SET is_summarized=1 "
            "WHERE session_id=? AND turn_number>=? AND turn_number<=?",
            (session_id, start, end)
        )
        await db.execute(
            "UPDATE sessions SET last_summarized_turn=MAX(last_summarized_turn, ?) "
            "WHERE session_id=?",
            (end, session_id)
        )
        await db.commit()