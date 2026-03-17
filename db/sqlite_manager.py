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

Raw turn cap (v3.3)
───────────────────
  get_turns_from_last_summary() now enforces a hard cap of
  _RAW_TURN_CAP = 20 most recent turns.

  WHY: Without a cap, a long session where summarization is behind
  (or never triggered) returns ALL unsummarized turns — potentially
  50-100 turns dumped raw into the prompt.

    50 turns × ~200 tokens each = ~10,000 tokens just from raw history
    → LLM context window fills up
    → Older summaries and memory get pushed out
    → Cost spikes with no quality benefit

  With the cap: only the 20 most recent turns are shown verbatim.
  Summaries already cover older turns — they don't need to appear raw.

  The cap is applied AFTER ordering by turn_number DESC so we always
  keep the MOST RECENT turns, not the oldest.
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

# Hard cap on raw turns returned by get_turns_from_last_summary()
# Prevents prompt bloat when summarization is behind on long sessions.
_RAW_TURN_CAP = 20


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


async def rename_session(session_id: str, name: str) -> bool:
    """Rename a session. Returns True if a row was updated."""
    name = name.strip()[:80] if name else "Untitled Session"
    async with get_db() as db:
        cur = await db.execute(
            "UPDATE sessions SET name=? WHERE session_id=?", (name, session_id)
        )
        await db.commit()
    return cur.rowcount > 0


async def delete_session(session_id: str) -> dict:
    """
    Delete a session and all its turn logs from SQLite.
    Redis summaries and MongoDB episodes are cleaned by server.py.
    """
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT COUNT(*) as c FROM turn_logs WHERE session_id=?", (session_id,)
        )
        row  = await cur.fetchone()
        turns = row["c"] if row else 0
        await db.execute("DELETE FROM turn_logs WHERE session_id=?", (session_id,))
        await db.execute("DELETE FROM sessions  WHERE session_id=?", (session_id,))
        await db.commit()
    return {"turns_deleted": turns, "session_deleted": 1}


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
    Returns turns AFTER the last summarized batch, capped at _RAW_TURN_CAP.

    FIX (v3.3): Added hard cap of _RAW_TURN_CAP (20) most recent turns.

    WHY THE CAP:
      Without it, a session where summarization is delayed returns ALL
      unsummarized turns — potentially 50-100 turns dumped raw into the
      prompt. At ~200 tokens per turn that's 10,000+ tokens of raw history
      which crowds out summaries, memories, and episodic context.

    HOW IT WORKS:
      1. Find the last summarized turn number
      2. Fetch ALL turns after it ordered DESC (newest first)
      3. Take only the first _RAW_TURN_CAP rows (most recent)
      4. Reverse to chronological order for the prompt

    This means if there are 40 unsummarized turns (T10-T50):
      Old: returns all 40 turns
      New: returns turns T31-T50 (the 20 most recent)
      Turns T10-T30 are not lost — they'll be summarized next cycle.

    Example (blind-spot fix still works with cap):
      Last summarized: turn 42
      Total turns: 50
      Cap: 20
      Returns: turns 43-50 (8 turns — well within cap, all returned)

    Example (cap kicks in for long unsummarized session):
      Last summarized: turn 5
      Total turns: 50
      Cap: 20
      Returns: turns 31-50 (20 most recent turns, not all 45)
    """
    last_summarized = await get_last_summarized_turn(session_id)
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            # ORDER BY DESC + LIMIT = get the most recent _RAW_TURN_CAP turns
            # Then we reverse below to restore chronological order
            "SELECT * FROM turn_logs "
            "WHERE session_id=? AND turn_number > ? "
            "ORDER BY turn_number DESC LIMIT ?",
            (session_id, last_summarized, _RAW_TURN_CAP)
        )
        rows = await cur.fetchall()
    # Reverse: DESC fetch gives newest-first, prompt needs oldest-first
    return [dict(r) for r in reversed(rows)]


async def get_unsummarized_turns(session_id: str) -> list:
    """All turns not yet summarized — used by summarizer to find pending batches.
    Note: intentionally has NO cap — summarizer needs the full list to batch correctly."""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM turn_logs WHERE session_id=? AND is_summarized=0 "
            "ORDER BY turn_number ASC",
            (session_id,)
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_last_turn(session_id: str) -> Optional[dict]:
    """Return the most recently stored turn for a session, including router_decision."""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM turn_logs WHERE session_id=? ORDER BY turn_number DESC LIMIT 1",
            (session_id,)
        )
        row = await cur.fetchone()
    if not row:
        return None
    result = dict(row)
    if result.get('router_decision'):
        try:
            result['router_decision'] = json.loads(result['router_decision'])
        except Exception:
            pass
    return result


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