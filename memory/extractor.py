"""
memory/extractor.py
LLM calls for conversation summarization only.
Memory extraction decisions are handled by core/router.py.

Functions:
  summarize_turns()      → level-0 summary (Redis)
  compress_summaries()   → level-1 meta-summary (Redis)
  create_episodic_memory() → rich narrative from turns (MongoDB)
"""
import json
from openai import AsyncOpenAI
from config import get_settings

settings = get_settings()
_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url
        )
    return _client


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {}


# ═══════════════════════════════════════════════════════════════
# 1. BATCH SUMMARIZATION → Redis level=0
# ═══════════════════════════════════════════════════════════════

SUMMARIZATION_SYSTEM = """You produce dense, information-rich conversation summaries.

Write in third-person. Capture: main topics, key decisions, user context revealed,
unresolved questions. Omit: filler, pleasantries, redundancy.
Target length: 3-5 sentences.

Return ONLY JSON (no markdown):
{
  "summary": "Dense 3-5 sentence summary...",
  "key_topics": ["topic1", "topic2"]
}"""


async def summarize_turns(turns: list[dict]) -> dict:
    """Summarise a batch of raw turns → level-0 summary for Redis."""
    client = get_client()
    text = "\n\n".join([
        f"[Turn {t['turn_number']}]\nUser: {t['user_msg']}\nAssistant: {t['assistant_msg']}"
        for t in turns
    ])
    response = await client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SUMMARIZATION_SYSTEM},
            {"role": "user", "content": f"Summarise:\n\n{text}"}
        ]
    )
    result = _parse_json(response.choices[0].message.content)
    return {
        "summary": result.get("summary", text[:300]),
        "key_topics": result.get("key_topics", [])
    }


# ═══════════════════════════════════════════════════════════════
# 2. L1 COMPRESSION — 3 L0s → 1 L1 (covers 9 turns)
# ═══════════════════════════════════════════════════════════════

L1_COMPRESSION_SYSTEM = """You compress 3 sequential conversation summaries into one concise window summary.

Rules:
- Merge overlapping themes, preserve key decisions and user facts revealed
- Be MORE compressed than any source — 3-4 sentences total
- Third person. Capture: main topics, user context revealed, decisions made, open threads.

Return ONLY JSON (no markdown):
{
  "summary": "Compressed 3-4 sentence window summary...",
  "key_topics": ["topic1", "topic2"]
}"""


async def compress_summaries(summaries: list[dict]) -> dict:
    """
    Compress a list of L0 summaries into one L1 summary.
    Accepts a list (replaces old 2-arg signature).
    """
    client = get_client()
    parts = []
    for s in summaries:
        parts.append(
            f"[T{s['batch_start']}-T{s['batch_end']}]: {s['summary_text']}"
        )
    prompt = "Compress these sequential summaries into one:\n\n" + "\n\n".join(parts)

    response = await client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=250,
        messages=[
            {"role": "system", "content": L1_COMPRESSION_SYSTEM},
            {"role": "user",   "content": prompt}
        ]
    )
    result = _parse_json(response.choices[0].message.content)
    return {
        "summary":    result.get("summary", ""),
        "key_topics": result.get("key_topics", []),
    }


# ═══════════════════════════════════════════════════════════════
# 3. L2 ARC COMPRESSION — 3 L1s → 1 L2 (covers 27 turns)
# ═══════════════════════════════════════════════════════════════

ARC_COMPRESSION_SYSTEM = """You compress 3 conversation window summaries into one high-level session arc.

This arc covers a large portion of a session. Your job is to capture:
- The main themes and how they evolved
- The most important facts revealed about the user
- Key decisions, outcomes, and open threads
- The emotional/motivational arc of the conversation

Be highly compressed — 4-5 sentences max. Write in third person.

Return ONLY JSON (no markdown):
{
  "summary": "High-level arc summary, 4-5 sentences...",
  "key_topics": ["main themes"],
  "key_facts": ["important user facts revealed in this arc"]
}"""


async def compress_to_arc(summaries: list[dict]) -> dict:
    """
    Compress L1 summaries into one L2 arc summary.
    Higher level of abstraction — captures the arc, not the details.
    """
    client = get_client()
    parts = []
    for s in summaries:
        topics = ", ".join(s.get("key_topics", []))
        parts.append(
            f"[T{s['batch_start']}-T{s['batch_end']} | topics: {topics}]:\n{s['summary_text']}"
        )
    prompt = "Compress into one high-level session arc:\n\n" + "\n\n".join(parts)

    response = await client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": ARC_COMPRESSION_SYSTEM},
            {"role": "user",   "content": prompt}
        ]
    )
    result = _parse_json(response.choices[0].message.content)
    return {
        "summary":    result.get("summary", ""),
        "key_topics": result.get("key_topics", []),
        "key_facts":  result.get("key_facts", []),
    }


# ═══════════════════════════════════════════════════════════════
# 4. CROSS-SESSION HANDOFF — "what happened last time"
# ═══════════════════════════════════════════════════════════════

HANDOFF_SYSTEM = """You write a cross-session handoff summary for an AI assistant.
This will be shown at the START of the user's NEXT conversation session.

Write in second person ("Last time we talked...").
Cover: what was discussed, what was decided or resolved, any open threads,
and anything the AI should follow up on proactively.
Max 3 sentences. Be warm and specific.

Return ONLY JSON (no markdown):
{
  "summary": "Last time we talked, you were... We also discussed... You mentioned wanting to...",
  "key_topics": ["topic1", "topic2"]
}"""


async def create_handoff_summary(best_summary: dict) -> dict:
    """
    Turn the best available session summary into a cross-session handoff.
    Written in second person — shown at top of next session.
    """
    client = get_client()
    prompt = (
        f"Session summary (turns {best_summary['batch_start']}-{best_summary['batch_end']}):\n"
        f"{best_summary['summary_text']}\n\n"
        f"Key topics: {', '.join(best_summary.get('key_topics', []))}\n\n"
        "Write a handoff for the next session."
    )
    response = await client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=200,
        messages=[
            {"role": "system", "content": HANDOFF_SYSTEM},
            {"role": "user",   "content": prompt}
        ]
    )
    result = _parse_json(response.choices[0].message.content)
    return {
        "summary":    result.get("summary", ""),
        "key_topics": result.get("key_topics", []),
    }


# ═══════════════════════════════════════════════════════════════
# 3. EPISODIC MEMORY NARRATIVE → MongoDB
# ═══════════════════════════════════════════════════════════════

EPISODIC_SYSTEM = """You write rich episodic memory records for an AI assistant.
These help future AI instances understand what happened, why, and what was resolved.

Write in third-person. Cover: situation, challenge, approach taken, resolution/outcome,
emotional arc, any open threads.

Return ONLY JSON (no markdown):
{
  "title": "Short evocative title, max 8 words",
  "content": "Rich 4-6 sentence narrative. Be specific — name the tools, concepts, decisions involved.",
  "outcome": "resolved|ongoing|abandoned|unclear",
  "importance_score": <float 1.0-10.0 — how significant is this for understanding the user?
    1-3: trivial small talk | 4-6: useful context | 7-8: significant event | 9-10: pivotal/life-changing>,
  "emotional_intensity": <int 1-5 — 1=neutral, 3=moderate, 5=very strong emotion>,
  "topic_cluster": "<one of: career|health|learning|finance|project|relationship|travel|personal|general>"
}"""


async def create_episodic_narrative(
    turns: list[dict],
    episodic_decision: dict
) -> dict:
    """
    Build a rich episodic narrative from a batch of turns.
    episodic_decision comes from the router (has tone, entities, tags).
    """
    client = get_client()
    text = "\n\n".join([
        f"[Turn {t['turn_number']}]\nUser: {t['user_msg']}\nAssistant: {t['assistant_msg']}"
        for t in turns
    ])
    hint = (
        f"\nContext hints — tone: {episodic_decision.get('emotional_tone','')}, "
        f"entities: {episodic_decision.get('key_entities', [])}"
    )
    response = await client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=350,
        messages=[
            {"role": "system", "content": EPISODIC_SYSTEM},
            {"role": "user", "content": f"Create episodic memory:\n{text}{hint}"}
        ]
    )
    result = _parse_json(response.choices[0].message.content)
    return {
        "title":              result.get("title", "Conversation episode"),
        "content":            result.get("content", text[:400]),
        "outcome":            result.get("outcome", "unclear"),
        "importance_score":   float(result.get("importance_score", 5.0)),
        "emotional_intensity": int(result.get("emotional_intensity", 2)),
        "topic_cluster":      result.get("topic_cluster", "general"),
    }