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
# 2. META-COMPRESSION → Redis level=1
# ═══════════════════════════════════════════════════════════════

META_COMPRESSION_SYSTEM = """You merge two sequential conversation summaries into a single meta-summary.

Rules:
- Merge overlapping themes, preserve critical decisions and user facts
- Describe how the conversation evolved between the two windows
- Be MORE compressed than either source (3-4 sentences total)
- Third person

Return ONLY JSON (no markdown):
{
  "meta_summary": "Compressed 3-4 sentence meta-summary...",
  "key_topics": ["merged", "topics"],
  "evolution_note": "One sentence on how conversation evolved across the two windows"
}"""


async def compress_summaries(summary_a: dict, summary_b: dict) -> dict:
    """
    Compress two level-0 summaries into one level-1 meta-summary.
    summary_a = earlier window, summary_b = later window.
    """
    client = get_client()
    prompt = (
        f"Summary A (turns {summary_a['batch_start']}-{summary_a['batch_end']}):\n"
        f"{summary_a['summary_text']}\n\n"
        f"Summary B (turns {summary_b['batch_start']}-{summary_b['batch_end']}):\n"
        f"{summary_b['summary_text']}\n\n"
        "Compress into one meta-summary."
    )
    response = await client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=250,
        messages=[
            {"role": "system", "content": META_COMPRESSION_SYSTEM},
            {"role": "user", "content": prompt}
        ]
    )
    result = _parse_json(response.choices[0].message.content)
    return {
        "meta_summary": result.get("meta_summary", ""),
        "key_topics": result.get("key_topics", []),
        "evolution_note": result.get("evolution_note", "")
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
  "outcome": "resolved|ongoing|abandoned|unclear"
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
        "title": result.get("title", "Conversation episode"),
        "content": result.get("content", text[:400]),
        "outcome": result.get("outcome", "unclear")
    }