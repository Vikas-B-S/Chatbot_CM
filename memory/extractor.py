"""
memory/extractor.py
LLM calls for conversation summarization only.
Memory extraction decisions are handled by core/router.py.

Model strategy (v3.3)
─────────────────────
  All 5 LLM calls in this file now use settings.extractor_model
  instead of settings.claude_model.

  WHY: Every function here is a structured JSON extraction task —
  the prompt has a clear schema, the output is short, and the content
  is formulaic (summaries, compressions, narratives). These tasks do
  NOT need a powerful model. They need a fast, cheap one.

  settings.extractor_model defaults to "openai/gpt-4o-mini":
    summarize_turns()          → L0 batch summary       (300 tokens max)
    compress_summaries()       → L1 window compression  (250 tokens max)
    compress_to_arc()          → L2 arc compression     (300 tokens max)
    create_handoff_summary()   → cross-session handoff  (200 tokens max)
    create_episodic_narrative()→ episodic narrative     (350 tokens max)

  The main model (settings.claude_model) is reserved exclusively for
  user-facing responses in core/agent.py.

Functions:
  summarize_turns()      → level-0 summary (Redis)
  compress_summaries()   → level-1 meta-summary (Redis)
  compress_to_arc()      → level-2 arc summary (Redis)
  create_handoff_summary() → cross-session handoff (Redis)
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
    """
    Summarise a batch of raw turns → level-0 summary for Redis.
    Uses extractor_model (cheap/fast) — structured JSON task, no need for main model.
    """
    client = get_client()
    text = "\n\n".join([
        f"[Turn {t['turn_number']}]\nUser: {t['user_msg']}\nAssistant: {t['assistant_msg']}"
        for t in turns
    ])
    response = await client.chat.completions.create(
        model=settings.extractor_model,   # ← was settings.claude_model
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
    Uses extractor_model (cheap/fast) — merging text, no reasoning needed.
    """
    client = get_client()
    parts = []
    for s in summaries:
        parts.append(
            f"[T{s['batch_start']}-T{s['batch_end']}]: {s['summary_text']}"
        )
    prompt = "Compress these sequential summaries into one:\n\n" + "\n\n".join(parts)

    response = await client.chat.completions.create(
        model=settings.extractor_model,   # ← was settings.claude_model
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
    Uses extractor_model (cheap/fast) — higher abstraction but still
    structured compression, not open-ended reasoning.
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
        model=settings.extractor_model,   # ← was settings.claude_model
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
    Uses extractor_model (cheap/fast) — templated 3-sentence output,
    no complex reasoning required.
    """
    client = get_client()
    prompt = (
        f"Session summary (turns {best_summary['batch_start']}-{best_summary['batch_end']}):\n"
        f"{best_summary['summary_text']}\n\n"
        f"Key topics: {', '.join(best_summary.get('key_topics', []))}\n\n"
        "Write a handoff for the next session."
    )
    response = await client.chat.completions.create(
        model=settings.extractor_model,   # ← was settings.claude_model
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
# 5. EPISODIC MEMORY NARRATIVE → MongoDB
# ═══════════════════════════════════════════════════════════════

EPISODIC_SYSTEM = """You write rich episodic memory records for an AI assistant.
These help future AI instances understand what happened, why, and what was resolved.

Write in third-person. Cover: situation, challenge, approach taken, resolution/outcome,
emotional arc, any open threads.

You will be given pre-computed metadata (importance, tone, entities, cluster) from the
router — treat these as ground truth. Do NOT re-derive them. Your job is ONLY to write
the narrative title, content, and outcome.

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
    Uses extractor_model (cheap/fast) — pure writing task, NOT a reasoning task.

    Router value passthrough (v3.3):
      The router already computed importance_score, emotional_tone,
      emotional_intensity, key_entities, and topic_cluster.
      These are passed directly into the return value — the LLM is NOT
      asked to re-derive them. This avoids paying twice for the same work.

      LLM now only produces: title, content, outcome (3 fields instead of 6).
      Router provides:        importance_score, emotional_tone,
                              emotional_intensity, topic_cluster, key_entities.

      max_tokens reduced 350 → 200 (LLM output is now much smaller).
    """
    client = get_client()
    text = "\n\n".join([
        f"[Turn {t['turn_number']}]\nUser: {t['user_msg']}\nAssistant: {t['assistant_msg']}"
        for t in turns
    ])

    # Pre-computed router values — passed as ground truth to the LLM
    tone      = episodic_decision.get("emotional_tone", "neutral")
    entities  = episodic_decision.get("key_entities", [])
    cluster   = episodic_decision.get("topic_cluster", "general")
    intensity = episodic_decision.get("emotional_intensity", 2)
    importance = episodic_decision.get("importance_score", 5.0)

    context = (
        f"\nPre-computed metadata (do not re-derive):"
        f"\n  tone: {tone}"
        f"\n  entities: {entities}"
        f"\n  cluster: {cluster}"
    )

    response = await client.chat.completions.create(
        model=settings.extractor_model,
        max_tokens=200,   # reduced — LLM only writes title, content, outcome now
        messages=[
            {"role": "system", "content": EPISODIC_SYSTEM},
            {"role": "user",   "content": f"Create episodic memory:\n{text}{context}"}
        ]
    )
    result = _parse_json(response.choices[0].message.content)
    return {
        # LLM-generated fields
        "title":               result.get("title", "Conversation episode"),
        "content":             result.get("content", text[:400]),
        "outcome":             result.get("outcome", "unclear"),
        # Router-provided fields — passed through directly, not re-derived
        "importance_score":    float(importance),
        "emotional_intensity": int(intensity),
        "emotional_tone":      tone,
        "topic_cluster":       cluster,
        "key_entities":        entities,
    }