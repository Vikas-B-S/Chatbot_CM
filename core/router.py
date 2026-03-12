"""
core/router.py — Memory Router with heuristic pre-filter

COST OPTIMISATION — Two-stage routing:
─────────────────────────────────────
  Stage 1: Heuristic pre-filter (FREE, ~0ms)
    Pattern-match the user message against known signal patterns.
    If clearly no memory to store → return empty decision immediately.
    No LLM call made. Saves ~60-70% of router LLM calls.

    Skip router entirely for:
      - Short greetings ("hi", "hello", "thanks")
      - Pure questions with no personal info ("what is X?", "how do I Y?")
      - Math/code/lookup requests ("calculate", "write a function", "what's 2+2")
      - Continuations ("ok", "got it", "yes", "no", "continue")

    Always call router for:
      - First-person statements ("I am", "I like", "I want", "I have")
      - Personal updates ("I switched", "I moved", "I got", "I decided")
      - Emotional content ("frustrated", "excited", "worried")
      - Significant events ("I got the job", "we launched")

  Stage 2: LLM router (only when heuristic says "maybe has memory")
    Uses a FAST cheap model (haiku/mini) with a tight 400 token limit.
    Same structured JSON output as before.

MODEL STRATEGY:
  Main LLM    → settings.claude_model (user-configured, quality matters)
  Router LLM  → fast cheap model, quality less critical
  Summarizer  → fast cheap model
"""

import json
import re
from openai import AsyncOpenAI
from dataclasses import dataclass, field
from typing import Optional
from config import get_settings

settings = get_settings()
_client  = None

# Use a faster/cheaper model for routing — does not need to be the best model
# Override this in .env as ROUTER_MODEL if needed
_ROUTER_MODEL = getattr(settings, "router_model", None) or "openai/gpt-4o-mini"


def _get_client() -> AsyncOpenAI:
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
        raw   = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class UserMemoryItem:
    memory_type:   str
    content:       str
    canonical_key: str
    confidence:    float = 1.0
    entities:      list  = field(default_factory=list)


@dataclass
class EpisodicDecision:
    should_store:   bool
    title:          str  = ""
    reason:         str  = ""
    emotional_tone: str  = "neutral"
    emotional_intensity: int = 2
    key_entities:   list = field(default_factory=list)
    tags:           list = field(default_factory=list)


@dataclass
class UserDbUpdate:
    should_update: bool
    fields:        dict = field(default_factory=dict)


@dataclass
class RoutingDecision:
    trigger_user_memory: bool
    trigger_episodic:    bool
    trigger_user_db:     bool
    user_memories:       list              = field(default_factory=list)
    episodic:            Optional[EpisodicDecision] = None
    user_db:             Optional[UserDbUpdate]     = None
    router_reasoning:    str               = ""
    skipped:             bool              = False   # True if pre-filter skipped LLM

    def to_dict(self) -> dict:
        return {
            "trigger_user_memory":   self.trigger_user_memory,
            "trigger_episodic":      self.trigger_episodic,
            "trigger_user_db":       self.trigger_user_db,
            "user_memories_count":   len(self.user_memories),
            "episodic_should_store": self.episodic.should_store if self.episodic else False,
            "router_reasoning":      self.router_reasoning,
            "skipped":               self.skipped,
        }


# ─── Heuristic pre-filter ─────────────────────────────────────────────────────

# Patterns that STRONGLY signal personal info worth storing
_STORE_SIGNALS = [
    r"\bi (am|'m|was|were|have|had|got|get|work|worked|live|lived|moved|study|studied|use|used|prefer|like|love|hate|want|need|decided|switched|chose|built|created|launched|started|finished)\b",
    r"\bmy (name|age|job|work|career|company|team|project|goal|plan|problem|issue|hobby|language|framework|stack|city|country|diet|budget|constraint)\b",
    r"\b(i switched|i moved|i got|i decided|i chose|i built|i launched|i finished|i completed|i failed|i succeeded)\b",
    r"\b(frustrated|excited|worried|anxious|happy|sad|stressed|overwhelmed|proud|nervous)\b",
    r"\b(years old|year old|years experience|i'm from|i live in|i work at|i work for|i study at)\b",
    r"\b(vegetarian|vegan|diabetic|allergic|disabled|pregnant)\b",
]

# Patterns that STRONGLY signal NO personal info (safe to skip router)
_SKIP_SIGNALS = [
    r"^(hi|hello|hey|thanks|thank you|ok|okay|got it|sure|yes|no|nope|yep|cool|great|awesome|nice|perfect|alright|fine|sounds good|makes sense|understood|correct|right|wrong)[\s!.?]*$",
    r"^(what is|what are|what does|what do|how do|how does|how can|how to|can you|could you|please|explain|tell me|show me|give me|list|describe|define|compare|contrast)\b",
    r"^(calculate|compute|convert|translate|write|code|create|generate|make|build|fix|debug|find|search|look up|summarize|summarise)\b",
    r"^(what'?s|where'?s|who'?s|when'?s|why'?s|how'?s)\b",
    r"^[\d\s\+\-\*\/\(\)=]+$",   # pure math
]

_COMPILED_STORE = [re.compile(p, re.IGNORECASE) for p in _STORE_SIGNALS]
_COMPILED_SKIP  = [re.compile(p, re.IGNORECASE) for p in _SKIP_SIGNALS]


def _should_skip_router(user_message: str) -> tuple[bool, str]:
    """
    Returns (skip, reason).
    skip=True  → return empty decision, don't call LLM
    skip=False → call LLM router
    """
    msg = user_message.strip()

    # Very short messages with no personal signal → skip
    if len(msg) < 20 and not any(p.search(msg) for p in _COMPILED_STORE):
        return True, "short message, no personal signal"

    # Explicit skip pattern matches → skip
    for pattern in _COMPILED_SKIP:
        if pattern.search(msg):
            # Double-check: does it also have a store signal? If so, don't skip
            if any(p.search(msg) for p in _COMPILED_STORE):
                return False, "has both skip and store signals"
            return True, f"matches skip pattern"

    # Has explicit store signal → don't skip
    for pattern in _COMPILED_STORE:
        if pattern.search(msg):
            return False, "has store signal"

    # Ambiguous — call router to be safe
    return False, "ambiguous"


def _empty_decision(reason: str) -> RoutingDecision:
    """Return a no-op decision without calling the LLM."""
    return RoutingDecision(
        trigger_user_memory=False,
        trigger_episodic=False,
        trigger_user_db=False,
        user_memories=[],
        episodic=EpisodicDecision(should_store=False),
        user_db=UserDbUpdate(should_update=False),
        router_reasoning=f"[pre-filter skipped: {reason}]",
        skipped=True,
    )


# ─── Router system prompt (tighter than before) ───────────────────────────────

ROUTER_SYSTEM = """Memory Router for an AI assistant. Analyse one exchange, decide what to store.

## Memory types:
USER MEMORY (Neo4j) — 4 types:
  fact:       stable truths — name, age, location, job, language spoken
  preference: likes/dislikes — coding_language, ui_theme, diet
  goal:       wants to achieve — project_goal, career_goal, learning_goal
  constraint: hard limits — budget_constraint, tech_constraint

canonical_key conventions (use EXACTLY, never invent for known concepts):
  name, age, location, occupation, employer, coding_language, ui_theme,
  diet, primary_goal, career_goal, project_goal, tech_constraint, budget_constraint

EPISODIC (MongoDB) — store when ANY of these apply:
  - Personal problem-solving or debugging (user is working through something)
  - Decisions made (chose X over Y, decided to use Z)
  - Emotional events (frustrated, excited, worried, proud)
  - Learning sessions tied to user's goals or project (studying fine-tuning for RAG, exploring a library for their project)
  - Technical deep-dives the user is actively exploring for their own use case
  - Multi-turn topic exploration where user is building understanding for a specific purpose
  - Progress on ongoing work ("I got X working", "I'm stuck on Y")

  Skip ONLY for:
  - Pure isolated factual lookups with no connection to user's goals ("what year was Python created")
  - Pure small talk or greetings ("hi", "thanks", "ok")
  - One-off questions clearly unrelated to anything the user is building or doing

## Output — ONLY this JSON, no markdown:
{
  "trigger_user_memory": true|false,
  "trigger_episodic": true|false,
  "trigger_user_db": false,
  "router_reasoning": "one sentence",
  "user_memories": [
    {"memory_type": "fact|preference|goal|constraint", "content": "...",
     "canonical_key": "...", "confidence": 0.0-1.0, "entities": []}
  ],
  "episodic": {
    "should_store": true|false, "title": "...", "reason": "...",
    "emotional_tone": "neutral|curious|frustrated|excited|worried|satisfied",
    "emotional_intensity": 1-5,
    "key_entities": [], "tags": []
  },
  "user_db": {"should_update": false, "fields": {}}
}

Rules:
- Only extract what USER explicitly states, never from assistant responses
- If nothing to store: all trigger_* false, empty arrays
- Use same canonical_key for same concept every time"""


# ─── Main route function ──────────────────────────────────────────────────────

async def route(
    user_message: str,
    assistant_response: str,
    conversation_context: str = "",
    turn_number: int = 0,
) -> RoutingDecision:
    """
    Two-stage router:
      1. Heuristic pre-filter (free, ~0ms)
      2. LLM router (only if pre-filter says needed)
    """

    # ── Stage 1: pre-filter ───────────────────────────────────
    skip, reason = _should_skip_router(user_message)
    if skip:
        print(f"  ↳ Router skipped [{reason}]")
        return _empty_decision(reason)

    # ── Stage 2: LLM router ───────────────────────────────────
    client = _get_client()
    prompt = (
        f"Turn #{turn_number}\n"
        f"Context: {conversation_context[:300]}\n\n"
        f"User: {user_message}\n"
        f"Assistant: {assistant_response[:200]}\n\n"
        "Route this exchange."
    )

    try:
        response = await client.chat.completions.create(
            model=_ROUTER_MODEL,
            max_tokens=400,   # down from 600
            temperature=0,    # deterministic — we want consistent routing
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user",   "content": prompt}
            ]
        )
        raw = _parse_json(response.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠ Router LLM failed: {e}")
        return _empty_decision("router LLM error")

    # ── Parse memories ────────────────────────────────────────
    user_memories = []
    for item in raw.get("user_memories", []):
        if not item.get("content") or not item.get("canonical_key") or not item.get("memory_type"):
            continue
        if item["memory_type"] not in {"fact", "preference", "goal", "constraint"}:
            continue
        user_memories.append(UserMemoryItem(
            memory_type=   item["memory_type"],
            content=       item["content"],
            canonical_key= item["canonical_key"],
            confidence=    float(item.get("confidence", 1.0)),
            entities=      item.get("entities", [])
        ))

    ep_raw   = raw.get("episodic", {})
    episodic = EpisodicDecision(
        should_store=       bool(ep_raw.get("should_store", False)),
        title=              ep_raw.get("title", ""),
        reason=             ep_raw.get("reason", ""),
        emotional_tone=     ep_raw.get("emotional_tone", "neutral"),
        emotional_intensity=int(ep_raw.get("emotional_intensity", 2)),
        key_entities=       ep_raw.get("key_entities", []),
        tags=               ep_raw.get("tags", [])
    )

    db_raw  = raw.get("user_db", {})
    user_db = UserDbUpdate(
        should_update=bool(db_raw.get("should_update", False)),
        fields=       db_raw.get("fields", {})
    )

    return RoutingDecision(
        trigger_user_memory=bool(raw.get("trigger_user_memory", False)) and len(user_memories) > 0,
        trigger_episodic=   bool(raw.get("trigger_episodic", False)) and episodic.should_store,
        trigger_user_db=    bool(raw.get("trigger_user_db", False)) and user_db.should_update,
        user_memories=      user_memories,
        episodic=           episodic,
        user_db=            user_db,
        router_reasoning=   raw.get("router_reasoning", ""),
        skipped=            False,
    )