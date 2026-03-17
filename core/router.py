"""
core/router.py — Memory Router with heuristic pre-filter

COST OPTIMISATION — Two-stage routing:
─────────────────────────────────────
  Stage 1: Heuristic pre-filter (FREE, ~0ms)
    Pattern-match the user message against known signal patterns.
    If clearly no memory to store → return empty decision immediately.

  Stage 2: LLM router (only when heuristic says "maybe has memory")
    Uses a FAST cheap model with a tight 400 token limit.

EPISODIC WORTHINESS GATE
─────────────────────────
  After the LLM router decides should_store=True, a hard gate runs:
  the episode must meet AT LEAST ONE of these to actually be stored:

    A) importance_score >= 6.0   (the LLM-assigned score)
    B) emotional_intensity >= 3  (the LLM-assigned intensity)
    C) title contains a milestone keyword (decided, solved, stuck, launched, etc.)

  This prevents episodic from firing for every curious question or minor
  learning moment. Only real turning points, blockers, and decisions get stored.

  Expected frequency: 1-3 episodes per multi-hour session, not every turn.
"""

import json
import re
from openai import AsyncOpenAI
from dataclasses import dataclass, field
from typing import Optional
from config import get_settings

settings = get_settings()
_client  = None

_ROUTER_MODEL = getattr(settings, "router_model", None) or "openai/gpt-4o-mini"

# ── Episodic worthiness thresholds ────────────────────────────────────────────
_EPISODIC_MIN_IMPORTANCE = 6.0   # importance_score must meet or exceed this
_EPISODIC_MIN_INTENSITY  = 3     # OR emotional_intensity must meet or exceed this

# Milestone keywords — presence in title/reason guarantees storage
# regardless of score thresholds (these are definitionally memorable)
_MILESTONE_KEYWORDS = {
    "decided", "decision", "chose", "switched", "moved",
    "launched", "shipped", "deployed", "built", "completed", "finished",
    "solved", "fixed", "figured out",
    "stuck", "blocked", "failed", "gave up", "abandoned",
    "frustrated", "excited", "worried", "anxious", "proud", "devastated",
    "got the job", "lost the job", "quit", "hired", "fired",
    "diagnosed", "surgery", "illness",
    "started", "began", "ended", "broke up", "got married",
}


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
    should_store:        bool
    title:               str   = ""
    reason:              str   = ""
    emotional_tone:      str   = "neutral"
    emotional_intensity: int   = 1
    importance_score:    float = 5.0   # LLM-assigned importance (1-10)
    key_entities:        list  = field(default_factory=list)
    tags:                list  = field(default_factory=list)


@dataclass
class RoutingDecision:
    trigger_user_memory: bool
    trigger_episodic:    bool
    user_memories:       list                       = field(default_factory=list)
    episodic:            Optional[EpisodicDecision] = None
    router_reasoning:    str                        = ""
    skipped:             bool                       = False

    def to_dict(self) -> dict:
        ep = self.episodic
        return {
            "trigger_user_memory":      self.trigger_user_memory,
            "trigger_episodic":         self.trigger_episodic,
            "user_memories_count":      len(self.user_memories),
            "episodic_should_store":    ep.should_store if ep else False,
            "episodic_importance":      ep.importance_score if ep else 0,
            "episodic_intensity":       ep.emotional_intensity if ep else 0,
            "router_reasoning":         self.router_reasoning,
            "skipped":                  self.skipped,
        }


# ─── Heuristic pre-filter ─────────────────────────────────────────────────────

_STORE_SIGNALS = [
    r"\bi (am|'m|was|were|have|had|got|get|work|worked|live|lived|moved|study|studied|use|used|prefer|like|love|hate|want|need|decided|switched|chose|built|created|launched|started|finished)\b",
    r"\bmy (name|age|job|work|career|company|team|project|goal|plan|problem|issue|hobby|language|framework|stack|city|country|diet|budget|constraint)\b",
    r"\b(i switched|i moved|i got|i decided|i chose|i built|i launched|i finished|i completed|i failed|i succeeded)\b",
    r"\b(frustrated|excited|worried|anxious|happy|sad|stressed|overwhelmed|proud|nervous)\b",
    r"\b(years old|year old|years experience|i'm from|i live in|i work at|i work for|i study at)\b",
    r"\b(vegetarian|vegan|diabetic|allergic|disabled|pregnant)\b",
]

_SKIP_SIGNALS = [
    # Pure acknowledgements — exact match (allow trailing punctuation/spaces)
    r"^(hi|hello|hey|thanks|thank you|ok|okay|got it|sure|yes|no|nope|yep|cool|great|awesome|nice|perfect|alright|fine|sounds good|makes sense|understood|correct|right|wrong)[\s!.?,]*$",
    # "Thank you + anything casual" — allow trailing text AND trailing punctuation
    r"^thank(s| you)[,\s]+(will|i will|i'll|sounds|ok|okay|great|will think|will consider|noted|got it|appreciate)[^!?]{0,80}[.!?,\s]*$",
    # Generic question starters with no personal content
    r"^(what is|what are|what does|what do|what all do|what all|how do|how does|how can|how to|can you|could you|please|explain|tell me|show me|give me|list|describe|define|compare|contrast)\b",
    # Task/action starters
    r"^(calculate|compute|convert|translate|write|code|create|generate|make|build|fix|debug|find|search|look up|summarize|summarise)\b",
    # Contraction question starters
    r"^(what'?s|where'?s|who'?s|when'?s|why'?s|how'?s)\b",
    # Pure math
    r"^[\d\s\+\-\*\/\(\)=]+$",
    # Follow-up continuation questions — no personal content
    r"^any (ideas|suggestions|tips|thoughts|advice|recommendations|alternatives|options)\b",
    r"^(what (should|would|do) (i|you|we))\b",
    r"^(how (should|would|do|can) (i|you|we))\b",
    r"^(which (is|would be|do you) (best|better|recommended|preferred|suggest))\b",
    r"^(should i|would you|do you think|what do you think|what would you|what would you recommend)\b",
    r"^(tell me more|can you elaborate|can you explain|more details|go on|continue|and then|what else)\b",
    r"^why\b[^,]{0,40}$",
    r"^(is it|is this|is that|are they|are these)\b",
]

_COMPILED_STORE = [re.compile(p, re.IGNORECASE) for p in _STORE_SIGNALS]
_COMPILED_SKIP  = [re.compile(p, re.IGNORECASE) for p in _SKIP_SIGNALS]


def _should_skip_router(user_message: str) -> tuple[bool, str]:
    msg = user_message.strip()

    if len(msg) < 20 and not any(p.search(msg) for p in _COMPILED_STORE):
        return True, "short message, no personal signal"

    for pattern in _COMPILED_SKIP:
        if pattern.search(msg):
            if any(p.search(msg) for p in _COMPILED_STORE):
                return False, "has both skip and store signals"
            return True, "matches skip pattern"

    for pattern in _COMPILED_STORE:
        if pattern.search(msg):
            return False, "has store signal"

    return False, "ambiguous"


def _empty_decision(reason: str) -> RoutingDecision:
    return RoutingDecision(
        trigger_user_memory=False,
        trigger_episodic=False,
        user_memories=[],
        episodic=EpisodicDecision(should_store=False),
        router_reasoning=f"[pre-filter skipped: {reason}]",
        skipped=True,
    )


# ─── Episodic worthiness gate ─────────────────────────────────────────────────

def _is_episodic_worthy(ep: EpisodicDecision) -> bool:
    """
    Hard gate applied AFTER the LLM router says should_store=True.
    Prevents episodic from being created for routine exchanges.

    An episode is worthy if it meets AT LEAST ONE of:
      A) importance_score >= 6.0  — significant event/decision/struggle
      B) emotional_intensity >= 3 — genuine emotional weight
      C) title/reason contains a milestone keyword — definitionally memorable

    Examples of what passes:
      "Debugging Async Race Condition for 3 Hours" (intensity=4) ✓
      "Decided to Switch from React to Vue" (milestone keyword) ✓
      "Got the Job Offer from Google" (importance=9) ✓

    Examples of what fails:
      "User asked about Python list comprehensions" (importance=3, intensity=1) ✗
      "Exploring FastAPI features" (importance=4, intensity=2) ✗
      "User curious about async/await" (importance=3, intensity=1) ✗
    """
    if not ep.should_store:
        return False

    # Gate A: importance
    if ep.importance_score >= _EPISODIC_MIN_IMPORTANCE:
        return True

    # Gate B: emotional intensity
    if ep.emotional_intensity >= _EPISODIC_MIN_INTENSITY:
        return True

    # Gate C: milestone keyword in title or reason
    combined = (ep.title + " " + ep.reason).lower()
    if any(kw in combined for kw in _MILESTONE_KEYWORDS):
        return True

    return False


# ─── Router system prompt ─────────────────────────────────────────────────────

ROUTER_SYSTEM = """Memory Router for an AI assistant. Analyse one exchange, decide what to store.

## Memory types:
USER MEMORY (Neo4j) — 4 types:
  fact:       stable truths — name, age, location, job, language spoken
  preference: likes/dislikes — coding_language, ui_theme, diet
  goal:       wants to achieve — project_goal, career_goal, learning_goal
  constraint: hard limits — budget_constraint, tech_constraint

canonical_key conventions — use EXACTLY these keys, never invent new ones:
  IDENTITY:   name, age, gender
  LOCATION:   location          ← NOT city, place, residence, hometown, current_city
  WORK:       occupation        ← NOT job, role, position, title, job_title
              employer          ← NOT company, workplace, works_at, organization
                                ← CRITICAL: only store employer when user CURRENTLY works there
                                   WRONG: "I got an offer from Amazon"  → do NOT store employer=Amazon
                                   WRONG: "I am thinking of joining Google" → do NOT store employer=Google
                                   WRONG: "I might leave Flipkart" → do NOT change employer
                                   RIGHT: "I joined Amazon" → store employer=Amazon
                                   RIGHT: "I now work at Google" → store employer=Google
                                   RIGHT: "I started at Microsoft" → store employer=Microsoft
                                   For offers, store as fact with key "received_offer" instead
              team              ← NOT department, squad, group
  TECH:       coding_language   ← NOT language, primary_language, tech, stack, pl
              framework         ← NOT library, tool (unless it IS a framework)
              os                ← NOT operating_system, platform
              editor            ← NOT ide, text_editor
  GOALS:      primary_goal      ← NOT goal, main_goal, objective
              career_goal       ← NOT job_goal, work_goal
              project_goal      ← NOT current_project_goal
              learning_goal     ← NOT study_goal, skill_goal
  CONSTRAINTS:budget_constraint ← NOT budget, money_limit, financial_constraint
              tech_constraint   ← NOT technical_constraint, stack_constraint
  LIFESTYLE:  diet              ← NOT food_preference, eating_habit
              health            ← NOT medical, condition
              hobby             ← NOT interest, pastime

CONFLICT RULE — if the user states something that contradicts a previous fact,
use the SAME canonical_key as the previous memory so the state machine can
transition it correctly. Example:
  Previous: employer="works at Google"
  New info:  "I joined Meta last month"
  Correct:   canonical_key="employer", content="works at Meta"  ← same key

EPISODIC (MongoDB) — store ONLY for high-value moments:
  ✓ STORE: Personal decisions made (chose X over Y, decided to switch)
  ✓ STORE: Real blockers or struggles (stuck on X for hours, can't figure out Y)
  ✓ STORE: Emotional events (genuinely frustrated, excited about a result, worried)
  ✓ STORE: Major progress milestones (got X working after days, launched Y)
  ✓ STORE: Life events (got a job, moved cities, started a project)
  ✓ STORE: Multi-session investigations where user is building something real

  ✗ SKIP: Casual questions about a topic ("how does X work?")
  ✗ SKIP: Brief curiosity without personal stakes
  ✗ SKIP: Simple lookups with no emotional or project connection
  ✗ SKIP: Small talk, greetings, continuations
  ✗ SKIP: Learning a concept without a specific goal or struggle attached

  importance_score 1-10:
    1-4: Routine learning, casual questions → these should NOT be stored
    5-6: Meaningful but not pivotal
    7-8: Significant event (major decision, real struggle, important milestone)
    9-10: Life-changing event (job change, major launch, serious problem)

## Output — ONLY this JSON, no markdown:
{
  "trigger_user_memory": true|false,
  "trigger_episodic": true|false,
  "router_reasoning": "one sentence",
  "user_memories": [
    {"memory_type": "fact|preference|goal|constraint", "content": "...",
     "canonical_key": "...", "confidence": 0.0-1.0, "entities": []}
  ],
  "episodic": {
    "should_store": true|false,
    "title": "...",
    "reason": "...",
    "emotional_tone": "neutral|curious|frustrated|excited|worried|satisfied",
    "emotional_intensity": 1-5,
    "importance_score": 1.0-10.0,
    "key_entities": [],
    "tags": []
  }
}

Rules:
- Extract ALL facts/preferences/goals/constraints from the message — a single message can contain several
- Only extract what USER explicitly states, never from assistant responses
- If nothing to store: all trigger_* false, empty arrays
- Use EXACT canonical_key from the list above — never deviate
- Set importance_score honestly — most turns are 1-4, not 7+"""


# ─── Main route function ──────────────────────────────────────────────────────

async def route(
    user_message: str,
    assistant_response: str,
    conversation_context: str = "",
    turn_number: int = 0,
) -> RoutingDecision:
    """
    Two-stage router with episodic worthiness gate:
      1. Heuristic pre-filter (free, ~0ms)
      2. LLM router (only if pre-filter says needed)
      3. Episodic worthiness gate (deterministic, free)
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
            max_tokens=600,   # 400 truncates when 3+ memories + episodic all need to be output
            temperature=0,
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

    ep_raw = raw.get("episodic", {})
    episodic = EpisodicDecision(
        should_store=        bool(ep_raw.get("should_store", False)),
        title=               ep_raw.get("title", ""),
        reason=              ep_raw.get("reason", ""),
        emotional_tone=      ep_raw.get("emotional_tone", "neutral"),
        emotional_intensity= int(ep_raw.get("emotional_intensity", 1)),
        importance_score=    float(ep_raw.get("importance_score", 5.0)),
        key_entities=        ep_raw.get("key_entities", []),
        tags=                ep_raw.get("tags", [])
    )

    # ── Stage 3: episodic worthiness gate ────────────────────
    # LLM said store — but does it actually meet the bar?
    episodic_triggered = (
        bool(raw.get("trigger_episodic", False))
        and episodic.should_store
        and _is_episodic_worthy(episodic)
    )

    if episodic.should_store and not episodic_triggered:
        print(
            f"  ↳ Episodic GATED OUT: importance={episodic.importance_score:.1f} "
            f"intensity={episodic.emotional_intensity} title='{episodic.title[:50]}'"
        )
        episodic.should_store = False  # Mark as not stored so caller knows

    return RoutingDecision(
        trigger_user_memory= bool(raw.get("trigger_user_memory", False)) and len(user_memories) > 0,
        trigger_episodic=    episodic_triggered,
        user_memories=       user_memories,
        episodic=            episodic,
        router_reasoning=    raw.get("router_reasoning", ""),
        skipped=             False,
    )