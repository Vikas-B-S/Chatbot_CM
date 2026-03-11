"""
core/router.py — Memory Router Node

After every turn, analyses the exchange and decides which memory systems
to activate. Returns a RoutingDecision executed by agent.py.

Memory nodes:
  User Memory → Graphiti/Neo4j  (facts, preferences, goals, constraints)
  Episodic    → MongoDB          (rich narrative episodes)
  User DB     → SQLite           (lightweight profile updates)

Note on Graphiti contradiction resolution:
  The router's job is to extract WHAT to remember and give it a canonical_key.
  Graphiti then handles HOW to store it — detecting contradictions, expiring
  old facts, and setting temporal edges automatically.
  e.g. if the user previously said "I love Python" and now says "I love C++",
  the router extracts {canonical_key: "coding_language", content: "User loves C++"}
  and Graphiti will expire the old Python edge automatically on add_episode().
"""
import json
from openai import AsyncOpenAI
from dataclasses import dataclass, field
from typing import Optional
from config import get_settings

settings = get_settings()
_client = None


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
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {}


# ─── Routing decision data structures ────────────────────────

@dataclass
class UserMemoryItem:
    memory_type:   str           # fact | preference | goal | constraint
    content:       str           # human-readable statement
    canonical_key: str           # dedup key — same concept always gets same key
    confidence:    float = 1.0
    entities:      list  = field(default_factory=list)  # [{name, type}]


@dataclass
class EpisodicDecision:
    should_store:   bool
    title:          str  = ""
    reason:         str  = ""
    emotional_tone: str  = "neutral"
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

    user_memories:   list[UserMemoryItem]      = field(default_factory=list)
    episodic:        Optional[EpisodicDecision] = None
    user_db:         Optional[UserDbUpdate]     = None
    router_reasoning: str                       = ""

    def to_dict(self) -> dict:
        return {
            "trigger_user_memory":   self.trigger_user_memory,
            "trigger_episodic":      self.trigger_episodic,
            "trigger_user_db":       self.trigger_user_db,
            "user_memories_count":   len(self.user_memories),
            "episodic_should_store": self.episodic.should_store if self.episodic else False,
            "user_db_update":        self.user_db.fields if self.user_db else {},
            "router_reasoning":      self.router_reasoning
        }


# ─── Router system prompt ─────────────────────────────────────

ROUTER_SYSTEM = """You are the Memory Router for a multi-memory AI assistant system.

Your job: analyse one conversation exchange and decide which memory systems to activate.

## Memory systems available:

### 1. USER MEMORY NODE → Graphiti/Neo4j (long-term, cross-session, temporal)
Stores 4 types of user memory:
  - fact:        Stable truths about the user. E.g. name, age, location, job, language
  - preference:  Likes/dislikes/preferred styles. E.g. "prefers dark mode", "likes Python"
  - goal:        Things user wants to achieve. E.g. "wants to build an iOS app", "learning ML"
  - constraint:  Hard limits. E.g. "cannot use paid APIs", "must use Python 3.9", "vegetarian"

IMPORTANT — canonical_key determines contradiction resolution:
  Graphiti automatically expires old facts when a new one shares the same canonical_key.
  Example: if you assign canonical_key="coding_language" to "loves Python" at T3,
  and then "loves C++" also gets canonical_key="coding_language" at T14,
  Graphiti will automatically mark "loves Python" as expired at T14.
  This means: ALWAYS use the same canonical_key for the same concept across turns.

canonical_key conventions (use these exactly, do not invent new ones for known concepts):
  fact keys:       name, age, location, occupation, employer, education, language, nationality
  preference keys: coding_language, ui_theme, communication_style, diet, [topic]_preference
  goal keys:       primary_goal, career_goal, learning_goal, project_goal, [topic]_goal
  constraint keys: tech_constraint, budget_constraint, diet_constraint, [topic]_constraint

For each memory item provide:
  - memory_type: one of fact|preference|goal|constraint
  - content: concise human-readable statement
  - canonical_key: short snake_case dedup key (see conventions above)
  - confidence: 1.0=explicit, 0.8=strong implication, 0.6=reasonable inference, 0.4=weak
  - entities: named entities [{name, type}] — type=Person|Location|Organization|Technology|Product

### 2. EPISODIC NODE → MongoDB (narrative memories of meaningful sessions)
Store episodically when the exchange involves:
  ✓ Personal problem-solving, debugging, planning, important decisions
  ✓ Emotional or significant life events
  ✓ Learning something complex together over multiple messages
  ✓ Rich back-and-forth that reveals the user's situation deeply
  ✗ Simple one-shot Q&A ("what is X?", "how do I do Y?")
  ✗ Trivial small talk

### 3. USER DB NODE → SQLite (lightweight profile updates)
Update when the user explicitly changes their display name or profile fields.

## Output format — return ONLY this JSON, no markdown:
{
  "trigger_user_memory": true|false,
  "trigger_episodic": true|false,
  "trigger_user_db": true|false,
  "router_reasoning": "1-2 sentence explanation",
  "user_memories": [
    {
      "memory_type": "fact|preference|goal|constraint",
      "content": "...",
      "canonical_key": "...",
      "confidence": 0.0-1.0,
      "entities": [{"name": "...", "type": "..."}]
    }
  ],
  "episodic": {
    "should_store": true|false,
    "title": "Short episode title",
    "reason": "Why episodic",
    "emotional_tone": "curious|frustrated|excited|neutral|worried|confident",
    "key_entities": ["entity1"],
    "tags": ["tag1", "tag2"]
  },
  "user_db": {
    "should_update": false,
    "fields": {}
  }
}

Rules:
- Only extract what is EXPLICITLY stated or unmistakably implied by the USER
- Do NOT extract from the assistant's responses or generic knowledge
- If nothing to extract: set all trigger_* to false, return empty arrays
- Never hallucinate or invent information
- canonical_key must be consistent across all turns — "name" always means the user's name
- If the user updates a preference (e.g. switches from Python to C++), use the SAME
  canonical_key as before so Graphiti can automatically expire the old value"""


async def route(
    user_message: str,
    assistant_response: str,
    conversation_context: str = "",
    turn_number: int = 0
) -> RoutingDecision:
    """
    Core router function. Takes one exchange and returns a RoutingDecision.
    Single LLM call — structured JSON output parsed into typed dataclasses.
    """
    client = _get_client()

    prompt = (
        f"Turn #{turn_number}\n\n"
        f"Recent context:\n{conversation_context}\n\n"
        f"User said:\n{user_message}\n\n"
        f"Assistant replied:\n{assistant_response}\n\n"
        "Produce routing decision for this exchange."
    )

    response = await client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user",   "content": prompt}
        ]
    )

    raw = _parse_json(response.choices[0].message.content)

    # ── Parse user memories ───────────────────────────────────
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

    # ── Parse episodic decision ───────────────────────────────
    ep_raw = raw.get("episodic", {})
    episodic = EpisodicDecision(
        should_store=   bool(ep_raw.get("should_store", False)),
        title=          ep_raw.get("title", ""),
        reason=         ep_raw.get("reason", ""),
        emotional_tone= ep_raw.get("emotional_tone", "neutral"),
        key_entities=   ep_raw.get("key_entities", []),
        tags=           ep_raw.get("tags", [])
    )

    # ── Parse user DB update ──────────────────────────────────
    db_raw = raw.get("user_db", {})
    user_db = UserDbUpdate(
        should_update= bool(db_raw.get("should_update", False)),
        fields=        db_raw.get("fields", {})
    )

    return RoutingDecision(
        trigger_user_memory= bool(raw.get("trigger_user_memory", False)) and len(user_memories) > 0,
        trigger_episodic=    bool(raw.get("trigger_episodic", False)) and episodic.should_store,
        trigger_user_db=     bool(raw.get("trigger_user_db", False)) and user_db.should_update,
        user_memories=       user_memories,
        episodic=            episodic,
        user_db=             user_db,
        router_reasoning=    raw.get("router_reasoning", "")
    )