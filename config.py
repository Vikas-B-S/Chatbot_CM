"""
config.py — Central settings via pydantic-settings

Model strategy
──────────────
  claude_model    — main LLM for user-facing responses (powerful, slower, expensive)
  router_model    — cheap fast model for routing decisions (~0ms extra latency)
  extractor_model — cheap fast model for ALL summarization tasks in extractor.py

  Why a separate extractor_model?
    summarize_turns()      → structured JSON, 3-5 sentences   → no need for main model
    compress_summaries()   → merge 3 summaries into 1         → no need for main model
    compress_to_arc()      → higher-level compression         → no need for main model
    create_handoff_summary() → 3 sentence handoff note        → no need for main model
    create_episodic_narrative() → narrative from turns        → no need for main model

    These are all structured extraction tasks with clear schemas.
    gpt-4o-mini handles them perfectly at ~20x lower cost than a full model.

  Cost example (100 turns/day, summarize every 3 turns):
    Old: ~33 summarization cycles × 5 calls × main model cost
    New: ~33 summarization cycles × 5 calls × mini model cost
    Saving: ~95% on summarization costs with identical output quality
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM models ────────────────────────────────────────────
    # Main model — used only for user-facing responses in agent.py
    claude_model: str = "anthropic/claude-3-5-sonnet"

    # Router model — cheap fast model for routing decisions
    # Already existed; kept separate so router and extractor can be tuned independently
    router_model: str = "openai/gpt-4o-mini"

    # Extractor model — cheap fast model for ALL summarization in extractor.py
    # Covers: L0 batch summary, L1 compression, L2 arc, handoff, episodic narrative
    # NEW (v3.3): was previously hardcoded to claude_model in extractor.py
    extractor_model: str = "openai/gpt-4o-mini"

    # ── OpenRouter ────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Neo4j ─────────────────────────────────────────────────
    neo4j_uri:      str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    # ── MongoDB ───────────────────────────────────────────────
    mongo_uri:                 str = "mongodb://localhost:27017"
    mongo_db:                  str = "mnemo"
    mongo_episodic_collection: str = "episodic_memories"

    # ── Redis ─────────────────────────────────────────────────
    redis_host:     str = "localhost"
    redis_port:     int = 6379
    redis_password: str = ""
    redis_db:       int = 0

    # ── SQLite ────────────────────────────────────────────────
    sqlite_db_path: str = "chatbot.db"

    # ── Summarization schedule ────────────────────────────────
    # summarize_at_turn: first summarization fires at this turn number
    # summarize_batch:   how many turns per L0 batch
    summarize_at_turn: int = 6
    summarize_batch:   int = 3

    class Config:
        env_file = ".env"
        extra    = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()