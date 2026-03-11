"""
config.py — Centralised settings with clear storage responsibilities.

Storage map:
  Neo4j + Graphiti → ALL user memory: facts, preferences, goals, constraints
                     Temporal edges, contradiction resolution, entity dedup
  MongoDB          → Episodic memories (rich narrative records)
  Redis            → Conversation summaries (level-0 batches + level-1 meta)
  SQLite           → Users, sessions, raw turn logs (operational data only)
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM via OpenRouter ────────────────────────────────────
    openrouter_api_key:  str = Field(...,    env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL")
    claude_model:        str = Field("anthropic/claude-haiku-4-5",   env="CLAUDE_MODEL")

    # ── Embeddings (used by Graphiti for semantic search) ─────
    embedding_model: str = Field("openai/text-embedding-3-small", env="EMBEDDING_MODEL")

    # ── Neo4j — Graphiti temporal knowledge graph ─────────────
    # Note: neo4j_database removed — Graphiti manages its own db scoping
    neo4j_uri:      str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j",    env="NEO4J_USERNAME")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")

    # ── MongoDB — episodic memories ───────────────────────────
    mongo_uri:                 str = Field("mongodb://localhost:27017", env="MONGO_URI")
    mongo_db:                  str = Field("chatbot",           env="MONGO_DB")
    mongo_episodic_collection: str = Field("episodic_memories", env="MONGO_EPISODIC_COLLECTION")

    # ── Redis — summaries ─────────────────────────────────────
    redis_host:     str = Field("localhost", env="REDIS_HOST")
    redis_port:     int = Field(6379,        env="REDIS_PORT")
    redis_password: str = Field("",          env="REDIS_PASSWORD")
    redis_db:       int = Field(0,           env="REDIS_DB")

    # ── SQLite — users / sessions / turn logs ─────────────────
    sqlite_db_path: str = Field("./chatbot.db", env="SQLITE_DB_PATH")

    # ── Context window ────────────────────────────────────────
    max_raw_turns:     int = Field(6, env="MAX_RAW_TURNS")
    summarize_at_turn: int = Field(6, env="SUMMARIZE_AT_TURN")
    summarize_batch:   int = Field(3, env="SUMMARIZE_BATCH")

    debug: bool = Field(False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
