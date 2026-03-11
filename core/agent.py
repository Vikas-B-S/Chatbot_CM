"""
core/agent.py
ChatAgent — per-turn orchestrator.

Pipeline:
  1. Build context  (Graphiti search + MongoDB + Redis + SQLite)
     ↑ user_message passed for targeted Graphiti hybrid search
  2. Generate response  (Claude via OpenRouter)
  3. Save turn counter  (SQLite)
  4. Parallel post-processing:
     a. Router   → extract memories → store_memories_batch() → Graphiti/Neo4j
     b. Summarizer → Redis L0/L1 + MongoDB episodic
  5. Save full turn to SQLite (with router decision logged)
"""
import asyncio
from openai import AsyncOpenAI
from config import get_settings
from db import sqlite_manager as sql
from db import neo4j_manager as neo4j
from db import mongo_manager as mongo
from core.router import route, RoutingDecision
from memory.context_builder import (
    build_context, format_context_for_prompt, format_context_for_router
)
from memory.summarizer import check_and_run_summarization

settings = get_settings()

_client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url
)

AGENT_SYSTEM = """You are a helpful, knowledgeable AI assistant with persistent memory of this user.

Use the context section below to personalise every response naturally:
- Reference known facts about the user when relevant ("Since you're based in Berlin…")
- Honour constraints without being preachy about them
- Connect to the user's goals when they're relevant
- If the user shares new personal info, acknowledge it briefly and naturally
- Never recite the context back verbatim

{context_section}

---
Answer the user's message helpfully, clearly, and concisely."""


async def chat(
    user_id: str,
    session_id: str,
    user_message: str
) -> dict:
    """Full per-turn pipeline. Returns structured result dict."""

    # ── 1. Build context ──────────────────────────────────────
    # Pass user_message so Graphiti does targeted search for relevant facts
    context = await build_context(session_id, user_id, user_message)
    context_str = format_context_for_prompt(context)

    # ── 2. Generate response ──────────────────────────────────
    response = await _client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=800,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM.format(context_section=context_str)},
            {"role": "user",   "content": user_message}
        ]
    )
    assistant_message = response.choices[0].message.content

    # ── 3. Save turn counter ──────────────────────────────────
    turn_number = await sql.increment_session_turn(session_id)
    router_context = format_context_for_router(context)

    # ── 4. Router + Summarizer in parallel ────────────────────
    decision, summarization_result = await asyncio.gather(
        route(
            user_message=user_message,
            assistant_response=assistant_message,
            conversation_context=router_context,
            turn_number=turn_number
        ),
        check_and_run_summarization(
            session_id=session_id,
            user_id=user_id,
            turn_number=turn_number
        ),
        return_exceptions=True
    )

    # ── 5. Save full turn to SQLite ───────────────────────────
    router_decision_dict = decision.to_dict() if isinstance(decision, RoutingDecision) else {}
    await sql.save_turn(
        session_id=session_id,
        user_id=user_id,
        turn_number=turn_number,
        user_msg=user_message,
        assistant_msg=assistant_message,
        router_decision=router_decision_dict
    )

    # ── 6. Execute routing decisions ──────────────────────────
    memories_stored = []
    episodic_stored = None
    user_db_updated = False

    if isinstance(decision, RoutingDecision):

        # Node A: User Memory → Graphiti (Neo4j)
        # Graphiti handles entity extraction + contradiction resolution
        # e.g. "I love C++" will automatically expire "I love Python"
        if decision.trigger_user_memory and decision.user_memories:
            mem_list = [
                {
                    "memory_type":   m.memory_type,
                    "content":       m.content,
                    "canonical_key": m.canonical_key,
                    "confidence":    m.confidence,
                    "entities":      m.entities
                }
                for m in decision.user_memories
            ]
            stored_ids = await neo4j.store_memories_batch(
                user_id=user_id,
                session_id=session_id,
                memories=mem_list,
                source_turn=turn_number
            )
            memories_stored = [
                {"id": sid, "type": m.memory_type, "content": m.content}
                for sid, m in zip(stored_ids, decision.user_memories)
            ]

        # Node B: Episodic → MongoDB
        # Only inline if this is NOT a summarization turn
        # (summarizer handles episodic at batch boundaries)
        if (
            decision.trigger_episodic
            and decision.episodic
            and not isinstance(summarization_result, dict)
        ):
            from memory.extractor import create_episodic_narrative
            recent_turns = context.get("raw_turns", [])
            episode_turns = recent_turns[-2:] + [{
                "turn_number":   turn_number,
                "user_msg":      user_message,
                "assistant_msg": assistant_message
            }]
            narrative = await create_episodic_narrative(
                episode_turns,
                {
                    "emotional_tone": decision.episodic.emotional_tone,
                    "key_entities":   decision.episodic.key_entities,
                    "tags":           decision.episodic.tags
                }
            )
            ep_id = await mongo.store_episodic_memory(
                user_id=user_id,
                session_id=session_id,
                title=narrative["title"],
                content=narrative["content"],
                outcome=narrative["outcome"],
                turn_start=episode_turns[0].get("turn_number", turn_number),
                turn_end=turn_number,
                key_entities=decision.episodic.key_entities,
                emotional_tone=decision.episodic.emotional_tone,
                emotional_intensity=getattr(decision.episodic, "emotional_intensity", 2),
                tags=decision.episodic.tags,
                topic_cluster=narrative.get("topic_cluster", "general"),
                importance_score=narrative.get("importance_score", 5.0),
            )
            episodic_stored = {"memory_id": ep_id, "title": narrative["title"]}

        # Node C: User DB update → SQLite
        if decision.trigger_user_db and decision.user_db and decision.user_db.fields:
            user_db_updated = True

    return {
        "response":      assistant_message,
        "turn_number":   turn_number,
        "routing": {
            "triggered_user_memory": isinstance(decision, RoutingDecision) and decision.trigger_user_memory,
            "triggered_episodic":    isinstance(decision, RoutingDecision) and decision.trigger_episodic,
            "triggered_user_db":     isinstance(decision, RoutingDecision) and decision.trigger_user_db,
            "reasoning":             decision.router_reasoning if isinstance(decision, RoutingDecision) else "",
        },
        "memories_stored":  memories_stored,
        "episodic_stored":  episodic_stored,
        "user_db_updated":  user_db_updated,
        "summarization":    summarization_result if isinstance(summarization_result, dict) else None,
        "context_used": {
            "memories_count":  len(context.get("memories", [])),
            "episodic_count":  len(context.get("episodic_memories", [])),
            "summaries_count": len(context.get("summaries", [])),
            "raw_turns_count": len(context.get("raw_turns", [])),
        }
    }