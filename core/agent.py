"""
core/agent.py — Per-turn orchestrator with latency optimisations

Latency optimisations applied
──────────────────────────────
  1. Background storage
     Router + Neo4j writes + MongoDB writes + Summarization all run AFTER
     the response is returned to the user. User never waits for storage.
     Fire-and-forget via asyncio.create_task().

  2. Embed once
     user_message is embedded ONCE in build_context(), the vector is
     reused by Neo4j cache, MongoDB, and Redis. No duplicate API calls.

  3. Neo4j 30s cache
     Neo4j/Graphiti result cached in Redis for 30 seconds.
     Cache hit: ~5ms instead of ~500ms.

  4. Streaming support
     chat_stream() yields tokens as they arrive from the LLM.
     User sees first token in ~300ms instead of waiting 2-4 seconds.
     Storage still happens in background after stream completes.

  5. Background summarisation
     Summarisation (L0/L1/L2/handoff) runs as a background task.
     No longer blocks the response on summarisation turns.
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

AGENT_SYSTEM = """You are Synapse — a helpful AI assistant with persistent memory of this user.

Use the context below to personalise every response naturally:
- Reference known facts about the user when relevant
- Honour their constraints and goals
- Connect to their ongoing projects when relevant
- If the user shares new info, acknowledge it briefly and naturally
- Never recite the context back verbatim

{context_section}

---
Answer the user's message helpfully, clearly, and concisely."""


# ─── Main chat — returns full response immediately ────────────────────────────

async def chat(user_id: str, session_id: str, user_message: str) -> dict:
    """
    Full per-turn pipeline.

    What the user waits for:
      1. build_context()     — parallel fetch from all stores
      2. LLM call            — generate response

    What runs in background (user does NOT wait):
      3. increment_session_turn()
      4. route()             — router LLM call
      5. store_memories()    — Neo4j writes
      6. store_episodic()    — MongoDB write
      7. save_turn()         — SQLite write
      8. summarization       — L0/L1/L2/handoff
    """

    # ── 1. Build context (embed once, parallel fetch) ─────────
    context     = await build_context(session_id, user_id, user_message)
    context_str = format_context_for_prompt(context)

    # ── 2. LLM response ───────────────────────────────────────
    response = await _client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=800,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM.format(context_section=context_str)},
            {"role": "user",   "content": user_message}
        ]
    )
    assistant_message = response.choices[0].message.content

    # ── 3. Fire background storage — user does NOT wait ───────
    asyncio.create_task(
        _background_store(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            context=context,
        )
    )

    return {
        "response":    assistant_message,
        "turn_number": None,   # filled after background task runs
        "context_used": {
            "memories_count":  len(context.get("memories", [])),
            "episodic_count":  len(context.get("episodic_memories", [])),
            "summaries_count": len(context.get("summaries", [])),
            "raw_turns_count": len(context.get("raw_turns", [])),
        }
    }


# ─── Streaming chat — yields tokens as they arrive ───────────────────────────

async def chat_stream(user_id: str, session_id: str, user_message: str):
    """
    Streaming version — yields tokens as they arrive from the LLM.
    Timing is logged to server console so latency is visible.
    """
    import time
    t0 = time.monotonic()

    # ── 1. Build context ──────────────────────────────────────
    context     = await build_context(session_id, user_id, user_message)
    context_str = format_context_for_prompt(context)
    t_ctx = time.monotonic()
    print(f"  ⏱ context_build: {(t_ctx-t0)*1000:.0f}ms  "
          f"[neo4j={len(context.get('memories',[]))} "
          f"mongo={len(context.get('episodic_memories',[]))} "
          f"redis={len(context.get('summaries',[]))} "
          f"sqlite={len(context.get('raw_turns',[]))}]")

    # ── 2. Stream LLM response ────────────────────────────────
    full_response  = ""
    first_token_at = None
    stream = await _client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=800,
        stream=True,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM.format(context_section=context_str)},
            {"role": "user",   "content": user_message}
        ]
    )

    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            if first_token_at is None:
                first_token_at = time.monotonic()
                print(f"  ⏱ first_token:   {(first_token_at-t0)*1000:.0f}ms total  "
                      f"(+{(first_token_at-t_ctx)*1000:.0f}ms LLM TTFT)")
            full_response += token
            yield {"token": token}

    t_end = time.monotonic()
    print(f"  ⏱ stream_done:   {(t_end-t0)*1000:.0f}ms total")

    # ── 3. Background storage — fire and forget ──────────────────
    # Storage runs fully in background. User never waits for it.
    # Badges/router log are populated by the UI's setTimeout refresh.
    asyncio.create_task(
        _background_store(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=full_response,
            context=context,
        )
    )

    yield {
        "done": True,
        "context_used": {
            "memories_count":  len(context.get("memories", [])),
            "episodic_count":  len(context.get("episodic_memories", [])),
            "summaries_count": len(context.get("summaries", [])),
            "raw_turns_count": len(context.get("raw_turns", [])),
        }
    }


# ─── Background storage ───────────────────────────────────────────────────────

async def _background_store(
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    context: dict,
) -> dict:
    """
    All storage operations. Now returns a result dict so chat_stream
    can include routing/episodic info in the SSE done chunk for UI badges.
    """
    result = {
        "turn_number":     None,
        "routing":         {},
        "memories_stored": [],
        "episodic_stored": None,
        "summarization":   None,
    }
    try:
        # Increment turn counter
        turn_number    = await sql.increment_session_turn(session_id)
        result["turn_number"] = turn_number
        router_context = format_context_for_router(context)

        # Router + summarization in parallel
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

        # Save raw turn to SQLite
        router_dict = decision.to_dict() if isinstance(decision, RoutingDecision) else {}
        await sql.save_turn(
            session_id=session_id,
            user_id=user_id,
            turn_number=turn_number,
            user_msg=user_message,
            assistant_msg=assistant_message,
            router_decision=router_dict
        )

        # Capture summarization result
        if isinstance(summarization_result, dict) and summarization_result.get("summarized"):
            result["summarization"] = summarization_result

        if not isinstance(decision, RoutingDecision):
            return result

        result["routing"] = decision.to_dict()

        # Neo4j — store facts via state machine
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
            await neo4j.store_memories_batch(
                user_id=user_id,
                session_id=session_id,
                memories=mem_list,
                source_turn=turn_number
            )
            result["memories_stored"] = mem_list
            # Invalidate Neo4j cache so next turn sees fresh facts
            await _invalidate_neo4j_cache(user_id)

        # MongoDB — store episodic if significant
        # NOTE: condition was previously `not isinstance(summarization_result, dict)`
        # which blocked episodic on every summarization turn — wrong.
        # Episodic and summarization are independent — both can trigger on same turn.
        if decision.trigger_episodic and decision.episodic:
            from memory.extractor import create_episodic_narrative
            recent_turns  = context.get("raw_turns", [])
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
            result["episodic_stored"] = {"memory_id": ep_id, "title": narrative["title"]}

    except Exception as e:
        print(f"⚠ Background store error: {e}")

    return result


async def _invalidate_neo4j_cache(user_id: str):
    """Delete the cached Neo4j result so next turn fetches fresh data."""
    try:
        from db.redis_manager import get_redis
        r = await get_redis()
        await r.delete(f"neo4j_cache:{user_id}")
    except Exception:
        pass