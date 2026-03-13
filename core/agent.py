"""
core/agent.py

KEY FIX (v3.2) — Why storage never ran before:
───────────────────────────────────────────────
  The SSE client in the UI closes the connection the moment it receives
  the final "done: true" chunk. FastAPI then cancels the async generator.
  asyncio.create_task(_background_store()) was at the END of the generator,
  after yielding all tokens — so the client always disconnected first and
  the task was never created. Nothing was ever stored.

  The fix: split chat_stream() into two functions:
    1. stream_tokens()              — pure generator, only yields tokens
    2. background_store_wrapper()   — registered via FastAPI BackgroundTasks
                                      in server.py BEFORE returning the
                                      StreamingResponse. Runs after the
                                      response is fully sent, guaranteed,
                                      even if the client disconnects early.

  The two functions share a `collected` dict that stream_tokens populates
  with the full response text and context while streaming. background_store_wrapper
  reads from it once streaming is done.

CACHE INVALIDATION FIX (v3.3):
────────────────────────────────
  _invalidate_neo4j_cache() is replaced by neo4j.invalidate_neo4j_cache()
  which now correctly accepts both user_id AND session_id.

  Old call:  await _invalidate_neo4j_cache(user_id)
             → deleted key: neo4j_cache:{user_id}          ← WRONG

  New call:  await neo4j.invalidate_neo4j_cache(user_id, session_id)
             → deleted key: neo4j_cache:{user_id}:{session_id}  ← CORRECT

  The cache is written in context_builder.py as:
    cache_key = f"neo4j_cache:{user_id}:{session_id}"

  The old invalidation key never matched — stale Neo4j data was served
  for up to 30 seconds after every memory write. Now it matches exactly.
"""
import asyncio
import time
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


# ─── 1. Pure streaming generator ─────────────────────────────────────────────
# Only job: build context, stream tokens, write results into `collected`.
# Does NOT create any background tasks — server.py does that via BackgroundTasks.

async def stream_tokens(
    user_id: str,
    session_id: str,
    user_message: str,
    collected: dict,          # shared dict — we populate, server.py reads
):
    """
    Pure token streaming generator. Populates `collected` dict with:
      collected["full_response"] — complete assistant message
      collected["context"]       — context dict from build_context()
    """
    t0 = time.monotonic()

    # ── Build context ─────────────────────────────────────────
    context     = await build_context(session_id, user_id, user_message)
    context_str = format_context_for_prompt(context)
    t_ctx = time.monotonic()
    print(f"  ⏱ context_build: {(t_ctx-t0)*1000:.0f}ms  "
          f"[neo4j={len(context.get('memories',[]))} "
          f"mongo={len(context.get('episodic_memories',[]))} "
          f"redis={len(context.get('summaries',[]))} "
          f"sqlite={len(context.get('raw_turns',[]))}]")

    # Write context into shared dict immediately
    collected["context"] = context

    # ── Stream LLM response ───────────────────────────────────
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
        if not chunk.choices:          # final [DONE] chunk from OpenRouter has empty choices
            continue
        token = chunk.choices[0].delta.content or ""
        if token:
            if first_token_at is None:
                first_token_at = time.monotonic()
                print(f"  ⏱ first_token:   {(first_token_at-t0)*1000:.0f}ms total  "
                      f"(+{(first_token_at-t_ctx)*1000:.0f}ms LLM TTFT)")
            full_response += token
            yield {"token": token}

    # Write full response into shared dict
    collected["full_response"] = full_response

    t_end = time.monotonic()
    print(f"  ⏱ stream_done:   {(t_end-t0)*1000:.0f}ms total  "
          f"({len(full_response)} chars)")

    yield {
        "done": True,
        "context_used": {
            "memories_count":  len(context.get("memories", [])),
            "episodic_count":  len(context.get("episodic_memories", [])),
            "summaries_count": len(context.get("summaries", [])),
            "raw_turns_count": len(context.get("raw_turns", [])),
        }
    }


# ─── 2. Background storage wrapper ───────────────────────────────────────────
# Called by FastAPI BackgroundTasks in server.py — guaranteed to run after
# the response is fully sent, even if the SSE client disconnects early.

async def background_store_wrapper(
    user_id: str,
    session_id: str,
    user_message: str,
    collected: dict,
):
    """
    Reads from `collected` (populated by stream_tokens) and runs all storage.
    Called via FastAPI BackgroundTasks — never cancelled by client disconnect.
    """
    full_response = collected.get("full_response", "")
    context       = collected.get("context")

    if not full_response:
        print("  ⚠ background_store: no response captured, skipping storage")
        return
    if context is None:
        print("  ⚠ background_store: no context captured, skipping storage")
        return

    print(f"  → background_store starting for session {session_id[:8]}...")
    await _background_store(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        assistant_message=full_response,
        context=context,
    )


# ─── Non-streaming chat (sync endpoint) ──────────────────────────────────────

async def chat(user_id: str, session_id: str, user_message: str) -> dict:
    context     = await build_context(session_id, user_id, user_message)
    context_str = format_context_for_prompt(context)

    response = await _client.chat.completions.create(
        model=settings.claude_model,
        max_tokens=800,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM.format(context_section=context_str)},
            {"role": "user",   "content": user_message}
        ]
    )
    assistant_message = response.choices[0].message.content

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
        "turn_number": None,
        "context_used": {
            "memories_count":  len(context.get("memories", [])),
            "episodic_count":  len(context.get("episodic_memories", [])),
            "summaries_count": len(context.get("summaries", [])),
            "raw_turns_count": len(context.get("raw_turns", [])),
        }
    }


# ─── Core storage pipeline ───────────────────────────────────────────────────
# Each store has its own try/except — one failure never blocks the others.

async def _background_store(
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    context: dict,
) -> dict:

    result = {
        "turn_number":     None,
        "routing":         {},
        "memories_stored": [],
        "episodic_stored": None,
        "summarization":   None,
    }

    # ── Step 1: SQLite turn counter ───────────────────────────
    turn_number = None
    try:
        turn_number = await sql.increment_session_turn(session_id)
        result["turn_number"] = turn_number
        print(f"  ✓ SQLite turn counter: T{turn_number}")
    except Exception as e:
        print(f"  ✗ SQLite increment_session_turn FAILED: {e}")
        return result

    router_context = format_context_for_router(context)

    # ── Step 2: Router + Summarization in parallel ────────────
    decision = None
    try:
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

        if isinstance(decision, Exception):
            print(f"  ✗ Router FAILED: {decision}")
            decision = None
        else:
            print(f"  ✓ Router: "
                  f"mem={decision.trigger_user_memory} "
                  f"ep={decision.trigger_episodic} "
                  f"skipped={decision.skipped}")

        if isinstance(summarization_result, Exception):
            print(f"  ✗ Summarization FAILED: {summarization_result}")
        elif summarization_result and summarization_result.get("summarized"):
            result["summarization"] = summarization_result
            print(f"  ✓ Summarization: "
                  f"T{summarization_result.get('batch_start')}"
                  f"-T{summarization_result.get('batch_end')}")

    except Exception as e:
        print(f"  ✗ Router/Summarization FAILED: {e}")

    # ── Step 3: SQLite save_turn ──────────────────────────────
    try:
        router_dict = decision.to_dict() if isinstance(decision, RoutingDecision) else {}
        await sql.save_turn(
            session_id=session_id,
            user_id=user_id,
            turn_number=turn_number,
            user_msg=user_message,
            assistant_msg=assistant_message,
            router_decision=router_dict
        )
        print(f"  ✓ SQLite save_turn: T{turn_number}")
    except Exception as e:
        print(f"  ✗ SQLite save_turn FAILED: {e}")

    if not isinstance(decision, RoutingDecision):
        return result

    result["routing"] = decision.to_dict()

    # ── Step 4: Neo4j ─────────────────────────────────────────
    if decision.trigger_user_memory and decision.user_memories:
        try:
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

            # ── FIX (v3.3): pass session_id so invalidation key matches
            # what context_builder.py stored: neo4j_cache:{user_id}:{session_id}
            # Old code deleted neo4j_cache:{user_id} — a key that never existed.
            await neo4j.invalidate_neo4j_cache(user_id, session_id)

            print(f"  ✓ Neo4j: stored {len(mem_list)} memories")
        except Exception as e:
            print(f"  ✗ Neo4j FAILED: {e}")
    else:
        print(f"  · Neo4j: skipped "
              f"(trigger={decision.trigger_user_memory}, "
              f"count={len(decision.user_memories) if decision.user_memories else 0})")

    # ── Step 5: MongoDB ───────────────────────────────────────
    if decision.trigger_episodic and decision.episodic:
        try:
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
            print(f"  ✓ MongoDB: [{narrative['title'][:40]}]")
        except Exception as e:
            print(f"  ✗ MongoDB FAILED: {e}")
    else:
        print(f"  · MongoDB: skipped (trigger={decision.trigger_episodic})")

    return result