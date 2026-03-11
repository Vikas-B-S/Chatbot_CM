"""
db/neo4j_manager.py — Graphiti temporal knowledge graph

Import paths adapt to graphiti-core >= 0.25 where OpenAIGenericClient
moved from graphiti_core.llm_client to graphiti_core.llm_client.openai_generic_client
and OpenAIEmbedder moved to graphiti_core.embedder.openai_embedder.
"""
from datetime import datetime, timezone
from typing import Optional
from config import get_settings

settings = get_settings()
_graphiti = None


from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig


def _build_graphiti() -> Graphiti:
    llm_config = LLMConfig(
        api_key=settings.openrouter_api_key,
        model=settings.claude_model,
        base_url=settings.openrouter_base_url,
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        embedding_model="openai/text-embedding-3-small"
    )
    return Graphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_username,
        password=settings.neo4j_password,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config)
    )


async def _get_graphiti() -> Graphiti:
    global _graphiti
    if _graphiti is None:
        _graphiti = _build_graphiti()
    return _graphiti


async def init_neo4j():
    g = await _get_graphiti()
    await g.build_indices_and_constraints()


async def close_driver():
    global _graphiti
    if _graphiti:
        await _graphiti.close()
        _graphiti = None


async def ensure_user_node(user_id: str, username: str):
    g = await _get_graphiti()
    await g.add_episode(
        name=f"user_init_{user_id}",
        episode_body=f"Personal AI memory graph for user '{username}'. User ID: {user_id}.",
        source=EpisodeType.text,
        source_description="User profile initialisation",
        reference_time=datetime.now(timezone.utc),
        group_id=user_id
    )


async def ensure_session_node(session_id: str, user_id: str):
    pass


async def store_memories_batch(user_id, session_id, memories, source_turn=None):
    g = await _get_graphiti()
    stored_ids = []
    frames = {
        "fact":       "User fact: {}",
        "preference": "User preference: {}",
        "goal":       "User goal: {}",
        "constraint": "User constraint: {}",
    }
    for i, m in enumerate(memories):
        if not m.get("content") or not m.get("memory_type"):
            continue
        mtype = m["memory_type"]
        ckey  = m.get("canonical_key", "unknown")
        name  = f"mem_{user_id}_{ckey}_t{source_turn}_{i}"
        await g.add_episode(
            name=name,
            episode_body=frames.get(mtype, "{}").format(m["content"]),
            source=EpisodeType.text,
            source_description=f"type:{mtype} key:{ckey} turn:{source_turn}",
            reference_time=datetime.now(timezone.utc),
            group_id=user_id
        )
        stored_ids.append(name)
    return stored_ids


async def get_user_memories(user_id, memory_type=None, query=None):
    g = await _get_graphiti()
    if query:
        queries = [query]
    elif memory_type:
        type_q = {
            "fact":       "user name location job age",
            "preference": "user likes prefers enjoys",
            "goal":       "user wants goal achieve",
            "constraint": "user cannot must limit avoid",
        }
        queries = [type_q.get(memory_type, memory_type)]
    else:
        queries = [
            "user name location job age",
            "user preferences likes dislikes",
            "user goals wants to achieve",
            "user constraints cannot must",
        ]

    seen, results = set(), []
    for q in queries:
        for edge in await g.search(query=q, group_ids=[user_id], num_results=8):
            if edge.uuid in seen:
                continue
            if getattr(edge, 'invalid_at', None) or getattr(edge, 'expired_at', None):
                continue
            seen.add(edge.uuid)
            results.append({
                "content":       edge.fact,
                "memory_type":   memory_type or _infer_type(edge.fact),
                "canonical_key": getattr(edge, 'name', ''),
                "valid_at":      edge.valid_at.isoformat() if getattr(edge, 'valid_at', None) else None,
                "is_current":    True,
            })
    return results


def _infer_type(fact):
    f = fact.lower()
    if any(w in f for w in ["prefer","like","love","enjoy","dislike","hate"]):
        return "preference"
    if any(w in f for w in ["want","goal","achieve","learn","build"]):
        return "goal"
    if any(w in f for w in ["cannot","can't","must","budget","limit","avoid"]):
        return "constraint"
    return "fact"


async def get_fact_history(user_id, topic):
    g = await _get_graphiti()
    edges = await g.search(query=topic, group_ids=[user_id], num_results=20)
    history = []
    for edge in edges:
        current = not getattr(edge, 'invalid_at', None) and not getattr(edge, 'expired_at', None)
        history.append({
            "fact":       edge.fact,
            "status":     "current" if current else "expired",
            "valid_at":   edge.valid_at.isoformat() if getattr(edge, 'valid_at', None) else None,
            "invalid_at": edge.invalid_at.isoformat() if getattr(edge, 'invalid_at', None) else None,
        })
    history.sort(key=lambda x: (x["status"] != "current", x["valid_at"] or ""))
    return history


async def get_full_memory_graph(user_id):
    all_memories = await get_user_memories(user_id)
    by_type = {"fact": [], "preference": [], "goal": [], "constraint": []}
    for m in all_memories:
        by_type.setdefault(m.get("memory_type", "fact"), []).append(m)
    return {
        "user_id":          user_id,
        "memories_by_type": by_type,
        "totals":           {t: len(v) for t, v in by_type.items()},
        "total_count":      len(all_memories),
    }