import asyncio
import os
from typing import Any, Dict, List

import aiohttp
import chainlit as cl
from chainlit.input_widget import Select

API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
ADMIN_KEY = os.getenv("RAG_ADMIN_API_KEY")

_last_backend = None
_last_model = None


async def query_backend(query: str, top_k: int = 5) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_BASE}/api/query",
            json={"query": query, "top_k": top_k, "filters": {}},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


async def hot_swap_generation(backend: str, model: str) -> Dict[str, Any]:
    headers = {"X-API-Key": ADMIN_KEY} if ADMIN_KEY else {}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_BASE}/api/admin/hot-swap/generation",
            params={"backend": backend, "model": model},
            headers=headers,
            timeout=30,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


def format_citations(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return "No citations."
    lines = []
    for c in citations:
        pages = ",".join(str(p) for p in c.get("pages", [])) or "-"
        lines.append(
            f"- {c.get('document','?')} (pages: {pages}) | score={c.get('relevance_score',0):.2f}"
        )
    return "\n".join(lines)


@cl.on_chat_start
async def start():
    settings = cl.ChatSettings(
        [
            Select(
                id="gen_backend",
                label="Generation Backend",
                values=["openai", "anthropic", "ollama"],
                initial_value="openai",
            ),
            Select(
                id="gen_model",
                label="Model",
                values=["gpt-4o", "claude-3-5-sonnet-20240620", "llama3"],
                initial_value="gpt-4o",
            ),
        ]
    )
    # initialize defaults in session for immediate availability
    cl.user_session.set("gen_backend", "openai")
    cl.user_session.set("gen_model", "gpt-4o")

    await cl.Message(
        content=(
            "Ask me anything about your ingested documents.\n"
            "Use the gear icon (top-right) to open Settings and switch models in real time.\n"
            "Alternatively, type /model <backend> <model> (e.g., /model openai gpt-4o)."
        )
    ).send()
    await settings.send()


@cl.on_settings_update
async def settings_update(settings: dict):
    global _last_backend, _last_model
    backend = settings.get("gen_backend")
    model = settings.get("gen_model")
    if backend:
        cl.user_session.set("gen_backend", backend)
    if model:
        cl.user_session.set("gen_model", model)
    if backend and model and (backend != _last_backend or model != _last_model):
        try:
            await hot_swap_generation(backend, model)
            _last_backend, _last_model = backend, model
            await cl.Message(content=f"Switched generation to {backend}:{model}").send()
        except Exception as e:  # noqa: BLE001
            await cl.Message(content=f"Failed to swap generation: {e}").send()


@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    if not query:
        await cl.Message(content="Please enter a query.").send()
        return

    # Slash command to swap model if settings panel is not visible
    if query.startswith("/model "):
        parts = query.split()
        if len(parts) >= 3:
            backend = parts[1]
            model = " ".join(parts[2:])
            try:
                await hot_swap_generation(backend, model)
                global _last_backend, _last_model
                _last_backend, _last_model = backend, model
                await cl.Message(content=f"Switched generation to {backend}:{model}").send()
            except Exception as e:  # noqa: BLE001
                await cl.Message(content=f"Failed to swap generation: {e}").send()
        else:
            await cl.Message(content="Usage: /model <backend> <model>").send()
        return

    # Ensure settings were applied at least once
    backend = cl.user_session.get("gen_backend")
    model = cl.user_session.get("gen_model")
    if backend and model and (backend != _last_backend or model != _last_model):
        try:
            await hot_swap_generation(backend, model)
            _last_backend, _last_model = backend, model
        except Exception:
            pass

    # Call backend
    try:
        result = await query_backend(query)
    except Exception as e:  # noqa: BLE001
        await cl.Message(content=f"Backend error: {e}").send()
        return

    # Render schema-compliant result
    answer = result.get("answer", {})
    citations = result.get("citations", [])
    perf = result.get("performance_metrics", {})
    meta = result.get("system_metadata", {})

    content = answer.get("content", "")
    reasoning = "\n".join(f"- {s}" for s in answer.get("reasoning_steps", []))
    conf = answer.get("confidence", 0.0)

    # Show only the citation(s) with the greatest relevance score
    filtered_citations: List[Dict[str, Any]] = []
    if citations:
        try:
            max_score = max(float(c.get("relevance_score", 0.0)) for c in citations)
            # include ties at max score
            filtered_citations = [
                c for c in citations if float(c.get("relevance_score", 0.0)) == max_score
            ]
            # sort deterministically
            filtered_citations.sort(key=lambda x: x.get("document", ""))
        except Exception:
            filtered_citations = citations[:1]

    md = f"""
### Answer
{content}

---
**Confidence**: {conf:.2f}

### Reasoning
{reasoning or 'N/A'}

### Top Citation(s)
{format_citations(filtered_citations)}

---
### Performance
- Retrieval latency: {perf.get('retrieval_latency_ms','-')} ms
- Generation latency: {perf.get('generation_latency_ms','-')} ms
- Total time: {perf.get('total_response_time_ms','-')} ms
- Tokens: {perf.get('tokens_processed','-')}
- Cost: ${perf.get('cost_estimate_usd','-')}

### System
- Embeddings: {meta.get('embedding_model','-')}
- Generation: {meta.get('generation_model','-')}
- Strategy: {meta.get('retrieval_strategy','-')}
- Time: {meta.get('timestamp','-')}
"""
    await cl.Message(content=md).send()


