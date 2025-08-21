import asyncio
import os
from typing import Any, Dict, List

import aiohttp
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
ADMIN_KEY = os.getenv("RAG_ADMIN_API_KEY")

_last_backend = None
_last_model = None

# Model registry for the UI
MODEL_REGISTRY = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229"
    ],
    "ollama": [
        "llama3.2",
        "llama3.2:1b",
        "llama3.2:3b", 
        "llama3.1",
        "llama3.1:8b",
        "llama3.1:70b",
        "mistral",
        "mixtral", 
        "codellama",
        "gemma2",
        "phi3"
    ]
}

def get_models_for_backend(backend: str) -> List[str]:
    """Get available models for a specific backend"""
    return MODEL_REGISTRY.get(backend, [])


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


async def check_ollama_status() -> bool:
    """Check if Ollama service is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:11434/api/tags", timeout=5) as resp:
                return resp.status == 200
    except:
        return False

async def get_available_ollama_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:11434/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [model["name"] for model in data.get("models", [])]
    except:
        pass
    return MODEL_REGISTRY["ollama"]  # Fallback to predefined list

@cl.on_chat_start
async def start():
    # Check Ollama status and get available models
    ollama_running = await check_ollama_status()
    available_ollama_models = await get_available_ollama_models() if ollama_running else MODEL_REGISTRY["ollama"]
    
    # Create dynamic model registry based on availability
    dynamic_model_registry = MODEL_REGISTRY.copy()
    dynamic_model_registry["ollama"] = available_ollama_models
    
    # Get all models for the initial dropdown
    all_models = []
    for backend, models in dynamic_model_registry.items():
        for model in models:
            all_models.append(f"{backend}:{model}")
    
    settings = cl.ChatSettings(
        [
            Select(
                id="gen_backend",
                label="🔧 Generation Backend",
                values=["openai", "anthropic", "ollama"],
                initial_value="ollama" if ollama_running and available_ollama_models else "openai",
            ),
            Select(
                id="gen_model", 
                label="🤖 Model",
                values=dynamic_model_registry.get("ollama" if ollama_running else "openai", ["llama3.2"]),
                initial_value=available_ollama_models[0] if ollama_running and available_ollama_models else "gpt-4o",
            ),
            Select(
                id="quick_model_select",
                label="⚡ Quick Model Select",
                values=all_models,
                initial_value=f"{'ollama' if ollama_running and available_ollama_models else 'openai'}:{available_ollama_models[0] if ollama_running and available_ollama_models else 'gpt-4o'}",
            ),
            Slider(
                id="top_k",
                label="📊 Retrieval Top K",
                initial=5,
                min=1,
                max=20,
                step=1,
            ),
            Switch(
                id="show_reasoning",
                label="🧠 Show Reasoning Steps",
                initial=True,
            ),
            Switch(
                id="show_citations",
                label="📚 Show All Citations",
                initial=False,
            ),
            Switch(
                id="json_detailed_view",
                label="🔧 Detailed JSON View",
                initial=True,
            ),
        ]
    )
    
    # Initialize defaults in session
    initial_backend = "ollama" if ollama_running and available_ollama_models else "openai"
    initial_model = available_ollama_models[0] if ollama_running and available_ollama_models else "gpt-4o"
    
    cl.user_session.set("gen_backend", initial_backend)
    cl.user_session.set("gen_model", initial_model)
    cl.user_session.set("top_k", 5)
    cl.user_session.set("show_reasoning", True)
    cl.user_session.set("show_citations", False)
    cl.user_session.set("json_detailed_view", True)
    cl.user_session.set("dynamic_model_registry", dynamic_model_registry)

    # Status message
    status_msg = "🤖 **Advanced RAG System** - Production Ready\n\n"
    if ollama_running:
        status_msg += f"✅ **Ollama Status**: Running ({len(available_ollama_models)} models available)\n"
        status_msg += f"🦙 **Current Model**: {initial_model}\n\n"
    else:
        status_msg += "⚠️ **Ollama Status**: Not running (using cloud models)\n\n"
    
    status_msg += (
        "**How to use:**\n"
        "• 🔧 Use the **gear icon** (top-right) to change models in real-time\n"
        "• ⚡ Use **Quick Model Select** for fast switching\n"
        "• 🔧 Toggle **Detailed JSON View** to see complete response structure\n"
        "• 💬 Type `/model <backend> <model>` for command-line switching\n"
        "• 📝 Type `/status` to check system status\n"
        "• 🔄 Type `/refresh` to refresh model list\n\n"
        "**Available Backends:**\n"
        f"• 🌐 OpenAI: {len(MODEL_REGISTRY['openai'])} models\n"
        f"• 🧠 Anthropic: {len(MODEL_REGISTRY['anthropic'])} models\n"
        f"• 🦙 Ollama: {len(available_ollama_models)} models\n\n"
        "**Response Modes:**\n"
        "• 🔧 **Detailed JSON View**: Shows complete response structure with all metadata\n"
        "• 📋 **Compact View**: Shows essential information only\n\n"
        "Ask me anything about your documents!"
    )

    await cl.Message(content=status_msg).send()
    await settings.send()


@cl.on_settings_update
async def settings_update(settings: dict):
    global _last_backend, _last_model
    
    # Handle quick model select
    quick_select = settings.get("quick_model_select")
    if quick_select and ":" in quick_select:
        backend, model = quick_select.split(":", 1)
        settings["gen_backend"] = backend
        settings["gen_model"] = model
    
    # Update backend and model
    backend = settings.get("gen_backend")
    model = settings.get("gen_model")
    
    # Update session variables
    for key, value in settings.items():
        if value is not None:
            cl.user_session.set(key, value)
    
    # Update model dropdown based on backend selection
    if backend:
        dynamic_registry = cl.user_session.get("dynamic_model_registry", MODEL_REGISTRY)
        available_models = dynamic_registry.get(backend, [])
        
        # If current model is not available for the new backend, select the first available
        if model not in available_models and available_models:
            model = available_models[0]
            cl.user_session.set("gen_model", model)
    
    # Perform hot swap if backend or model changed
    if backend and model and (backend != _last_backend or model != _last_model):
        try:
            await hot_swap_generation(backend, model)
            _last_backend, _last_model = backend, model
            
            # Success message with model info
            if backend == "ollama":
                emoji = "🦙"
            elif backend == "openai":
                emoji = "🤖"
            elif backend == "anthropic":
                emoji = "🧠"
            else:
                emoji = "⚙️"
                
            await cl.Message(content=f"{emoji} **Model switched successfully!**\n\n**Backend**: {backend}\n**Model**: {model}").send()
        except Exception as e:
            await cl.Message(content=f"❌ **Failed to swap generation**: {e}\n\nFalling back to previous model.").send()


@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    if not query:
        await cl.Message(content="Please enter a query.").send()
        return

    # Handle slash commands
    if query.startswith("/"):
        await handle_slash_command(query)
        return

    # Ensure settings were applied at least once
    backend = cl.user_session.get("gen_backend")
    model = cl.user_session.get("gen_model")
    top_k = cl.user_session.get("top_k", 5)
    
    global _last_backend, _last_model
    if backend and model and (backend != _last_backend or model != _last_model):
        try:
            await hot_swap_generation(backend, model)
            _last_backend, _last_model = backend, model
        except Exception:
            pass

    # Call backend with dynamic top_k
    try:
        result = await query_backend(query, top_k)
    except Exception as e:  # noqa: BLE001
        await cl.Message(content=f"❌ **Backend error**: {e}").send()
        return

    # Get user preferences
    show_reasoning = cl.user_session.get("show_reasoning", True)
    show_all_citations = cl.user_session.get("show_citations", False)
    json_detailed_view = cl.user_session.get("json_detailed_view", True)
    
    # Get the complete JSON response structure
    query_id = result.get("query_id", "N/A")
    answer = result.get("answer", {})
    citations = result.get("citations", [])
    alternative_answers = result.get("alternative_answers", [])
    context_analysis = result.get("context_analysis", {})
    perf = result.get("performance_metrics", {})
    meta = result.get("system_metadata", {})

    content = answer.get("content", "")
    reasoning_steps = answer.get("reasoning_steps", [])
    conf = answer.get("confidence", 0.0)
    uncertainty_factors = answer.get("uncertainty_factors", [])

    # Handle citations based on user preference
    if show_all_citations:
        display_citations = citations
    else:
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
        display_citations = filtered_citations

    # Build response based on user preference
    backend_emoji = "🦙" if backend == "ollama" else "🤖" if backend == "openai" else "🧠"
    
    if json_detailed_view:
        # Detailed JSON-structured response
        md = f"""# 🎯 RAG Response

## 📋 Query Information
- **Query ID**: `{query_id}`
- **Current Model**: {backend_emoji} **{backend}:{model}**

## 💬 Answer
{content}

---
### 📊 Answer Metadata
- **🎯 Confidence**: {conf:.2f} ({get_confidence_label(conf)})
- **⚠️ Uncertainty Factors**: {', '.join(uncertainty_factors) if uncertainty_factors else 'None'}"""
    else:
        # Compact response (original format)
        md = f"""## 💬 Response

{content}

---
**🎯 Confidence**: {conf:.2f} | **{backend_emoji} Model**: {backend}:{model}"""

    if show_reasoning and reasoning_steps:
        if json_detailed_view:
            md += f"""

## 🧠 Reasoning Steps ({len(reasoning_steps)} steps)
"""
            for i, step in enumerate(reasoning_steps, 1):
                md += f"{i}. {step}\n"
        else:
            md += f"""

## 🧠 Reasoning Steps
"""
            md += "\n".join(f"- {s}" for s in reasoning_steps)

    # Citations section 
    if json_detailed_view:
        # Detailed citations with full JSON structure
        citations_title = f"📚 Citations ({len(display_citations)} shown)" if show_all_citations else f"🏆 Top Citations ({len(display_citations)} shown)"
        md += f"""

## {citations_title}
"""
        
        if display_citations:
            for i, citation in enumerate(display_citations, 1):
                pages_str = f"Pages: {citation.get('pages', [])}" if citation.get('pages') else "Pages: N/A"
                md += f"""
### Citation {i}
- **📄 Document**: {citation.get('document', 'Unknown')}
- **🆔 Chunk ID**: `{citation.get('chunk_id', 'N/A')}`
- **📍 {pages_str}**
- **🎯 Relevance Score**: {citation.get('relevance_score', 0):.3f}
- **🔍 Credibility Score**: {citation.get('credibility_score', 0):.3f}
- **⚙️ Extraction Method**: {citation.get('extraction_method', 'N/A')}
- **📝 Excerpt**: _{citation.get('excerpt', 'No excerpt available')[:200]}..._
"""
        else:
            md += "No citations available."

        # Alternative answers section
        if alternative_answers:
            md += f"""

## 🔄 Alternative Answers ({len(alternative_answers)} available)
"""
            for i, alt in enumerate(alternative_answers, 1):
                alt_confidence = alt.get('confidence', 0)
                supporting_citations = alt.get('supporting_citations', [])
                md += f"""
### Alternative {i}
- **🎯 Confidence**: {alt_confidence:.2f}
- **📚 Supporting Citations**: {len(supporting_citations)}
- **💬 Content**: {alt.get('content', 'No content')[:150]}...
"""

        # Context Analysis section
        md += f"""

## 📈 Context Analysis
- **📊 Total Chunks Analyzed**: {context_analysis.get('total_chunks_analyzed', 0)}
- **🔍 Retrieval Methods Used**: {', '.join(context_analysis.get('retrieval_methods_used', []))}
- **🔗 Cross-Document Connections**: {context_analysis.get('cross_document_connections', 0)}
- **⏰ Temporal Relevance**: {context_analysis.get('temporal_relevance', 'N/A')}"""

        # Performance metrics section
        md += f"""

## ⚡ Performance Metrics
- **🔍 Retrieval Latency**: {perf.get('retrieval_latency_ms', 0)} ms
- **🤖 Generation Latency**: {perf.get('generation_latency_ms', 0)} ms
- **⏱️ Total Response Time**: {perf.get('total_response_time_ms', 0)} ms
- **🔢 Tokens Processed**: {perf.get('tokens_processed', 0):,}
- **💰 Cost Estimate**: ${perf.get('cost_estimate_usd', 0):.4f}"""

        # System metadata section
        md += f"""

## 🔧 System Metadata
- **🧮 Embedding Model**: {meta.get('embedding_model', 'N/A')}
- **🤖 Generation Model**: {meta.get('generation_model', 'N/A')}
- **📋 Retrieval Strategy**: {meta.get('retrieval_strategy', 'N/A')}
- **🕐 Timestamp**: {meta.get('timestamp', 'N/A')}

---
*Complete JSON structure preserved in response*"""
    else:
        # Compact citations (original format)
        citations_title = "📚 All Citations" if show_all_citations else "🏆 Top Citation(s)"
        md += f"""

## {citations_title}
{format_citations(display_citations)}

---
## ⚡ Performance Metrics
- **Retrieval**: {perf.get('retrieval_latency_ms','-')} ms
- **Generation**: {perf.get('generation_latency_ms','-')} ms
- **Total**: {perf.get('total_response_time_ms','-')} ms
- **Tokens**: {perf.get('tokens_processed','-')}
- **Cost**: ${perf.get('cost_estimate_usd','-')}

## 🔧 System Info
- **Embeddings**: {meta.get('embedding_model','-')}
- **Generation**: {meta.get('generation_model','-')}
- **Strategy**: {meta.get('retrieval_strategy','-')}
- **Timestamp**: {meta.get('timestamp','-')}"""

    await cl.Message(content=md).send()


def get_confidence_label(confidence: float) -> str:
    """Get a descriptive label for confidence score."""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High" 
    elif confidence >= 0.6:
        return "Moderate"
    elif confidence >= 0.4:
        return "Low"
    else:
        return "Very Low"


async def handle_slash_command(query: str):
    """Handle various slash commands"""
    parts = query.split()
    command = parts[0].lower()
    
    if command == "/model":
        if len(parts) >= 3:
            backend = parts[1]
            model = " ".join(parts[2:])
            try:
                await hot_swap_generation(backend, model)
                global _last_backend, _last_model
                _last_backend, _last_model = backend, model
                
                # Update session
                cl.user_session.set("gen_backend", backend)
                cl.user_session.set("gen_model", model)
                
                emoji = "🦙" if backend == "ollama" else "🤖" if backend == "openai" else "🧠"
                await cl.Message(content=f"{emoji} **Model switched via command!**\n\n**Backend**: {backend}\n**Model**: {model}").send()
            except Exception as e:
                await cl.Message(content=f"❌ **Failed to swap generation**: {e}").send()
        else:
            await cl.Message(content="**Usage**: `/model <backend> <model>`\n\n**Examples**:\n• `/model ollama llama3.2`\n• `/model openai gpt-4o`\n• `/model anthropic claude-3-5-sonnet-20241022`").send()
    
    elif command == "/status":
        backend = cl.user_session.get("gen_backend", "unknown")
        model = cl.user_session.get("gen_model", "unknown")
        top_k = cl.user_session.get("top_k", 5)
        
        # Check Ollama status
        ollama_running = await check_ollama_status()
        available_ollama_models = await get_available_ollama_models() if ollama_running else []
        
        status_msg = "📊 **System Status**\n\n"
        status_msg += f"**Current Model**: {backend}:{model}\n"
        status_msg += f"**Retrieval Top K**: {top_k}\n"
        status_msg += f"**Ollama Status**: {'✅ Running' if ollama_running else '❌ Not running'}\n"
        
        if ollama_running:
            status_msg += f"**Ollama Models**: {len(available_ollama_models)} available\n"
            if available_ollama_models:
                status_msg += f"**Available**: {', '.join(available_ollama_models[:5])}"
                if len(available_ollama_models) > 5:
                    status_msg += f" (+{len(available_ollama_models) - 5} more)"
                status_msg += "\n"
        
        status_msg += f"\n**API Base**: {API_BASE}"
        await cl.Message(content=status_msg).send()
    
    elif command == "/refresh":
        # Refresh model list
        ollama_running = await check_ollama_status()
        available_ollama_models = await get_available_ollama_models() if ollama_running else MODEL_REGISTRY["ollama"]
        
        # Update dynamic registry
        dynamic_model_registry = MODEL_REGISTRY.copy()
        dynamic_model_registry["ollama"] = available_ollama_models
        cl.user_session.set("dynamic_model_registry", dynamic_model_registry)
        
        refresh_msg = "🔄 **Model list refreshed!**\n\n"
        refresh_msg += f"**Ollama Status**: {'✅ Running' if ollama_running else '❌ Not running'}\n"
        if ollama_running:
            refresh_msg += f"**Available Models**: {len(available_ollama_models)}\n"
            if available_ollama_models:
                refresh_msg += f"**Models**: {', '.join(available_ollama_models)}"
        
        await cl.Message(content=refresh_msg).send()
    
    elif command == "/help":
        help_msg = """🆘 **Available Commands**

**Model Management:**
• `/model <backend> <model>` - Switch model
• `/status` - Show system status  
• `/refresh` - Refresh model list

**Examples:**
• `/model ollama llama3.2` - Switch to Llama 3.2
• `/model openai gpt-4o` - Switch to GPT-4o
• `/model anthropic claude-3-5-sonnet-20241022` - Switch to Claude

**Available Backends:**
• 🦙 **ollama** - Local Llama models
• 🤖 **openai** - GPT models  
• 🧠 **anthropic** - Claude models

Use the gear icon ⚙️ for GUI model selection!"""
        await cl.Message(content=help_msg).send()
    
    else:
        await cl.Message(content=f"❓ **Unknown command**: {command}\n\nType `/help` for available commands.").send()


