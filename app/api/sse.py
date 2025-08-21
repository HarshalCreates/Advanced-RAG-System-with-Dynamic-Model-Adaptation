"""Server-Sent Events (SSE) endpoint for progressive answer building."""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.schemas import QueryRequest
from app.pipeline.manager import PipelineManager
from app.api.query_manager import QueryManager


sse_router = APIRouter()

# Global query manager instance
_query_manager: QueryManager | None = None


def get_query_manager() -> QueryManager:
    """Get or create the global query manager."""
    global _query_manager
    if _query_manager is None:
        _query_manager = QueryManager()
        # Start in background (will be handled by FastAPI lifespan in production)
        asyncio.create_task(_query_manager.start())
    return _query_manager


class SSEQueryRequest(BaseModel):
    """SSE Query request model."""
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = {}
    user_id: str | None = None
    priority: int = 1


async def create_sse_response(query_id: str, pipeline_manager: PipelineManager, 
                            query_request: QueryRequest) -> AsyncGenerator[str, None]:
    """Create SSE response stream for a query."""
    
    def format_sse_data(event_type: str, data: Dict[str, Any]) -> str:
        """Format data as SSE message."""
        return f"event: {event_type}\\ndata: {json.dumps(data)}\\n\\n"
    
    try:
        # Send initial event
        yield format_sse_data("start", {
            "query_id": query_id,
            "message": "Query processing started",
            "timestamp": time.time()
        })
        
        # Process query with streaming
        start_time = time.time()
        
        # Step 1: Retrieval phase
        yield format_sse_data("phase", {
            "phase": "retrieval",
            "message": "Searching for relevant documents...",
            "progress": 0.1
        })
        
        # Get retrieval results
        retrieval_start = time.time()
        fused = pipeline_manager.retriever.search(
            query_request.query, 
            top_k=query_request.top_k, 
            filters=query_request.filters
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        yield format_sse_data("retrieval_complete", {
            "documents_found": len(fused),
            "retrieval_time_ms": retrieval_time,
            "progress": 0.4
        })
        
        # Step 2: Context preparation
        yield format_sse_data("phase", {
            "phase": "context_preparation", 
            "message": "Preparing context for generation...",
            "progress": 0.5
        })
        
        # Prepare context (simplified version of pipeline logic)
        context_texts = []
        for doc_id, score in fused[:3]:  # Top 3 for context
            text = pipeline_manager.retriever.get_text(doc_id)
            if text:
                context_texts.append(text[:500])  # Truncate for context
        
        yield format_sse_data("context_ready", {
            "context_chunks": len(context_texts),
            "progress": 0.6
        })
        
        # Step 3: Generation phase
        yield format_sse_data("phase", {
            "phase": "generation",
            "message": "Generating response...",
            "progress": 0.7
        })
        
        # Stream generation if supported
        generation_start = time.time()
        
        # Try to get streaming generation
        try:
            accumulated_response = ""
            async for chunk in pipeline_manager.stream(query_request.query, query_request.top_k):
                if chunk.get("event") == "token":
                    accumulated_response += chunk.get("content", "")
                    yield format_sse_data("token", {
                        "content": chunk.get("content", ""),
                        "accumulated": accumulated_response,
                        "progress": min(0.9, 0.7 + (len(accumulated_response) / 1000) * 0.2)
                    })
                elif chunk.get("event") == "complete":
                    break
                    
        except Exception:
            # Fallback to regular query
            result = pipeline_manager.query(query_request)
            yield format_sse_data("token", {
                "content": result.answer.content,
                "accumulated": result.answer.content,
                "progress": 0.9
            })
            accumulated_response = result.answer.content
        
        generation_time = (time.time() - generation_start) * 1000
        
        # Step 4: Finalization
        yield format_sse_data("phase", {
            "phase": "finalization",
            "message": "Finalizing response...",
            "progress": 0.95
        })
        
        # Get final result for metadata
        final_result = pipeline_manager.query(query_request)
        
        # Send completion event
        total_time = (time.time() - start_time) * 1000
        
        yield format_sse_data("complete", {
            "query_id": query_id,
            "response": accumulated_response,
            "citations": [
                {
                    "document": citation.document,
                    "excerpt": citation.excerpt[:200],
                    "relevance_score": citation.relevance_score
                }
                for citation in final_result.citations
            ],
            "confidence": final_result.answer.confidence,
            "performance": {
                "total_time_ms": total_time,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time
            },
            "progress": 1.0
        })
        
    except asyncio.CancelledError:
        yield format_sse_data("cancelled", {
            "query_id": query_id,
            "message": "Query was cancelled",
            "timestamp": time.time()
        })
        
    except Exception as e:
        yield format_sse_data("error", {
            "query_id": query_id,
            "error": str(e),
            "timestamp": time.time()
        })


@sse_router.post("/stream")
async def stream_query_sse(
    request: SSEQueryRequest,
    http_request: Request,
    pipeline_manager: PipelineManager = Depends(lambda: PipelineManager.get_instance()),
    query_manager: QueryManager = Depends(get_query_manager)
):
    """Stream query response using Server-Sent Events."""
    
    # Convert to internal query request
    query_request = QueryRequest(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters
    )
    
    # Submit query to query manager for resource management
    async def execute_query():
        return await create_sse_response(
            query_id="sse_" + str(int(time.time())),
            pipeline_manager=pipeline_manager,
            query_request=query_request
        )
    
    query_id = await query_manager.submit_query(
        query_text=request.query,
        query_function=execute_query,
        user_id=request.user_id,
        priority=request.priority,
        estimated_resources={"memory_mb": 150.0, "cpu_percent": 15.0}
    )
    
    # Check if query was queued or rejected
    query_status = await query_manager.get_query_status(query_id)
    if query_status and query_status.status.value in ["failed"]:
        raise HTTPException(
            status_code=503, 
            detail=f"Query rejected: {query_status.cancellation_reason}"
        )
    
    async def event_stream():
        """Generate SSE event stream."""
        # Wait for query to start processing
        max_wait = 30  # 30 seconds max wait
        wait_count = 0
        
        while wait_count < max_wait:
            query_status = await query_manager.get_query_status(query_id)
            if query_status and query_status.status.value == "running":
                break
            elif query_status and query_status.status.value in ["failed", "cancelled"]:
                yield f"event: error\\ndata: {json.dumps({'error': query_status.cancellation_reason})}\\n\\n"
                return
            
            await asyncio.sleep(1)
            wait_count += 1
        
        # If still not running, return timeout
        if wait_count >= max_wait:
            yield f"event: error\\ndata: {json.dumps({'error': 'Query timeout in queue'})}\\n\\n"
            return
        
        # Stream the actual query results
        try:
            async for event in await execute_query():
                # Check for client disconnect
                if await http_request.is_disconnected():
                    await query_manager.cancel_query(query_id, "Client disconnected")
                    break
                
                yield event
                
        except asyncio.CancelledError:
            await query_manager.cancel_query(query_id, "Stream cancelled")
        except Exception as e:
            yield f"event: error\\ndata: {json.dumps({'error': str(e)})}\\n\\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@sse_router.get("/query/{query_id}/status")
async def get_query_status_sse(
    query_id: str,
    query_manager: QueryManager = Depends(get_query_manager)
):
    """Get current status of a query."""
    
    status = await query_manager.get_query_status(query_id)
    if not status:
        raise HTTPException(status_code=404, detail="Query not found")
    
    return {
        "query_id": status.query_id,
        "status": status.status.value,
        "start_time": status.start_time,
        "end_time": status.end_time,
        "duration_seconds": (status.end_time - status.start_time) if status.end_time else None,
        "cancellation_reason": status.cancellation_reason,
        "user_id": status.user_id
    }


@sse_router.post("/query/{query_id}/cancel")
async def cancel_query_sse(
    query_id: str,
    query_manager: QueryManager = Depends(get_query_manager)
):
    """Cancel a running or queued query."""
    
    success = await query_manager.cancel_query(query_id, "User requested cancellation")
    
    if not success:
        raise HTTPException(status_code=404, detail="Query not found or already completed")
    
    return {"message": "Query cancelled successfully", "query_id": query_id}


@sse_router.get("/system/stats")
async def get_system_stats(
    query_manager: QueryManager = Depends(get_query_manager)
):
    """Get current system resource usage and query statistics."""
    
    return query_manager.get_system_stats()


@sse_router.get("/queries/history")
async def get_query_history(
    user_id: str | None = None,
    limit: int = 50,
    query_manager: QueryManager = Depends(get_query_manager)
):
    """Get query history for a user or all users."""
    
    history = query_manager.get_query_history(user_id=user_id, limit=limit)
    
    return {
        "queries": [
            {
                "query_id": q.query_id,
                "user_id": q.user_id,
                "query_text": q.query_text[:100] + "..." if len(q.query_text) > 100 else q.query_text,
                "status": q.status.value,
                "start_time": q.start_time,
                "end_time": q.end_time,
                "duration_seconds": (q.end_time - q.start_time) if q.end_time else None,
                "cancellation_reason": q.cancellation_reason
            }
            for q in history
        ]
    }
