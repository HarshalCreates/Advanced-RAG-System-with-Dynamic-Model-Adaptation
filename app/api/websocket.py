from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.pipeline.manager import PipelineManager


ws_router = APIRouter()


@ws_router.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    manager = PipelineManager.get_instance()
    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)
            query_text = payload.get("query", "")
            top_k = int(payload.get("top_k", 5))
            t0 = time.time()
            async for chunk in manager.stream(query_text=query_text, top_k=top_k):
                await websocket.send_text(json.dumps(chunk))
            await websocket.send_text(
                json.dumps({"event": "complete", "latency_ms": int((time.time() - t0) * 1000)})
            )
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_text(json.dumps({"event": "error", "detail": str(e)}))
        await asyncio.sleep(0)


