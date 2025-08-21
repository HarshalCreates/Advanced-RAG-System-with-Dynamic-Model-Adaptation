from __future__ import annotations

import time
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    HealthResponse,
    IngestRequest,
    QueryRequest,
    RAGResponse,
    Answer,
    RetrievedChunk,
    ContextAnalysis,
    PerformanceMetrics,
    SystemMetadata,
)
from fastapi import UploadFile, File, Form
from app.pipeline.ingestion import IngestionPipeline
from app.pipeline.manager import PipelineManager


api_router = APIRouter()


def get_ingestion() -> IngestionPipeline:
    return PipelineManager.get_ingestion()


def get_pipeline_manager() -> PipelineManager:
    return PipelineManager.get_instance()


@api_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@api_router.post("/ingest")
def ingest(payload: IngestRequest, ingestion: IngestionPipeline = Depends(get_ingestion)) -> Dict[str, int]:
    count = ingestion.process(payload.documents, overwrite=payload.overwrite)
    return {"ingested": count}


@api_router.post("/ingest/upload")
async def ingest_upload(
    files: List[UploadFile] = File(...),
    overwrite: bool = Form(False),
    ingestion: IngestionPipeline = Depends(get_ingestion),
) -> Dict[str, int]:
    from app.models.schemas import UploadDocument
    import base64

    docs: List[UploadDocument] = []
    for f in files:
        content = await f.read()
        docs.append(
            UploadDocument(
                filename=f.filename,
                mime_type=f.content_type or "application/octet-stream",
                content_base64=base64.b64encode(content).decode("utf-8"),
            )
        )
    count = ingestion.process(docs, overwrite=overwrite)
    return {"ingested": count}


@api_router.post("/ingest/directory")
def ingest_directory(path: str = Form(...), ingestion: IngestionPipeline = Depends(get_ingestion)) -> Dict[str, int]:
    count = ingestion.ingest_from_directory(path)
    # return index size hint
    from app.retrieval.service import HybridRetrievalService
    size = len(ingestion.retriever.documents.texts)
    return {"ingested": count, "index_size": size}


@api_router.post("/query", response_model=RAGResponse)
def query(payload: QueryRequest, manager: PipelineManager = Depends(get_pipeline_manager)) -> RAGResponse:
    try:
        t0 = time.time()
        result = manager.query(payload)
        total_ms = int((time.time() - t0) * 1000)
        result.performance_metrics.total_response_time_ms = total_ms
        return result
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"query_failed: {e}")


