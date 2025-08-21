from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


# ---------- Input Schemas ----------


class UploadDocument(BaseModel):
    id: Optional[str] = None
    filename: str
    mime_type: str
    content_base64: str | None = None
    url: HttpUrl | None = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_source(self) -> "UploadDocument":
        if not self.content_base64 and not self.url:
            raise ValueError("Either content_base64 or url must be provided")
        return self


class IngestRequest(BaseModel):
    documents: List[UploadDocument]
    overwrite: bool = False


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Dict[str, Any] = Field(default_factory=dict)
    language: Optional[str] = None


# ---------- Retrieval & Generation ----------


class RetrievedChunk(BaseModel):
    document: str
    pages: List[int] = Field(default_factory=list)
    chunk_id: str
    excerpt: str
    relevance_score: float
    credibility_score: float
    extraction_method: Literal[
        "dense_retrieval",
        "sparse_retrieval",
        "graph_retrieval",
        "fusion",
    ]


class Answer(BaseModel):
    content: str
    reasoning_steps: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_factors: List[str]


class AlternativeAnswer(BaseModel):
    content: str
    confidence: float
    supporting_citations: List[RetrievedChunk] = Field(default_factory=list)


class ContextAnalysis(BaseModel):
    total_chunks_analyzed: int
    retrieval_methods_used: List[str]
    cross_document_connections: int
    temporal_relevance: Literal["past", "current", "future"]


class PerformanceMetrics(BaseModel):
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_response_time_ms: int
    tokens_processed: int
    cost_estimate_usd: float


class SystemMetadata(BaseModel):
    embedding_model: str
    generation_model: str
    retrieval_strategy: str
    timestamp: str


class RAGResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer: Answer
    citations: List[RetrievedChunk]
    alternative_answers: List[AlternativeAnswer] = Field(default_factory=list)
    context_analysis: ContextAnalysis
    performance_metrics: PerformanceMetrics
    system_metadata: SystemMetadata


class HealthResponse(BaseModel):
    status: str = "ok"
    time: datetime = Field(default_factory=datetime.utcnow)


