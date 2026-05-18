from __future__ import annotations
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)
    stream: bool = Field(False)


class SourceRef(BaseModel):
    document: str
    page: int | None
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: float
    tokens_used: int
    latency_ms: int
    model: str


class IngestRequest(BaseModel):
    documents_dir: str


class IngestResponse(BaseModel):
    status: str
    documents_dir: str


class HealthResponse(BaseModel):
    status: str
    model: str
    index_ready: bool
