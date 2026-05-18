"""
End-to-End RAG Pipeline Orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.core.generator import MiMoGenerator
from src.core.ingestion import DocumentIngestionPipeline
from src.core.retriever import VectorRetriever
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    confidence: float
    tokens_used: int
    latency_ms: int
    model: str


class RAGPipeline:
    def __init__(self) -> None:
        self.ingestion = DocumentIngestionPipeline(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        self.retriever = VectorRetriever()
        self.generator = MiMoGenerator()
        self._ready = False

    def build_index(self, documents_dir: str | Path) -> None:
        chunks = self.ingestion.load_directory(documents_dir)
        if not chunks: raise ValueError(f"No documents found in: {documents_dir}")
        self.retriever.index(chunks)
        self._ready = True

    def load_index(self, index_path: str) -> None:
        self.retriever.load(index_path)
        self._ready = True

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        if not self._ready: raise RuntimeError("Pipeline not ready.")
        chunks = self.retriever.search(question, top_k=top_k or settings.MAX_RETRIEVED_CHUNKS)
        if not chunks:
            return RAGResponse(answer="I cannot find relevant information.", sources=[], confidence=0.0, tokens_used=0, latency_ms=0, model=settings.MIMO_MODEL)
        result = self.generator.generate(question, chunks)
        return RAGResponse(
            answer=result["answer"],
            sources=[{"document": c["document"], "page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks],
            confidence=round(min(sum(c["score"] for c in chunks) / len(chunks), 1.0), 4),
            tokens_used=result["tokens_used"], latency_ms=result["latency_ms"], model=result["model"],
        )

    def stream_query(self, question: str, top_k: int | None = None) -> Iterator[str]:
        if not self._ready: raise RuntimeError("Pipeline not ready.")
        chunks = self.retriever.search(question, top_k=top_k or settings.MAX_RETRIEVED_CHUNKS)
        yield from self.generator.stream(question, chunks)
