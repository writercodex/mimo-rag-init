"""
Vector Store & Retrieval Module for MiMo-RAG.
FAISS-backed semantic search with SentenceTransformers.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.ingestion import DocumentChunk
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    def __init__(self) -> None:
        self.embed_model = SentenceTransformer(settings.EMBED_MODEL)
        self._index = None
        self._chunks: list[DocumentChunk] = []

    def index(self, chunks: list[DocumentChunk]) -> None:
        import faiss
        texts = [c.text for c in chunks]
        embeddings = np.array(self.embed_model.encode(texts, show_progress_bar=True, batch_size=64), dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._chunks = chunks
        logger.info(f"Index built — {self._index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._index is None: raise RuntimeError("Index not built.")
        import faiss
        qv = np.array(self.embed_model.encode([query], normalize_embeddings=True), dtype=np.float32)
        scores, indices = self._index.search(qv, top_k)
        return [
            {"chunk_id": self._chunks[i].chunk_id, "document": self._chunks[i].document,
             "page": self._chunks[i].page, "text": self._chunks[i].text, "score": float(s)}
            for s, i in zip(scores[0], indices[0]) if i != -1
        ]

    def save(self, path: str) -> None:
        import faiss, pickle
        faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.chunks.pkl", "wb") as f: pickle.dump(self._chunks, f)

    def load(self, path: str) -> None:
        import faiss, pickle
        self._index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.chunks.pkl", "rb") as f: self._chunks = pickle.load(f)
        logger.info(f"Index loaded — {self._index.ntotal} vectors")
