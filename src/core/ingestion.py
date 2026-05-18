"""
Document Ingestion & Chunking Pipeline for MiMo-RAG.
Supports: PDF, DOCX, Markdown, HTML, plain text.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    chunk_id: str
    document: str
    page: int | None
    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


class DocumentIngestionPipeline:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html"}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_directory(self, directory: str | Path) -> List[DocumentChunk]:
        directory = Path(directory)
        all_chunks: list[DocumentChunk] = []
        files = [f for f in directory.rglob("*") if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        logger.info(f"Found {len(files)} documents")
        for file_path in files:
            try:
                all_chunks.extend(self.load_file(file_path))
            except Exception as exc:
                logger.warning(f"Failed: {file_path}: {exc}")
        return all_chunks

    def load_file(self, file_path: str | Path) -> List[DocumentChunk]:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf": return self._load_pdf(file_path)
        elif suffix == ".docx": return self._load_docx(file_path)
        elif suffix in {".md", ".txt"}: return self._load_text(file_path)
        elif suffix == ".html": return self._load_html(file_path)
        else: raise ValueError(f"Unsupported file type: {suffix}")

    def _split_text(self, text: str, document_name: str, page: int | None = None) -> List[DocumentChunk]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_id = hashlib.md5(f"{document_name}_{start}".encode()).hexdigest()[:12]
            chunks.append(DocumentChunk(chunk_id=chunk_id, document=document_name, page=page, text=chunk_text))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _load_pdf(self, path: Path) -> List[DocumentChunk]:
        try:
            import fitz
            doc = fitz.open(str(path))
            chunks = []
            for i, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip(): chunks.extend(self._split_text(text, path.name, page=i))
            doc.close()
            return chunks
        except ImportError:
            return self._load_text(path)

    def _load_docx(self, path: Path) -> List[DocumentChunk]:
        from docx import Document
        doc = Document(str(path))
        return self._split_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()), path.name)

    def _load_text(self, path: Path) -> List[DocumentChunk]:
        return self._split_text(path.read_text(encoding="utf-8", errors="replace"), path.name)

    def _load_html(self, path: Path) -> List[DocumentChunk]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        return self._split_text(soup.get_text(separator="\n"), path.name)
