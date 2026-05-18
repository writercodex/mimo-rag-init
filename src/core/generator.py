"""
MiMo-RAG Generator Module
Handles LLM generation using Xiaomi MiMo Open Platform API.
"""

from __future__ import annotations

import time
from typing import Iterator

from openai import OpenAI

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a precise and trustworthy document assistant.
Your task is to answer questions strictly based on the provided context.

Rules:
- Only use information from the provided context.
- If the answer is not in the context, respond with "I cannot find this information in the provided documents."
- Always cite the source document and page number when available.
- Keep answers concise but complete.
- Use markdown formatting for structured responses.
"""


class MiMoGenerator:
    """
    Wrapper around the Xiaomi MiMo API for RAG generation.
    Uses the OpenAI-compatible interface provided by the MiMo Open Platform.
    """

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.MIMO_API_KEY,
            base_url=settings.MIMO_API_BASE_URL,
        )
        self.model = settings.MIMO_MODEL
        logger.info(f"MiMoGenerator initialized with model: {self.model}")

    def _build_prompt(self, question: str, context_chunks: list[dict]) -> str:
        context_text = "\n\n".join(
            f"[Source: {chunk['document']}, Page {chunk.get('page', 'N/A')}]\n{chunk['text']}"
            for chunk in context_chunks
        )
        return f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer (with citations):"

    def generate(self, question: str, context_chunks: list[dict], temperature: float = 0.1, max_tokens: int = 2048) -> dict:
        prompt = self._build_prompt(question, context_chunks)
        t0 = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        answer = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        logger.info(f"Generation complete — tokens={tokens_used}, latency={latency_ms}ms")
        return {"answer": answer, "model": self.model, "tokens_used": tokens_used, "latency_ms": latency_ms}

    def stream(self, question: str, context_chunks: list[dict], temperature: float = 0.1, max_tokens: int = 2048) -> Iterator[str]:
        prompt = self._build_prompt(question, context_chunks)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
