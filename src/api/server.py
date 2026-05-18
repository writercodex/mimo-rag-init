from __future__ import annotations

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse, HealthResponse
from src.core.pipeline import RAGPipeline
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="MiMo-RAG API", description="Intelligent Document QA powered by Xiaomi MiMo API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipeline = RAGPipeline()


@app.on_event("startup")
async def startup_event():
    try:
        pipeline.load_index(settings.INDEX_PATH)
    except Exception:
        logger.warning("No pre-built index found.")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(status="ok", model=settings.MIMO_MODEL, index_ready=pipeline._ready)


@app.post("/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    if not pipeline._ready:
        raise HTTPException(status_code=503, detail="Index not ready.")
    if request.stream:
        async def generate():
            for token in pipeline.stream_query(request.question, top_k=request.top_k):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    r = pipeline.query(request.question, top_k=request.top_k)
    return QueryResponse(answer=r.answer, sources=r.sources, confidence=r.confidence,
                         tokens_used=r.tokens_used, latency_ms=r.latency_ms, model=r.model)


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(pipeline.build_index, request.documents_dir)
    return IngestResponse(status="ingestion_started", documents_dir=request.documents_dir)
