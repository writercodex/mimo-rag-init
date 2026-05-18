#!/usr/bin/env python3
"""CLI: ingest documents into MiMo-RAG. Usage: python scripts/ingest.py --input ./docs/"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.ingestion import DocumentIngestionPipeline
from src.core.retriever import VectorRetriever
from src.utils.logger import get_logger
logger = get_logger("ingest")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="./data/index/mimo_rag")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=64)
    args = p.parse_args()
    pipeline = DocumentIngestionPipeline(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = pipeline.load_directory(args.input)
    if not chunks: logger.error("No documents found."); sys.exit(1)
    r = VectorRetriever()
    r.index(chunks)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    r.save(args.output)
    logger.info(f"Done! {len(chunks)} chunks -> {args.output}")

if __name__ == "__main__": main()
