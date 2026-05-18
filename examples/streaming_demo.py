"""Streaming demo — real-time tokens from Xiaomi MiMo API."""
from src.core.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load_index("./data/index/mimo_rag")

print("MiMo: ", end="", flush=True)
for token in pipeline.stream_query("Summarize the key findings."):
    print(token, end="", flush=True)
print()
