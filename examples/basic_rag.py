"""Basic RAG example using MiMo-RAG."""
from src.core.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.build_index("./data/documents")

question = "What are the main risk factors in the annual report?"
response = pipeline.query(question, top_k=5)

print(f"Answer ({response.model}):\n{response.answer}")
print(f"Confidence: {response.confidence:.0%} | {response.latency_ms}ms | {response.tokens_used} tokens")
for s in response.sources:
    print(f"  - {s['document']}, page {s.get('page')}")
