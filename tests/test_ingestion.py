import pytest
from src.core.ingestion import DocumentIngestionPipeline, DocumentChunk

@pytest.fixture
def pipeline():
    return DocumentIngestionPipeline(chunk_size=100, chunk_overlap=20)

@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text(" ".join([f"word{i}" for i in range(300)]))
    return f

def test_load_text_file(pipeline, sample_txt):
    chunks = pipeline.load_file(sample_txt)
    assert len(chunks) > 0 and all(isinstance(c, DocumentChunk) for c in chunks)

def test_chunk_size(pipeline, sample_txt):
    for chunk in pipeline.load_file(sample_txt)[:-1]:
        assert len(chunk.text.split()) <= pipeline.chunk_size

def test_chunk_ids_unique(pipeline, sample_txt):
    ids = [c.chunk_id for c in pipeline.load_file(sample_txt)]
    assert len(ids) == len(set(ids))

def test_document_name_preserved(pipeline, sample_txt):
    assert all(c.document == sample_txt.name for c in pipeline.load_file(sample_txt))

def test_unsupported_raises(pipeline, tmp_path):
    f = tmp_path / "data.xyz"; f.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"): pipeline.load_file(f)

def test_load_directory(pipeline, tmp_path):
    for i in range(3): (tmp_path / f"doc_{i}.txt").write_text(" ".join([f"tok{j}" for j in range(200)]))
    assert len(pipeline.load_directory(tmp_path)) > 0

def test_empty_directory(pipeline, tmp_path):
    assert pipeline.load_directory(tmp_path) == []
