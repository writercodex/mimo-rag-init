# Contributing to MiMo-RAG

## Setup
```bash
git clone https://github.com/writercodex/mimo-rag-init.git
cd mimo-rag-init && pip install -r requirements.txt
```

## Code Style
- `black src/` for formatting
- `ruff check src/` for linting

## Tests
```bash
pytest tests/ -v
```

## PR Guide
Branch `feat/your-feature` -> commit with `feat:` prefix -> PR to `main`
