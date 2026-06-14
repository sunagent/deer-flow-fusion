# Contributing

Thanks for helping improve CodeGraph RAG.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

For local Jina embeddings:

```bash
pip install -e .
```

The first local embedding run downloads the configured model into the local
HuggingFace cache.

## Checks

```bash
python -m compileall src
python -m pytest
```

Before opening a PR, also run:

```bash
rg "sk-[A-Za-z0-9]{20,}" -n --hidden .
```

Do not commit local indexes, benchmark datasets, model caches, `.env`, or API
keys.

## Architecture Boundary

Keep the core package independent from host applications:

- No DeerFlow dependency in `src/codegraph_rag`.
- No LangChain/LangGraph dependency in the core package.
- Host integrations belong under `integrations/`.
- Retrieval weights are owned by CodeGraph RAG, not by clients.
