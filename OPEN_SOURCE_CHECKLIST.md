# Open-Source Checklist

## Repository Boundary

- [x] Core RAG engine extracted to `src/codegraph_rag`.
- [x] DeerFlow/LangChain tool adapter excluded from the core package.
- [x] README states that this is the standalone RAG engine, not the full AI
  programming app.
- [x] Benchmark claims are retrieval-only unless explicitly marked otherwise.
- [x] Deterministic runtime behavior is documented.
- [x] `search_explain()` exposes routing/fusion metadata for reproducibility.

## Safety

- [x] No hardcoded `sk-...` API key in this standalone repository.
- [x] `.env`, SQLite indexes, local model caches, and benchmark datasets ignored.
- [x] DeepSeek/SWE-bench generation claims are not marketed as retrieval results.

## Before Publishing

- [ ] Run `python -m compileall src`.
- [ ] Run `rg "sk-[A-Za-z0-9]{20,}" -n --hidden .`.
- [ ] Run a tiny local index/search smoke test.
- [ ] Create a clean GitHub repository and push only the standalone files.
- [ ] Add CI for compile/import smoke tests.

## Suggested GitHub Description

Standalone local code RAG engine with Graph + Vector + BM25/FTS5 RRF fusion,
CallGraph expansion, local Jina code embeddings, and IDE-aware context routing.
