# Deterministic Runtime

CodeGraph RAG separates predictable retrieval from less predictable LLM
generation. The retrieval layer is designed to be reproducible when the same
workspace, index, config, embedding model, query, and context are used.

## What Is Deterministic

The following retrieval decisions are fixed by the engine:

- Two-mode routing:
  - Code/Symbol and Function Memory: Graph 0.90 / Vector 0.05 / BM25 0.05
  - NL/Issue: Graph 0.30 / Vector 0.50 / BM25 0.20
- RRF `rrf_k`.
- Context routing mode.
- P1 reranker settings.
- Stable tie-breaker for equal-scored candidates.
- Config hash and index snapshot id in explain traces.

## Stable Tie-Breaker

When candidates have the same score, CodeGraph RAG orders them by:

1. Score, descending.
2. File path, normalized and lower-cased.
3. Line number.
4. Qualified symbol or symbol name.
5. Chunk id.

This avoids accidental ordering drift from Python dictionary/list insertion
order, channel arrival order, or same-score RRF collisions.

Config:

```yaml
codegraph:
  search:
    deterministic_sort: true
```

## Explain Trace

Use `search_explain()` to inspect routing and fusion decisions:

```python
from codegraph_rag import CodeGraphRAG

rag = CodeGraphRAG("C:/dev/project")
payload = rag.search_explain(
    "find the function that validates login",
    top_k=5,
    active_file="src/auth/service.py",
    language="python",
)

print(payload["trace"])
print(payload["results"][0]["channel_ranks"])
```

CLI:

```bash
codegraph-rag search C:/dev/project "find login validator" --language python --explain
```

Trace fields include:

- `route`
- `query_type`
- `weights`
- `rrf_k`
- `rerank_enabled`
- `context_routing_mode`
- `routed_language`
- `embedding_provider`
- `embedding_model`
- `config_hash`
- `index_snapshot_id`
- `channel_counts`

Each result also includes:

- final rank
- channel ranks
- stable sort key

## Index Snapshot Id

`index_snapshot_id` is derived from the persisted SQLite index path, file size,
and modification timestamp. It is lightweight and intended for runtime
diagnostics, not cryptographic content verification.

If the index changes after incremental indexing, the snapshot id changes. A host
application can log this value per LLM request to explain why retrieved context
changed between runs.

## Product Guidance

For an AI coding tool:

- Bind each LLM request to one retrieval trace.
- Log `config_hash`, `index_snapshot_id`, route, weights, and Top-K result ids.
- Use `temperature=0` and fixed prompt templates on the LLM side.
- Treat retrieval metrics and generation/repair metrics separately.
- Avoid querying while an index update is half-applied; prefer a long-lived
  service that serializes index writes and search reads.

## Current Boundary

CodeGraph RAG now guarantees stable ranking for equal scores and exposes enough
trace data to reproduce a retrieval decision. It does not guarantee identical
LLM output, because remote models, prompts, and generation backends can change
outside the retrieval engine.
