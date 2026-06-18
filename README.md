# CodeGraph RAG

Standalone local code RAG engine for AI coding tools.

CodeGraph RAG is built for AI coding tools that need reliable local-project
retrieval: finding functions, signatures, related callers/callees, code context,
and issue-relevant files without sending source code to a remote embedding API.

## What It Is

- Local code indexing for Python, TypeScript/JavaScript, Go, Rust, and Java.
- Three-mode retrieval: Graph + local Jina vector + BM25/FTS5.
- Weighted RRF fusion with a frozen two-mode router.
- CallGraph expansion for symbol and function-memory queries.
- `search_document` dual-vector recall for natural-language descriptions.
- Lightweight P1 reranker.
- Deterministic tie-breaker and explain traces for reproducible retrieval.
- Optional INT8 vector storage and function-level incremental embedding cache.

This repository is the RAG engine only. It does not include any host IDE,
frontend, agent runtime, sandbox, or orchestration system.

## Frozen Retrieval Policy

| Mode | Query Class | Graph | Vector | BM25/FTS5 |
|---|---|---:|---:|---:|
| Code/Symbol and Function Memory | code snippets, function names, signatures, purpose-to-function lookup | 0.90 | 0.05 | 0.05 |
| NL/Issue | strong bug reports, issue descriptions, traceback-heavy queries | 0.30 | 0.50 | 0.20 |

RRF uses `rrf_k = 60`. Clients pass product facts such as `active_file`,
`language`, `repo`, and `workspace`; they do not pass retrieval weights.

## Install

```bash
pip install -e .
```

The default local embedding model is:

```text
jinaai/jina-embeddings-v2-base-code
```

It is downloaded by `sentence-transformers` on first use and then loaded from
the local HuggingFace cache. No embedding API key is required.

## Quick Start

```python
from codegraph_rag import CodeGraphRAG

rag = CodeGraphRAG("C:/dev/my-project")

results = rag.search(
    "find the function that validates user login",
    top_k=5,
    active_file="src/auth/service.py",
    language="python",
    repo="my-project",
)

for result in results:
    print(result.score, result.qualified_name, result.file_path, result.line_number)
```

CLI:

```bash
codegraph-rag index C:/dev/my-project
codegraph-rag search C:/dev/my-project "read json config file" --language python
codegraph-rag search C:/dev/my-project "read json config file" --language python --explain
```

Disable embeddings and run lexical/graph-only smoke tests:

```bash
codegraph-rag --embedding-provider none index C:/dev/my-project
codegraph-rag --embedding-provider none search C:/dev/my-project "parse_config" --language python
```

## Configuration

```python
from codegraph_rag import CodeGraphConfig, CodeGraphRAG

config = CodeGraphConfig(enabled=True)
config.embedding.provider = "local"
config.embedding.model_name = "jinaai/jina-embeddings-v2-base-code"
config.storage.vector_dtype = "float32"  # or "int8"
config.search.context_routing_mode = "language"

rag = CodeGraphRAG("C:/dev/my-project", config=config)
```

## IDE / AI Client Contract

Minimum useful context:

```json
{
  "active_file": "src/api/user.ts",
  "language": "typescript"
}
```

Recommended context:

```json
{
  "active_file": "src/api/user.ts",
  "language": "typescript",
  "repo": "checkout-service",
  "workspace": "C:/dev/checkout-service"
}
```

Cross-language search should be explicit:

```json
{
  "active_file": "src/model.py",
  "language": "python",
  "allow_cross_language": true
}
```

## Benchmarks

Current frozen-package verification:

| Check | Samples | MRR | P@1 | Hit@10 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| RepoQA 500, context-aware rerank | 500 | 0.9199 | 86.8% | 98.8% | 101.2 ms |
| P1 smoke | 100 | 0.9950 | 99.0% | 100.0% | 83.7 ms |
| P1 RAG, overall | 500 | 0.9953 | 99.4% | 99.8% | 78.2 ms |
| P1 RAG, NL slice | 436 | 0.9946 | 99.31% | 99.77% | 79.6 ms |
| P1 RAG, code slice | 64 | 1.0000 | 100.0% | 100.0% | 68.8 ms |

Product retrieval returns `default_top_k=50` by default and uses
`rerank_top_k=20`. Ordinary queries use an internal `candidate_pool_top_k=50`;
function-memory queries with concrete repo/path/module context use
`context_candidate_pool_top_k=150` internally to avoid premature truncation
while still returning only Top50.

Engineering verification for the real three-mode path:

| Check | Result |
|---|---:|
| 10K-line full index time | 88.1 s |
| Code embeddings | 10000 |
| Document embeddings | 10000 |
| Avg search latency | 27.4 ms |
| 10-thread errors | 0 |
| 10-thread P99 latency | 396.2 ms |

Recent indexing optimization:

- Full `search_document` text is still stored for BM25 and reranking.
- Only the doc-vector embedding input is compacted for `chunk_doc_embeddings`.
- 10K real three-mode build time dropped from `156.6s` to `88.1s`.
- P1 smoke stayed flat or slightly better: `0.960 -> 0.965`.
- P1 RAG 500 stayed effectively unchanged: `0.973 -> 0.971`, `P@1 97.2% -> 97.0%`.

Historical large-benchmark results such as CodeSearchNet full, CoIR-CCR,
CodeNeedle-style memory, and SWE-bench retrieval-only are kept in
[docs/open-source-release.md](docs/open-source-release.md) with their metric
boundary notes.

## Predictable Runtime

CodeGraph RAG includes a deterministic tie-breaker and a structured
`search_explain()` API:

```python
payload = rag.search_explain(
    "find the function that validates user login",
    top_k=5,
    active_file="src/auth/service.py",
    language="python",
)

print(payload["trace"]["route"])
print(payload["trace"]["config_hash"])
print(payload["trace"]["index_snapshot_id"])
```

See [docs/deterministic-runtime.md](docs/deterministic-runtime.md).

## Known Limits

- Product-level parser depth still needs hardening across all languages.
- Pure natural-language reverse lookup is harder than symbol/signature lookup.
- High-concurrency use should run CodeGraph as a long-lived service so model and
  SQLite handles are reused.
- Full bug repair quality depends on the LLM and agent loop, not only retrieval.

## License

Apache-2.0.
