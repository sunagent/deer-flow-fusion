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

Current frozen-architecture retrieval results:

| Benchmark | Samples | MRR | P@1 | Hit@10 | Avg Latency |
|---|---:|---:|---:|---:|---:|
| CodeSearchNet full | 22,150 | 0.8095 | 76.66% | 86.28% | 671.3 ms |
| CoIR-CCR | 14,918 | 0.9623 | 93.26% | 99.93% | 413.6 ms |
| RepoQA 500, language-routed context | 500 | 0.6607 | 56.0% | 85.0% | 67.1 ms |
| CodeNeedle-style memory | 3,000 | 0.8530 | 85.30% | 85.30% | 48.2 ms |

SWE-bench Lite Astropy full-repo retrieval-only, 6 cases:

- Target file Hit@1: 83.33%
- Target file Hit@3/5/10/30: 100%
- Target hunk any Hit@5/10/30: 100%
- Target hunk all-covered rate: 83.33%

These are retrieval metrics. They are not full SWE-bench repair resolve rates.
See [docs/open-source-release.md](docs/open-source-release.md) for the complete
metric boundary.

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

MIT.
