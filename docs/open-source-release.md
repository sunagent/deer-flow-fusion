# CodeGraph Open-Source Release Notes

Date: 2026-06-14

This document freezes the current CodeGraph retrieval architecture, benchmark
boundaries, and open-source readiness notes for the standalone RAG release.

## 1. Release Scope

This release focuses on local AI coding retrieval:

- Local project code indexing and retrieval.
- Function memory and code context recall.
- Graph + Vector + BM25/FTS5 three-mode RRF fusion.
- CallGraph expansion.
- Local Jina code embedding.
- Lightweight P1 reranking.
- IDE/client context routing.

The release does not claim a full SWE-bench Lite repair resolve rate. SWE-bench
repair is an agent and LLM generation benchmark, while this release freezes the
retrieval layer and reports retrieval-only results separately.

## 2. Frozen Architecture

Core retrieval stack:

- Graph search: symbol, AST, file/path, dependency, and CallGraph expansion.
- Vector search: local `jinaai/jina-embeddings-v2-base-code`, 768 dimensions.
- Document vector search: `search_document` summary embeddings.
- Text search: BM25/FTS5 lexical recall.
- Fusion: weighted RRF, `rrf_k = 60`.
- Reranking: lightweight P1 reranker.

Dynamic routing is reduced to two public modes:

| Mode | Query Class | Graph | Vector | BM25/FTS5 |
|---|---|---:|---:|---:|
| Code/Symbol and Function Memory | code snippets, function names, signatures, purpose-to-function lookup | 0.90 | 0.05 | 0.05 |
| NL/Issue | strong bug reports, issue descriptions, traceback-heavy queries | 0.30 | 0.50 | 0.20 |

Product rule: IDEs and AI programming clients pass facts such as active file,
language, repo, and workspace. They do not pass Graph/Vector/BM25 weights.
CodeGraph owns the routing and fusion policy.

## 3. Local Embedding Baseline

Default local model:

```yaml
codegraph:
  embedding:
    provider: "local"
    model_name: "jinaai/jina-embeddings-v2-base-code"
```

The model runs locally and returns 768-dimensional vectors. No API key is needed
for embedding inference after the model is available in the local HuggingFace
cache.

## 4. Retrieval Benchmarks

All results below use the current frozen retrieval architecture unless noted.

### CodeSearchNet Full

| Metric | Value |
|---|---:|
| Samples | 22,150 |
| MRR | 0.8095 |
| P@1 | 76.66% |
| Hit@5 | 86.04% |
| Hit@10 | 86.28% |
| Zero hit | 2,993 |
| Average latency | 671.3 ms |

### CoIR-CCR Code Context Retrieval

| Metric | Value |
|---|---:|
| Samples | 14,918 |
| Corpus size | 29,918 |
| Distractors | 15,000 |
| MRR | 0.9623 |
| P@1 | 93.26% |
| Hit@5 | 99.84% |
| Hit@10 | 99.93% |
| Zero hit | 5 |
| Average latency | 413.6 ms |

### RepoQA 500

RepoQA is a function-memory style benchmark. The product path should provide
language/path/repo context when the caller is an IDE or AI coding client.

| Case | MRR | P@1 | Hit@10 | Hit@150 | Misses | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| No context | 0.4930 | 40.0% | 67.4% | 85.4% | 73 | 128.8 |
| Language/path/repo context | 0.6054 | 50.2% | 80.8% | 98.0% | 10 | 128.0 |
| Language-routed context | 0.6607 | 56.0% | 85.0% | 98.0% | 10 | 67.1 |

Use `Language-routed context` as the product-context number. Use `No context`
only when comparing pure query-only behavior.

### CodeNeedle-Style Memory Recall

| Metric | Value |
|---|---:|
| Functions | 1,000 |
| Queries | 3,000 |
| Overall MRR | 0.8530 |
| Overall P@1 | 85.30% |
| Average latency | 48.2 ms |
| Function-name MRR/P@1 | 1.0000 / 100.0% |
| Signature MRR/P@1 | 1.0000 / 100.0% |
| NL-description MRR/P@1 | 0.5590 / 55.9% |

The function-name and signature paths are the strongest product paths today.
Pure NL description recall remains an optimization target.

## 5. Engineering Benchmarks

Synthetic 100,000-line local project:

| Item | Value |
|---|---:|
| Lines | 100,001 |
| Files | 100 |
| Functions | 14,243 |
| Index time | 10.286 s |
| Embedding time | 193.562 s |
| Symbol search hit rate | 100% |
| Symbol search avg / p99 | 64.6 ms / 88.0 ms |
| NL search hit rate | 100% |
| NL search avg / p99 | 90.9 ms / 132.0 ms |
| Incremental single-file update | 1,998.2 ms |
| Incremental recall after update | true |
| Persistence reopen | 4,565.7 ms |
| First search after restart | 66.0 ms |
| Recall after restart | true |
| 10-worker error rate / p99 | 0% / 495.4 ms |
| 50-worker error rate / p99 | 0% / 2,585.7 ms |

Known product optimization targets:

- INT8 vector storage for lower memory and faster search.
- Function-level incremental embedding cache to reduce save-time reindexing.
- Long-running service mode for shared model and storage handles under high
  concurrency.

## 6. SWE-bench Retrieval-Only Result

SWE-bench Lite Astropy full-repo retrieval-only, 6 cases after NL/format alias
fix:

| Metric | Value |
|---|---:|
| Cases | 6 |
| Target file Hit@1 | 83.33% |
| Target file Hit@3/5/10/30 | 100% |
| Target hunk any Hit@5/10/30 | 100% |
| Target hunk all-covered rate | 83.33% |
| Average search latency | 0.8955 s |

This is a retrieval-only result. It should not be described as SWE-bench Lite
resolve rate.

## 7. Generation Metrics Are Separate

Generation and repair quality depend on prompt construction, LLM capability,
patch formatting, test execution, and retry policy. Keep them separate from RAG
retrieval metrics.

Current internal notes:

- RepoBench DeepSeek strict next-line exact match: 28.0%.
- ContextBench gold-context diff-like hit: 92%.
- ContextBench path-normalized touches-gold-file hit: 82%.
- SWE target-scope small samples passed in controlled runs, but this is not a
  full public SWE-bench Lite resolve-rate claim.

## 8. IDE / AI Client Context Contract

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

## 9. Known Limits

- Symbol extraction is strongest for Python today. Multi-language extraction
  exists in the benchmark path, but product-level parser depth still needs
  continued hardening for Java, TypeScript/JavaScript, Rust, and Go.
- Pure natural-language reverse lookup is harder than code and symbol lookup.
  RepoQA shows strong gains with language-routed context, but P@1 is still below
  the symbol/signature path.
- Full SWE-bench repair is out of scope for this retrieval release.
- Benchmark datasets, generated indexes, and local model caches are not part of
  the source release.

## 10. Open-Source Safety Checklist

Before publishing:

- Keep API keys in environment variables only.
- Do not commit `.env`, `.codegraph/`, `.deer-flow/`, local SQLite indexes, or
  benchmark datasets.
- Keep large datasets in external storage such as `F:/codex-cache/datasets`.
- Keep generated benchmark artifacts outside the repository unless they are
  intentionally curated reports.
- Re-run a secret scan before publishing.
- Re-run `py_compile` for touched Python files.

## 11. Reproducibility Artifacts

Local benchmark artifacts used for this report:

- `F:\codex-cache\benchmarks\csn_full_current_results.json`
- `F:\codex-cache\benchmarks\coir_ccr_14918_current_results.json`
- `F:\codex-cache\benchmarks\repoqa_500_context_prior_results.json`
- `F:\codex-cache\benchmarks\REPOQA_500_CONTEXT_PRIOR_REPORT.md`
- `F:\codex-cache\benchmarks\codeneedle_1000_current_results.json`
- `F:\codex-cache\benchmarks\engineering_current_results.json`
- `F:\codex-cache\benchmarks\swebench_fullrepo_retrieval_astropy_6_after_alias_summary.json`
