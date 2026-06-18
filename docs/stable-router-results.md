# CodeGraph Stable Router Results

Date: 2026-06-15

## Problem

The old router treated query type as a hard NL/Code switch. When a natural
language query contained symbols, Markdown, parentheses, paths, or words like
`error` and `expected`, it could jump to the wrong profile.

That made fusion unstable because the weight gap is large:

- Graph-first: graph 0.90 / vector 0.05 / BM25 0.05
- NL/Issue: graph 0.30 / vector 0.50 / BM25 0.20

One routing mistake could therefore dominate ranking.

## Stable Router

The router now separates text-processing mode from fusion-weight profile:

- Code/Symbol: graph-first
- Function Memory / Purpose Description: graph-first + NL document/reranker features
- Strong Issue/Bug/Traceback: vector-heavy NL/Issue

Function Memory has priority over weak issue words. For example:

```text
1. **Purpose**: To verify that expected errors are handled.
```

This stays Function Memory because the target is still a function/symbol lookup,
not an open-ended bug report.

Strong issue examples still route to NL/Issue:

```text
Bug report: calling parse_config crashes with TypeError...
### Description
The parser does not handle ascii.rst files and raises ValueError traceback.
```

## Route Self-Check

RepoQA 500 after stable router v2:

- Function Memory: 500
- Issue: 0
- Effective dynamic weights: graph 0.90 / vector 0.05 / BM25 0.05

Manual probes:

| Query | Profile | Weights |
|---|---|---|
| `def read_json(path):` | Code | 0.90 / 0.05 / 0.05 |
| `read_json_file` | Code | 0.90 / 0.05 / 0.05 |
| `1. **Purpose**: ... expected errors ...` | Function Memory | 0.90 / 0.05 / 0.05 |
| `Bug report: ... crashes with TypeError ...` | Issue | 0.30 / 0.50 / 0.20 |

## RepoQA 500 Result

Current verified frozen baseline with language-routed context and
context-aware P1 reranking:

| Profile | MRR | P@1 | Hit@10 | Hit@150 | Misses | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| Stable dynamic router + context-aware P1 | 0.9199 | 86.8% | 98.8% | 98.8% | 6 | 101.2 |

Route self-check stays the same: RepoQA-style purpose/function-memory prompts
remain on the graph-first profile instead of flipping into the issue profile.

Runtime defaults return `default_top_k=50` and rerank `rerank_top_k=20`.
Ordinary queries use `candidate_pool_top_k=50`; function-memory queries with
concrete repo/path/module context use `context_candidate_pool_top_k=150`
internally, still returning only Top50.

## Smoke Regression

| Benchmark | Samples | MRR | P@1 | Hit@10 | Zero hit | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| CodeSearchNet smoke | 50 | 0.9629 | 0.9400 | 1.0000 | 0 | 65.0 |
| CoIR-CCR smoke | 50 | 0.9700 | 0.9400 | 1.0000 | 0 | 100.8 |
| CodeNeedle NL smoke | 50 | 1.0000 | 1.0000 | 1.0000 | 0 | 43.0 |

## Rerun Confirmation

Current package verification after routing, indexing, and concurrency fixes:

| Benchmark | Samples | MRR | P@1 | Hit@10 | Zero hit | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| P1 smoke | 100 | 0.9950 | 99.0% | 100.0% | 0 | 83.7 |
| P1 RAG overall | 500 | 0.9953 | 99.4% | 99.8% | 1 | 78.2 |
| P1 RAG NL slice | 436 | 0.9946 | 99.31% | 99.77% | 1 | 79.6 |
| P1 RAG code slice | 64 | 1.0000 | 100.0% | 100.0% | 0 | 68.8 |

## Conclusion

The unstable routing issue is fixed for the tested paths. The key design change
is not trying to make NL/Code classification perfect. Instead, low-confidence
and function-memory natural-language queries default to graph-first, while only
strong issue/bug signals switch to vector-heavy NL/Issue.

This keeps the product behavior aligned with local AI coding agents: symbols,
files, and graph structure remain the stable default, while natural-language
document vectors and summary reranking still help as secondary signals.

Artifacts:

- `F:\codex-cache\benchmarks\repoqa_500_stable_router_v2.log`
- `F:\codex-cache\benchmarks\repoqa_500_stable_router_rerun_20260614.log`
- `F:\codex-cache\benchmarks\fixed_architecture_smoke_stable_router.log`
- `F:\codex-cache\benchmarks\fixed_architecture_smoke_stable_router_rerun_20260614.log`
- `F:\codex-cache\benchmarks\REPOQA_500_P0_P1_PROFILE_COMPARE.md`
- `F:\codex-cache\benchmarks\repoqa_500_context_rerank_fixed_20260616.json`
- `F:\codex-cache\benchmarks\repoqa_500_rust_context_pool_fixed_20260616.json`
- `F:\codex-cache\benchmarks\REPOQA_500_RERANK_ROOT_CAUSE_FIX_20260616.md`
- `F:\codex-cache\benchmarks\p1_smoke_100.py`
- `F:\codex-cache\benchmarks\p1_rag_500.py`
