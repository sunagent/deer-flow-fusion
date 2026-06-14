# CodeGraph Stable Router Results

Date: 2026-06-14

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

| Profile | MRR | P@1 | Hit@5 | Hit@10 | Hit@50 | Misses | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stable dynamic router | 0.5061 | 0.418 | 0.624 | 0.674 | 0.772 | 114 | 41.3 |
| Forced graph-first | 0.5061 | 0.418 | 0.624 | 0.674 | 0.772 | 114 | 41.3 |

Stable dynamic router now matches forced graph-first exactly on RepoQA 500.

## Smoke Regression

| Benchmark | Samples | MRR | P@1 | Hit@10 | Zero hit | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| CodeSearchNet smoke | 50 | 0.9629 | 0.9400 | 1.0000 | 0 | 65.0 |
| CoIR-CCR smoke | 50 | 0.9700 | 0.9400 | 1.0000 | 0 | 100.8 |
| CodeNeedle NL smoke | 50 | 1.0000 | 1.0000 | 1.0000 | 0 | 43.0 |

## Rerun Confirmation

The stable router was rerun after implementation without changing the index.

RepoQA 500 rerun:

| Profile | MRR | P@1 | Hit@5 | Hit@10 | Hit@50 | Misses | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stable dynamic router | 0.5061 | 0.418 | 0.624 | 0.674 | 0.772 | 114 | 42.1 |
| Forced graph-first | 0.5061 | 0.418 | 0.624 | 0.674 | 0.772 | 114 | 41.9 |

Smoke rerun:

| Benchmark | Samples | MRR | P@1 | Hit@10 | Zero hit | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| CodeSearchNet smoke | 50 | 0.9629 | 0.9400 | 1.0000 | 0 | 75.3 |
| CoIR-CCR smoke | 50 | 0.9700 | 0.9400 | 1.0000 | 0 | 119.7 |
| CodeNeedle NL smoke | 50 | 1.0000 | 1.0000 | 1.0000 | 0 | 46.6 |

The rerun matches the first stable-router metrics. Latency moved slightly due to
runtime/GPU noise, while retrieval metrics stayed stable.

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
