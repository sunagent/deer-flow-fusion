# Rust Rerank Optimization

Date: 2026-06-16

## Problem

Rust was the weakest RepoQA language slice after the first context-aware P1
fix:

| Language | MRR | P@1 | Hit@10 |
|---|---:|---:|---:|
| Rust before | 0.6636 | 55.0% | 93.0% |

The target was usually retrievable, but Rust candidates were not ranked well
because Rust did not have a language-specific NL context profile. It fell back
to the generic NL reranker, which underweights Rust-specific structure such as:

- `impl` blocks
- traits
- module paths
- crate/use imports
- macro-style APIs
- short parser/test helper names

## Fix

Two changes were added:

1. Rust-specific P1 weights
   - Added Rust entries to `NL_CONTEXT_WEIGHTS_BY_LANGUAGE`.
   - Added Rust entries to `NL_SPECIFIC_CONTEXT_WEIGHTS_BY_LANGUAGE`.
   - Increased the importance of concrete context, owner/module tokens, symbol
     soft match, and Rust structural hints.

2. Return Top50, keep a larger product-context candidate pool
   - Product output still returns `default_top_k=50`.
   - Ordinary queries use `candidate_pool_top_k=50`.
   - Function-memory queries with concrete repo/path/module context use
     `context_candidate_pool_top_k=150` internally, then return Top50.

This prevents the RRF candidate pool from truncating relevant Rust targets
before the reranker sees them.

## Result

RepoQA 500 with product context:

| Metric | Before | After |
|---|---:|---:|
| Overall MRR | 0.8735 | 0.9199 |
| Overall P@1 | 81.4% | 86.8% |
| Overall Hit@10 | 97.4% | 98.8% |
| Rust MRR | 0.6636 | 0.8954 |
| Rust P@1 | 55.0% | 82.0% |
| Rust Hit@10 | 93.0% | 100.0% |

Regression checks:

| Check | MRR | P@1 | Hit@10 | Avg ms |
|---|---:|---:|---:|---:|
| P1 smoke 100 | 0.9950 | 99.0% | 100.0% | 83.7 |
| P1 RAG 500 overall | 0.9953 | 99.4% | 99.8% | 78.2 |

Artifact:

- `F:\codex-cache\benchmarks\repoqa_500_rust_context_pool_fixed_20260616.json`

