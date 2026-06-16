# RepoQA Rerank Root Cause Fix

Date: 2026-06-16

## Conclusion

RepoQA 500 was not recall-bound. The target was already present in the
candidate pool:

| Metric | Before |
|---|---:|
| Hit@150 | 98.8% |
| Misses | 6 / 500 |

The failure was P1 precision. The reranker could not reliably separate the
right function from same-language, semantically similar distractors.

## Root Cause

The context signal collapsed into a mostly constant language prior:

- `context_routing_mode="language"` filtered candidates to the right language.
- `_context_match_score()` only matched repo/path tokens against `file_path`.
- RepoQA uses synthetic local file paths such as `f_00409_go.go`.
- The real repo/path/module metadata lived in `search_document`, but P1 context
  scoring did not read it.

As a result, most same-language candidates received nearly the same
`context_prior`, so final ranking fell back to generic text overlap and
single-channel rank evidence.

## Negative Check

Lowering or gating `channel_boost` was tested first. It hurt quality, proving
that the channel signal was not the root cause:

| Variant | MRR | P@1 | Hit@10 |
|---|---:|---:|---:|
| Baseline | 0.6714 | 57.8% | 85.8% |
| Channel removed / redistributed | 0.6484 | 54.2% | 84.8% |
| Channel gated | 0.6427 | 53.4% | 85.0% |

## Fix

The fix keeps Graph + Vector + BM25 RRF unchanged and only improves P1 context
reranking:

- Add a cached context surface that combines `file_path` and `search_document`.
- Add `_context_specific_match_score()` for repo/path/module agreement.
- Keep language as routing context, not a strong P1 discriminator.
- Add `context_specific_match` and `context_has_specific` to reranker
  candidates.
- Use a dedicated NL specific-context weight profile only when the caller
  provides repo/path/module context.

## Result

| Evaluation | MRR | P@1 | Hit@10 | Hit@150 | Misses | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| Before: language-routed context | 0.6714 | 57.8% | 85.8% | 98.8% | 6 | 69.8 |
| After: context-aware P1 | 0.8735 | 81.4% | 97.4% | 98.8% | 6 | 96.6 |

Runtime defaults are now `default_top_k=50` and `rerank_top_k=20`. Top150 was
used only to prove recall headroom; after the fix, Hit@50 and Hit@150 are both
98.8%.

By language after the fix:

| Language | MRR | P@1 | Hit@10 | Misses |
|---|---:|---:|---:|---:|
| Python | 0.9387 | 90.0% | 99.0% | 1 |
| Java | 0.8942 | 83.0% | 98.0% | 2 |
| TypeScript | 0.9567 | 92.0% | 100.0% | 0 |
| Rust | 0.6636 | 55.0% | 93.0% | 0 |
| Go | 0.9145 | 87.0% | 97.0% | 3 |

## Regression Checks

| Check | MRR | P@1 | Hit@10 | Zero hits | Avg ms |
|---|---:|---:|---:|---:|---:|
| P1 smoke 100 | 0.9650 | 96.0% | 97.0% | 3 | 42.3 |
| P1 RAG 500 overall | 0.9710 | 97.0% | 97.2% | 14 | 37.7 |

Artifacts:

- `F:\codex-cache\benchmarks\repoqa_500_rerank_root_cause_20260616.json`
- `F:\codex-cache\benchmarks\repoqa_500_rerank_ab_channel_20260616.json`
- `F:\codex-cache\benchmarks\repoqa_500_rerank_ab_context_20260616.json`
- `F:\codex-cache\benchmarks\repoqa_500_context_rerank_fixed_20260616.json`
