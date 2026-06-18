# SWE Pro Routing Fix

Date: 2026-06-18

## Summary

This fix addresses a routing bug observed in SWE Pro full-repo retrieval-only
validation where issue-style natural-language queries were sometimes routed to
the graph-first profile:

- Expected NL/Issue weights: graph 0.30 / vector 0.50 / BM25 0.20
- Wrong weights seen in trace: graph 0.90 / vector 0.05 / BM25 0.05

The bug was real. It was caused by query profile detection and weight
selection getting out of sync.

## Symptoms

Observed in:

- `F:\codex-cache\benchmarks\codegraph_retrieval_validation_swepro4_20260618_053359.md`
- `F:\codex-cache\benchmarks\codegraph_retrieval_validation_swepro4_20260618_053359.jsonl`

Representative bad trace before fix:

- `query_type = "nl"`
- `query_profile = "function_memory"`
- `weights = {"graph": 0.9, "vector": 0.05, "bm25": 0.05}`

This happened on issue reports that also contained:

- `Description:`
- `the function`
- `Failing tests:`
- test names, file paths, or traceback-style content

## Root Cause

Two issues combined:

1. `function_memory` had higher priority than `issue`

In `SearchEngine._detect_query_profile()`, the router checked
`_is_function_memory_query()` before `_is_strong_issue_query()`.
That meant issue reports containing words such as `description:` or
`the function` were classified as `function_memory`.

2. Weight selection did not reuse the resolved profile

`search()` computed `qtype` and `profile`, but `_get_dynamic_weights(query)`
ran its own profile detection again from raw query text. This made the routing
path harder to reason about and easier to drift.

## Files Changed

Runtime package actually used by SWE Pro validation:

- [src/codegraph_rag/search.py](/F:/codegraph-rag/src/codegraph_rag/search.py:59)

Mirror copy in harness repo:

- [search.py](/F:/新建文件夹/deer-flow-main/backend/packages/harness/deerflow/tools/codegraph/search.py:58)

Regression tests:

- [tests/test_search_routing.py](/F:/codegraph-rag/tests/test_search_routing.py:48)
- [test_codegraph_search_routing.py](/F:/新建文件夹/deer-flow-main/backend/tests/test_codegraph_search_routing.py:48)

## Fix Applied

### 1. Make issue routing win over function-memory cues

`_detect_query_profile()` now checks strong issue signals before
`function_memory`.

### 2. Reuse the resolved profile when selecting weights

`search()` now passes `qtype` and `profile` into `_get_dynamic_weights(...)`
instead of letting that method independently re-detect from raw query text.

### 3. Expand strong issue markers for SWE-style reports

Added high-confidence issue markers:

- `current behavior`
- `failing tests:`
- `failing test:`
- `test failure`
- `stack trace`

## Why The First Patch Seemed Not To Work

The initial patch was applied to the harness copy first, but the SWE Pro
validation script actually imports the standalone package from:

- `F:\codegraph-rag`

This is because `run_swebench_pro_ide.py` injects:

- `external_package_path = codegraph_root`

So the effective runtime patch had to be applied to `codegraph-rag` as well.

## Verification

### Unit tests

Passed:

- `backend/tests/test_codegraph_search_routing.py` -> `3 passed`
- `codegraph-rag/tests/test_search_routing.py` -> `3 passed`

### SWE Pro rerun

Artifacts after fix:

- [report](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_20260618_054804.md)
- [jsonl](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_20260618_054804.jsonl)

Verified routing behavior after fix:

- NodeBB sample: `query_profile = issue`, `weights = 0.3 / 0.5 / 0.2`
- qutebrowser moved-test sample: `query_profile = issue`, `weights = 0.3 / 0.5 / 0.2`
- ansible sample: `query_profile = issue`, `weights = 0.3 / 0.5 / 0.2`
- qutebrowser changelog sample: `query_profile = issue`, `weights = 0.3 / 0.5 / 0.2`

## What This Fix Does Not Solve

This fix corrects routing. It does not fully solve full-repo ranking quality in
SWE Pro.

Remaining issue:

- `Failing tests:` content still strongly boosts test files and nearby test
  symbols.
- In several cases the implementation file is still below the test file, even
  though the query now correctly routes to the NL/Issue profile.

Examples after fix:

- qutebrowser moved-test case returns `tests/unit/utils/test_qtlog.py` at rank 1
  and `qutebrowser/utils/log.py` at rank 2
- qutebrowser changelog case still returns `tests/unit/config/test_configfiles.py`
  above implementation/config/docs files

## Recommended Next Step

Do not change the frozen weight values.

Instead, improve issue-query handling by splitting retrieval intent:

1. treat problem statement and behavior text as the main retrieval query
2. treat `Failing tests:` as auxiliary evidence
3. apply a small test-file penalty during issue-mode reranking unless the test
   file is itself an expected target

That should improve SWE Pro full-repo retrieval more than further tuning
`0.3 / 0.5 / 0.2`.

## Sync Checklist

When syncing to cloud Git, keep these changes together:

1. `F:\codegraph-rag\src\codegraph_rag\search.py`
2. `F:\codegraph-rag\tests\test_search_routing.py`
3. `F:\新建文件夹\deer-flow-main\backend\packages\harness\deerflow\tools\codegraph\search.py`
4. `F:\新建文件夹\deer-flow-main\backend\tests\test_codegraph_search_routing.py`
5. this document and the harness mirror document

If only one repo is updated, SWE Pro validation and local harness behavior can
diverge again.
