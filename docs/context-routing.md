# Context Routing

Date: 2026-06-15

CodeGraph RAG owns retrieval routing and fusion. IDEs or AI programming clients
only pass stable product facts:

```python
from codegraph_rag import CodeGraphRAG

rag = CodeGraphRAG("C:/work/my-project")
results = rag.search(
    "find the function that validates expected errors",
    active_file="packages/api/src/errors.ts",
    language="typescript",
    repo="my-project",
)
```

Clients must not pass Graph/Vector/BM25 weights. The engine keeps the frozen
two-mode router and applies context internally.

## Built-In Behavior

When no context is provided, CodeGraph keeps the stable router:

- Code/Symbol and Function Memory queries: graph-first profile.
- Strong Issue/Bug/Traceback queries: NL/Issue profile.
- No language hard filter is applied.

When context is provided and `context_routing_mode = "language"`:

- `active_file` or `language` is normalized to the current language.
- BM25, code vector, document vector, and graph candidates are recalled from the
  same-language pool by default.
- Path/repo/workspace context is used as a soft RRF/P1 reranker prior.
- `allow_cross_language=True` disables the same-language pool for migration,
  translation, and cross-framework analogy tasks.

## Public API

```python
CodeGraphRAG.search(
    query: str,
    *,
    top_k: int = 10,
    active_file: str | None = None,
    language: str | None = None,
    repo: str | None = None,
    allow_cross_language: bool = False,
)
```

Host applications can wrap this method in their own tool protocol. The core
package intentionally has no LangChain, LangGraph, or DeerFlow runtime
dependency.

## Config

Default:

```yaml
codegraph:
  search:
    enable_context_prior: true
    context_routing_mode: "language"
    context_rrf_boost: 0.003
    context_rerank_boost: 0.10
```

This default is safe because it has no effect unless context is present.

## Metric Boundary

Do not mix these RepoQA 500 result families:

| Evaluation Mode | MRR | P@1 | Hit@10 | Hit@150 | Notes |
|---|---:|---:|---:|---:|---|
| No context | 0.4930 | 40.0% | 67.4% | 85.4% | Pure query only |
| Language/path/repo context | 0.6054 | 50.2% | 80.8% | 98.0% | Context soft prior |
| Language-routed context | 0.6714 | 57.8% | 85.8% | 98.8% | Product-context path, 69.8 ms avg |

Use `Language-routed context` when the caller knows the active file, current
language, or workspace, which is normal for an IDE/AI programming integration.

## Recommended Client Contract

Minimum useful context:

```json
{
  "active_file": "src/api/user.ts",
  "language": "typescript"
}
```

Better context:

```json
{
  "active_file": "src/api/user.ts",
  "language": "typescript",
  "repo": "checkout-service",
  "workspace": "C:/dev/checkout-service"
}
```

Cross-language search:

```json
{
  "active_file": "src/model.py",
  "language": "python",
  "allow_cross_language": true
}
```

Use cross-language only when the user explicitly asks for migration,
translation, or cross-framework analogies.
