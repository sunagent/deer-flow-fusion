# Host Integrations

CodeGraph RAG is intentionally host-agnostic. The core package does not depend
on any specific IDE, agent framework, tool protocol, or runtime.

Recommended integration shape:

1. Keep `codegraph_rag.CodeGraphRAG` as the retrieval engine.
2. Let the host application pass stable facts:
   - `active_file`
   - `language`
   - `repo`
   - `workspace`
   - `allow_cross_language`
3. Wrap `CodeGraphRAG.search()` in the host tool system.
4. Do not expose Graph/Vector/BM25 weights to the client.

Minimal wrapper:

```python
from codegraph_rag import CodeGraphRAG

rag = CodeGraphRAG("/path/to/workspace")

def retrieve_code(query: str, active_file: str | None = None, language: str | None = None):
    return rag.search(
        query,
        top_k=10,
        active_file=active_file,
        language=language,
    )
```

The host owns authentication, user sessions, prompt construction, and LLM calls.
CodeGraph RAG owns local indexing, retrieval routing, fusion, and reranking.
