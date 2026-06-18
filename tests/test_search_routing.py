from codegraph_rag.search import SearchEngine


NODEBB_ISSUE_QUERY = """**Title: Email Validation Status Not Handled Correctly in ACP and Confirmation Logic**

**Description:**

The Admin Control Panel (ACP) does not accurately reflect the email validation status of users.

**What is expected:**

Accurate display of email status in ACP.

Failing tests:
- test/database.js | Test database test/database/keys.js::Key methods should return multiple keys and null if key doesn't exist
"""


QUTE_ISSUE_QUERY = """# Qt warning filtering tests moved to appropriate module

## Description

The `hide_qt_warning` function and its associated tests have been moved from `log.py` to `qtlog.py`.

## Expected Behavior

Qt warning filtering should work identically after the code reorganization.

## Current Behavior

The functionality has been moved but tests need to be updated to reflect the new module organization.

Failing tests:
- tests/unit/utils/test_qtlog.py::TestHideQtWarning::test_unfiltered
"""


FUNCTION_MEMORY_QUERY = """Purpose: find the function used to parse config files.

Description: this function is used to load settings from disk and return a configuration object.
"""


def _engine() -> SearchEngine:
    return SearchEngine.__new__(SearchEngine)


def test_issue_profile_wins_over_function_memory_markers() -> None:
    assert SearchEngine._detect_query_type(NODEBB_ISSUE_QUERY) == "nl"
    assert SearchEngine._detect_query_profile(NODEBB_ISSUE_QUERY) == "issue"

    engine = _engine()
    assert engine._get_dynamic_weights(
        NODEBB_ISSUE_QUERY,
        qtype="nl",
        profile="issue",
    ) == (0.3, 0.5, 0.2)


def test_failing_test_issue_stays_in_issue_mode() -> None:
    assert SearchEngine._detect_query_type(QUTE_ISSUE_QUERY) == "nl"
    assert SearchEngine._detect_query_profile(QUTE_ISSUE_QUERY) == "issue"

    engine = _engine()
    assert engine._get_dynamic_weights(
        QUTE_ISSUE_QUERY,
        qtype="nl",
        profile="issue",
    ) == (0.3, 0.5, 0.2)


def test_plain_function_memory_query_stays_graph_first() -> None:
    assert SearchEngine._detect_query_profile(FUNCTION_MEMORY_QUERY) == "function_memory"

    engine = _engine()
    assert engine._get_dynamic_weights(
        FUNCTION_MEMORY_QUERY,
        qtype="nl",
        profile="function_memory",
    ) == (0.9, 0.05, 0.05)


def test_issue_query_split_and_test_hints() -> None:
    raw = "\"# Title\\n\\n### Description\\nSomething broke\\n\\nFailing tests:\\n- tests/unit/utils/test_qtlog.py::TestHideQtWarning::test_unfiltered\\n- test/user/emails.js | email confirmation canSendValidation\""

    normalized = SearchEngine._normalize_query_text(raw)
    body, test_lines = SearchEngine._split_issue_query(normalized)
    focus = SearchEngine._build_issue_focus_query(body)
    hints = SearchEngine._build_issue_test_hint_query(test_lines)

    assert not normalized.startswith('"')
    assert "Failing tests:" not in body
    assert "Something broke" in body
    assert "Something broke" in focus
    assert test_lines
    assert "qtlog" in hints
    assert "emails" in hints
    assert "email" in hints



# ---------------- P0 issue-mode internal optimizations ----------------

CHANGELOG_ISSUE = """## Description

The settings reference under `qutebrowser/config/configdata.yml` is out of
sync with the documented behavior in `doc/changelog.asciidoc`, and the
upgrade path defined in `qutebrowser/app.py` should be revisited.
"""


def test_extract_issue_keywords_config_doc_changelog() -> None:
    keywords = SearchEngine._extract_issue_keywords(CHANGELOG_ISSUE)
    assert "config" in keywords
    assert "settings" in keywords
    assert "changelog" in keywords
    assert "upgrade" in keywords


def test_is_test_path_detects_common_test_locations() -> None:
    class _R:
        def __init__(self, file_path: str, symbol_name: str = "", tags=None) -> None:
            self.file_path = file_path
            self.symbol_name = symbol_name
            self.tags = list(tags or [])

    assert SearchEngine._is_test_path(_R("tests/unit/utils/test_qtlog.py"))
    assert SearchEngine._is_test_path(_R("test/database.js"))
    assert SearchEngine._is_test_path(_R("src/foo_test.py"))
    assert SearchEngine._is_test_path(_R("packages/app/foo.test.ts"))
    assert SearchEngine._is_test_path(_R("src/foo.py", symbol_name="test_something"))
    assert SearchEngine._is_test_path(_R("src/foo.py", tags=["test"]))

    # Real implementation surfaces must not be flagged.
    assert not SearchEngine._is_test_path(_R("qutebrowser/app.py"))
    assert not SearchEngine._is_test_path(_R("src/controllers/admin/users.js"))
    assert not SearchEngine._is_test_path(_R("doc/changelog.asciidoc"))
    assert not SearchEngine._is_test_path(_R("src/foo.py", symbol_name="tester"))


def test_extract_issue_keywords_empty_for_unrelated_text() -> None:
    assert SearchEngine._extract_issue_keywords("") == []
    assert SearchEngine._extract_issue_keywords("plain text no markers") == []



# ---------------- Dual-lane wiring smoke ----------------

class _FakeEmbedding:
    is_available = False
    def get_embedding(self, *_a, **_k):
        return None


class _FakeStorage:
    """Storage stub that records bm25_search calls."""

    def __init__(self):
        self.bm25_calls = []

    def bm25_search(self, query, **kwargs):
        self.bm25_calls.append((query, kwargs))
        # Return a single fake result so the lane runs through merge logic.
        from codegraph_rag.models import SearchResult, ChunkType
        return [
            SearchResult(
                id=f"id::{query[:20]}",
                symbol_name="x",
                qualified_name="x",
                file_path="src/foo.py",
                line_number=1,
                chunk_type=ChunkType.FUNCTION,
                score=0.5,
                snippet="...",
                tags=[],
                language="python",
            )
        ]

    def vector_search(self, *_a, **_k):
        return []

    def document_vector_search(self, *_a, **_k):
        return []

    def get_search_document(self, *_a, **_k):
        return ""


class _FakeConfig:
    class search:
        enable_bm25 = True
        enable_vector_search = False
        enable_graph_search = False
        enable_rerank = False
        enable_nl_query_enhance = False
        enable_context_prior = False
        default_top_k = 10
        candidate_pool_top_k = 50
        context_candidate_pool_top_k = 150
        rerank_top_k = 20
        nl_rrf_k = None
        rrf_k = 60
        weight_graph = 0.3
        weight_vector = 0.5
        weight_bm25 = 0.2
        merge_mode = "rrf"
        context_routing_mode = "off"
        context_rrf_boost = 0.0
        context_rerank_boost = 0.0
        deterministic_sort = True
    storage = type("s", (), {"persist_directory": ":memory:"})()


def test_issue_dual_lane_uses_test_hints() -> None:
    engine = SearchEngine.__new__(SearchEngine)
    engine._storage = _FakeStorage()
    engine._embedding = _FakeEmbedding()
    engine._config = _FakeConfig()
    engine._graph_engine = None
    engine._call_graph = None
    engine._expand_code_tokens = False
    engine._context_doc_cache = {}
    engine._reranker = None
    engine._last_trace = None

    engine.search(QUTE_ISSUE_QUERY, top_k=10)

    queries = [q for q, _ in engine._storage.bm25_calls]
    assert len(queries) >= 2, queries
    # First call: focused issue body, second call: failing-test hints
    assert any("qtlog" in q or "test_qtlog" in q or "test_unfiltered" in q or "unfiltered" in q for q in queries[1:]), queries



# ---------------- Title-lane wiring smoke ----------------


def test_extract_issue_title_inline_marker() -> None:
    body = "**Title: Email Validation Status Not Handled Correctly**\n\nDesc"
    assert SearchEngine._extract_issue_title(body) == "Email Validation Status Not Handled Correctly"


def test_extract_issue_title_heading_lookahead() -> None:
    body = "Title:\n# Qt warning filtering tests moved to appropriate module\n\n## Description"
    assert "Qt warning" in SearchEngine._extract_issue_title(body)


class _TitleBM25Storage(_FakeStorage):
    pass


def test_issue_title_bm25_lane_invoked() -> None:
    """Issue mode with a distinct title runs a third BM25 lane."""
    engine = SearchEngine.__new__(SearchEngine)
    engine._storage = _TitleBM25Storage()
    engine._embedding = _FakeEmbedding()
    engine._config = _FakeConfig()
    engine._graph_engine = None
    engine._call_graph = None
    engine._expand_code_tokens = False
    engine._context_doc_cache = {}
    engine._reranker = None
    engine._last_trace = None

    engine.search(NODEBB_ISSUE_QUERY, top_k=10)

    queries = [q for q, _ in engine._storage.bm25_calls]
    # body + title + hint  ->  at least 3 BM25 calls
    assert len(queries) >= 3, queries
    assert any("Email Validation" in q for q in queries), queries


class _TitleVectorEmbedding:
    is_available = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_embedding(self, text):
        self.calls.append(text)
        # Return a tiny fake vector so callers proceed to vector_search.
        return [0.0, 1.0, 0.0]


class _TitleVectorStorage:
    def __init__(self) -> None:
        self.bm25_calls: list[tuple[str, dict]] = []
        self.doc_vec_calls: list[str] = []
        self.code_vec_calls: list[str] = []

    def bm25_search(self, query, **kwargs):
        self.bm25_calls.append((query, kwargs))
        from codegraph_rag.models import SearchResult, ChunkType
        return [
            SearchResult(
                id=f"bm25::{query[:24]}",
                symbol_name="x",
                qualified_name="x",
                file_path="src/foo.py",
                line_number=1,
                chunk_type=ChunkType.FUNCTION,
                score=0.5,
                snippet="...",
                tags=[],
                language="python",
            )
        ]

    def document_vector_search(self, emb, **kwargs):
        # _TitleVectorEmbedding always returns a fixed embedding, so we can't
        # discriminate via the embedding contents. Track call ordering only.
        self.doc_vec_calls.append("doc")
        return []

    def vector_search(self, emb, **kwargs):
        self.code_vec_calls.append("code")
        from codegraph_rag.models import SearchResult, ChunkType
        return [
            SearchResult(
                id=f"vec::{len(self.code_vec_calls)}",
                symbol_name="y",
                qualified_name="y",
                file_path="src/bar.py",
                line_number=1,
                chunk_type=ChunkType.FUNCTION,
                score=0.5,
                snippet="...",
                tags=[],
                language="python",
            )
        ]

    def get_search_document(self, *_a, **_k):
        return ""


class _TitleVectorConfig(_FakeConfig):
    class search(_FakeConfig.search):
        enable_vector_search = True


def test_issue_title_vector_lane_invoked() -> None:
    """Title vector lane is wired: distinct title triggers extra embedding/vector call."""
    engine = SearchEngine.__new__(SearchEngine)
    engine._storage = _TitleVectorStorage()
    engine._embedding = _TitleVectorEmbedding()
    engine._config = _TitleVectorConfig()
    engine._graph_engine = None
    engine._call_graph = None
    engine._expand_code_tokens = False
    engine._context_doc_cache = {}
    engine._reranker = None
    engine._last_trace = None

    engine.search(NODEBB_ISSUE_QUERY, top_k=10)

    # Body + title + hint embeddings should be requested.
    embed_calls = engine._embedding.calls
    assert len(embed_calls) >= 3, embed_calls
    assert any("Email Validation" in c for c in embed_calls), embed_calls
    # Each embedding call drives a vector_search (doc_vec returns [], so
    # the code-vec fallback fires exactly once per embedding).
    assert len(engine._storage.code_vec_calls) >= 3, engine._storage.code_vec_calls
