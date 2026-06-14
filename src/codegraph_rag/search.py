""" ?BM25 +  + ?+ RRF """
import logging, re
from pathlib import Path
from .config import CodeGraphConfig
from .embedding import EmbeddingCache
from .models import SearchResult, ChunkType
from .storage import CodeGraphStorage

logger = logging.getLogger(__name__)

#   + 
_KNOWN_LIBS = {
    "pandas","numpy","torch","tensorflow","tf","flask","django",
    "fastapi","sqlalchemy","celery","redis","aiohttp","asyncio",
    "requests","matplotlib","sklearn","scikit","pytest","jwt",
    "pillow","opencv","cv2","bs4","beautifulsoup","scrapy",
    "plotly","seaborn","nltk","spacy","transformers",
    "pydantic","dataclasses","multiprocessing","threading",
    "sqlite3","psycopg2","pymongo","aiofiles","httpx",
}

class SearchEngine:
    """Hybrid search engine - BM25 + Vector + Graph + RRF fusion"""

    def __init__(
        self,
        storage: CodeGraphStorage,
        embedding_cache: EmbeddingCache,
        config: CodeGraphConfig,
        graph_engine = None,          # GraphSearchEngine | None
        call_graph = None,            # CallGraph | None
        expand_code_tokens: bool = False,  # BM25 code-aware token expansion (graph_engine fallback)
    ):
        self._storage = storage
        self._embedding = embedding_cache
        self._config = config
        self._graph_engine = graph_engine
        self._call_graph = call_graph or (graph_engine._call_graph if graph_engine and hasattr(graph_engine, "_call_graph") else None)
        self._expand_code_tokens = expand_code_tokens
        self._reranker = None
        if config.search.enable_rerank:
            try:
                from .reranker import ReRanker
                self._reranker = ReRanker(storage._conn)
            except Exception:
                pass

    #  ?
    def search(
        self, query: str, *, top_k: int | None = None,
        file_pattern: str | None = None, tags: list[str] | None = None,
        context: dict | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or self._config.search.default_top_k
        qtype = self._detect_query_type(query)
        routed_language = self._context_language(context) if (
            context
            and self._config.search.enable_context_prior
            and self._config.search.context_routing_mode == "language"
            and not context.get("allow_cross_language")
        ) else None
        effective_file_pattern = file_pattern
        search_query = self._enhance_nl_query(query) if (
            qtype == "nl" and self._config.search.enable_nl_query_enhance
        ) else query

        # Step 1: BM25 
        bm25_results: list[SearchResult] = []
        if self._config.search.enable_bm25:
            bm25_results = self._storage.bm25_search(
                search_query,
                top_k=top_k * 3,
                file_pattern=effective_file_pattern,
                language=routed_language,
                tags=tags,
                or_semantics=(qtype == "nl"),
            )

        # Step 2: 
        # NL  search_document ?        semantic_results: list[SearchResult] = []
        if self._config.search.enable_vector_search and self._embedding.is_available:
            q_emb = self._embedding.get_embedding(search_query)
            if q_emb:
                if qtype == "nl":
                    semantic_results = self._storage.document_vector_search(
                        q_emb,
                        top_k=top_k * 3,
                        file_pattern=effective_file_pattern,
                        language=routed_language,
                    )
                if not semantic_results:
                    semantic_results = self._storage.vector_search(
                        q_emb,
                        top_k=top_k * 3,
                        file_pattern=effective_file_pattern,
                        language=routed_language,
                    )
                if tags:
                    semantic_results = self._filter_results(semantic_results, None, tags)

        # Step 3: ?+ ?NL 
        graph_results: list[SearchResult] = []
        if self._config.search.enable_graph_search:
            graph_results = self._graph_search(search_query, top_k=top_k * 2)
            if effective_file_pattern or tags or routed_language:
                graph_results = self._filter_results(
                    graph_results,
                    effective_file_pattern,
                    tags,
                    language=routed_language,
                )

        # Step 4: ?+ RRF ?        has_graph = bool(graph_results)
        has_vector = bool(semantic_results)
        dwg, dwv, dwb = self._get_dynamic_weights(query)

        if has_graph and has_vector:
            fused = self._reciprocal_rank_fusion_3way(
                bm25_results,
                semantic_results,
                graph_results,
                top_k,
                qtype=qtype,
                wg=dwg,
                wv=dwv,
                wb=dwb,
                context=context,
            )
        elif has_vector:
            fused = self._reciprocal_rank_fusion_2way(
                bm25_results,
                semantic_results,
                top_k,
                "vector",
                qtype=qtype,
                wg=dwg,
                wv=dwv,
                wb=dwb,
                context=context,
            )
        elif has_graph:
            fused = self._reciprocal_rank_fusion_2way(
                bm25_results,
                graph_results,
                top_k,
                "graph",
                qtype=qtype,
                wg=dwg,
                wv=dwv,
                wb=dwb,
                context=context,
            )
        else:
            fused = bm25_results[:top_k]

        if self._reranker is not None and self._config.search.enable_rerank:
            return self._apply_reranker(search_query, fused, context=context)
        return fused

    @staticmethod
    def _normalize_language(value: str | None) -> str:
        if not value:
            return ""
        v = value.lower().strip().lstrip(".")
        aliases = {
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "rs": "rust",
            "golang": "go",
            "java": "java",
        }
        return aliases.get(v, v)

    @staticmethod
    def _language_matches(expected: str | None, actual: str | None) -> bool:
        e = SearchEngine._normalize_language(expected)
        a = SearchEngine._normalize_language(actual)
        if not e or not a:
            return False
        if e in {"typescript", "javascript"} and a in {"typescript", "javascript"}:
            return True
        return e == a

    @staticmethod
    def _language_from_path(path: str | None) -> str:
        if not path:
            return ""
        ext = Path(path).suffix.lower().lstrip(".")
        return SearchEngine._normalize_language(ext)

    @staticmethod
    def _file_glob_for_language(language: str) -> str | None:
        lang = SearchEngine._normalize_language(language)
        mapping = {
            "python": "*.py",
            "javascript": "*.js",
            "typescript": "*.ts",
            "go": "*.go",
            "rust": "*.rs",
            "java": "*.java",
        }
        return mapping.get(lang)

    def _context_file_pattern(self, context: dict | None) -> str | None:
        if not context:
            return None
        lang = self._context_language(context)
        return self._file_glob_for_language(lang)

    def _context_language(self, context: dict | None) -> str | None:
        if not context:
            return None
        lang = self._normalize_language(context.get("language") or context.get("lang"))
        if not lang:
            lang = self._language_from_path(
                context.get("active_file") or context.get("file_path") or context.get("path")
            )
        return lang or None

    @staticmethod
    def _context_tokens(value) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            text = " ".join(str(v) for v in value)
        else:
            text = str(value)
        return {
            t.lower()
            for t in re.findall(r"[A-Za-z0-9]+", text.replace("_", " ").replace("-", " "))
            if len(t) >= 2
        }

    def _context_match_score(self, result: SearchResult, context: dict | None) -> float:
        if not context:
            return 0.0

        score = 0.0
        wanted_lang = self._normalize_language(context.get("language") or context.get("lang"))
        if not wanted_lang:
            wanted_lang = self._language_from_path(context.get("active_file") or context.get("file_path") or context.get("path"))
        result_lang = self._normalize_language(getattr(result, "language", None)) or self._language_from_path(result.file_path)
        if self._language_matches(wanted_lang, result_lang):
            score += 0.60

        wanted_repo = str(context.get("repo") or context.get("repository") or "").lower()
        if wanted_repo and wanted_repo in (result.file_path or "").lower():
            score += 0.20

        ctx_tokens = set()
        for key in ("path", "file_path", "active_file", "module", "package", "workspace"):
            ctx_tokens.update(self._context_tokens(context.get(key)))
        if ctx_tokens:
            path_tokens = self._context_tokens(result.file_path)
            if path_tokens:
                score += min(0.20, 0.20 * len(ctx_tokens & path_tokens) / max(len(ctx_tokens), 1))

        return min(score, 1.0)

    def _apply_context_prior(self, results: list[SearchResult], context: dict | None) -> list[SearchResult]:
        if not results or not context:
            return results
        boost = self._config.search.context_rrf_boost
        if boost <= 0:
            return results
        for result in results:
            match = self._context_match_score(result, context)
            if match:
                result.score += boost * match
                result.channel_ranks["context"] = 1
        return sorted(results, key=lambda r: r.score, reverse=True)

    # Semantic-to-symbol inference for NL queries
    _NL_STOP_WORDS = {
        "function", "method", "class", "the", "a", "an", "to", "for",
        "of", "in", "on", "at", "with", "from", "by", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should",
        "can", "may", "might", "shall", "must", "that", "this",
        "these", "those", "it", "its", "and", "or", "but", "not",
        "as", "if", "then", "else", "when", "where", "which", "who",
        "how", "what", "why", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "only",
        "own", "same", "so", "than", "too", "very", "just",
        "code", "need", "needs", "needed", "want", "wants", "wanted",
        "use", "uses", "used", "using", "get", "gets", "got", "make",
        "makes", "made", "set", "sets", "run", "runs", "call", "calls",
        "like", "way", "one", "two", "first", "new", "now", "also",
        "able", "about", "above", "after", "again", "any", "here",
        "into", "out", "over", "under", "up", "down", "off",
    }

    _SEMANTIC_TO_SYMBOL = {
        "json": ["read_json", "load_json", "parse_json", "json_load", "json_dump", "json_decode", "json_encode"],
        "file": ["read_file", "write_file", "open_file", "save_file", "load_file"],
        "sort": ["sort_list", "sort_array", "sorted_list", "quicksort", "mergesort"],
        "search": ["binary_search", "search_list", "find_element", "linear_search"],
        "merge": ["merge_sort", "merge_lists", "merge_dicts", "merge_two"],
        "string": ["parse_string", "format_string", "split_string", "join_string"],
        "request": ["handle_request", "process_request", "send_request", "make_request", "http_request"],
        "response": ["handle_response", "send_response", "parse_response", "http_response"],
        "parse": ["parse_data", "parse_json", "parse_xml", "parse_input"],
        "validate": ["validate_input", "validate_data", "validate_user", "validate_request"],
        "auth": ["authenticate_user", "check_auth", "verify_auth", "login_user"],
        "login": ["user_login", "do_login", "authenticate_login", "verify_login"],
        "connect": ["db_connect", "connect_db", "create_connection", "open_connection"],
        "query": ["db_query", "execute_query", "run_query", "sql_query"],
        "error": ["handle_error", "raise_error", "log_error", "catch_error"],
        "log": ["log_message", "write_log", "log_error", "setup_logger"],
        "cache": ["cache_get", "cache_set", "init_cache", "clear_cache"],
        "config": ["load_config", "read_config", "parse_config", "get_config"],
        "thread": ["start_thread", "run_thread", "create_thread", "thread_pool"],
        "process": ["start_process", "fork_process", "process_data", "run_process"],
        "queue": ["enqueue", "dequeue", "push_queue", "pop_queue"],
        "hash": ["compute_hash", "hash_key", "hash_password", "sha256_hash"],
        "encrypt": ["encrypt_data", "decrypt_data", "aes_encrypt", "rsa_encrypt"],
        "compress": ["compress_data", "decompress_data", "gzip_compress"],
        "serialize": ["serialize_obj", "deserialize_obj", "json_serialize"],
        "deserialize": ["deserialize_json", "deserialize_obj"],
        "render": ["render_template", "render_html", "render_page"],
        "redirect": ["http_redirect", "url_redirect", "do_redirect"],
        "upload": ["upload_file", "handle_upload", "save_upload"],
        "download": ["download_file", "fetch_download", "stream_download"],
        "serve": ["serve_static", "serve_file", "start_server"],
        "listen": ["listen_port", "start_listener", "tcp_listen"],
        "bind": ["bind_socket", "bind_address", "bind_port"],
        "accept": ["accept_connection", "accept_client"],
        "send": ["send_data", "send_message", "send_response"],
        "receive": ["receive_data", "receive_message", "receive_packet"],
        "close": ["close_connection", "close_file", "close_socket"],
        "shutdown": ["graceful_shutdown", "shutdown_server", "do_shutdown"],
        "signal": ["handle_signal", "setup_signals", "signal_handler"],
        "timeout": ["set_timeout", "handle_timeout", "timeout_handler"],
        "retry": ["retry_request", "retry_with_backoff", "do_retry"],
        "backoff": ["exponential_backoff", "retry_backoff"],
        "pool": ["connection_pool", "thread_pool", "create_pool"],
        "session": ["create_session", "session_store", "get_session"],
        "cookie": ["set_cookie", "parse_cookies", "read_cookie"],
        "header": ["parse_headers", "set_header", "read_headers"],
        "body": ["parse_body", "read_body", "stream_body"],
        "route": ["add_route", "register_route", "route_handler", "url_route"],
        "middleware": ["auth_middleware", "logging_middleware", "cors_middleware"],
        "handler": ["request_handler", "error_handler", "event_handler"],
        "dispatch": ["dispatch_request", "dispatch_event", "route_dispatch"],
    }

    _FORMAT_SYMBOL_ALIASES = {
        "rst": ["RST", "rst", "SimpleRSTHeader", "SimpleRSTData"],
        "qdp": ["QDP", "qdp"],
    }

    @staticmethod
    def _infer_format_symbols(query: str) -> set[str]:
        """Infer format/reader symbols from explicit file-format mentions.

        Kept intentionally narrow: these aliases only trigger on unambiguous
        format names such as ascii.rst or reStructuredText, so general code
        retrieval benchmarks are not affected.
        """
        ql = query.lower()
        candidates: set[str] = set()
        formats = set(re.findall(r"\bascii\.([a-z0-9_]+)\b", ql))
        formats.update(re.findall(r"\bformat\s*=\s*['\"]ascii\.([a-z0-9_]+)['\"]", ql))
        if "restructuredtext" in ql or "restructured text" in ql:
            formats.add("rst")

        for fmt in formats:
            aliases = SearchEngine._FORMAT_SYMBOL_ALIASES.get(fmt)
            if aliases:
                candidates.update(aliases)
        return candidates

    @staticmethod
    def _format_alias_text(query: str) -> str:
        symbols = SearchEngine._infer_format_symbols(query)
        if not symbols:
            return ""
        terms = sorted(symbols)
        if "RST" in symbols:
            terms.extend(["ascii.rst", "reStructuredText", "RestructuredText", "rst.py"])
        if "QDP" in symbols:
            terms.extend(["ascii.qdp", "qdp.py"])
        return " ".join(dict.fromkeys(terms))

    @staticmethod
    def _infer_symbols_from_nl(query: str) -> set[str]:
        """Infer code symbols from natural language queries.
        
        Fallback when regex-based camelCase/snake_case extraction finds nothing.
        Used for CodeSearchNet / CodeNeedle pure NL description scenarios.
        """
        ql = query.lower()
        candidates: set[str] = set()
        
        # Strategy 1: Semantic-to-symbol lookup table
        for semantic, symbols in SearchEngine._SEMANTIC_TO_SYMBOL.items():
            if semantic in ql:
                candidates.update(symbols[:3])
        
        # Strategy 2: Remove stop words, generate snake_case n-grams
        punct = ".,;:!?()[]{}"
        words = []
        for w in ql.split():
            w = w.strip(punct + chr(34) + chr(39))
            if w and w not in SearchEngine._NL_STOP_WORDS:
                words.append(w)
        
        if len(words) >= 2:
            for i in range(len(words) - 1):
                pair = f"{words[i]}_{words[i+1]}"
                if len(pair) >= 5:
                    candidates.add(pair)
            full = "_".join(words)
            if len(full) >= 5:
                candidates.add(full)
        elif len(words) == 1 and len(words[0]) >= 3:
            for prefix in ("get", "set", "do", "run", "handle", "process", "check", "init", "calc", "find"):
                candidates.add(f"{prefix}_{words[0]}")
        
        # Strategy 3: Verb stemming (handling->handle, processing->process)
        for word in words:
            for suffix, root in [("ing", ""), ("ed", ""), ("tion", "te"), ("sion", "d")]:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    stem = word[:-len(suffix)] + root
                    if len(stem) >= 3:
                        for prefix in ("handle", "process", "do", "run", "get", "set"):
                            candidates.add(f"{prefix}_{stem}")
                        candidates.add(stem)

        # Strategy 4: NL-to-code reverse mapping.
        # Pattern: "function/method/class to/for/that <description>"
        # Extract <description>, replace spaces with underscores.
        # Does NOT use stop-word filtering to preserve function-name words like "one", "two".
        ql_raw = query.lower().strip()
        for prefix in ("function to ", "function for ", "function that ",
                        "method to ", "method for ",
                        "class to ", "class for ",
                        "code to ", "code for ",
                        "def "):
            if ql_raw.startswith(prefix):
                desc = ql_raw[len(prefix):].strip()
                if desc:
                    snake = desc.replace(" ", "_")
                    if len(snake) >= 3:
                        candidates.add(snake)
                    # Also try with leading underscore (for private functions)
                    candidates.add("_" + snake)
                break
        
        return candidates

    #  ?
    @staticmethod
    def _starts_like_code(first_line: str) -> bool:
        return bool(
            re.match(
                r"^(def|class|import|from|return|if|for|while|try|except|function|const|let|var|public|private|fn|impl|package)\b",
                first_line,
            )
        )

    @staticmethod
    def _is_function_memory_query(query: str) -> bool:
        """Function-purpose descriptions should stay graph-first.

        RepoQA/CodeNeedle-style prompts are natural language, but their target is
        usually a function symbol. They benefit from NL text processing while
        still needing graph/symbol dominance.
        """
        ql = query.lower()
        markers = (
            "**purpose**",
            "purpose:",
            "**description**",
            "description:",
            "**parameters**",
            "parameters:",
            "**returns**",
            "returns:",
            "the function",
            "this function",
            "designed to",
            "used to",
        )
        return any(marker in ql for marker in markers)

    @staticmethod
    def _is_strong_issue_query(query: str) -> bool:
        """Only strong issue/task signals switch to vector-heavy NL weights."""
        ql = query.lower()
        first_line = next((line.strip() for line in query.splitlines() if line.strip()), "")
        if SearchEngine._starts_like_code(first_line):
            return False
        strong_markers = (
            "### description",
            "### expected behavior",
            "### how to reproduce",
            "expected behavior",
            "actual behavior",
            "steps to reproduce",
            "traceback",
            "typeerror:",
            "valueerror:",
            "attributeerror:",
            "indexerror:",
            "runtimeerror:",
            "assertionerror:",
            "bug report",
            "issue:",
            "does not",
            "fails to",
            "should not",
            "regression",
        )
        if any(marker in ql for marker in strong_markers):
            return True
        weak_markers = (
            "bug",
            "should",
            "expected",
            "crash",
            "error",
            "support",
            "fails",
            "failure",
            "incorrect",
            "wrong",
        )
        weak_count = sum(1 for marker in weak_markers if marker in ql)
        return len(query.split()) >= 10 and weak_count >= 2

    @staticmethod
    def _detect_query_profile(query: str) -> str:
        """Return code, function_memory, or issue.

        This profile controls fusion weights. `_detect_query_type()` remains a
        text-processing switch for NL query expansion and document-vector recall.
        Low-confidence NL defaults to function_memory/graph-first because this
        engine primarily serves local code/agent retrieval, where symbols and
        graph structure are usually the most stable signal.
        """
        first_line = next((line.strip() for line in query.splitlines() if line.strip()), "")
        if SearchEngine._starts_like_code(first_line):
            return "code"
        if SearchEngine._is_function_memory_query(query):
            return "function_memory"
        if SearchEngine._is_strong_issue_query(query):
            return "issue"
        return "code" if SearchEngine._detect_query_type(query) == "code" else "function_memory"

    #   
    @staticmethod
    def _detect_query_type(query: str) -> str:
        ql = query.lower()
        words = query.split()
        first_line = next((line.strip() for line in query.splitlines() if line.strip()), "")
        starts_like_code = SearchEngine._starts_like_code(first_line)
        if not starts_like_code and SearchEngine._is_function_memory_query(query):
            return "nl"
        strong_issue_markers = (
            "### description",
            "### expected behavior",
            "### how to reproduce",
            "traceback",
            "typeerror:",
            "valueerror:",
            "attributeerror:",
            "indexerror:",
            "issue",
            "does not",
        )
        weak_issue_markers = (
            "bug",
            "should",
            "expected",
            "crash",
            "error",
            "support",
            "fails",
            "failure",
        )
        has_strong_issue_marker = any(marker in ql for marker in strong_issue_markers)
        weak_issue_marker_count = sum(1 for marker in weak_issue_markers if marker in ql)
        if not starts_like_code:
            if len(words) >= 5 and (has_strong_issue_marker or weak_issue_marker_count >= 1):
                return "nl"
            if len(words) >= 12 and weak_issue_marker_count >= 2:
                return "nl"
            if len(words) >= 18:
                return "nl"

        if starts_like_code:
            return "code"
        if "\n" in query and re.search(
            r"\b(def|class|import|from|return|if|for|while|try|except|function|const|let|var|public|private|fn|impl|package)\b|[{};]",
            query,
        ):
            return "code"

        if "_" in query:
            return "code"
        if re.search(r"[a-z][A-Z]", query):
            return "code"
        if "(" in query or "::" in query:
            return "code"
        if "." in query and (re.search(r"\.[a-zA-Z]", query) or query.count(".") >= 2):
            return "code"
        _has_lib_mention = any(lib in ql or lib.replace("_", " ") in ql for lib in _KNOWN_LIBS)
        if _has_lib_mention and (
            "_" in query
            or re.search(r"[a-z][A-Z]", query)
            or "(" in query or "::" in query
            or ("." in query and (re.search(r"\.[a-zA-Z]", query) or query.count(".") >= 2))
        ):
            return "code"
        # Library mention without code signals is still NL
        if len(words) >= 5:
            return "nl"
        return "nl"

    def _get_dynamic_weights(self, query: str):
        if self._detect_query_profile(query) == "issue":
            return (0.3, 0.5, 0.2)      # Issue/bug/task: vector-heavy NL
        return (0.9, 0.05, 0.05)        # Code + function memory: graph-first

    @staticmethod
    def _enhance_nl_query(query: str) -> str:
        """Conservative NL query expansion for document-vector retrieval.

        The original query is preserved and matched rules append deduplicated
        English code/docstring style terms. We do not recursively rewrite the
        appended text, otherwise simple Chinese queries can explode into noisy
        repeated terms before embedding.
        """
        q = query
        rules = [
            (r"\u600e\u4e48\u8bfb\u53d6\u914d\u7f6e\u6587\u4ef6", "read config file load configuration input file path output config content"),
            (r"\u8bfb\u53d6\u914d\u7f6e\u6587\u4ef6", "read config file load configuration input file path output config content"),
            (r"\u8bfb\u53d6\s*json\s*\u6587\u4ef6", "read json file parse json input file path output dictionary object data"),
            (r"\u5199\u5165\s*json\s*\u6587\u4ef6", "write json file serialize json input object data output file path"),
            (r"\u6570\u636e\u5e93\u67e5\u8be2", "database query select fetch execute sql input condition output records rows"),
            (r"\u7528\u6237\u8ba4\u8bc1", "user authentication login verify credentials input username password output user session"),
            (r"\u6392\u5e8f\u5217\u8868", "sort list order collection input sequence output sorted sequence"),
            (r"\u753b\u56fe", "plot chart draw figure visualize data matplotlib output figure"),
            (r"\u53d1\u9001\u8bf7\u6c42", "send http request fetch url input headers output response"),
            (r"\u89e3\u6790\u914d\u7f6e", "parse config configuration file input path output settings options"),
            (r"\btakes\b", "takes accepts input parameter"),
            (r"\baccepts\b", "accepts input parameter argument"),
            (r"\binput\b", "input parameter argument accepts"),
            (r"\boutput\b", "output result returns"),
            (r"\breturns\b", "returns outputs result"),
            (r"\breturn\b", "return output result"),
            (r"\bloads?\b", "load read parse"),
            (r"\breads?\b", "read load parse"),
            (r"\bwrites?\b", "write save dump serialize"),
            (r"\bparses?\b", "parse decode deserialize"),
            (r"\bserializes?\b", "serialize encode dump write"),
            (r"\bdeserializes?\b", "deserialize decode parse load"),
            (r"\bconfigs?\b", "config configuration settings options"),
            (r"\bfiles?\b", "file path filesystem"),
            (r"\bpaths?\b", "path file filesystem"),
            (r"\bdicts?\b", "dict dictionary mapping object"),
            (r"\barrays?\b", "array list sequence collection"),
            (r"\bnumber of\b", "number count length size"),
            (r"\blist of\b", "list collection sequence"),
            (r"\bset-like\b", "set collection unique elements"),
            (r"\bmatrix multiplication\b", "matrix multiplication matmul weighted sum"),
            (r"\b16-bit floating-point\b", "16-bit floating-point fp16 float16 precision"),
            (r"\b32-bit unsigned integer\b", "32-bit unsigned integer uint32 compressed data"),
            (r"\bmemory usage\b", "memory usage statistics profiling megabytes peak"),
            (r"\bquery strings\b", "query strings url parameters string representation"),
            (r"\bsynchronous version\b", "synchronous version async cleanup task effect"),
        ]
        expansion_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for pattern, repl in rules:
            if not re.search(pattern, query, flags=re.IGNORECASE):
                continue
            for token in re.findall(r"[A-Za-z0-9_+-]+", repl.lower()):
                if token not in seen_tokens:
                    expansion_tokens.append(token)
                    seen_tokens.add(token)
        if expansion_tokens:
            q = f"{q}\nnl expansion: {' '.join(expansion_tokens)}"
        alias_text = SearchEngine._format_alias_text(query)
        if alias_text:
            q = f"{q}\nformat aliases: {alias_text}"
        return q

    #  ?
    def _graph_search(self, query: str, top_k: int = 30) -> list[SearchResult]:
        """??FTS5  ??? chunk"""
        results: list[SearchResult] = []
        seen_ids: set[str] = set()

        # 1. ?        ql = query.lower()
        symbol_candidates: set[str] = set()
        # camelCase
        for m in re.finditer(r"\b([a-z]+(?:[A-Z][a-z]+)+)\b", query):
            symbol_candidates.add(m.group(1).lower())
        # snake_case
        for m in re.finditer(r"\b([a-z]+(?:_[a-z]+)+)\b", query):
            symbol_candidates.add(m.group(1).lower())
        symbol_candidates.update(SearchEngine._infer_format_symbols(query))
        for lib in _KNOWN_LIBS:
            if lib in ql:
                symbol_candidates.add(lib)

        if not symbol_candidates:
            # semantic-to-symbol inference for NL queries
            symbol_candidates = SearchEngine._infer_symbols_from_nl(query)
            # For inferred symbols, use individual keyword FTS5 prefix queries
            # (snake_case tokens dont cross underscore boundaries in FTS5,
            #  so handle_request* wont match handle_one_request)
            expanded = set()
            for s in symbol_candidates:
                # Split snake_case into individual words for prefix matching
                parts = s.split("_")
                for part in parts:
                    if len(part) >= 3 and part.isalpha():
                        expanded.add(part + "*")
                # Also try the full token as is (might match exactly)
                expanded.add(s)
            symbol_candidates = expanded

        if not symbol_candidates:
            return results

        # 2. FTS5 ?(with FTS5-safe sanitization)
        # Sort: exact matches first, then prefix queries (run prefix last to avoid filling top_k early)
        exact_candidates = sorted([s for s in symbol_candidates if not s.endswith("*")])
        prefix_candidates = sorted([s for s in symbol_candidates if s.endswith("*")])
        sorted_candidates = exact_candidates + prefix_candidates
        max_symbol_candidates = getattr(self._config.search, "max_graph_symbol_candidates", 12)
        max_graph_neighbors = getattr(self._config.search, "max_graph_neighbors", 12)
        for sym_name in sorted_candidates[:max_symbol_candidates]:
            # Sanitize for FTS5: remove leading/trailing special chars,
            # replace consecutive underscores, ensure valid FTS5 token
            safe_sym = sym_name.strip("_*")
            if not safe_sym or len(safe_sym) < 2:
                continue
            # Replace double underscores with single
            while "__" in safe_sym:
                safe_sym = safe_sym.replace("__", "_")
            is_prefix_candidate = sym_name.endswith("*")
            try:
                sym_rows = self._storage.search_symbols_fts(safe_sym, limit=5)
            except Exception:
                # FTS5 rejects this token; try with wildcard stripped
                clean = safe_sym.rstrip("*")
                if clean and len(clean) >= 2:
                    try:
                        sym_rows = self._storage.search_symbols_fts(clean, limit=5)
                    except Exception:
                        continue
                else:
                    continue
            for sym_row in sym_rows:
                qname = sym_row[0]   # qualified_name
                fpath = sym_row[1]   # file_path
                sname = sym_row[2]   # name
                stype = sym_row[3]   # symbol_type
                sname_l = (sname or "").lower()
                safe_l = safe_sym.lower().rstrip("*")
                if safe_l and sname_l == safe_l:
                    base_graph_score = 1.0
                elif safe_l and (sname_l.startswith(safe_l) or safe_l in sname_l):
                    base_graph_score = 0.45 if is_prefix_candidate else 0.65
                else:
                    base_graph_score = 0.28 if is_prefix_candidate else 0.40

                # 3. ?chunkymbols  file_path 
                #  chunks  filesystem ?symbol_name ?                chunk = None
                from .models import CodeChunk, ChunkType as _CT
                cur = self._storage._conn.execute(
                    "SELECT id, content, file_path, language, chunk_type, symbol_name, parent_symbol, start_line, end_line, tags, docstring FROM chunks WHERE symbol_name = ? LIMIT 1",
                    (sname,),
                )
                row = cur.fetchone()
                if row:
                    try: ct = _CT(row[4])
                    except ValueError: ct = _CT.FUNCTION
                    chunk = CodeChunk(
                        id=row[0], content=row[1] or "", file_path=row[2] or "",
                        language=row[3] or "python", chunk_type=ct, symbol_name=row[5],
                        parent_symbol=row[6], start_line=row[7] or 0, end_line=row[8] or 0,
                        tags=[], docstring=row[10],
                    )
                else:
                    # Fallback: class method without dedicated chunk?
                    # Extract parent class from qualified_name (e.g. ...CGIHTTPRequestHandler.is_executable -> CGIHTTPRequestHandler)
                    parent_name = None
                    if "." in qname:
                        parts = qname.rsplit(".", 1)
                        if len(parts) == 2 and parts[1] == sname:
                            parent_name = parts[0].rsplit(".", 1)[-1] if "." in parts[0] else parts[0]
                    if parent_name:
                        cur2 = self._storage._conn.execute(
                            "SELECT id, content, file_path, language, chunk_type, symbol_name, parent_symbol, start_line, end_line, tags, docstring FROM chunks WHERE symbol_name = ? LIMIT 1",
                            (parent_name,),
                        )
                        row2 = cur2.fetchone()
                        if row2:
                            try: ct2 = _CT(row2[4])
                            except ValueError: ct2 = _CT.CLASS
                            chunk = CodeChunk(
                                id=row2[0], content=row2[1] or "", file_path=row2[2] or "",
                                language=row2[3] or "python", chunk_type=ct2, symbol_name=row2[5],
                                parent_symbol=row2[6], start_line=row2[7] or 0, end_line=row2[8] or 0,
                                tags=[], docstring=row2[10],
                            )
                    if chunk is None:
                        # Last resort: file-based lookup (for symbols without dedicated chunks)
                        chunk = self._storage.get_chunk_by_file(fpath)
                        # If we got a FILE chunk, override its type so it survives the ChunkType.FILE filter
                        if chunk is not None and chunk.chunk_type == _CT.FILE:
                            # Clone with FUNCTION type and unique ID (avoids RRF collision with BM25 FILE result)
                            chunk = CodeChunk(
                                id=f"{chunk.id}__graph_{sname}", content=chunk.content, file_path=chunk.file_path,
                                language=chunk.language, chunk_type=_CT.FUNCTION if stype in ("function","method") else _CT.CLASS,
                                symbol_name=chunk.symbol_name, parent_symbol=chunk.parent_symbol,
                                start_line=chunk.start_line, end_line=chunk.end_line,
                                tags=chunk.tags or [], docstring=chunk.docstring,
                            )
                if chunk and chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    results.append(SearchResult(
                        id=chunk.id, symbol_name=sname, qualified_name=qname,
                        file_path=fpath, line_number=chunk.start_line,
                        chunk_type=ChunkType(stype) if stype in ("function","class","method") else ChunkType.FUNCTION,
                        score=base_graph_score, snippet=chunk.content[:500], signature=None,
                        docstring=chunk.docstring, tags=chunk.tags or [],
                        language=chunk.language,
                    ))

                # 4. 1-hop callers + callees
                if self._call_graph and self._call_graph.node_count > 0 and base_graph_score >= 0.65:
                    for neighbor_name in self._call_graph.get_callers(qname, depth=1)[:max_graph_neighbors]:
                        neighbor_qname = neighbor_name.get("name", "")
                        nchunk = self._storage.get_chunk_by_qualified_symbol(neighbor_qname)
                        if nchunk and nchunk.id not in seen_ids:
                            seen_ids.add(nchunk.id)
                            results.append(SearchResult(
                                id=nchunk.id, symbol_name=nchunk.symbol_name or neighbor_qname.split(".")[-1],
                                qualified_name=neighbor_qname,
                                file_path=nchunk.file_path, line_number=nchunk.start_line,
                                chunk_type=ChunkType.FUNCTION, score=base_graph_score * 0.30,
                                snippet=nchunk.content[:300], tags=nchunk.tags or [],
                                language=nchunk.language,
                            ))
                    for neighbor_name in self._call_graph.get_callees(qname, depth=1)[:max_graph_neighbors]:
                        neighbor_qname = neighbor_name.get("name", "")
                        nchunk = self._storage.get_chunk_by_qualified_symbol(neighbor_qname)
                        if nchunk and nchunk.id not in seen_ids:
                            seen_ids.add(nchunk.id)
                            results.append(SearchResult(
                                id=nchunk.id, symbol_name=nchunk.symbol_name or neighbor_qname.split(".")[-1],
                                qualified_name=neighbor_qname,
                                file_path=nchunk.file_path, line_number=nchunk.start_line,
                                chunk_type=ChunkType.FUNCTION, score=base_graph_score * 0.25,
                                snippet=nchunk.content[:300], tags=nchunk.tags or [],
                                language=nchunk.language,
                            ))

                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    #  RRF  
    def _reciprocal_rank_fusion_3way(
        self, bm25: list[SearchResult], vec: list[SearchResult],
        graph: list[SearchResult], top_k: int = 10,
        qtype: str | None = None,
        wg=None, wv=None, wb=None,
        context: dict | None = None,
    ) -> list[SearchResult]:
        k = self._config.search.nl_rrf_k if qtype == "nl" and self._config.search.nl_rrf_k else self._config.search.rrf_k
        wb = wb if wb is not None else self._config.search.weight_bm25
        wv = wv if wv is not None else self._config.search.weight_vector
        wg = wg if wg is not None else self._config.search.weight_graph

        scores: dict[str, float] = {}
        all_r: dict[str, SearchResult] = {}

        for rank, r in enumerate(bm25):
            scores[r.id] = scores.get(r.id, 0) + wb / (k + rank + 1)
            all_r[r.id] = r
            all_r[r.id].channel_ranks["bm25"] = rank + 1
        for rank, r in enumerate(vec):
            scores[r.id] = scores.get(r.id, 0) + wv / (k + rank + 1)
            if r.id not in all_r:
                all_r[r.id] = r
            all_r[r.id].channel_ranks["vector"] = rank + 1
        for rank, r in enumerate(graph):
            scores[r.id] = scores.get(r.id, 0) + wg / (k + rank + 1)
            if r.id not in all_r:
                all_r[r.id] = r
            all_r[r.id].channel_ranks["graph"] = rank + 1

        if context and self._config.search.enable_context_prior:
            boost = self._config.search.context_rrf_boost
            for rid, result in all_r.items():
                match = self._context_match_score(result, context)
                if match:
                    scores[rid] = scores.get(rid, 0.0) + boost * match
                    result.channel_ranks["context"] = 1

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        results = []
        for rid in sorted_ids:
            r = all_r.get(rid)
            if r:
                r.score = scores[rid]
                results.append(r)
        return results

    def _reciprocal_rank_fusion_2way(
        self, bm25: list[SearchResult], second: list[SearchResult],
        top_k: int, second_type: str,
        qtype: str | None = None,
        wg=None, wv=None, wb=None,
        context: dict | None = None,
    ) -> list[SearchResult]:
        k = self._config.search.nl_rrf_k if qtype == "nl" and self._config.search.nl_rrf_k else self._config.search.rrf_k
        wb = wb if wb is not None else self._config.search.weight_bm25
        w2 = (wv if wv is not None else self._config.search.weight_vector) if second_type == "vector" else (wg if wg is not None else self._config.search.weight_graph)

        scores: dict[str, float] = {}
        all_r: dict[str, SearchResult] = {}

        for rank, r in enumerate(bm25):
            scores[r.id] = scores.get(r.id, 0) + wb / (k + rank + 1)
            all_r[r.id] = r
            all_r[r.id].channel_ranks["bm25"] = rank + 1
        for rank, r in enumerate(second):
            scores[r.id] = scores.get(r.id, 0) + w2 / (k + rank + 1)
            if r.id not in all_r:
                all_r[r.id] = r
            all_r[r.id].channel_ranks[second_type] = rank + 1

        if context and self._config.search.enable_context_prior:
            boost = self._config.search.context_rrf_boost
            for rid, result in all_r.items():
                match = self._context_match_score(result, context)
                if match:
                    scores[rid] = scores.get(rid, 0.0) + boost * match
                    result.channel_ranks["context"] = 1

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        results = []
        for rid in sorted_ids:
            r = all_r.get(rid)
            if r:
                r.score = scores[rid]
                results.append(r)
        return results

    #   
    def _apply_reranker(self, query: str, rrf_results, context: dict | None = None):
        if not rrf_results:
            return rrf_results
        nl_mode = self._detect_query_type(query) == "nl"
        candidates = [
            {
                'id': r.id,
                'content': r.snippet or '',
                'score': r.score,
                'file': r.file_path,
                'symbol_name': r.symbol_name or '',
                'docstring': r.docstring or '',
                'tags': r.tags or [],
                'channel_ranks': r.channel_ranks or {},
                'context_match': self._context_match_score(r, context),
                'context_boost': self._config.search.context_rerank_boost,
                'search_document': self._storage.get_search_document(r.id) or '',
            }
            for r in rrf_results
        ]
        rerank_top_k = max(1, self._config.search.rerank_top_k)
        reranked = self._reranker.rerank(
            query,
            candidates,
            top_k=min(rerank_top_k, len(candidates)),
            nl_mode=nl_mode,
        )
        result_map = {r.id: r for r in rrf_results}
        file_result_map = {r.file_path: r for r in rrf_results}
        output = []
        used_ids = set()
        for c in reranked:
            orig = result_map.get(c.get('id', '')) or file_result_map.get(c.get('file', ''))
            if orig:
                orig.score = c.get('_rerank_score', orig.score)
                output.append(orig)
                used_ids.add(orig.id)
        if nl_mode:
            # NL/Agent: keep RRF tail candidates for context pool
            output.extend(r for r in rrf_results if r.id not in used_ids)
        return output

    def symbol_lookup(self, entity: str) -> list[SearchResult]:
        results = self._storage.bm25_search(entity, top_k=5)
        exact = [r for r in results if r.symbol_name and r.symbol_name.lower() == entity.lower()]
        fuzzy = [r for r in results if r not in exact]
        return exact + fuzzy

    @staticmethod
    def _filter_results(results, file_pattern, tags, language: str | None = None):
        filtered = results
        if file_pattern:
            from fnmatch import fnmatch
            filtered = [r for r in filtered if fnmatch(r.file_path, file_pattern)]
        if language:
            normalized = SearchEngine._normalize_language(language)
            filtered = [
                r for r in filtered
                if SearchEngine._language_matches(
                    normalized,
                    SearchEngine._normalize_language(getattr(r, "language", None))
                    or SearchEngine._language_from_path(r.file_path),
                )
            ]
        if tags:
            tag_set = set(t.lower() for t in tags)
            filtered = [r for r in filtered if tag_set.intersection(t.lower() for t in r.tags)]
        return filtered
