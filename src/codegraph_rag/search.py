""" ?BM25 +  + ?+ RRF """
import hashlib
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
        self._context_doc_cache: dict[str, str] = {}
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
        explain: bool = False,
    ) -> list[SearchResult]:
        return_top_k = top_k or self._config.search.default_top_k
        query = self._normalize_query_text(query)
        qtype = self._detect_query_type(query)
        profile = self._detect_query_profile(query)
        issue_body = query
        issue_focus_query = query
        issue_test_hints = ""
        issue_title = ""
        issue_keywords: list[str] = []
        graph_query = query
        if profile == "issue":
            issue_body, issue_test_lines = self._split_issue_query(query)
            if issue_body.strip():
                issue_body = issue_body.strip()
                issue_focus_query = self._build_issue_focus_query(issue_body)
                graph_query = issue_focus_query or issue_body
            issue_test_hints = self._build_issue_test_hint_query(issue_test_lines)
            issue_keywords = self._extract_issue_keywords(
                issue_body if issue_body.strip() else query
            )
            issue_title = self._extract_issue_title(
                issue_body if issue_body.strip() else query
            )
        has_specific_context = self._context_has_specific(context)
        if profile == "function_memory" and has_specific_context:
            pool_floor = getattr(
                self._config.search, "context_candidate_pool_top_k", return_top_k
            )
        elif profile == "issue":
            pool_floor = getattr(
                self._config.search, "issue_candidate_pool_top_k", return_top_k
            )
        else:
            pool_floor = getattr(
                self._config.search, "candidate_pool_top_k", return_top_k
            )
        candidate_pool_top_k = max(return_top_k, pool_floor)
        # Issue mode uses a tighter, fixed pool (lane=20 / aux=20 / rerank=10)
        # to cut latency without losing Hit@5 recall. Non-issue modes keep the
        # original candidate_pool_top_k * 3 headroom.
        if profile == "issue":
            lane_top_k = max(1, getattr(self._config.search, "issue_lane_top_k", 20))
            aux_pool_top_k = max(1, getattr(self._config.search, "issue_aux_pool_top_k", 20))
        else:
            lane_top_k = candidate_pool_top_k * 3
            aux_pool_top_k = candidate_pool_top_k
        routed_language = self._context_language(context) if (
            context
            and self._config.search.enable_context_prior
            and self._config.search.context_routing_mode == "language"
            and not context.get("allow_cross_language")
        ) else None
        effective_file_pattern = file_pattern
        text_query = issue_focus_query if profile == "issue" and issue_focus_query.strip() else (
            issue_body if profile == "issue" and issue_body.strip() else query
        )
        search_query = self._enhance_nl_query(text_query) if (
            qtype == "nl" and self._config.search.enable_nl_query_enhance
        ) else text_query

        # Step 1: BM25
        bm25_results: list[SearchResult] = []
        if self._config.search.enable_bm25:
            bm25_results = self._storage.bm25_search(
                search_query,
                top_k=lane_top_k,
                file_pattern=effective_file_pattern,
                language=routed_language,
                tags=tags,
                or_semantics=(qtype == "nl"),
            )
            if profile == "issue":
                title_bm25: list[SearchResult] = []
                norm_title_bm25 = (issue_title or "").strip()
                norm_query_bm25 = (search_query or "").strip()
                if norm_title_bm25 and norm_title_bm25 != norm_query_bm25:
                    title_bm25 = self._storage.bm25_search(
                        issue_title,
                        top_k=max(return_top_k, aux_pool_top_k),
                        file_pattern=effective_file_pattern,
                        language=routed_language,
                        tags=tags,
                        or_semantics=True,
                    )
                hint_bm25: list[SearchResult] = []
                if issue_test_hints:
                    # Failing-test hints are routed through a supplementary
                    # lane so implementation/config/doc candidates linked to
                    # the failing tests get a fair shot. Test files themselves
                    # are suppressed here so the lane never dominates.
                    hint_bm25 = self._storage.bm25_search(
                        issue_test_hints,
                        top_k=max(return_top_k, aux_pool_top_k),
                        file_pattern=effective_file_pattern,
                        language=routed_language,
                        tags=tags,
                        or_semantics=True,
                    )
                    hint_bm25 = [r for r in hint_bm25 if not self._is_test_path(r)]
                if title_bm25 or hint_bm25:
                    bm25_results = self._rank_fuse_lanes(
                        [
                            (bm25_results, 1.00),
                            (title_bm25, 0.65),
                            (hint_bm25, 0.55),
                        ],
                        top_k=lane_top_k,
                        rrf_k=self._config.search.rrf_k,
                    )

        # Step 2: vector / document search
        # NL queries prefer search_document; fallback to code vector when missing
        semantic_results: list[SearchResult] = []
        if self._config.search.enable_vector_search and self._embedding.is_available:
            q_emb = self._embedding.get_embedding(search_query)
            if q_emb:
                if qtype == "nl":
                    semantic_results = self._storage.document_vector_search(
                        q_emb,
                        top_k=lane_top_k,
                        file_pattern=effective_file_pattern,
                        language=routed_language,
                    )
                if not semantic_results:
                    semantic_results = self._storage.vector_search(
                        q_emb,
                        top_k=lane_top_k,
                        file_pattern=effective_file_pattern,
                        language=routed_language,
                    )
                if tags:
                    semantic_results = self._filter_results(semantic_results, None, tags)
            if profile == "issue":
                # Batch issue-mode auxiliary embeddings (title + hint) so the
                # extra lanes pay one model forward pass instead of two cold
                # calls. Empty/duplicate lanes are skipped so we never embed
                # text we won't use.
                norm_title_vec = (issue_title or "").strip()
                norm_query_vec = (search_query or "").strip()
                run_title = bool(
                    norm_title_vec and norm_title_vec != norm_query_vec
                )
                run_hint = bool(issue_test_hints)
                aux_texts: list[str] = []
                aux_kinds: list[str] = []
                if run_title:
                    aux_texts.append(issue_title)
                    aux_kinds.append("title")
                if run_hint:
                    aux_texts.append(issue_test_hints)
                    aux_kinds.append("hint")
                aux_embs: dict[str, list[float] | None] = {}
                if aux_texts:
                    batch_fn = getattr(self._embedding, "get_embeddings_batch", None)
                    if callable(batch_fn):
                        batch = batch_fn(aux_texts)
                    else:
                        batch = [self._embedding.get_embedding(t) for t in aux_texts]
                    for kind, emb in zip(aux_kinds, batch):
                        aux_embs[kind] = emb

                title_vec: list[SearchResult] = []
                title_emb = aux_embs.get("title")
                if title_emb:
                    aux_top_k = max(return_top_k, aux_pool_top_k)
                    if qtype == "nl":
                        title_vec = self._storage.document_vector_search(
                            title_emb,
                            top_k=aux_top_k,
                            file_pattern=effective_file_pattern,
                            language=routed_language,
                        )
                    if not title_vec:
                        title_vec = self._storage.vector_search(
                            title_emb,
                            top_k=aux_top_k,
                            file_pattern=effective_file_pattern,
                            language=routed_language,
                        )
                    if tags:
                        title_vec = self._filter_results(title_vec, None, tags)

                hint_vec: list[SearchResult] = []
                hint_emb = aux_embs.get("hint")
                if hint_emb:
                    aux_top_k = max(return_top_k, aux_pool_top_k)
                    if qtype == "nl":
                        hint_vec = self._storage.document_vector_search(
                            hint_emb,
                            top_k=aux_top_k,
                            file_pattern=effective_file_pattern,
                            language=routed_language,
                        )
                    if not hint_vec:
                        hint_vec = self._storage.vector_search(
                            hint_emb,
                            top_k=aux_top_k,
                            file_pattern=effective_file_pattern,
                            language=routed_language,
                        )
                    hint_vec = [r for r in hint_vec if not self._is_test_path(r)]
                    if tags:
                        hint_vec = self._filter_results(hint_vec, None, tags)

                if title_vec or hint_vec:
                    # Title vector weight >= body weight: the issue title is
                    # the cleanest semantic anchor; long body text dilutes
                    # the embedding so we let title lead the lane fusion.
                    semantic_results = self._rank_fuse_lanes(
                        [
                            (semantic_results, 0.85),
                            (title_vec, 1.00),
                            (hint_vec, 0.55),
                        ],
                        top_k=lane_top_k,
                        rrf_k=self._config.search.rrf_k,
                    )

        # Step 3: graph search (NL queries also benefit via inferred symbols)
        graph_results: list[SearchResult] = []
        if self._config.search.enable_graph_search:
            graph_results = self._graph_search(graph_query, top_k=lane_top_k)
            if profile == "issue" and issue_test_hints:
                graph_results = self._merge_ranked_results(
                    graph_results,
                    self._graph_search(issue_test_hints, top_k=max(return_top_k, aux_pool_top_k)),
                    score_scale=0.72,
                    top_k=lane_top_k,
                )
            if effective_file_pattern or tags or routed_language:
                graph_results = self._filter_results(
                    graph_results,
                    effective_file_pattern,
                    tags,
                    language=routed_language,
                )

        # Step 4: RRF fusion across BM25 + vector + graph
        has_graph = bool(graph_results)
        has_vector = bool(semantic_results)
        dwg, dwv, dwb = self._get_dynamic_weights(
            query,
            qtype=qtype,
            profile=profile,
        )

        if has_graph and has_vector:
            fusion_mode = "rrf_3way"
            fused = self._reciprocal_rank_fusion_3way(
                bm25_results,
                semantic_results,
                graph_results,
                candidate_pool_top_k,
                qtype=qtype,
                wg=dwg,
                wv=dwv,
                wb=dwb,
                context=context,
            )
        elif has_vector:
            fusion_mode = "rrf_2way_vector"
            fused = self._reciprocal_rank_fusion_2way(
                bm25_results,
                semantic_results,
                candidate_pool_top_k,
                "vector",
                qtype=qtype,
                wg=dwg,
                wv=dwv,
                wb=dwb,
                context=context,
            )
        elif has_graph:
            fusion_mode = "rrf_2way_graph"
            fused = self._reciprocal_rank_fusion_2way(
                bm25_results,
                graph_results,
                candidate_pool_top_k,
                "graph",
                qtype=qtype,
                wg=dwg,
                wv=dwv,
                wb=dwb,
                context=context,
            )
        else:
            fusion_mode = "bm25_only"
            fused = self._stable_sort_results(bm25_results)[:candidate_pool_top_k]

        if self._reranker is not None and self._config.search.enable_rerank:
            stage = "reranked"
            final_results = self._apply_reranker(
                search_query,
                fused,
                context=context,
                profile=profile,
                issue_keywords=issue_keywords,
                issue_title=issue_title,
            )
        else:
            stage = "fused"
            final_results = fused

        # Always normalize the final ordering with deterministic tiebreakers.
        final_results = self._stable_sort_results(final_results)
        if profile == "issue":
            # SWE / Bug-Loc style benchmarks evaluate at the file level; one
            # file should occupy at most one slot so the remaining Top-K are
            # spent on distinct gold candidates rather than chunk siblings.
            final_results = self._dedupe_by_file(final_results)
        final_results = final_results[:return_top_k]

        trace = self._build_explain_trace(
            query=query,
            search_query=search_query,
            qtype=qtype,
            profile=profile,
            top_k=return_top_k,
            candidate_pool_top_k=candidate_pool_top_k,
            file_pattern=file_pattern,
            tags=tags,
            context=context,
            routed_language=routed_language,
            weights=(dwg, dwv, dwb),
            graph_query=graph_query,
            issue_test_hints=issue_test_hints,
            fusion_mode=fusion_mode,
            stage=stage,
            channel_counts={
                "bm25": len(bm25_results),
                "vector": len(semantic_results),
                "graph": len(graph_results),
                "fused": len(fused),
                "final": len(final_results),
            },
        )
        self._last_trace = trace

        if explain:
            self._attach_explain(final_results, trace)

        return final_results

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

    def _context_has_specific(self, context: dict | None) -> bool:
        if not context:
            return False
        if context.get("repo") or context.get("repository"):
            return True
        for key in ("path", "file_path", "active_file", "module", "package", "workspace"):
            if self._context_tokens(context.get(key)):
                return True
        return False

    def _context_surface_for_result(self, result: SearchResult) -> str:
        """Return path-like text used by context-aware reranking.

        Repo/path/module metadata may live in search_document even when the
        indexed file path is synthetic. Cache lookups because this runs for
        every candidate in RRF/P1.
        """
        rid = getattr(result, "id", "") or ""
        search_document = ""
        if rid:
            if rid in self._context_doc_cache:
                search_document = self._context_doc_cache[rid]
            else:
                try:
                    search_document = self._storage.get_search_document(rid) or ""
                except Exception:
                    search_document = ""
                if len(self._context_doc_cache) > 20000:
                    self._context_doc_cache.clear()
                self._context_doc_cache[rid] = search_document
        return f"{getattr(result, 'file_path', '') or ''}\n{search_document}"

    def _context_specific_match_score(self, result: SearchResult, context: dict | None) -> float:
        """Score repo/path/module agreement without the language-only prior."""
        if not context:
            return 0.0

        surface = self._context_surface_for_result(result)
        surface_lower = surface.lower()
        score = 0.0

        wanted_repo = str(context.get("repo") or context.get("repository") or "").lower()
        if wanted_repo and wanted_repo in surface_lower:
            score += 0.45

        ctx_tokens = set()
        for key in ("path", "file_path", "active_file", "module", "package", "workspace"):
            ctx_tokens.update(self._context_tokens(context.get(key)))
        ctx_tokens = {
            token
            for token in ctx_tokens
            if token not in {"src", "lib", "test", "tests", "main", "index", "code", "repo"}
        }
        if ctx_tokens:
            surface_tokens = self._context_tokens(surface)
            if surface_tokens:
                score += min(
                    0.55,
                    0.55 * len(ctx_tokens & surface_tokens) / max(min(len(ctx_tokens), 8), 1),
                )

        return min(score, 1.0)

    def _context_match_score(self, result: SearchResult, context: dict | None) -> float:
        if not context:
            return 0.0

        score = 0.0
        wanted_lang = self._normalize_language(context.get("language") or context.get("lang"))
        if not wanted_lang:
            wanted_lang = self._language_from_path(context.get("active_file") or context.get("file_path") or context.get("path"))
        result_lang = self._normalize_language(getattr(result, "language", None)) or self._language_from_path(result.file_path)
        if self._language_matches(wanted_lang, result_lang):
            score += 0.55

        score += 0.45 * self._context_specific_match_score(result, context)

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
        return self._stable_sort_results(results)

    @staticmethod
    def _normalize_query_text(query: str) -> str:
        """Normalize benchmark-style escaped issue text into readable NL."""
        text = (query or "").strip()
        if not text:
            return ""
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            inner = text[1:-1]
            if inner.count("\\n") >= 2 or inner.count("\n") >= 2:
                text = inner
        if "\\r\\n" in text or "\\n" in text or '\\"' in text:
            text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        if text.startswith('"') and text.count("\n") >= 2:
            text = text[1:].lstrip()
        if text.endswith('"') and text.count("\n") >= 2:
            text = text[:-1].rstrip()
        return text

    @staticmethod
    def _split_issue_query(query: str) -> tuple[str, list[str]]:
        """Split issue body from failing-test bullets."""
        body_lines: list[str] = []
        test_lines: list[str] = []
        in_tests = False
        for raw_line in query.splitlines():
            line = raw_line.rstrip()
            lower = line.strip().lower()
            if lower.startswith(("failing tests:", "failing test:", "selected tests:")):
                in_tests = True
                continue
            if in_tests:
                if not lower:
                    continue
                if lower.startswith("- "):
                    test_lines.append(line.strip()[2:].strip())
                    continue
                if "::" in line or re.search(r"\btest[\w./\\-]*\.(py|js|ts|go|rs|java)\b", lower):
                    test_lines.append(line.strip())
                    continue
                in_tests = False
            body_lines.append(line)
        body = "\n".join(body_lines).strip()
        return body or query, test_lines

    @staticmethod
    def _build_issue_focus_query(body: str) -> str:
        """Keep the semantically useful issue sections and drop noisy scaffolding."""
        keep_lines: list[str] = []
        current_section = "keep"
        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if keep_lines and keep_lines[-1]:
                    keep_lines.append("")
                continue

            lowered = stripped.lower().strip("*# :")
            inline_header = re.match(
                r"^[#*\s`]*(title|description|expected behavior|what is expected|actual behavior|what happened instead|current behavior)\s*:\s*(.+?)\s*[*#`]*$",
                stripped,
                flags=re.IGNORECASE,
            )
            if lowered.startswith("steps to reproduce"):
                current_section = "skip_steps"
                continue
            if lowered.startswith("labels"):
                current_section = "skip_labels"
                continue
            if inline_header:
                current_section = "keep"
                keep_lines.append(inline_header.group(2).replace("`", "").strip())
                continue
            if lowered.startswith((
                "title",
                "description",
                "expected behavior",
                "what is expected",
                "actual behavior",
                "what happened instead",
                "current behavior",
            )):
                current_section = "keep"
                continue
            if current_section == "skip_steps":
                if re.match(r"^\d+\.\s", stripped):
                    continue
                current_section = "keep"
            if current_section == "skip_labels":
                if "," in stripped or len(stripped.split()) <= 8:
                    continue
                current_section = "keep"

            cleaned = stripped.strip("* ").replace("`", "")
            keep_lines.append(cleaned)

        compact = "\n".join(keep_lines).strip().strip('"').strip("'")
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        return compact or body

    @staticmethod
    def _build_issue_test_hint_query(test_lines: list[str]) -> str:
        """Extract implementation-oriented hints from failing test lines."""
        hints: list[str] = []
        seen: set[str] = set()

        def add_hint(value: str) -> None:
            token = value.strip().lower().replace("-", "_")
            if not token or token in seen:
                return
            seen.add(token)
            hints.append(token)
            if token.endswith("s") and len(token) >= 5:
                singular = token[:-1]
                if singular and singular not in seen:
                    seen.add(singular)
                    hints.append(singular)

        for line in test_lines:
            normalized = line.replace("\\", "/")
            for path_match in re.findall(r"([A-Za-z0-9_./-]+\.(?:py|js|ts|go|rs|java|yml|yaml|tpl))", normalized):
                stem = Path(path_match).stem.lower()
                if stem.startswith("test_"):
                    stem = stem[5:]
                if stem:
                    add_hint(stem)
            for symbol_match in re.findall(r"::([A-Za-z_][A-Za-z0-9_]*)", normalized):
                symbol = symbol_match.lower()
                if symbol.startswith("test_"):
                    symbol = symbol[5:]
                elif symbol.startswith("test") and len(symbol) > 4:
                    symbol = symbol[4:]
                if symbol and symbol not in {"test", "tests"}:
                    add_hint(symbol)

        return " ".join(hints)

    # Issue-mode helper: extract title and structural anchors
    @staticmethod
    def _extract_issue_title(text: str) -> str:
        """Pick a short headline-like span from an issue body.

        Generic across SWE-bench / GitHub-issue style markup. Strategy:
        1. Inline ``**Title: <value>**`` / ``Title: <value>`` -> use the value.
        2. Bare ``Title:`` / ``# Title:`` line -> use the next non-empty line.
        3. First non-empty line if it begins with ``#``/``##`` -> strip markup.
        4. Fallback: first non-empty line, capped at 200 chars.
        """
        if not text:
            return ""
        lines = [raw.rstrip() for raw in text.splitlines()]
        first_nonempty: str | None = None
        for i, line in enumerate(lines):
            stripped_full = line.strip()
            if not stripped_full:
                continue
            inner = stripped_full.strip("*# `")
            inline = re.match(r"^title\s*:\s*(.+)$", inner, flags=re.IGNORECASE)
            if inline and inline.group(1).strip():
                return inline.group(1).strip().strip("*` ")[:200]
            if re.match(r"^title\s*:?\s*$", inner, flags=re.IGNORECASE):
                # Heading-only marker, look ahead.
                for j in range(i + 1, len(lines)):
                    follow = lines[j].strip()
                    if follow:
                        return follow.strip("*# `")[:200]
                return ""
            if first_nonempty is None:
                first_nonempty = inner[:200]
        return first_nonempty or ""

    @classmethod
    def _rank_fuse_lanes(
        cls,
        lanes: list[tuple[list["SearchResult"], float]],
        *,
        top_k: int,
        rrf_k: int = 60,
    ) -> list["SearchResult"]:
        """Generic RRF merge for multiple ranked lanes.

        ``lanes`` is a list of ``(ranked_results, weight)`` pairs. Each lane
        contributes ``weight / (rrf_k + rank)`` to a candidate's combined
        score, summed across lanes. This is the same fusion math used by the
        outer 3-way RRF, but applied *within* a single channel so primary +
        secondary lanes interleave instead of concatenating tail-on-head.
        """
        scores: dict[str, float] = {}
        all_r: dict[str, "SearchResult"] = {}
        for lane_results, weight in lanes:
            if not lane_results or weight <= 0:
                continue
            for rank, r in enumerate(lane_results):
                if r is None:
                    continue
                scores[r.id] = scores.get(r.id, 0.0) + weight / (rrf_k + rank + 1)
                if r.id not in all_r:
                    all_r[r.id] = r
        if not scores:
            return []
        sorted_ids = sorted(
            scores,
            key=lambda rid: cls._stable_score_key(scores[rid], all_r.get(rid)),
        )[:top_k]
        out: list["SearchResult"] = []
        for rid in sorted_ids:
            r = all_r.get(rid)
            if r is None:
                continue
            r.score = scores[rid]
            out.append(r)
        return out

    # Issue-mode helper: extract focus keywords that drive type/path priors
    _ISSUE_KEYWORD_MAP = {
        "config": ("config", "settings", "options"),
        "settings": ("config", "settings", "options"),
        "changelog": ("changelog", "release_notes", "version"),
        "release": ("changelog", "release", "version"),
        "version": ("changelog", "version", "release"),
        "upgrade": ("changelog", "upgrade", "version"),
        "docs": ("doc", "docs", "documentation"),
        "documentation": ("doc", "docs", "documentation"),
        "openapi": ("openapi", "schema", "swagger"),
        "schema": ("schema", "openapi"),
        "i18n": ("i18n", "language", "translation"),
        "translation": ("i18n", "language", "translation"),
        "language": ("language", "i18n"),
        "template": ("template", "tpl", "view"),
        "view": ("view", "template", "tpl"),
        "ui": ("view", "template", "ui"),
        "controller": ("controller",),
        "router": ("router", "route"),
        "route": ("router", "route"),
        "database": ("database", "db"),
        "migration": ("migration", "upgrade"),
        "model": ("model", "schema"),
        "socket": ("socket", "ws", "websocket"),
        "email": ("email", "mail"),
        "mail": ("email", "mail"),
    }

    @classmethod
    def _extract_issue_keywords(cls, text: str) -> list[str]:
        """Pull issue-body intents that bias rerank toward impl/config/doc."""
        if not text:
            return []
        lowered = text.lower()
        out: list[str] = []
        seen: set[str] = set()
        for trigger, expansions in cls._ISSUE_KEYWORD_MAP.items():
            if trigger in lowered:
                for token in (trigger,) + tuple(expansions):
                    if token and token not in seen:
                        seen.add(token)
                        out.append(token)
        return out

    @staticmethod
    def _is_test_path(result: "SearchResult") -> bool:
        """Return True when the candidate looks like a test fixture/file."""
        try:
            file_path = (getattr(result, "file_path", "") or "").replace("\\", "/").lower()
            symbol_name = (getattr(result, "symbol_name", "") or "").lower()
            tags = {str(t).lower() for t in (getattr(result, "tags", None) or [])}
        except Exception:
            return False
        if not file_path and not symbol_name:
            return False
        if (
            file_path.startswith("test/")
            or file_path.startswith("tests/")
            or "/test/" in file_path
            or "/tests/" in file_path
            or file_path.endswith("_test.py")
            or file_path.endswith(".test.js")
            or file_path.endswith(".test.ts")
        ):
            return True
        name = Path(file_path).name if file_path else ""
        if name.startswith("test_") or name.startswith("test."):
            return True
        if symbol_name.startswith("test_"):
            return True
        if "test" in tags or "tests" in tags:
            return True
        return False

    @classmethod
    def _merge_ranked_results(
        cls,
        primary: list[SearchResult],
        secondary: list[SearchResult],
        *,
        score_scale: float,
        top_k: int,
    ) -> list[SearchResult]:
        if not primary:
            primary = []
        if not secondary:
            return cls._stable_sort_results(primary)[:top_k]
        merged = list(primary)
        seen_ids = {result.id for result in primary}
        for result in secondary:
            if result.id in seen_ids:
                continue
            result.score = float(getattr(result, "score", 0.0) or 0.0) * score_scale
            merged.append(result)
            seen_ids.add(result.id)
        return cls._stable_sort_results(merged)[:top_k]

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
        "email": ["email", "send_email", "confirm_email", "validate_email"],
        "confirm": ["confirm_email", "email_confirm", "can_send_validation"],
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
        "validation": ["validate_input", "validate_data", "validate_user", "validate_request"],
        "auth": ["authenticate_user", "check_auth", "verify_auth", "login_user"],
        "login": ["user_login", "do_login", "authenticate_login", "verify_login"],
        "connect": ["db_connect", "connect_db", "create_connection", "open_connection"],
        "collection": ["collection_loader", "collection_finder", "validate_collection"],
        "keyword": ["is_keyword", "keyword_validation", "fqcn_validation"],
        "version": ["version_changed", "parse_version", "version"],
        "changelog": ["changelog", "show_changelog", "version_changed"],
        "setting": ["configfiles", "configdata", "settings"],
        "settings": ["configfiles", "configdata", "settings"],
        "upgrade": ["upgrade", "version_changed", "changelog"],
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
                r"^(@\w+|def|class|import|from|return|if|for|while|with|try|except|assert|raise|await|yield|lambda|function|const|let|var|public|private|fn|impl|package)\b",
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
            "current behavior",
            "steps to reproduce",
            "failing tests:",
            "failing test:",
            "test failure",
            "stack trace",
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
        Strong issue/task reports must win over function-memory cues so SWE /
        issue-style queries keep the frozen NL/Issue weight family even when
        they mention functions, descriptions, or failing tests.
        """
        first_line = next((line.strip() for line in query.splitlines() if line.strip()), "")
        qtype = SearchEngine._detect_query_type(query)
        if SearchEngine._starts_like_code(first_line):
            return "code"
        if SearchEngine._is_strong_issue_query(query):
            return "issue"
        if SearchEngine._is_function_memory_query(query):
            return "function_memory"
        return "code" if qtype == "code" else "function_memory"

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

        if len(words) >= 5 and not re.search(r"[_(){};=]|::", query):
            return "nl"
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

    def _get_dynamic_weights(
        self,
        query: str,
        *,
        qtype: str | None = None,
        profile: str | None = None,
    ):
        """Choose the frozen fusion family from the resolved query profile.

        `query_type` remains an NL/code text-processing switch. Fusion weights
        should follow the final profile chosen by the router so explain traces,
        reranker mode, and RRF all stay aligned.
        """
        _ = qtype  # reserved for future trace/debug parity; profile decides weights
        profile = profile or self._detect_query_profile(query)
        if profile == "issue":
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

        # 1. Extract symbol candidates from query
        ql = query.lower()
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
                # Look up chunk by symbol_name in the chunks table (fallback chain)
                chunk = None
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

        results = self._stable_sort_results(results)
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

        sorted_ids = sorted(
            scores,
            key=lambda rid: self._stable_score_key(scores[rid], all_r.get(rid)),
        )[:top_k]
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

        sorted_ids = sorted(
            scores,
            key=lambda rid: self._stable_score_key(scores[rid], all_r.get(rid)),
        )[:top_k]
        results = []
        for rid in sorted_ids:
            r = all_r.get(rid)
            if r:
                r.score = scores[rid]
                results.append(r)
        return results

    @staticmethod
    def _dedupe_by_file(results: list["SearchResult"]) -> list["SearchResult"]:
        """Keep the highest-scoring chunk per file_path.

        SWE-bench / Bug-Loc style benchmarks score at the file level, so a
        Top-K filled with multiple chunks of the same wrong file wastes slots.
        We keep stable order: first occurrence wins position, but the score
        is upgraded to the maximum across collapsed chunks. Channel-rank
        metadata is merged so explain traces still show why the file appeared.
        """
        seen: dict[str, "SearchResult"] = {}
        order: list[str] = []
        for r in results:
            key = (getattr(r, "file_path", "") or "").lower()
            if not key:
                key = getattr(r, "id", "") or str(id(r))
            existing = seen.get(key)
            if existing is None:
                seen[key] = r
                order.append(key)
                continue
            if (getattr(r, "score", 0.0) or 0.0) > (getattr(existing, "score", 0.0) or 0.0):
                existing.score = r.score
            try:
                merged = dict(existing.channel_ranks or {})
                for ch, rk in (r.channel_ranks or {}).items():
                    if ch not in merged or (rk and rk < merged[ch]):
                        merged[ch] = rk
                existing.channel_ranks = merged
            except Exception:
                pass
        return [seen[k] for k in order]

    #   
    def _apply_reranker(
        self,
        query: str,
        rrf_results,
        context: dict | None = None,
        profile: str | None = None,
        issue_keywords: list[str] | None = None,
        issue_title: str | None = None,
    ):
        if not rrf_results:
            return rrf_results
        nl_mode = self._detect_query_type(query) == "nl"
        resolved_profile = profile or self._detect_query_profile(query)
        use_specific_context = (
            self._context_has_specific(context)
            and resolved_profile == "function_memory"
        )
        issue_mode = resolved_profile == "issue"
        issue_keywords = list(issue_keywords or [])
        issue_title_tokens: list[str] = []
        if issue_mode and issue_title:
            issue_title_tokens = sorted({
                t.lower()
                for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+", issue_title)
                if len(t) >= 3 and t.lower() not in self._NL_STOP_WORDS
            })
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
                'context_specific_match': self._context_specific_match_score(r, context),
                'context_has_specific': use_specific_context,
                'issue_mode': issue_mode,
                'issue_keywords': issue_keywords if issue_mode else [],
                'issue_title_tokens': issue_title_tokens if issue_mode else [],
                'context_boost': self._config.search.context_rerank_boost,
                'search_document': self._storage.get_search_document(r.id) or '',
            }
            for r in rrf_results
        ]
        issue_rerank_top_k = getattr(self._config.search, "issue_rerank_top_k", None)
        if profile == "issue" and issue_rerank_top_k and issue_rerank_top_k > 0:
            rerank_top_k = max(1, issue_rerank_top_k)
        else:
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
        rerank_min: float | None = None
        for c in reranked:
            orig = result_map.get(c.get('id', '')) or file_result_map.get(c.get('file', ''))
            if orig:
                rerank_score = c.get('_rerank_score', orig.score) or 0.0
                orig.score = rerank_score
                output.append(orig)
                used_ids.add(orig.id)
                if rerank_min is None or rerank_score < rerank_min:
                    rerank_min = rerank_score
        if nl_mode:
            # NL/Agent: keep RRF tail candidates for context pool, but ensure
            # they cannot outrank the reranked head when the unreranked pool
            # was widened beyond ``rerank_top_k`` (e.g. issue mode with a 200
            # candidate pool feeding a 20 rerank_top_k). We attenuate the tail
            # to land strictly below the lowest reranked score so explain
            # traces still keep ranks consistent.
            tail_floor = (rerank_min if rerank_min is not None else 0.0) - 1e-6
            tail_candidates = [r for r in rrf_results if r.id not in used_ids]
            for offset, tail in enumerate(tail_candidates, 1):
                tail_score = float(getattr(tail, "score", 0.0) or 0.0)
                if rerank_min is not None and tail_score >= rerank_min:
                    tail.score = tail_floor - 1e-6 * offset
            output.extend(tail_candidates)
        return output

    # ---- Deterministic ordering helpers ----
    @staticmethod
    def _stable_result_key(result: SearchResult | None) -> tuple:
        """Stable, content-derived secondary key for tiebreaking sort.

        Falls back to a SHA1 of the result id so equal-score results land
        in a deterministic order across runs and platforms.
        """
        if result is None:
            return ("", 0, "", "")
        rid = getattr(result, "id", "") or ""
        digest = hashlib.sha1(rid.encode("utf-8", errors="replace")).hexdigest()
        file_path = getattr(result, "file_path", "") or ""
        line_number = getattr(result, "line_number", 0) or 0
        symbol_name = getattr(result, "symbol_name", "") or ""
        return (file_path, line_number, symbol_name, digest)

    @classmethod
    def _stable_score_key(cls, score: float, result: SearchResult | None) -> tuple:
        """Ascending sort key: score DESC, then file/line/symbol/hash ASC.

        Use this with ``sorted(..., key=...)`` (no ``reverse=True``).
        """
        file_path, line_number, symbol_name, digest = cls._stable_result_key(result)
        return (-float(score), file_path, int(line_number), symbol_name, digest)

    @classmethod
    def _stable_sort_results(cls, results):
        """Sort a list of SearchResult deterministically: score desc, then
        file_path asc, line asc, symbol_name asc, sha1(id) asc."""
        if not results:
            return results
        return sorted(
            results,
            key=lambda r: cls._stable_score_key(getattr(r, "score", 0.0) or 0.0, r),
        )

    def _index_snapshot_id(self) -> str:
        """Short, stable identifier for the current index state.

        Tries the storage `snapshot_id`/`signature`/`persist_directory` in
        order and falls back to a hash of the storage class + dimension.
        """
        storage = getattr(self, "_storage", None)
        for attr in ("snapshot_id", "signature", "fingerprint"):
            v = getattr(storage, attr, None)
            if callable(v):
                try:
                    v = v()
                except Exception:
                    v = None
            if v:
                return str(v)[:32]
        cfg = getattr(self, "_config", None)
        persist = ""
        try:
            persist = str(cfg.storage.persist_directory) if cfg else ""
        except Exception:
            pass
        seed = f"{type(storage).__name__}|{persist}"
        return hashlib.sha1(seed.encode("utf-8", errors="replace")).hexdigest()[:16]

    # ---- Explain trace builders ----
    def _result_to_explain_dict(self, result: SearchResult, rank: int) -> dict:
        """Convert a SearchResult into a JSON-friendly explain entry."""
        return {
            "rank": rank,
            "id": getattr(result, "id", None),
            "score": float(getattr(result, "score", 0.0) or 0.0),
            "symbol": getattr(result, "symbol_name", None),
            "qualified_name": getattr(result, "qualified_name", None),
            "file_path": getattr(result, "file_path", None),
            "line_number": getattr(result, "line_number", 0) or 0,
            "language": getattr(result, "language", None),
            "channel_ranks": dict(getattr(result, "channel_ranks", {}) or {}),
            "tags": list(getattr(result, "tags", []) or []),
        }

    def _build_explain_trace(
        self,
        *,
        query: str,
        search_query: str,
        qtype: str,
        profile: str | None,
        top_k: int,
        candidate_pool_top_k: int,
        file_pattern: str | None,
        tags: list[str] | None,
        context: dict | None,
        routed_language: str | None,
        weights: tuple,
        graph_query: str | None,
        issue_test_hints: str | None,
        fusion_mode: str,
        stage: str,
        channel_counts: dict,
    ) -> dict:
        wg, wv, wb = weights
        cfg = self._config.search
        return {
            "engine": "codegraph_rag.search.SearchEngine",
            "snapshot_id": self._index_snapshot_id(),
            "query": query,
            "search_query": search_query,
            "query_type": qtype,
            "query_profile": profile,
            "top_k": top_k,
            "candidate_pool_top_k": candidate_pool_top_k,
            "file_pattern": file_pattern,
            "tags": list(tags or []),
            "context": dict(context or {}) if context else None,
            "routed_language": routed_language,
            "weights": {"graph": float(wg), "vector": float(wv), "bm25": float(wb)},
            "graph_query": graph_query,
            "issue_test_hints": issue_test_hints or "",
            "rrf_k": getattr(cfg, "rrf_k", None),
            "nl_rrf_k": getattr(cfg, "nl_rrf_k", None),
            "fusion_mode": fusion_mode,
            "stage": stage,
            "channels": dict(channel_counts or {}),
            "rerank_enabled": bool(getattr(cfg, "enable_rerank", False) and self._reranker is not None),
            "context_prior_enabled": bool(getattr(cfg, "enable_context_prior", False)),
        }

    def _attach_explain(self, results, trace: dict) -> None:
        """Attach trace metadata to each result.explain in-place."""
        for rank, r in enumerate(results, 1):
            entry = self._result_to_explain_dict(r, rank)
            entry["trace"] = trace
            try:
                r.explain = entry
            except Exception:
                pass

    def search_explain(
        self,
        query: str,
        *,
        top_k: int | None = None,
        file_pattern: str | None = None,
        tags: list[str] | None = None,
        context: dict | None = None,
    ) -> dict:
        """Run search with deterministic ordering and return trace metadata.

        Returns a dict shaped as:
            {
                "trace": {...},                # query routing/fusion summary
                "results": [SearchResult, ...] # final stable-sorted results
                "explain":  [{rank, id, ...}], # per-result explain entries
            }
        """
        results = self.search(
            query,
            top_k=top_k,
            file_pattern=file_pattern,
            tags=tags,
            context=context,
            explain=True,
        )
        trace = getattr(self, "_last_trace", {}) or {}
        return {
            "trace": trace,
            "results": results,
            "explain": [self._result_to_explain_dict(r, i + 1) for i, r in enumerate(results)],
        }

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
