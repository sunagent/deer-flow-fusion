"""CodeGraph ReRanker — 两阶段检索精排模块

Stage 1: RRF 粗召 Top-50
Stage 2: 代码特征精排 Top-10
"""
import re, math, json
from pathlib import Path
from dataclasses import dataclass, field

# Known Python libraries for import matching
KNOWN_LIBS = {
    'pandas', 'numpy', 'torch', 'tensorflow', 'tf', 'flask', 'django',
    'fastapi', 'sqlalchemy', 'celery', 'redis', 'aiohttp', 'asyncio',
    'requests', 'matplotlib', 'sklearn', 'scikit', 'pytest', 'jwt',
    'pillow', 'opencv', 'cv2', 'bs4', 'beautifulsoup', 'scrapy',
    'plotly', 'seaborn', 'nltk', 'spacy', 'transformers', 'huggingface',
    'pydantic', 'dataclasses', 'collections', 'itertools', 'functools',
    'multiprocessing', 'threading', 'subprocess', 'os', 'sys', 'io',
    'json', 'yaml', 'toml', 'csv', 'sqlite3', 'psycopg2', 'pymongo',
}

QUERY_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'by',
    'can', 'code', 'designed', 'do', 'does', 'for', 'from', 'function',
    'had', 'has', 'have', 'in', 'into', 'is', 'it', 'its', 'method',
    'not', 'of', 'on', 'or', 'purpose', 'that', 'the', 'this', 'to',
    'used', 'using', 'was', 'were', 'which', 'with', 'input', 'output',
    'procedure', 'returns', 'return', 'takes', 'parameters',
}


def split_identifier(name: str | None) -> list[str]:
    if not name:
        return []
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    text = text.replace('_', ' ').replace('-', ' ')
    return [t.lower() for t in re.findall(r'[A-Za-z0-9]+', text) if len(t) >= 2]


def normalize_tokens(text: str | None, *, keep_stop_words: bool = False) -> set[str]:
    if not text:
        return set()
    tokens = {t.lower() for t in re.findall(r'[A-Za-z0-9_]+', text)}
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(split_identifier(token))
    if not keep_stop_words:
        expanded = {t for t in expanded if len(t) >= 2 and t not in QUERY_STOP_WORDS}
    return expanded


def language_from_candidate(candidate: dict | None, file_path: str | None) -> str:
    """Infer candidate language from search_document first, then file suffix."""
    search_document = (candidate or {}).get("search_document") or ""
    for line in search_document.splitlines()[:20]:
        match = re.match(r"\s*language:\s*([A-Za-z0-9_+-]+)", line, flags=re.I)
        if match:
            lang = match.group(1).lower()
            return "typescript" if lang in {"ts", "tsx"} else ("javascript" if lang in {"js", "jsx", "mjs"} else lang)

    suffix = Path(file_path or "").suffix.lower()
    return {
        ".py": "python",
        ".java": "java",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".rs": "rust",
        ".go": "go",
    }.get(suffix, "")


def overlap_score(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    # Recall-style overlap is better for long code/doc candidates than Jaccard.
    return overlap / max(len(query_tokens), 1)


def alias_tokens_for_symbol(name: str | None) -> set[str]:
    if not name:
        return set()
    lowered = name.lower()
    parts = set(split_identifier(name))
    aliases: set[str] = set()
    special = {
        "__len__": "length count size number elements collection cardinality",
        "__post_init__": "post init initialize object state configuration setup properties",
        "__init__": "constructor initialize object state setup configuration",
        "__str__": "string representation convert display text",
        "__repr__": "debug representation printable string",
        "has": "has contains exists member key property presence check object field",
        "min": "minimum smallest lower bound compare two values",
        "max": "maximum largest upper bound compare two values",
        "run": "execute start process task command",
        "alpha": "alphabetic alpha hash cracking cipher decoder",
    }
    if lowered in special:
        aliases.update(normalize_tokens(special[lowered]))
    if "len" in parts or "length" in parts or "count" in parts:
        aliases.update(normalize_tokens("length count size number elements collection"))
    if {"try", "stat"} <= parts or lowered in {"trystat", "try_stat"}:
        aliases.update(normalize_tokens("safely attempt file metadata filesystem stat without throwing missing file"))
    if "stat" in parts:
        aliases.update(normalize_tokens("metadata status statistics file information"))
    if {"random", "string"} <= parts:
        aliases.update(normalize_tokens("generate random alphanumeric string sequence"))
    if "opts" in parts or "options" in parts:
        aliases.update(normalize_tokens("options configuration command line arguments defaults settings"))
    if "u64" in parts or ("64" in parts and "u" in parts):
        aliases.update(normalize_tokens("unsigned 64 bit integer numeric identifier id"))
    if "into" in parts:
        aliases.update(normalize_tokens("convert transform into output value"))
    if "null" in parts:
        aliases.update(normalize_tokens("null none empty missing value type check"))
    if "json" in parts:
        aliases.update(normalize_tokens("json serialization deserialization element object value"))
    return aliases


def owner_tokens_from_search_document(search_document: str | None, file_path: str | None) -> set[str]:
    if not search_document:
        base = normalize_tokens(str(file_path or ""))
        return base
    owner_lines = []
    capture = False
    for line in search_document.splitlines():
        low = line.lower()
        if low.startswith("owner context:"):
            capture = True
            continue
        if capture and low.startswith(("module path:", "module words:", "imports:", "type:", "signature:")):
            capture = False
        if capture or low.startswith(
            (
                "qualified name:",
                "owner:",
                "owner words:",
                "parent owner:",
                "parent owner words:",
                "java package:",
                "java package words:",
                "java owner type:",
                "java extends:",
                "java implements:",
                "js export:",
                "js assigned function:",
                "js object owner:",
                "js object method:",
                "rust modules:",
                "rust use imports:",
                "rust impl:",
                "rust traits:",
                "module words:",
            )
        ):
            owner_lines.append(line)
    owner_lines.append(str(file_path or ""))
    return normalize_tokens("\n".join(owner_lines))


def summary_tokens_from_search_document(search_document: str | None) -> set[str]:
    if not search_document:
        return set()
    lines = []
    capture = False
    for line in search_document.splitlines():
        low = line.lower()
        if low.startswith(
            (
                "function summary:",
                "function purpose:",
                "function input:",
                "function output:",
                "function categories:",
                "function dependencies:",
                "function module context:",
            )
        ):
            lines.append(line)
            capture = True
            continue
        if capture and low.startswith(
            (
                "language:",
                "symbol:",
                "symbol words:",
                "symbol aliases:",
                "parent:",
                "owner context:",
                "module path:",
            )
        ):
            capture = False
        elif capture:
            lines.append(line)
    return normalize_tokens("\n".join(lines))


@dataclass
class CodeFeatures:
    """Extracted code features for a candidate."""
    chunk_id: str
    symbol_name: str
    symbol_type: str
    params: list[dict]
    param_count: int
    return_type: str | None
    decorators: list[str]
    tags: list[str]
    imports: set[str]
    code_tokens: set[str]
    doc_tokens: set[str]
    symbol_tokens: set[str]
    param_tokens: set[str]
    return_tokens: set[str]
    search_doc_tokens: set[str]
    summary_tokens: set[str]
    owner_tokens: set[str]
    alias_tokens: set[str]
    language: str = ""

def extract_imports(code: str) -> set[str]:
    """Extract imported library names from code."""
    imports = set()
    for m in re.finditer(r'(?:import|from)\s+(\w+)', code):
        lib = m.group(1).lower()
        if lib in KNOWN_LIBS:
            imports.add(lib)
    return imports

def extract_query_signals(query: str) -> dict:
    """Extract code-relevant signals from a natural language query."""
    signals = {
        'mentioned_libs': set(),
        'potential_func_names': [],
        'is_async': False,
        'is_decorator': False,
        'param_count_hint': 0,
        'wants_typed': False,
    }
    ql = query.lower()

    # Library mentions
    for lib in KNOWN_LIBS:
        if lib in ql:
            signals['mentioned_libs'].add(lib)

    # Function name extraction: camelCase, snake_case patterns
    func_patterns = re.findall(r'\b([a-z]+(?:[A-Z][a-z]+)+)\b', query)  # camelCase
    func_patterns += re.findall(r'\b([a-z]+(?:_[a-z]+)+)\b', query)      # snake_case
    signals['potential_func_names'] = list(set(func_patterns[:3]))

    # Async detection
    if any(w in ql for w in ['async', 'await', 'coroutine', 'asyncio']):
        signals['is_async'] = True

    # Decorator detection
    if any(w in ql for w in ['decorator', 'wraps', 'decorated']):
        signals['is_decorator'] = True

    # Parameter count hints
    if 'multiple' in ql or 'several' in ql or 'many' in ql:
        signals['param_count_hint'] = 3
    elif 'single' in ql or 'one parameter' in ql:
        signals['param_count_hint'] = 1

    # Type hint awareness
    if any(w in ql for w in ['type hint', 'typed', 'return type', 'annotation']):
        signals['wants_typed'] = True

    return signals


class ReRanker:
    """轻量精排模块：基于代码特征二次打分重排 Top-50 → Top-10."""

    def __init__(self, storage_conn):
        self._conn = storage_conn
        self._features_by_id = {}
        self._features_by_file = {}
        cur = self._conn.execute(
            """
            SELECT c.id, c.file_path, c.symbol_name, c.chunk_type, c.tags, c.docstring,
                   s.name, s.symbol_type, s.parameters, s.return_type, s.decorators,
                   s.signature
            FROM chunks c
            LEFT JOIN symbols s ON s.name = c.symbol_name AND s.file_path = c.file_path
            """
        )
        for row in cur:
            chunk_id, file_path = row[0], row[1]
            symbol_name = row[6] or row[2] or ''
            params = json.loads(row[8]) if row[8] else []
            return_type = row[9]
            decorators = json.loads(row[10]) if row[10] else []
            tags = json.loads(row[4]) if row[4] else []
            signature = row[11] or ''
            param_text = ' '.join(
                f"{p.get('name', '')} {p.get('type_annotation') or ''}"
                for p in params
            )
            meta = {
                'id': chunk_id,
                'file_path': file_path,
                'name': symbol_name,
                'symbol_type': row[7] or row[3] or 'function',
                'parameters': params,
                'return_type': return_type,
                'decorators': decorators,
                'tags': tags,
                'doc_tokens': normalize_tokens(row[5]),
                'symbol_tokens': set(split_identifier(symbol_name)) | normalize_tokens(signature),
                'param_tokens': normalize_tokens(param_text),
                'return_tokens': normalize_tokens(return_type),
                'search_doc_tokens': set(),
                'summary_tokens': set(),
            }
            self._features_by_id[chunk_id] = meta
            self._features_by_file.setdefault(file_path, meta)

    def _get_features(
        self,
        chunk_id: str,
        chunk_file: str,
        code_snippet: str,
        candidate: dict | None = None,
    ) -> CodeFeatures | None:
        """Extract features for a candidate."""
        sym = self._features_by_id.get(chunk_id) or self._features_by_file.get(chunk_file)
        if not sym:
            candidate = candidate or {}
            symbol_name = candidate.get('symbol_name') or ''
            tags = candidate.get('tags') or []
            if not symbol_name and not tags:
                return None
            sym = {
                'id': chunk_id or chunk_file,
                'name': symbol_name,
                'symbol_type': 'function',
                'parameters': [],
                'return_type': None,
                'decorators': [],
                'tags': tags,
                'doc_tokens': normalize_tokens(candidate.get('docstring')),
                'symbol_tokens': set(split_identifier(symbol_name)),
                'param_tokens': set(),
                'return_tokens': set(),
                'search_doc_tokens': set(),
                'summary_tokens': set(),
            }
        tags = sym['tags']
        search_document = (candidate or {}).get('search_document')
        search_doc_tokens = normalize_tokens(search_document)
        summary_tokens = summary_tokens_from_search_document(search_document)

        return CodeFeatures(
            chunk_id=sym['id'],
            symbol_name=sym['name'],
            symbol_type=sym['symbol_type'],
            params=sym['parameters'],
            param_count=len(sym['parameters']),
            return_type=sym['return_type'],
            decorators=sym['decorators'],
            tags=tags,
            imports=extract_imports(code_snippet),
            code_tokens=normalize_tokens(code_snippet),
            doc_tokens=sym['doc_tokens'],
            symbol_tokens=sym['symbol_tokens'],
            param_tokens=sym['param_tokens'],
            return_tokens=sym['return_tokens'],
            search_doc_tokens=search_doc_tokens or sym.get('search_doc_tokens', set()),
            summary_tokens=summary_tokens or sym.get('summary_tokens', set()),
            owner_tokens=owner_tokens_from_search_document((candidate or {}).get('search_document'), chunk_file),
            alias_tokens=alias_tokens_for_symbol(sym['name']),
            language=language_from_candidate(candidate, chunk_file),
        )

    # NL-optimized weights (for docstring / natural language queries)
    NL_WEIGHTS = {
        'func_name': 0.08,
        'import_match': 0.04,
        'ast_struct': 0.05,
        'token_overlap': 0.14,
        'doc_overlap': 0.20,
        'summary_overlap': 0.14,
        'search_doc_overlap': 0.04,
        'symbol_soft': 0.10,
        'param_return': 0.07,
        'channel_boost': 0.06,
        'rrf_prior': 0.08,
    }
    NL_CONTEXT_WEIGHTS = {
        'func_name': 0.10,
        'import_match': 0.02,
        'ast_struct': 0.03,
        'token_overlap': 0.12,
        'doc_overlap': 0.16,
        'summary_overlap': 0.16,
        'search_doc_overlap': 0.16,
        'symbol_soft': 0.12,
        'param_return': 0.05,
        'channel_boost': 0.04,
        'context_prior': 0.03,
        'rrf_prior': 0.01,
    }
    NL_CONTEXT_WEIGHTS_BY_LANGUAGE = {
        # RepoQA/SNF-style pure NL lookup benefits from different signals by
        # language. These profiles only apply when caller context is present.
        "python": {
            'func_name': 0.08,
            'import_match': 0.01,
            'ast_struct': 0.02,
            'token_overlap': 0.09,
            'doc_overlap': 0.13,
            'summary_overlap': 0.16,
            'search_doc_overlap': 0.15,
            'symbol_soft': 0.09,
            'owner_match': 0.10,
            'alias_match': 0.06,
            'param_return': 0.06,
            'channel_boost': 0.03,
            'context_prior': 0.01,
            'rrf_prior': 0.01,
        },
        "java": {
            'func_name': 0.13,
            'import_match': 0.01,
            'ast_struct': 0.02,
            'token_overlap': 0.08,
            'doc_overlap': 0.12,
            'summary_overlap': 0.14,
            'search_doc_overlap': 0.13,
            'symbol_soft': 0.16,
            'owner_match': 0.09,
            'alias_match': 0.04,
            'param_return': 0.04,
            'channel_boost': 0.03,
            'context_prior': 0.01,
            'rrf_prior': 0.00,
        },
        "typescript": {
            'func_name': 0.08,
            'import_match': 0.01,
            'ast_struct': 0.02,
            'token_overlap': 0.08,
            'doc_overlap': 0.13,
            'summary_overlap': 0.22,
            'search_doc_overlap': 0.20,
            'symbol_soft': 0.08,
            'owner_match': 0.05,
            'alias_match': 0.03,
            'param_return': 0.05,
            'channel_boost': 0.03,
            'context_prior': 0.01,
            'rrf_prior': 0.01,
        },
        "javascript": {
            'func_name': 0.08,
            'import_match': 0.01,
            'ast_struct': 0.02,
            'token_overlap': 0.08,
            'doc_overlap': 0.13,
            'summary_overlap': 0.22,
            'search_doc_overlap': 0.20,
            'symbol_soft': 0.08,
            'owner_match': 0.05,
            'alias_match': 0.03,
            'param_return': 0.05,
            'channel_boost': 0.03,
            'context_prior': 0.01,
            'rrf_prior': 0.01,
        },
        "go": {
            'func_name': 0.13,
            'import_match': 0.01,
            'ast_struct': 0.02,
            'token_overlap': 0.08,
            'doc_overlap': 0.12,
            'summary_overlap': 0.14,
            'search_doc_overlap': 0.13,
            'symbol_soft': 0.16,
            'owner_match': 0.09,
            'alias_match': 0.04,
            'param_return': 0.04,
            'channel_boost': 0.03,
            'context_prior': 0.01,
            'rrf_prior': 0.00,
        },
    }
    # Code-oriented weights (for symbol-bearing queries)
    CODE_WEIGHTS = {'func_name': 0.25, 'import_match': 0.20, 'ast_struct': 0.25, 'token_overlap': 0.20, 'rrf_prior': 0.10}

    def score(
        self,
        query: str,
        chunk_file: str,
        code_snippet: str,
        rrf_score: float,
        nl_mode: bool = False,
        chunk_id: str = '',
        candidate: dict | None = None,
    ) -> float:
        """Compute re-ranking score for a candidate.
        nl_mode=True uses NL-optimized weights (token_overlap=0.35, func_name=0.10)."""
        feat = self._get_features(chunk_id, chunk_file, code_snippet, candidate)
        if feat is None:
            return rrf_score * 0.1  # No features → fall back to RRF

        qs = extract_query_signals(query)
        q_tokens = normalize_tokens(query)
        scores = {}

        # F1: Function Name Similarity (0.25)
        f1 = 0.0
        for pf_name in qs['potential_func_names']:
            # Jaccard on character trigrams
            q_tri = set(pf_name[i:i+3] for i in range(len(pf_name)-2))
            s_tri = set(feat.symbol_name[i:i+3] for i in range(len(feat.symbol_name)-2))
            if q_tri or s_tri:
                sim = len(q_tri & s_tri) / max(len(q_tri | s_tri), 1)
                f1 = max(f1, sim)
        # Also check token-level symbol overlap.
        symbol_overlap = overlap_score(q_tokens, feat.symbol_tokens)
        if symbol_overlap:
            f1 = max(f1, min(0.9, symbol_overlap))
        scores['func_name'] = f1

        # F2: Import/Dependency Match (0.20)
        f2 = 0.0
        if qs['mentioned_libs']:
            overlap = len(qs['mentioned_libs'] & feat.imports)
            total = len(qs['mentioned_libs'])
            f2 = overlap / max(total, 1)
        scores['import_match'] = f2

        # F3: AST Structure Alignment (0.25)
        f3 = 0.0
        # Async alignment
        if qs['is_async'] and 'async' in feat.tags:
            f3 += 0.4
        # Decorator alignment
        if qs['is_decorator'] and feat.decorators:
            f3 += 0.3
        # Type hint alignment
        if qs['wants_typed'] and ('type_hint' in feat.tags or feat.return_type):
            f3 += 0.3
        # Parameter count alignment
        if qs['param_count_hint'] > 0:
            pc_diff = abs(qs['param_count_hint'] - feat.param_count)
            f3 += max(0, 0.3 - pc_diff * 0.1)
        scores['ast_struct'] = min(f3, 1.0)

        # F4: Query-code overlap — recall style for long code snippets.
        scores['token_overlap'] = overlap_score(q_tokens, feat.code_tokens)

        # F5: Query-docstring overlap. This is the strongest NL/SNF signal and
        # reuses only indexed code comments/docstrings.
        scores['doc_overlap'] = overlap_score(q_tokens, feat.doc_tokens)

        # F5b: Query-search_document overlap. This lets P1 see the same
        # function/module/import/AST summary text used by the P0 doc-vector index.
        scores['search_doc_overlap'] = overlap_score(q_tokens, feat.search_doc_tokens)
        scores['summary_overlap'] = overlap_score(q_tokens, feat.summary_tokens)

        # F6: Soft symbol-name overlap after snake/camel splitting.
        scores['symbol_soft'] = overlap_score(q_tokens, feat.symbol_tokens)

        # F6b/F6c: Owner/package/module and idiomatic alias signals. These are
        # especially helpful for short names such as has/min and magic methods.
        scores['owner_match'] = overlap_score(q_tokens, feat.owner_tokens)
        scores['alias_match'] = overlap_score(q_tokens, feat.alias_tokens)

        # F7: Parameter and return-type language.
        scores['param_return'] = overlap_score(
            q_tokens, feat.param_tokens | feat.return_tokens
        )

        # F8: Single-channel rank boost. This preserves strong BM25/Graph/DocVec
        # evidence that may not be visible in local token features.
        channel_ranks = (candidate or {}).get('channel_ranks') or {}
        channel_scores = []
        for channel, weight in (("graph", 1.0), ("bm25", 0.9), ("vector", 0.7)):
            rank = channel_ranks.get(channel)
            if rank:
                channel_scores.append(weight / max(rank, 1))
        scores['channel_boost'] = min(1.0, max(channel_scores) if channel_scores else 0.0)

        # F9: RRF Prior.
        scores['rrf_prior'] = min(rrf_score, 1.0)

        # F10: Optional workspace context prior. The caller may know active
        # language/repo/path; use it as a soft boost, never as a hard filter.
        context_match = float((candidate or {}).get('context_match') or 0.0)
        context_boost = float((candidate or {}).get('context_boost') or 0.0)
        scores['context_prior'] = min(1.0, max(0.0, context_match * context_boost * 10.0))

        # Weighted sum — auto-select NL vs Code weights
        has_context = bool((candidate or {}).get('context_match'))
        if nl_mode and has_context:
            weights = self.NL_CONTEXT_WEIGHTS_BY_LANGUAGE.get(feat.language, self.NL_CONTEXT_WEIGHTS)
        else:
            weights = self.NL_WEIGHTS if nl_mode else self.CODE_WEIGHTS
        final = sum(weights[k] * scores[k] for k in weights)
        return final

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10, nl_mode: bool = False) -> list[dict]:
        """Re-rank candidates and return top_k. nl_mode=True uses NL-optimized weights."""
        for c in candidates:
            chunk_id = c.get('id', '')
            file_path = c.get('file', '')
            snippet = c.get('content', '')
            rrf_score = c.get('score', 0)
            c['_rerank_score'] = self.score(
                query,
                file_path,
                snippet,
                rrf_score,
                nl_mode=nl_mode,
                chunk_id=chunk_id,
                candidate=c,
            )

        candidates.sort(
            key=lambda x: (
                -float(x.get('_rerank_score', 0) or 0.0),
                str(x.get('file') or '').replace('\\', '/').lower(),
                str(x.get('symbol_name') or '').lower(),
                str(x.get('id') or ''),
            )
        )
        return candidates[:top_k]


# For standalone testing
if __name__ == '__main__':
    import sqlite3
    conn = sqlite3.connect(r'F:\codex-cache\csn997_index\codegraph.db')
    rr = ReRanker(conn)

    # Test a few queries
    tests = [
        ("read the directory of images into DataFrame", "F:\\codex-cache\\csn997_corpus\\f_00000.py",
         "def readImages(path, sc=None): ..."),
        ("how to send an async activity with type hints", "F:\\codex-cache\\csn997_corpus\\f_00001.py",
         "def send_activity(self, *activity_or_text: Union[Activity, str]) -> ResourceResponse"),
    ]
    for q, fp, code in tests:
        s = rr.score(q, fp, code, 0.5)
        print(f'Query: {q[:60]}')
        print(f'  Score: {s:.4f}')
        print()
