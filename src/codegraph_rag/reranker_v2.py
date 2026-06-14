"""CodeGraph ReRanker v2 — 两阶段精排模块

Stage 1: RRF 粗召 Top-50
Stage 2: 6维代码特征精排 Top-10

设计原则:
- 零新增计算: 复用粗召已计算的向量/AST/BM25
- 极致轻量: 加权线性打分, 无训练
- 延迟 <10ms
"""
import re, math, json
from pathlib import Path
from dataclasses import dataclass

KNOWN_LIBS = {
    'pandas', 'numpy', 'torch', 'tensorflow', 'tf', 'flask', 'django',
    'fastapi', 'sqlalchemy', 'celery', 'redis', 'aiohttp', 'asyncio',
    'requests', 'matplotlib', 'sklearn', 'scikit', 'pytest', 'jwt',
    'pillow', 'opencv', 'cv2', 'bs4', 'beautifulsoup', 'scrapy',
    'plotly', 'seaborn', 'nltk', 'spacy', 'transformers',
    'pydantic', 'dataclasses', 'multiprocessing', 'threading',
    'sqlite3', 'psycopg2', 'pymongo', 'aiofiles', 'httpx',
}

class ReRankerV2:
    """6维二值化特征精排器"""

    def __init__(self, storage_conn):
        self._conn = storage_conn

        # Preload symbols: file_path -> {name, type, imports}
        cur = self._conn.execute(
            "SELECT file_path, name, symbol_type, parameters, return_type, decorators FROM symbols"
        )
        self._symbols = {}
        for row in cur:
            fp = row[0]
            self._symbols[fp] = {
                'name': row[1] or '',
                'symbol_type': row[2] or '',
                'param_count': len(json.loads(row[3])) if row[3] else 0,
                'return_type': row[4],
                'decorators': json.loads(row[5]) if row[5] else [],
            }

        # Preload chunk tags
        cur = self._conn.execute("SELECT file_path, tags FROM chunks")
        self._chunk_tags = {}
        for row in cur:
            self._chunk_tags[row[0]] = json.loads(row[1]) if row[1] else []

    def score(self, query: str, chunk_file: str, code_snippet: str, rrf_score: float, vec_sim: float = 0) -> float:
        """6维加权线性打分.

        Features:
        F1 (0.35): 函数名精确匹配 - query包含symbol_name子串 → 1
        F2 (0.25): 导入依赖匹配 - query提到的库在import中 → 1
        F3 (0.15): AST结构匹配 - chunk类型与查询意图匹配 → 1
        F4 (0.15): RRF粗召得分 - 归一化后保留
        F5 (0.07): 文本重叠率 - Jaccard(去停用词)
        F6 (0.03): 向量余弦相似度 - 弱语义兜底
        """
        sym = self._symbols.get(chunk_file, {})
        sym_name = sym.get('name', '').lower()
        sym_type = sym.get('symbol_type', '')
        tags = self._chunk_tags.get(chunk_file, [])

        ql = query.lower()
        q_tokens = set(re.findall(r'\w+', ql))

        # F1: 函数名精确匹配 (0.35)
        f1 = 0.0
        if sym_name and len(sym_name) >= 2:
            # Check if query contains the full symbol name
            if sym_name in ql:
                f1 = 1.0
            # Or the symbol name tokens all appear in query
            elif all(tok in q_tokens for tok in re.findall(r'\w+', sym_name)):
                f1 = 0.8
        # Also check: query-extracted potential function names
        camel_matches = re.findall(r'\b([a-z]+(?:[A-Z][a-z]+)+)\b', query)
        snake_matches = re.findall(r'\b([a-z]+(?:_[a-z]+)+)\b', query)
        for pf in camel_matches + snake_matches:
            if pf.lower() == sym_name or (len(pf) >= 4 and pf.lower() in sym_name):
                f1 = max(f1, 1.0)
                break

        # F2: 导入依赖匹配 (0.25)
        f2 = 0.0
        mentioned_libs = {lib for lib in KNOWN_LIBS if lib in ql}
        if mentioned_libs:
            code_libs = set()
            for m in re.finditer(r'(?:import|from)\s+(\w+)', code_snippet):
                lib = m.group(1).lower()
                if lib in KNOWN_LIBS:
                    code_libs.add(lib)
            if mentioned_libs & code_libs:
                f2 = 1.0
            elif not code_libs and not any('import' in code_snippet.lower() for _ in [1]):
                f2 = 0.0  # No imports at all → no penalty, no bonus
        # No libs mentioned → skip this feature (treat as 0)

        # F3: AST结构匹配 (0.15)
        f3 = 0.0
        type_hints = {
            'function': ['def ', 'function', 'method', 'func'],
            'class': ['class ', 'object'],
            'variable': ['var', 'variable', 'const', 'constant'],
        }
        query_type = None
        for t, hints in type_hints.items():
            if any(h in ql for h in hints):
                query_type = t
                break
        if query_type and sym_type == query_type:
            f3 = 1.0
        elif not query_type:
            # No type hint in query → bonus if it's a function (most common)
            f3 = 0.3 if sym_type == 'function' else 0.0

        # F4: RRF粗召得分 (0.15) — normalize to [0, 1]
        if rrf_score < 0:
            f4 = 1.0 / (1.0 + abs(rrf_score))
        else:
            f4 = min(rrf_score, 1.0) if rrf_score <= 1.0 else 1.0 / (1.0 + rrf_score)

        # F5: 文本重叠率 (0.07) — Jaccard with stopword filtering
        STOPWORDS = {'the','a','an','is','in','to','of','for','and','or','it','on','at',
                     'with','from','by','as','be','this','that','not','are','was','were',
                     'has','have','had','do','does','did','will','would','can','could',
                     'should','may','might','shall','get','set','use','using','used'}
        q_filtered = q_tokens - STOPWORDS
        c_tokens = set(re.findall(r'\w+', code_snippet.lower())) - STOPWORDS
        if q_filtered and c_tokens:
            f5 = len(q_filtered & c_tokens) / max(len(q_filtered | c_tokens), 1)
        else:
            f5 = 0.0

        # F6: 向量余弦相似度 (0.03) — weak semantic safety net
        f6 = max(0.0, min(1.0, vec_sim))

        # Weighted sum
        weights = [0.35, 0.25, 0.15, 0.15, 0.07, 0.03]
        features = [f1, f2, f3, f4, f5, f6]
        final = sum(w * feat for w, feat in zip(weights, features))
        return final

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10, vec_sims: dict | None = None) -> list[dict]:
        """Re-rank candidates. vec_sims: {file_path: cosine_similarity}."""
        for c in candidates:
            fp = c.get('file', '')
            snippet = c.get('content', '')
            rrf_score = c.get('score', 0)
            vec_sim = (vec_sims or {}).get(fp, 0.0)
            c['_rerank_score'] = self.score(query, fp, snippet, rrf_score, vec_sim)

        candidates.sort(key=lambda x: x.get('_rerank_score', 0), reverse=True)
        return candidates[:top_k]
