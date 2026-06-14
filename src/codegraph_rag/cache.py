"""多级缓存系统 — L1 静态 + L2 子图 + L3 查询，SHA-256 失效"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: str           # JSON 序列化的结果
    created_at: float
    expires_at: float    # 过期时间戳，0=永不过期
    file_hash: str       # 关联文件的 SHA-256，用于失效检测
    version: int = 1


class MultiLevelCache:
    """三级缓存系统

    L1 静态缓存：依赖层结构、不可变区图索引 — 永久缓存
    L2 子图缓存：常用子图结构（模块结构、调用链） — 24h
    L3 查询缓存：精确查询结果、上下文生成结果 — 5min

    失效机制：
    - 基于内容：每个文件计算 SHA-256，内容改变时失效
    - 基于时间：L2/L3 有 TTL
    - 原子更新：所有操作在 SQLite 事务中
    """

    def __init__(self, persist_path: Path) -> None:
        self._db_path = persist_path / "cache.db"
        self._conn: sqlite3.Connection | None = None
        self._file_hashes: dict[str, str] = {}  # file_path → SHA-256
        self._init_db()

    def _init_db(self) -> None:
        """初始化缓存数据库"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                level INTEGER NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0,
                file_hash TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_level ON cache_entries(level)
        """)
        self._conn.commit()

    # ── 文件哈希管理 ──

    def compute_file_hash(self, file_path: str, content: str) -> str:
        """计算文件内容的 SHA-256 哈希"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def update_file_hash(self, file_path: str, content: str) -> str:
        """更新文件哈希，返回新哈希"""
        new_hash = self.compute_file_hash(file_path, content)
        old_hash = self._file_hashes.get(file_path)
        self._file_hashes[file_path] = new_hash

        if old_hash and old_hash != new_hash:
            # 文件内容变化，失效相关缓存
            self._invalidate_by_file_hash(old_hash)

        return new_hash

    def is_file_changed(self, file_path: str, content: str) -> bool:
        """检查文件内容是否变化"""
        new_hash = self.compute_file_hash(file_path, content)
        old_hash = self._file_hashes.get(file_path)
        return old_hash != new_hash

    # ── 缓存读写 ──

    def get(self, key: str) -> Any | None:
        """获取缓存值"""
        if self._conn is None:
            return None

        row = self._conn.execute(
            "SELECT value, expires_at, file_hash FROM cache_entries WHERE key = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        value_json, expires_at, file_hash = row

        # 检查时间过期
        if expires_at > 0 and time.time() > expires_at:
            self._delete(key)
            return None

        # 检查文件哈希失效
        if file_hash:
            # 查找是否有任何关联文件发生了变化
            hash_still_valid = any(
                h == file_hash for h in self._file_hashes.values()
            )
            if not hash_still_valid and file_hash not in self._file_hashes.values():
                # 关联文件可能已变化（哈希不在当前映射中说明文件被更新过）
                pass  # 宽松策略：只在明确检测到变化时失效

        try:
            return json.loads(value_json)
        except (json.JSONDecodeError, TypeError):
            return value_json

    def set(
        self,
        key: str,
        value: Any,
        level: int = 3,
        ttl_seconds: float = 0,
        file_hash: str = "",
    ) -> None:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值（会被 JSON 序列化）
            level: 缓存层级 (1=L1永久, 2=L2子图, 3=L3查询)
            ttl_seconds: 生存时间（秒），0=永不过期
            file_hash: 关联文件哈希，用于失效检测
        """
        if self._conn is None:
            return

        # 层级默认 TTL
        if ttl_seconds == 0:
            ttl_map = {1: 0, 2: 86400, 3: 300}  # L1永久, L2 24h, L3 5min
            ttl_seconds = ttl_map.get(level, 300)

        expires_at = (time.time() + ttl_seconds) if ttl_seconds > 0 else 0
        value_json = json.dumps(value, ensure_ascii=False, default=str)

        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO cache_entries
                   (key, level, value, created_at, expires_at, file_hash, version)
                   VALUES (?, ?, ?, ?, ?, ?, 1)""",
                (key, level, value_json, time.time(), expires_at, file_hash),
            )

    def delete(self, key: str) -> None:
        """删除缓存条目"""
        self._delete(key)

    def _delete(self, key: str) -> None:
        if self._conn is None:
            return
        with self._conn:
            self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))

    # ── 批量操作 ──

    def get_layer_cache(self, layer_key: str) -> Any | None:
        """获取 L1/L2 层级缓存"""
        return self.get(f"layer:{layer_key}")

    def set_layer_cache(self, layer_key: str, value: Any, level: int = 1) -> None:
        """设置 L1/L2 层级缓存"""
        self.set(f"layer:{layer_key}", value, level=level, ttl_seconds=0)

    def get_subgraph_cache(self, subgraph_key: str) -> Any | None:
        """获取 L2 子图缓存"""
        return self.get(f"subgraph:{subgraph_key}")

    def set_subgraph_cache(self, subgraph_key: str, value: Any, file_hash: str = "") -> None:
        """设置 L2 子图缓存"""
        self.set(f"subgraph:{subgraph_key}", value, level=2, file_hash=file_hash)

    def get_query_cache(self, query_key: str) -> Any | None:
        """获取 L3 查询缓存"""
        return self.get(f"query:{query_key}")

    def set_query_cache(self, query_key: str, value: Any, file_hash: str = "") -> None:
        """设置 L3 查询缓存"""
        self.set(f"query:{query_key}", value, level=3, file_hash=file_hash)

    # ── 失效管理 ──

    def _invalidate_by_file_hash(self, file_hash: str) -> None:
        """根据文件哈希失效相关缓存"""
        if self._conn is None:
            return
        with self._conn:
            self._conn.execute(
                "DELETE FROM cache_entries WHERE file_hash = ? AND level >= 2",
                (file_hash,),
            )
        logger.debug(f"Invalidated cache entries for file_hash={file_hash}")

    def invalidate_file(self, file_path: str) -> None:
        """失效与某文件相关的所有缓存"""
        file_hash = self._file_hashes.get(file_path)
        if file_hash:
            self._invalidate_by_file_hash(file_hash)

    def invalidate_all(self, level: int | None = None) -> None:
        """清空缓存（可按层级）"""
        if self._conn is None:
            return
        with self._conn:
            if level is not None:
                self._conn.execute("DELETE FROM cache_entries WHERE level = ?", (level,))
            else:
                self._conn.execute("DELETE FROM cache_entries")

    # ── 统计与维护 ──

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计"""
        if self._conn is None:
            return {}
        total = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        by_level = {}
        for level in [1, 2, 3]:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE level = ?", (level,)
            ).fetchone()[0]
            by_level[f"L{level}"] = count
        return {"total": total, **by_level}

    def cleanup_expired(self) -> int:
        """清理过期缓存，返回清理数量"""
        if self._conn is None:
            return 0
        now = time.time()
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE expires_at > 0 AND expires_at < ?",
                (now,),
            )
            return cursor.rowcount

    def close(self) -> None:
        """关闭缓存"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
