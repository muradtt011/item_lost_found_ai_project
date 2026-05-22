from __future__ import annotations

import asyncio
import json
import logging
import shutil
import struct
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.core.exceptions import ItemNotFoundError, StorageError
from src.models import ItemRecord, ItemStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL – PostgreSQL schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS items (
    id              TEXT        PRIMARY KEY,
    status          TEXT        NOT NULL,
    user_text       TEXT        NOT NULL DEFAULT '',
    image_path      TEXT        NOT NULL,
    vlm_description JSONB,
    embedding       BYTEA,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ItemRepository(ABC):

    @abstractmethod
    async def save(self, item: ItemRecord) -> ItemRecord:
        """Persist a new item and return it."""

    @abstractmethod
    async def get(self, item_id: str) -> ItemRecord:
        """Return the item or raise ItemNotFoundError."""

    @abstractmethod
    async def list_items(
        self, status: Optional[ItemStatus] = None
    ) -> list[ItemRecord]:
        """Return all items, optionally filtered by status."""

    @abstractmethod
    async def update(self, item: ItemRecord) -> ItemRecord:
        """Overwrite an existing item record."""

    @abstractmethod
    async def delete(self, item_id: str) -> None:
        """Remove an item; raises ItemNotFoundError if absent."""

    async def store_image(self, source_path: Path, storage_dir: Path) -> Path:
        storage_dir.mkdir(parents=True, exist_ok=True)
        dest = storage_dir / source_path.name
        if dest.exists():
            stem, suffix = source_path.stem, source_path.suffix
            dest = storage_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        shutil.copy2(source_path, dest)
        logger.debug("Stored image blob: %s → %s", source_path, dest)
        return dest


# ---------------------------------------------------------------------------
# PostgreSQL repository (asyncpg)
# ---------------------------------------------------------------------------

class PostgreSQLRepository(ItemRepository):

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        # Strip SQLAlchemy dialect suffix if present
        self._dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None  # asyncpg.Pool (imported lazily)

    async def connect(self) -> None:
        try:
            import asyncpg  # noqa: PLC0415 — lazy import
        except ModuleNotFoundError as exc:
            raise StorageError(
                "asyncpg is not installed. Run: pip install asyncpg"
            ) from exc

        logger.info("PostgreSQLRepository connecting: %s", self._dsn)
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
        logger.info("PostgreSQLRepository ready (pool min=%d max=%d)",
                    self._min_size, self._max_size)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQLRepository pool closed")

    async def __aenter__(self) -> "PostgreSQLRepository":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pool_or_raise(self) -> Any:
        if self._pool is None:
            raise StorageError(
                "PostgreSQLRepository is not connected. Call connect() first."
            )
        return self._pool

    @staticmethod
    def _embed_to_bytes(embedding: list[float] | None) -> bytes | None:
        if embedding is None:
            return None
        arr = np.array(embedding, dtype=np.float32)
        return arr.tobytes()

    @staticmethod
    def _bytes_to_embed(raw: bytes | None) -> list[float] | None:
        if raw is None:
            return None
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.tolist()

    @staticmethod
    def _row_to_record(row: Any) -> ItemRecord:
        return ItemRecord(
            id=row["id"],
            status=ItemStatus(row["status"]),
            user_text=row["user_text"],
            image_path=row["image_path"],
            vlm_description=json.loads(row["vlm_description"])
            if row["vlm_description"] is not None
            else None,
            embedding=PostgreSQLRepository._bytes_to_embed(row["embedding"]),
            created_at=row["created_at"].replace(tzinfo=timezone.utc)
            if row["created_at"].tzinfo is None
            else row["created_at"],
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def save(self, item: ItemRecord) -> ItemRecord:
        pool = self._pool_or_raise()
        vlm_json = (
            json.dumps(item.vlm_description)
            if item.vlm_description is not None
            else None
        )
        embed_bytes = self._embed_to_bytes(item.embedding)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO items
                        (id, status, user_text, image_path,
                         vlm_description, embedding, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    item.id,
                    item.status.value,
                    item.user_text,
                    item.image_path,
                    vlm_json,
                    embed_bytes,
                    item.created_at,
                )
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                raise StorageError(f"Item {item.id!r} already exists") from exc
            raise StorageError(f"Failed to save item: {exc}") from exc
        logger.info("PostgreSQL: saved item %s (status=%s)", item.id, item.status)
        return item

    async def get(self, item_id: str) -> ItemRecord:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM items WHERE id = $1", item_id
            )
        if row is None:
            raise ItemNotFoundError(item_id)
        return self._row_to_record(row)

    async def list_items(
        self, status: Optional[ItemStatus] = None
    ) -> list[ItemRecord]:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            if status is None:
                rows = await conn.fetch(
                    "SELECT * FROM items ORDER BY created_at DESC"
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM items WHERE status = $1 ORDER BY created_at DESC",
                    status.value,
                )
        return [self._row_to_record(r) for r in rows]

    async def update(self, item: ItemRecord) -> ItemRecord:
        pool = self._pool_or_raise()
        vlm_json = (
            json.dumps(item.vlm_description)
            if item.vlm_description is not None
            else None
        )
        embed_bytes = self._embed_to_bytes(item.embedding)
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE items
                SET status = $2,
                    user_text = $3,
                    image_path = $4,
                    vlm_description = $5,
                    embedding = $6
                WHERE id = $1
                """,
                item.id,
                item.status.value,
                item.user_text,
                item.image_path,
                vlm_json,
                embed_bytes,
            )
        if result == "UPDATE 0":
            raise ItemNotFoundError(item.id)
        logger.debug("PostgreSQL: updated item %s", item.id)
        return item

    async def delete(self, item_id: str) -> None:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM items WHERE id = $1", item_id
            )
        if result == "DELETE 0":
            raise ItemNotFoundError(item_id)
        logger.info("PostgreSQL: deleted item %s", item_id)


# ---------------------------------------------------------------------------
# In-memory repository (tests / demos)
# ---------------------------------------------------------------------------

class InMemoryRepository(ItemRepository):
    """Thread-safe async in-memory store; suitable for tests and development."""

    def __init__(self) -> None:
        self._store: dict[str, ItemRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, item: ItemRecord) -> ItemRecord:
        async with self._lock:
            if item.id in self._store:
                raise StorageError(f"Item {item.id!r} already exists; use update()")
            self._store[item.id] = item
            logger.info("Saved item %s (status=%s)", item.id, item.status)
            return item

    async def get(self, item_id: str) -> ItemRecord:
        async with self._lock:
            if item_id not in self._store:
                raise ItemNotFoundError(item_id)
            return self._store[item_id]

    async def list_items(
        self, status: Optional[ItemStatus] = None
    ) -> list[ItemRecord]:
        async with self._lock:
            items = list(self._store.values())
        if status is not None:
            items = [i for i in items if i.status == status]
        items.sort(key=lambda i: i.created_at, reverse=True)
        return items

    async def update(self, item: ItemRecord) -> ItemRecord:
        async with self._lock:
            if item.id not in self._store:
                raise ItemNotFoundError(item.id)
            self._store[item.id] = item
            logger.debug("Updated item %s", item.id)
            return item

    async def delete(self, item_id: str) -> None:
        async with self._lock:
            if item_id not in self._store:
                raise ItemNotFoundError(item_id)
            del self._store[item_id]
            logger.info("Deleted item %s", item_id)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# JSON file-backed repository (persistent fallback, no DB server needed)
# ---------------------------------------------------------------------------

class JsonRepository(ItemRepository):

    def __init__(self, json_path: str | Path = "./lostfound.json") -> None:
        self._path = Path(json_path)
        self._lock = asyncio.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    def _load(self) -> dict[str, ItemRecord]:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return {r["id"]: ItemRecord.model_validate(r) for r in raw}

    def _dump(self, store: dict[str, ItemRecord]) -> None:
        data = [r.model_dump(mode="json") for r in store.values()]
        self._path.write_text(
            json.dumps(data, default=str, indent=2), encoding="utf-8"
        )

    async def save(self, item: ItemRecord) -> ItemRecord:
        async with self._lock:
            store = self._load()
            if item.id in store:
                raise StorageError(f"Item {item.id!r} already exists")
            store[item.id] = item
            self._dump(store)
            logger.info("Saved item %s to JSON store", item.id)
            return item

    async def get(self, item_id: str) -> ItemRecord:
        async with self._lock:
            store = self._load()
        if item_id not in store:
            raise ItemNotFoundError(item_id)
        return store[item_id]

    async def list_items(
        self, status: Optional[ItemStatus] = None
    ) -> list[ItemRecord]:
        async with self._lock:
            store = self._load()
        items = list(store.values())
        if status is not None:
            items = [i for i in items if i.status == status]
        items.sort(key=lambda i: i.created_at, reverse=True)
        return items

    async def update(self, item: ItemRecord) -> ItemRecord:
        async with self._lock:
            store = self._load()
            if item.id not in store:
                raise ItemNotFoundError(item.id)
            store[item.id] = item
            self._dump(store)
            logger.debug("Updated item %s in JSON store", item.id)
            return item

    async def delete(self, item_id: str) -> None:
        async with self._lock:
            store = self._load()
            if item_id not in store:
                raise ItemNotFoundError(item_id)
            del store[item_id]
            self._dump(store)
            logger.info("Deleted item %s from JSON store", item_id)


# ---------------------------------------------------------------------------
# Smart factory
# ---------------------------------------------------------------------------

def make_repository(
    database_url: str = "",
    json_path: str | Path = "./lostfound.json",
) -> ItemRepository:
    url = (database_url or "").strip()
    if url.startswith("postgresql") or url.startswith("postgres://"):
        logger.info("Storage backend: PostgreSQL (%s)", url.split("@")[-1])
        return PostgreSQLRepository(dsn=url)

    logger.info("Storage backend: JSON file (%s)", json_path)
    return JsonRepository(json_path=json_path)
