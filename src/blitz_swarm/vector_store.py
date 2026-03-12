"""Semantic vector memory with LanceDB-first and JSON fallback behavior."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from blitz_swarm.interfaces import EmbeddingClient
from blitz_swarm.models import MemoryRecord
from blitz_swarm.utils import cosine_similarity, json_dumps

try:
    import lancedb  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    lancedb = None


class SemanticMemoryStore:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self._fallback_path = self.storage_dir / "fallback_vectors.json"
        self._fallback_index: dict[str, dict[str, Any]] = {}
        self._db: Any = None
        self._table: Any = None

    async def connect(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if self._fallback_path.exists():
            self._fallback_index = json.loads(self._fallback_path.read_text(encoding="utf-8"))
        if lancedb is not None:
            self._db = lancedb.connect(str(self.storage_dir))
            if "memory_records" in self._db.table_names():
                self._table = self._db.open_table("memory_records")

    async def close(self) -> None:
        if self._fallback_index:
            await asyncio.to_thread(self._fallback_path.write_text, json_dumps(self._fallback_index), "utf-8")

    async def upsert(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        if lancedb is not None and self._db is not None:
            rows = [self._record_to_row(record) for record in records]
            if self._table is None:
                self._table = self._db.create_table("memory_records", rows, mode="overwrite")
            else:
                self._table.add(rows)
        for record in records:
            self._fallback_index[record.record_id] = self._record_to_row(record)
        await asyncio.to_thread(self._fallback_path.write_text, json_dumps(self._fallback_index), "utf-8")

    async def query(
        self,
        text: str,
        *,
        limit: int,
        embedding_client: EmbeddingClient,
        source_type: str | None = None,
    ) -> list[MemoryRecord]:
        query_vector = (await embedding_client.embed_texts([text]))[0]
        records: list[MemoryRecord] = []
        if self._table is not None:
            try:
                search = self._table.search(query_vector)
                if source_type is not None:
                    search = search.where(f"source_type = '{source_type}'")
                rows = search.limit(limit).to_list()
                records.extend(self._row_to_record(row) for row in rows)
            except Exception:
                records = []
        if records:
            return records[:limit]
        candidates = [
            self._row_to_record(row)
            for row in self._fallback_index.values()
            if source_type is None or row["source_type"] == source_type
        ]
        ranked = sorted(
            candidates,
            key=lambda record: cosine_similarity(record.embedding or [], query_vector),
            reverse=True,
        )
        return ranked[:limit]

    async def delete(self, record_ids: list[str]) -> None:
        for record_id in record_ids:
            self._fallback_index.pop(record_id, None)
        await asyncio.to_thread(self._fallback_path.write_text, json_dumps(self._fallback_index), "utf-8")

    def _record_to_row(self, record: MemoryRecord) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "source_id": record.source_id,
            "chunk_id": record.chunk_id,
            "content": record.content,
            "source_type": record.source_type,
            "uri": record.uri,
            "metadata": record.metadata,
            "embedding": record.embedding or [],
            "access_count": record.access_count,
            "reward_score": record.reward_score,
            "dependency_importance": record.dependency_importance,
            "novelty_score": record.novelty_score,
            "utility_score": record.utility_score,
            "hot": record.hot,
        }

    def _row_to_record(self, row: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            record_id=row["record_id"],
            source_id=row["source_id"],
            chunk_id=row["chunk_id"],
            content=row["content"],
            source_type=row["source_type"],
            uri=row["uri"],
            metadata=row.get("metadata", {}),
            embedding=list(row.get("embedding", [])),
            access_count=row.get("access_count", 0),
            reward_score=row.get("reward_score", 0.0),
            dependency_importance=row.get("dependency_importance", 0.0),
            novelty_score=row.get("novelty_score", 0.0),
            utility_score=row.get("utility_score", 0.0),
            hot=bool(row.get("hot", True)),
        )
