"""Durable SQLite-backed storage with a single async writer queue."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from blitz_swarm.models import DocumentSource, EventRecord, MemoryRecord, RunConfig
from blitz_swarm.utils import json_dumps


class SQLiteDurableStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._connection: sqlite3.Connection | None = None
        self._writer_queue: asyncio.Queue[tuple[str, dict[str, Any], asyncio.Future[None]] | None] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await asyncio.to_thread(self._open_connection)
        await asyncio.to_thread(self._create_schema)
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def close(self) -> None:
        await self._writer_queue.put(None)
        if self._writer_task is not None:
            await self._writer_task
        if self._connection is not None:
            await asyncio.to_thread(self._connection.close)
            self._connection = None

    async def record_run(self, run_config: RunConfig) -> None:
        await self._enqueue(
            "record_run",
            {
                "run_id": run_config.run_id,
                "workflow_name": run_config.workflow_name,
                "brief": run_config.brief,
                "inputs_json": json_dumps(run_config.inputs),
                "urls_json": json_dumps(run_config.urls),
                "max_rounds": run_config.max_rounds,
                "output_dir": run_config.output_dir,
            },
        )

    async def upsert_source(self, source: DocumentSource) -> None:
        await self._enqueue(
            "upsert_source",
            {
                "source_id": source.source_id,
                "uri": source.uri,
                "source_type": source.source_type,
                "title": source.title,
                "content": source.content,
                "metadata_json": json_dumps(source.metadata),
                "loaded_at": source.loaded_at.isoformat(),
            },
        )

    async def upsert_memory_records(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        await self._enqueue(
            "upsert_memory_records",
            {
                "records": [record.model_dump(mode="json") for record in records],
            },
        )

    async def add_dependencies(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        await self._enqueue("add_dependencies", {"edges": edges})

    async def record_event(self, event: EventRecord) -> None:
        await self._enqueue(
            "record_event",
            {
                "event_id": event.event_id,
                "run_id": event.run_id,
                "event_type": event.event_type,
                "payload_json": json_dumps(event.payload),
                "occurred_at": event.occurred_at.isoformat(),
            },
        )

    async def save_snapshot(
        self,
        run_id: str,
        version: int,
        payload: dict[str, Any],
        *,
        is_full: bool,
        base_version: int | None,
    ) -> None:
        await self._enqueue(
            "save_snapshot",
            {
                "run_id": run_id,
                "version": version,
                "payload_json": json_dumps(payload),
                "is_full": 1 if is_full else 0,
                "base_version": base_version,
            },
        )

    async def update_memory_metrics(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        await self._enqueue(
            "update_memory_metrics",
            {
                "records": [record.model_dump(mode="json") for record in records],
            },
        )

    async def mark_records_cold(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        await self._enqueue("mark_records_cold", {"record_ids": record_ids})

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_run_sync, run_id)

    async def get_events(self, run_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_events_sync, run_id)

    async def get_memory_records(
        self,
        *,
        limit: int | None = None,
        source_type: str | None = None,
        hot_only: bool = False,
    ) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._get_memory_records_sync, limit, source_type, hot_only)

    async def get_sources(self, *, source_type: str | None = None) -> list[DocumentSource]:
        return await asyncio.to_thread(self._get_sources_sync, source_type)

    async def get_dependency_edges(self) -> list[tuple[str, str]]:
        return await asyncio.to_thread(self._get_dependency_edges_sync)

    async def get_snapshots(self, run_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_snapshots_sync, run_id)

    async def _enqueue(self, operation: str, payload: dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        await self._writer_queue.put((operation, payload, future))
        await future

    async def _writer_loop(self) -> None:
        while True:
            item = await self._writer_queue.get()
            if item is None:
                break
            operation, payload, future = item
            try:
                await asyncio.to_thread(self._apply_write, operation, payload)
            except Exception as exc:  # pragma: no cover - defensive propagation
                future.set_exception(exc)
            else:
                future.set_result(None)

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=5000;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def _create_schema(self) -> None:
        assert self._connection is not None
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                brief TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                urls_json TEXT NOT NULL,
                max_rounds INTEGER NOT NULL,
                output_dir TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                uri TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                loaded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_records (
                record_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                uri TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT,
                access_count INTEGER NOT NULL,
                reward_score REAL NOT NULL,
                dependency_importance REAL NOT NULL,
                novelty_score REAL NOT NULL,
                utility_score REAL NOT NULL,
                hot INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dependencies (
                dependency_id TEXT NOT NULL,
                dependent_id TEXT NOT NULL,
                PRIMARY KEY (dependency_id, dependent_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                occurred_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                run_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                is_full INTEGER NOT NULL,
                base_version INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, version)
            );
            """
        )
        self._connection.commit()

    def _apply_write(self, operation: str, payload: dict[str, Any]) -> None:
        assert self._connection is not None
        cursor = self._connection.cursor()
        if operation == "record_run":
            cursor.execute(
                """
                INSERT INTO runs (run_id, workflow_name, brief, inputs_json, urls_json, max_rounds, output_dir)
                VALUES (:run_id, :workflow_name, :brief, :inputs_json, :urls_json, :max_rounds, :output_dir)
                ON CONFLICT(run_id) DO UPDATE SET
                    workflow_name = excluded.workflow_name,
                    brief = excluded.brief,
                    inputs_json = excluded.inputs_json,
                    urls_json = excluded.urls_json,
                    max_rounds = excluded.max_rounds,
                    output_dir = excluded.output_dir
                """,
                payload,
            )
        elif operation == "upsert_source":
            cursor.execute(
                """
                INSERT INTO sources (source_id, uri, source_type, title, content, metadata_json, loaded_at)
                VALUES (:source_id, :uri, :source_type, :title, :content, :metadata_json, :loaded_at)
                ON CONFLICT(source_id) DO UPDATE SET
                    uri = excluded.uri,
                    source_type = excluded.source_type,
                    title = excluded.title,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    loaded_at = excluded.loaded_at
                """,
                payload,
            )
        elif operation == "upsert_memory_records":
            cursor.executemany(
                """
                INSERT INTO memory_records (
                    record_id, source_id, chunk_id, content, source_type, uri, metadata_json,
                    embedding_json, access_count, reward_score, dependency_importance,
                    novelty_score, utility_score, hot, created_at, updated_at
                )
                VALUES (
                    :record_id, :source_id, :chunk_id, :content, :source_type, :uri, :metadata_json,
                    :embedding_json, :access_count, :reward_score, :dependency_importance,
                    :novelty_score, :utility_score, :hot, :created_at, :updated_at
                )
                ON CONFLICT(record_id) DO UPDATE SET
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json,
                    access_count = excluded.access_count,
                    reward_score = excluded.reward_score,
                    dependency_importance = excluded.dependency_importance,
                    novelty_score = excluded.novelty_score,
                    utility_score = excluded.utility_score,
                    hot = excluded.hot,
                    updated_at = excluded.updated_at
                """,
                [
                    {
                        **record,
                        "metadata_json": json_dumps(record["metadata"]),
                        "embedding_json": json_dumps(record["embedding"]) if record["embedding"] is not None else None,
                        "hot": 1 if record["hot"] else 0,
                    }
                    for record in payload["records"]
                ],
            )
        elif operation == "add_dependencies":
            cursor.executemany(
                """
                INSERT OR IGNORE INTO dependencies (dependency_id, dependent_id)
                VALUES (?, ?)
                """,
                payload["edges"],
            )
        elif operation == "record_event":
            cursor.execute(
                """
                INSERT INTO events (event_id, run_id, event_type, payload_json, occurred_at)
                VALUES (:event_id, :run_id, :event_type, :payload_json, :occurred_at)
                ON CONFLICT(event_id) DO UPDATE SET
                    event_type = excluded.event_type,
                    payload_json = excluded.payload_json,
                    occurred_at = excluded.occurred_at
                """,
                payload,
            )
        elif operation == "save_snapshot":
            cursor.execute(
                """
                INSERT INTO snapshots (run_id, version, payload_json, is_full, base_version)
                VALUES (:run_id, :version, :payload_json, :is_full, :base_version)
                ON CONFLICT(run_id, version) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    is_full = excluded.is_full,
                    base_version = excluded.base_version
                """,
                payload,
            )
        elif operation == "update_memory_metrics":
            cursor.executemany(
                """
                UPDATE memory_records
                SET access_count = :access_count,
                    reward_score = :reward_score,
                    dependency_importance = :dependency_importance,
                    novelty_score = :novelty_score,
                    utility_score = :utility_score,
                    hot = :hot,
                    updated_at = :updated_at
                WHERE record_id = :record_id
                """,
                [
                    {
                        "record_id": record["record_id"],
                        "access_count": record["access_count"],
                        "reward_score": record["reward_score"],
                        "dependency_importance": record["dependency_importance"],
                        "novelty_score": record["novelty_score"],
                        "utility_score": record["utility_score"],
                        "hot": 1 if record["hot"] else 0,
                        "updated_at": record["updated_at"],
                    }
                    for record in payload["records"]
                ],
            )
        elif operation == "mark_records_cold":
            cursor.executemany(
                """
                UPDATE memory_records
                SET hot = 0
                WHERE record_id = ?
                """,
                [(record_id,) for record_id in payload["record_ids"]],
            )
        else:
            raise ValueError(f"Unknown write operation: {operation}")
        self._connection.commit()

    def _get_run_sync(self, run_id: str) -> dict[str, Any] | None:
        assert self._connection is not None
        row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def _get_events_sync(self, run_id: str) -> list[dict[str, Any]]:
        assert self._connection is not None
        rows = self._connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY occurred_at ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]

    def _get_memory_records_sync(
        self,
        limit: int | None,
        source_type: str | None,
        hot_only: bool,
    ) -> list[MemoryRecord]:
        assert self._connection is not None
        clauses: list[str] = []
        params: list[Any] = []
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type)
        if hot_only:
            clauses.append("hot = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        rows = self._connection.execute(
            f"SELECT * FROM memory_records {where} ORDER BY updated_at DESC {limit_clause}",
            tuple(params),
        ).fetchall()
        return [self._memory_record_from_row(row) for row in rows]

    def _get_sources_sync(self, source_type: str | None) -> list[DocumentSource]:
        assert self._connection is not None
        if source_type is None:
            rows = self._connection.execute("SELECT * FROM sources ORDER BY loaded_at DESC").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM sources WHERE source_type = ? ORDER BY loaded_at DESC",
                (source_type,),
            ).fetchall()
        return [
            DocumentSource(
                source_id=row["source_id"],
                uri=row["uri"],
                source_type=row["source_type"],
                title=row["title"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                loaded_at=row["loaded_at"],
            )
            for row in rows
        ]

    def _get_dependency_edges_sync(self) -> list[tuple[str, str]]:
        assert self._connection is not None
        rows = self._connection.execute("SELECT dependency_id, dependent_id FROM dependencies").fetchall()
        return [(row["dependency_id"], row["dependent_id"]) for row in rows]

    def _get_snapshots_sync(self, run_id: str) -> list[dict[str, Any]]:
        assert self._connection is not None
        rows = self._connection.execute(
            "SELECT * FROM snapshots WHERE run_id = ? ORDER BY version ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]

    def _memory_record_from_row(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            record_id=row["record_id"],
            source_id=row["source_id"],
            chunk_id=row["chunk_id"],
            content=row["content"],
            source_type=row["source_type"],
            uri=row["uri"],
            metadata=json.loads(row["metadata_json"]),
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            access_count=row["access_count"],
            reward_score=row["reward_score"],
            dependency_importance=row["dependency_importance"],
            novelty_score=row["novelty_score"],
            utility_score=row["utility_score"],
            hot=bool(row["hot"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
