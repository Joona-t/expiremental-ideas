from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from blitz_swarm.blackboard import RedisBlackboardStore
from blitz_swarm.fetcher import HTTPUrlFetcher
from blitz_swarm.models import EventRecord, MemoryRecord
from blitz_swarm.providers.mock import HashEmbeddingClient
from blitz_swarm.storage import SQLiteDurableStore
from blitz_swarm.vector_store import SemanticMemoryStore


class TimeoutFetcher(HTTPUrlFetcher):
    async def _fetch_bytes(self, url: str) -> tuple[str, str]:
        raise RuntimeError(f"Failed to fetch URL '{url}': timeout")


class FakeRedisModule:
    @staticmethod
    def from_url(url: str, decode_responses: bool = True) -> object:
        del url, decode_responses

        class Client:
            async def ping(self) -> None:
                raise RuntimeError("redis unavailable")

        return Client()


class FailurePathTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetcher_surfaces_timeout_failure(self) -> None:
        fetcher = TimeoutFetcher()
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            await fetcher.fetch("https://example.com")

    async def test_blackboard_connect_surfaces_redis_unavailable(self) -> None:
        store = RedisBlackboardStore("redis://127.0.0.1:6379/0")
        from blitz_swarm import blackboard as blackboard_module

        original = blackboard_module.redis_async
        blackboard_module.redis_async = FakeRedisModule()
        try:
            with self.assertRaisesRegex(RuntimeError, "redis unavailable"):
                await store.connect()
        finally:
            blackboard_module.redis_async = original

    async def test_sqlite_writer_handles_queued_event_burst(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteDurableStore(Path(tmpdir) / "blitz.db")
            await store.connect()
            try:
                await asyncio.gather(
                    *[
                        store.record_event(
                            EventRecord(
                                event_id=f"evt_{index}",
                                run_id="run_1",
                                event_type="agent_result",
                                payload={"index": index},
                            )
                        )
                        for index in range(50)
                    ]
                )
                events = await store.get_events("run_1")
                self.assertEqual(len(events), 50)
            finally:
                await store.close()

    async def test_vector_store_falls_back_when_lancedb_search_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SemanticMemoryStore(Path(tmpdir))
            await store.connect()
            try:
                record = MemoryRecord(
                    record_id="mem_1",
                    source_id="src_1",
                    chunk_id="chunk_1",
                    content="Redis blackboard architecture",
                    source_type="seed",
                    uri="file://seed",
                    embedding=[0.1] * 24,
                )
                await store.upsert([record])

                class BrokenTable:
                    def search(self, query_vector):  # noqa: ANN001
                        raise RuntimeError("corrupt index")

                store._table = BrokenTable()
                results = await store.query(
                    "Redis blackboard",
                    limit=1,
                    embedding_client=HashEmbeddingClient(),
                    source_type="seed",
                )
                self.assertEqual(results[0].record_id, "mem_1")
            finally:
                await store.close()
