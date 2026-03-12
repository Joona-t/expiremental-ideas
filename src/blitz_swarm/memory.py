"""Memory ingestion, retrieval, dependency tracking, and retention."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from blitz_swarm.dependency_graph import DependencyGraph
from blitz_swarm.interfaces import EmbeddingClient, UrlFetcher
from blitz_swarm.models import DocumentSource, MemoryRecord
from blitz_swarm.retention import RetentionManager
from blitz_swarm.storage import SQLiteDurableStore
from blitz_swarm.utils import chunk_text, deterministic_embedding, read_text_file, stable_id
from blitz_swarm.vector_store import SemanticMemoryStore


class MemoryCoordinator:
    def __init__(
        self,
        *,
        durable_store: SQLiteDurableStore,
        semantic_store: SemanticMemoryStore,
        embedding_client: EmbeddingClient,
        dependency_graph: DependencyGraph,
        retention_manager: RetentionManager,
        blackboard: object | None = None,
    ) -> None:
        self.durable_store = durable_store
        self.semantic_store = semantic_store
        self.embedding_client = embedding_client
        self.dependency_graph = dependency_graph
        self.retention_manager = retention_manager
        self.blackboard = blackboard

    async def ingest_paths(self, paths: Iterable[Path]) -> list[MemoryRecord]:
        sources = [
            DocumentSource(
                source_id=stable_id("src", str(path.resolve())),
                uri=str(path.resolve()),
                source_type="seed" if "compass_artifact" in path.name else "file",
                title=path.name,
                content=read_text_file(path),
                metadata={"path": str(path.resolve())},
            )
            for path in paths
        ]
        return await self.ingest_sources(sources)

    async def ingest_urls(self, urls: Iterable[str], fetcher: UrlFetcher) -> list[MemoryRecord]:
        sources = [await fetcher.fetch(url) for url in urls]
        return await self.ingest_sources(sources)

    async def ingest_sources(self, sources: list[DocumentSource]) -> list[MemoryRecord]:
        created_records: list[MemoryRecord] = []
        for source in sources:
            await self.durable_store.upsert_source(source)
            chunks = chunk_text(source.content)
            embeddings = await self.embedding_client.embed_texts(chunks or [source.content])
            records: list[MemoryRecord] = []
            for index, chunk in enumerate(chunks or [source.content]):
                record_id = stable_id("mem", source.source_id, str(index))
                novelty = self._novelty_score(chunk)
                embedding = embeddings[index] if index < len(embeddings) else deterministic_embedding(chunk)
                record = MemoryRecord(
                    record_id=record_id,
                    source_id=source.source_id,
                    chunk_id=f"{source.source_id}:{index}",
                    content=chunk,
                    source_type=source.source_type,
                    uri=source.uri,
                    metadata={"title": source.title, **source.metadata, "chunk_index": index},
                    embedding=embedding,
                    novelty_score=novelty,
                    utility_score=novelty * 0.2,
                )
                self.dependency_graph.add_node(record.record_id)
                records.append(record)
            await self.durable_store.upsert_memory_records(records)
            await self.semantic_store.upsert(records)
            if self.blackboard is not None and hasattr(self.blackboard, "cache_memory_records"):
                await self.blackboard.cache_memory_records([record.model_dump(mode="json") for record in records])
            created_records.extend(records)
        await self.apply_retention()
        return created_records

    async def query(
        self,
        query_text: str,
        *,
        limit: int = 5,
        source_type: str | None = None,
    ) -> list[MemoryRecord]:
        records = await self.semantic_store.query(
            query_text,
            limit=limit,
            embedding_client=self.embedding_client,
            source_type=source_type,
        )
        for record in records:
            record.access_count += 1
            self.retention_manager.score_record(record)
        await self.durable_store.update_memory_metrics(records)
        if self.blackboard is not None and hasattr(self.blackboard, "cache_memory_records"):
            await self.blackboard.cache_memory_records([record.model_dump(mode="json") for record in records])
        return records

    async def add_dependencies(self, edges: list[tuple[str, str]]) -> None:
        for dependency_id, dependent_id in edges:
            self.dependency_graph.add_edge(dependency_id, dependent_id)
        await self.durable_store.add_dependencies(edges)

    async def reward(self, record_ids: list[str], reward_value: float) -> list[MemoryRecord]:
        all_records = await self.durable_store.get_memory_records()
        record_map = {record.record_id: record for record in all_records}
        rewarded = self.retention_manager.reward(
            [record_map[record_id] for record_id in record_ids if record_id in record_map],
            reward_value,
        )
        await self.durable_store.update_memory_metrics(rewarded)
        return rewarded

    async def apply_retention(self) -> list[MemoryRecord]:
        records = await self.durable_store.get_memory_records(hot_only=True)
        evicted = self.retention_manager.evictable_records(records)
        if not evicted:
            return []
        for record in evicted:
            record.hot = False
        record_ids = [record.record_id for record in evicted]
        await self.durable_store.mark_records_cold(record_ids)
        await self.durable_store.update_memory_metrics(evicted)
        await self.semantic_store.delete(record_ids)
        if self.blackboard is not None and hasattr(self.blackboard, "evict_cached_records"):
            await self.blackboard.evict_cached_records(record_ids)
        return evicted

    def _novelty_score(self, text: str) -> float:
        words = [word for word in text.lower().split() if word]
        if not words:
            return 0.0
        return min(len(set(words)) / len(words), 1.0)
