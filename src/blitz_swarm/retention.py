"""Composite utility scoring and hot-memory eviction."""

from __future__ import annotations

from typing import Iterable

from blitz_swarm.dependency_graph import DependencyGraph
from blitz_swarm.models import MemoryRecord


class RetentionManager:
    def __init__(self, dependency_graph: DependencyGraph, hot_limit: int = 250) -> None:
        self.dependency_graph = dependency_graph
        self.hot_limit = hot_limit

    def score_record(self, record: MemoryRecord) -> float:
        dependency_importance = max(
            record.dependency_importance,
            min(self.dependency_graph.reference_count(record.record_id) / 10.0, 1.0),
        )
        score = (
            0.3 * min(record.access_count / 10.0, 1.0)
            + 0.3 * max(record.reward_score, 0.0)
            + 0.2 * dependency_importance
            + 0.2 * max(record.novelty_score, 0.0)
        )
        record.utility_score = round(score, 6)
        return record.utility_score

    def score_records(self, records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        scored = list(records)
        for record in scored:
            self.score_record(record)
        return scored

    def reward(self, records: Iterable[MemoryRecord], reward_value: float) -> list[MemoryRecord]:
        updated: list[MemoryRecord] = []
        for record in records:
            record.reward_score = record.reward_score + 0.5 * (reward_value - record.reward_score)
            self.score_record(record)
            updated.append(record)
        return updated

    def evictable_records(self, records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        hot_records = [record for record in records if record.hot]
        if len(hot_records) <= self.hot_limit:
            return []
        ordered = sorted(self.score_records(hot_records), key=lambda item: item.utility_score)
        evicted: list[MemoryRecord] = []
        remaining = len(hot_records)
        for record in ordered:
            if remaining <= self.hot_limit:
                break
            if not self.dependency_graph.can_evict(record.record_id):
                continue
            evicted.append(record)
            remaining -= 1
        return evicted
