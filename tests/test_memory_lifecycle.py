from __future__ import annotations

import unittest

from blitz_swarm.compression import SnapshotDeltaCompressor
from blitz_swarm.dependency_graph import DependencyGraph
from blitz_swarm.models import MemoryRecord
from blitz_swarm.retention import RetentionManager


class MemoryLifecycleTests(unittest.TestCase):
    def test_delta_round_trip_restores_snapshot(self) -> None:
        compressor = SnapshotDeltaCompressor()
        before = {"plan": "A", "stats": {"round": 1}, "tasks": ["scan"]}
        after = {"plan": "B", "stats": {"round": 2}, "tasks": ["scan", "synthesize"]}
        delta = compressor.make_delta(before, after)
        restored = compressor.apply_delta(before, delta)
        self.assertEqual(restored, after)

    def test_dependency_graph_tracks_descendants(self) -> None:
        graph = DependencyGraph()
        graph.add_edge("chunk_1", "obs_1")
        graph.add_edge("obs_1", "plan_1")
        self.assertEqual(graph.descendants("chunk_1"), {"obs_1", "plan_1"})
        self.assertFalse(graph.can_evict("chunk_1"))

    def test_retention_reward_updates_utility_score(self) -> None:
        graph = DependencyGraph()
        manager = RetentionManager(graph, hot_limit=1)
        record = MemoryRecord(
            record_id="mem_1",
            source_id="src_1",
            chunk_id="chunk_1",
            content="Redis and SQLite",
            source_type="seed",
            uri="file://seed",
            novelty_score=0.6,
        )
        before = manager.score_record(record)
        [updated] = manager.reward([record], 1.0)
        self.assertGreater(updated.utility_score, before)
        self.assertGreater(updated.reward_score, 0.0)

    def test_safe_eviction_preserves_dependency_roots(self) -> None:
        graph = DependencyGraph()
        graph.add_edge("root", "dependent")
        manager = RetentionManager(graph, hot_limit=1)
        protected = MemoryRecord(
            record_id="root",
            source_id="src_1",
            chunk_id="chunk_1",
            content="Foundational context",
            source_type="seed",
            uri="file://seed",
            utility_score=0.1,
            novelty_score=0.1,
        )
        disposable = MemoryRecord(
            record_id="dependent",
            source_id="src_2",
            chunk_id="chunk_2",
            content="Downstream synthesis",
            source_type="observation",
            uri="run://obs",
            utility_score=0.9,
            novelty_score=0.9,
        )
        evicted = manager.evictable_records([protected, disposable])
        self.assertEqual([record.record_id for record in evicted], ["dependent"])
