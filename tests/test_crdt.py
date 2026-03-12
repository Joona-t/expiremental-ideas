from __future__ import annotations

import unittest

from blitz_swarm.crdt import GCounter, LWWRegister, MVRegister, ORSet


class CRDTTests(unittest.TestCase):
    def test_lww_register_merge_prefers_latest_timestamp(self) -> None:
        older = LWWRegister(value="old", timestamp="2026-03-12T05:00:00+00:00", node_id="a")
        newer = LWWRegister(value="new", timestamp="2026-03-12T05:01:00+00:00", node_id="b")
        merged = older.merge(newer)
        self.assertEqual(merged.value, "new")

    def test_or_set_merge_respects_adds_and_removes(self) -> None:
        left = ORSet()
        left.add("redis", tag="a1")
        right = ORSet()
        right.add("sqlite", tag="b1")
        right.add("redis", tag="b2")
        left.remove("redis")
        merged = left.merge(right)
        self.assertEqual(merged.values(), {"sqlite", "redis"})

    def test_gcounter_merge_uses_max_for_each_node(self) -> None:
        left = GCounter(counts={"a": 2, "b": 1})
        right = GCounter(counts={"a": 1, "b": 4, "c": 3})
        merged = left.merge(right)
        self.assertEqual(merged.value(), 9)

    def test_mv_register_preserves_concurrent_values_until_resolution(self) -> None:
        register = MVRegister()
        register.write({"plan": "A"})
        register.write({"plan": "B"})
        self.assertEqual(len(register.values), 2)
        resolved = register.resolve({"plan": "Merged"})
        self.assertEqual(resolved["plan"], "Merged")
        self.assertEqual(register.values, [{"plan": "Merged"}])
