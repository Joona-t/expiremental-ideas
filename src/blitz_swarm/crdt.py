"""Conflict-free replicated data types used by the blackboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class LWWRegister:
    value: Any
    timestamp: str = field(default_factory=_timestamp)
    node_id: str = "local"

    def assign(self, value: Any, *, timestamp: str | None = None, node_id: str | None = None) -> "LWWRegister":
        return LWWRegister(
            value=value,
            timestamp=timestamp or _timestamp(),
            node_id=node_id or self.node_id,
        )

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        if (other.timestamp, other.node_id) >= (self.timestamp, self.node_id):
            return other
        return self


@dataclass
class ORSet:
    adds: dict[str, set[str]] = field(default_factory=dict)
    removes: dict[str, set[str]] = field(default_factory=dict)

    def add(self, value: str, *, tag: str) -> None:
        self.adds.setdefault(value, set()).add(tag)

    def remove(self, value: str) -> None:
        tags = self.adds.get(value, set())
        if tags:
            self.removes.setdefault(value, set()).update(tags)

    def values(self) -> set[str]:
        current: set[str] = set()
        for value, tags in self.adds.items():
            removed = self.removes.get(value, set())
            if tags - removed:
                current.add(value)
        return current

    def merge(self, other: "ORSet") -> "ORSet":
        merged = ORSet()
        keys = set(self.adds) | set(other.adds)
        for key in keys:
            merged.adds[key] = set(self.adds.get(key, set())) | set(other.adds.get(key, set()))
        keys = set(self.removes) | set(other.removes)
        for key in keys:
            merged.removes[key] = set(self.removes.get(key, set())) | set(other.removes.get(key, set()))
        return merged


@dataclass
class GCounter:
    counts: dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str, amount: int = 1) -> None:
        self.counts[node_id] = self.counts.get(node_id, 0) + amount

    def value(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        merged = GCounter()
        for key in set(self.counts) | set(other.counts):
            merged.counts[key] = max(self.counts.get(key, 0), other.counts.get(key, 0))
        return merged


@dataclass
class MVRegister:
    values: list[dict[str, Any]] = field(default_factory=list)

    def write(self, value: dict[str, Any]) -> None:
        self.values.append(value)

    def merge(self, other: "MVRegister") -> "MVRegister":
        merged = MVRegister()
        seen: set[str] = set()
        for entry in [*self.values, *other.values]:
            key = str(entry)
            if key in seen:
                continue
            seen.add(key)
            merged.values.append(entry)
        return merged

    def resolve(self, resolution: dict[str, Any]) -> dict[str, Any]:
        self.values = [resolution]
        return resolution
