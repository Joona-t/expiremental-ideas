"""Dependency graph tracking for memory retention and provenance."""

from __future__ import annotations

from collections import defaultdict, deque


class DependencyGraph:
    """Tracks dependency -> dependent edges."""

    def __init__(self) -> None:
        self._forward: dict[str, set[str]] = defaultdict(set)
        self._reverse: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node_id: str) -> None:
        self._forward.setdefault(node_id, set())
        self._reverse.setdefault(node_id, set())

    def add_edge(self, dependency_id: str, dependent_id: str) -> None:
        self.add_node(dependency_id)
        self.add_node(dependent_id)
        self._forward[dependency_id].add(dependent_id)
        self._reverse[dependent_id].add(dependency_id)

    def descendants(self, node_id: str) -> set[str]:
        visited: set[str] = set()
        queue: deque[str] = deque(self._forward.get(node_id, set()))
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._forward.get(current, set()))
        return visited

    def reference_count(self, node_id: str) -> int:
        return len(self._forward.get(node_id, set()))

    def dependency_count(self, node_id: str) -> int:
        return len(self._reverse.get(node_id, set()))

    def can_evict(self, node_id: str) -> bool:
        return not self.descendants(node_id)

    def edges(self) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for dependency, dependents in self._forward.items():
            results.extend((dependency, dependent) for dependent in dependents)
        return results
