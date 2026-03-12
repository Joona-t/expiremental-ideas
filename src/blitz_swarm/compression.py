"""Delta compression for blackboard snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class SnapshotDeltaCompressor:
    """Stores keyed updates and removals for JSON-like payloads."""

    def make_delta(self, previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
        if previous is None:
            return {"full": True, "value": deepcopy(current)}
        updates: dict[str, Any] = {}
        removals: list[str] = []
        self._diff("", previous, current, updates, removals)
        return {"full": False, "updates": updates, "removals": removals}

    def apply_delta(self, base: dict[str, Any] | None, delta: dict[str, Any]) -> dict[str, Any]:
        if delta.get("full") or base is None:
            return deepcopy(delta["value"])
        result = deepcopy(base)
        for path in delta.get("removals", []):
            self._delete_path(result, path)
        for path, value in delta.get("updates", {}).items():
            self._set_path(result, path, deepcopy(value))
        return result

    def _diff(
        self,
        prefix: str,
        previous: Any,
        current: Any,
        updates: dict[str, Any],
        removals: list[str],
    ) -> None:
        if previous == current:
            return
        if isinstance(previous, Mapping) and isinstance(current, Mapping):
            previous_keys = set(previous)
            current_keys = set(current)
            for removed in previous_keys - current_keys:
                removals.append(self._path(prefix, removed))
            for key in current_keys:
                child_prefix = self._path(prefix, key)
                if key not in previous:
                    updates[child_prefix] = current[key]
                    continue
                self._diff(child_prefix, previous[key], current[key], updates, removals)
            return
        updates[prefix or "$"] = current

    def _path(self, prefix: str, key: str) -> str:
        return f"{prefix}.{key}" if prefix else key

    def _set_path(self, root: dict[str, Any], path: str, value: Any) -> None:
        if path == "$":
            root.clear()
            root.update(value)
            return
        cursor = root
        segments = path.split(".")
        for segment in segments[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[segments[-1]] = value

    def _delete_path(self, root: dict[str, Any], path: str) -> None:
        cursor = root
        segments = path.split(".")
        for segment in segments[:-1]:
            next_cursor = cursor.get(segment)
            if not isinstance(next_cursor, dict):
                return
            cursor = next_cursor
        cursor.pop(segments[-1], None)
