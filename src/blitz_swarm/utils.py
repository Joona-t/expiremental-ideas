"""Utility helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def random_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            split_at = normalized.rfind("\n\n", start, end)
            if split_at > start + 300:
                end = split_at
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - overlap, 0)
    return [chunk for chunk in chunks if chunk]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if not av or not bv or len(av) != len(bv):
        return 0.0
    numerator = sum(x * y for x, y in zip(av, bv, strict=True))
    a_norm = math.sqrt(sum(x * x for x in av))
    b_norm = math.sqrt(sum(y * y for y in bv))
    if not a_norm or not b_norm:
        return 0.0
    return numerator / (a_norm * b_norm)


def deterministic_embedding(text: str, *, dimensions: int = 24) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(dimensions):
        byte = digest[index % len(digest)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def deep_copy_json(value: Any) -> Any:
    return deepcopy(value)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")
