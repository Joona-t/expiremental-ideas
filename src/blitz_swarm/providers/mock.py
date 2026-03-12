"""Deterministic local providers for tests and offline runs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from blitz_swarm.utils import deterministic_embedding


class HashEmbeddingClient:
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [deterministic_embedding(text) for text in texts]


class RuleBasedLLMClient:
    async def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
        del temperature
        prompt = f"{system_prompt}\n{user_prompt}".lower()
        lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        salient = self._salient_terms(user_prompt)
        summary = ", ".join(salient[:6]) if salient else "the available evidence"
        if "planner" in prompt:
            return (
                "Research plan:\n"
                f"1. Review the strongest sources about {summary}.\n"
                "2. Extract architecture choices, conflict semantics, and retention policy.\n"
                "3. Synthesize a recommendation with implementation tradeoffs."
            )
        if "critic" in prompt:
            return (
                "Open risks:\n"
                "- External dependencies may be unavailable locally.\n"
                "- Retention policy needs validation against real workloads.\n"
                "- URL evidence quality depends on fetched source content."
            )
        if "arbiter" in prompt or "finalizer" in prompt:
            return (
                "Final decision:\n"
                "Use Redis for the blackboard, SQLite for durability, and an embedded vector index for retrieval. "
                "Adopt append-only observations, optimistic plan updates, and utility-scored retention."
            )
        if "synthesizer" in prompt:
            return (
                "Synthesis:\n"
                f"The collected evidence converges on {summary}. "
                "The architecture favors correctness, append-only observations, and layered storage."
            )
        return (
            "Findings:\n"
            + "\n".join(f"- {line}" for line in lines[:4])
            if lines
            else "Findings:\n- No additional evidence was provided."
        )

    def _salient_terms(self, text: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
        counts = Counter(words)
        stop_words = {"with", "that", "from", "this", "have", "into", "your", "will", "about", "using"}
        filtered = [(word, count) for word, count in counts.items() if word not in stop_words]
        filtered.sort(key=lambda item: (-item[1], item[0]))
        return [word for word, _ in filtered]
