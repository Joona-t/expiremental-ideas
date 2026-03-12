"""Protocol definitions for the swarm runtime."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from blitz_swarm.models import AgentResult, BlackboardSnapshot, DocumentSource, MemoryRecord, RunConfig


class LLMClient(Protocol):
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
    ) -> str: ...


class EmbeddingClient(Protocol):
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class UrlFetcher(Protocol):
    async def fetch(self, url: str) -> DocumentSource: ...


class BlackboardStore(Protocol):
    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def initialize_run(self, run_config: RunConfig) -> None: ...

    async def get_snapshot(self, run_id: str) -> BlackboardSnapshot: ...

    async def publish_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> str: ...

    async def append_result(self, run_id: str, result: AgentResult, round_index: int) -> None: ...

    async def set_final_answer(self, run_id: str, final_answer: str) -> None: ...

    async def mutate_plan(self, run_id: str, patch: dict[str, Any]) -> tuple[bool, int]: ...


class Agent(Protocol):
    name: str
    role: str

    async def run(self, context: Any) -> AgentResult: ...


class Workflow(Protocol):
    name: str

    async def setup(self, run_config: RunConfig) -> None: ...

    def build_agents(self) -> list[Agent]: ...

    async def finalize(self, snapshot: BlackboardSnapshot, output_dir: str) -> tuple[str, dict[str, Any]]: ...

    async def query_memory(
        self,
        brief: str,
        *,
        limit: int = 5,
        source_type: str | None = None,
    ) -> list[MemoryRecord]: ...
