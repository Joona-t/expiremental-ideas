"""Pydantic models used across the runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class Citation(BaseModel):
    source_id: str
    uri: str
    chunk_id: str | None = None
    quote: str | None = None


class DocumentSource(BaseModel):
    source_id: str
    uri: str
    source_type: Literal["file", "url", "seed", "observation"]
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    loaded_at: datetime = Field(default_factory=utc_now)


class Observation(BaseModel):
    observation_id: str
    agent_name: str
    category: str
    content: str
    confidence: float = 0.5
    citations: list[Citation] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    round_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanPatch(BaseModel):
    patch_id: str
    agent_name: str
    summary: str
    content: str
    base_version: int = 0
    proposed_version: int = 1
    confidence: float = 0.5
    dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictBundle(BaseModel):
    conflict_id: str
    key: str
    values: list[dict[str, Any]] = Field(default_factory=list)
    resolved: bool = False
    resolution: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)


class AgentStatus(BaseModel):
    agent_name: str
    state: str
    detail: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class AgentResult(BaseModel):
    agent_name: str
    observations: list[Observation] = Field(default_factory=list)
    plan_patch: PlanPatch | None = None
    entities_added: list[str] = Field(default_factory=list)
    tasks_added: list[str] = Field(default_factory=list)
    status: AgentStatus | None = None
    conflicts: list[ConflictBundle] = Field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    record_id: str
    source_id: str
    chunk_id: str
    content: str
    source_type: str
    uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    access_count: int = 0
    reward_score: float = 0.0
    dependency_importance: float = 0.0
    novelty_score: float = 0.0
    utility_score: float = 0.0
    hot: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class BlackboardSnapshot(BaseModel):
    run_id: str
    round_index: int
    brief: str
    plan: str = ""
    plan_version: int = 0
    observations: list[Observation] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    statuses: dict[str, AgentStatus] = Field(default_factory=dict)
    conflicts: list[ConflictBundle] = Field(default_factory=list)
    final_answer: str | None = None
    stable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    run_id: str
    workflow_name: str
    brief: str
    inputs: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    max_rounds: int = 4
    output_dir: str


class EventRecord(BaseModel):
    event_id: str
    run_id: str
    event_type: str
    payload: dict[str, Any]
    occurred_at: datetime = Field(default_factory=utc_now)
