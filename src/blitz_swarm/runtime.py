"""Runtime orchestration for Blitz Swarm."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from blitz_swarm.blackboard import RedisBlackboardStore
from blitz_swarm.compression import SnapshotDeltaCompressor
from blitz_swarm.config import AppSettings
from blitz_swarm.dependency_graph import DependencyGraph
from blitz_swarm.fetcher import HTTPUrlFetcher
from blitz_swarm.models import (
    AgentResult,
    AgentStatus,
    BlackboardSnapshot,
    Citation,
    ConflictBundle,
    DocumentSource,
    EventRecord,
    MemoryRecord,
    Observation,
    PlanPatch,
    RunConfig,
)
from blitz_swarm.memory import MemoryCoordinator
from blitz_swarm.providers.local_cli import LocalCliLLMClient
from blitz_swarm.providers.mock import HashEmbeddingClient, RuleBasedLLMClient
from blitz_swarm.retention import RetentionManager
from blitz_swarm.storage import SQLiteDurableStore
from blitz_swarm.utils import random_id
from blitz_swarm.vector_store import SemanticMemoryStore


class AgentContext:
    def __init__(
        self,
        *,
        runtime: "SwarmRuntime",
        run_config: RunConfig,
        snapshot: BlackboardSnapshot,
        round_index: int,
        agent_name: str,
    ) -> None:
        self.runtime = runtime
        self.run_config = run_config
        self.snapshot = snapshot
        self.round_index = round_index
        self.agent_name = agent_name

    @property
    def brief(self) -> str:
        return self.run_config.brief

    @property
    def urls(self) -> list[str]:
        return list(self.run_config.urls)

    async def query_memory(
        self,
        query_text: str,
        *,
        limit: int = 5,
        source_type: str | None = None,
    ) -> list[MemoryRecord]:
        return await self.runtime.memory.query(query_text, limit=limit, source_type=source_type)

    async def fetch_and_ingest_url(self, url: str) -> list[MemoryRecord]:
        return await self.runtime.memory.ingest_urls([url], self.runtime.fetcher)

    def observation(
        self,
        *,
        category: str,
        content: str,
        citations: list[Citation],
        confidence: float,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        return Observation(
            observation_id=random_id("obs"),
            agent_name=self.agent_name,
            category=category,
            content=content,
            confidence=confidence,
            citations=citations,
            dependencies=dependencies or [],
            round_index=self.round_index,
            metadata=metadata or {},
        )

    def plan_patch(
        self,
        *,
        summary: str,
        content: str,
        confidence: float,
        dependencies: list[str] | None = None,
    ) -> PlanPatch:
        return PlanPatch(
            patch_id=random_id("plan"),
            agent_name=self.agent_name,
            summary=summary,
            content=content,
            base_version=self.snapshot.plan_version,
            proposed_version=self.snapshot.plan_version + 1,
            confidence=confidence,
            dependencies=dependencies or [],
        )

    def status(self, state: str, detail: str | None = None) -> AgentStatus:
        return AgentStatus(agent_name=self.agent_name, state=state, detail=detail)

    def conflict(self, key: str, values: list[dict[str, Any]]) -> ConflictBundle:
        return ConflictBundle(
            conflict_id=random_id("conflict"),
            key=key,
            values=values,
        )


class RoundController:
    def __init__(self, runtime: "SwarmRuntime") -> None:
        self.runtime = runtime
        self.compressor = SnapshotDeltaCompressor()

    async def execute(self, workflow: Any, run_config: RunConfig) -> dict[str, Any]:
        await self.runtime.blackboard.initialize_run(run_config)
        await self.runtime.storage.record_run(run_config)
        await self.runtime._prime_sources(run_config)
        await workflow.setup(run_config)

        agents = workflow.build_agents()
        trace: dict[str, Any] = {
            "run_id": run_config.run_id,
            "workflow": workflow.name,
            "brief": run_config.brief,
            "config": run_config.model_dump(mode="json"),
            "rounds": [],
            "artifacts": {},
        }
        previous_snapshot_payload: dict[str, Any] | None = None
        convergence_reason = "max_rounds_reached"

        for round_index in range(1, run_config.max_rounds + 1):
            snapshot = await self.runtime.blackboard.get_snapshot(run_config.run_id)
            round_trace: dict[str, Any] = {
                "round_index": round_index,
                "starting_plan_version": snapshot.plan_version,
                "agent_results": [],
            }
            tasks = [
                self._run_agent(agent, workflow, run_config, snapshot, round_index)
                for agent in agents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            high_value_observations = 0
            plan_changed = False
            for result in results:
                if isinstance(result, Exception):
                    await self.runtime.record_event(
                        run_config.run_id,
                        "agent_error",
                        {"error": str(result), "round_index": round_index},
                    )
                    round_trace["agent_results"].append({"error": str(result)})
                    continue
                if result.plan_patch is not None:
                    plan_changed = True
                high_value_observations += sum(1 for item in result.observations if item.confidence >= 0.65)
                await self.runtime.blackboard.append_result(run_config.run_id, result, round_index)
                await self.runtime.persist_agent_result(run_config, result, round_index)
                await self.runtime.record_event(
                    run_config.run_id,
                    "agent_result",
                    {
                        "agent": result.agent_name,
                        "round_index": round_index,
                        "observations": len(result.observations),
                        "plan_patch": result.plan_patch.model_dump(mode="json") if result.plan_patch else None,
                        "final_answer": bool(result.final_answer),
                    },
                )
                round_trace["agent_results"].append(
                    {
                        "agent": result.agent_name,
                        "observations": len(result.observations),
                        "tasks_added": result.tasks_added,
                        "entities_added": result.entities_added,
                        "plan_patch": result.plan_patch.summary if result.plan_patch else None,
                        "final_answer": result.final_answer,
                    }
                )

            snapshot = await self.runtime.blackboard.get_snapshot(run_config.run_id)
            snapshot_payload = snapshot.model_dump(mode="json")
            delta = self.compressor.make_delta(previous_snapshot_payload, snapshot_payload)
            is_full = round_index == 1 or round_index % self.runtime.settings.snapshot_interval == 0
            await self.runtime.storage.save_snapshot(
                run_config.run_id,
                round_index,
                snapshot_payload if is_full else delta,
                is_full=is_full,
                base_version=None if is_full else round_index - 1,
            )
            previous_snapshot_payload = snapshot_payload
            round_trace["ending_plan_version"] = snapshot.plan_version
            round_trace["final_answer_present"] = bool(snapshot.final_answer)
            round_trace["conflicts"] = len(snapshot.conflicts)
            trace["rounds"].append(round_trace)

            if snapshot.stable and snapshot.final_answer:
                convergence_reason = "arbiter_finalized"
                break
            if high_value_observations == 0 and not plan_changed and round_index > 1:
                convergence_reason = "no_high_value_observations"
                break

        final_snapshot = await self.runtime.blackboard.get_snapshot(run_config.run_id)
        report_markdown, workflow_trace = await workflow.finalize(final_snapshot, run_config.output_dir)
        report_path = Path(run_config.output_dir) / "report.md"
        trace_path = Path(run_config.output_dir) / "trace.json"
        report_path.write_text(report_markdown, encoding="utf-8")
        trace["convergence_reason"] = convergence_reason
        trace["workflow_summary"] = workflow_trace
        trace["artifacts"] = {
            "report": str(report_path),
            "trace": str(trace_path),
        }
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=True), encoding="utf-8")
        await self.runtime.record_event(
            run_config.run_id,
            "workflow_completed",
            {
                "convergence_reason": convergence_reason,
                "report_path": str(report_path),
                "trace_path": str(trace_path),
            },
        )
        return trace

    async def _run_agent(
        self,
        agent: Any,
        workflow: Any,
        run_config: RunConfig,
        snapshot: BlackboardSnapshot,
        round_index: int,
    ) -> AgentResult:
        context = AgentContext(
            runtime=self.runtime,
            run_config=run_config,
            snapshot=snapshot,
            round_index=round_index,
            agent_name=agent.name,
        )
        return await agent.run(context)


class SwarmRuntime:
    def __init__(
        self,
        *,
        settings: AppSettings,
        llm_client: Any,
        embedding_client: Any,
    ) -> None:
        self.settings = settings
        self.blackboard = RedisBlackboardStore(settings.redis_url)
        self.storage = SQLiteDurableStore(settings.sqlite_path)
        self.semantic_store = SemanticMemoryStore(settings.vector_dir)
        self.dependency_graph = DependencyGraph()
        self.retention = RetentionManager(self.dependency_graph, hot_limit=settings.retention_hot_limit)
        self.fetcher = HTTPUrlFetcher()
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.memory = MemoryCoordinator(
            durable_store=self.storage,
            semantic_store=self.semantic_store,
            embedding_client=self.embedding_client,
            dependency_graph=self.dependency_graph,
            retention_manager=self.retention,
            blackboard=self.blackboard,
        )
        self.round_controller = RoundController(self)

    @classmethod
    def from_env(cls) -> "SwarmRuntime":
        settings = AppSettings.from_env()
        settings.ensure_directories()
        # Embeddings have no local-CLI equivalent, so the deterministic hash
        # embedder is always used regardless of LLM backend (see
        # providers/local_cli.py docstring).
        embedding_client = HashEmbeddingClient()
        if settings.llm_backend == "cli":
            llm_client = LocalCliLLMClient(
                model=settings.llm_cli_model,
                command=settings.llm_cli_command,
            )
        else:
            llm_client = RuleBasedLLMClient()
        return cls(settings=settings, llm_client=llm_client, embedding_client=embedding_client)

    async def __aenter__(self) -> "SwarmRuntime":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        self.settings.ensure_directories()
        await self.blackboard.connect()
        await self.storage.connect()
        await self.semantic_store.connect()
        for dependency_id, dependent_id in await self.storage.get_dependency_edges():
            self.dependency_graph.add_edge(dependency_id, dependent_id)

    async def close(self) -> None:
        await self.semantic_store.close()
        await self.storage.close()
        await self.blackboard.close()

    async def ingest(self, *, paths: list[Path], urls: list[str]) -> dict[str, Any]:
        records: list[MemoryRecord] = []
        if paths:
            records.extend(await self.memory.ingest_paths(paths))
        if urls:
            records.extend(await self.memory.ingest_urls(urls, self.fetcher))
        return {
            "records_ingested": len(records),
            "sources": sorted({record.source_id for record in records}),
        }

    async def run_workflow(self, workflow: Any, *, brief: str, inputs: list[Path], urls: list[str], max_rounds: int | None = None) -> dict[str, Any]:
        run_id = random_id("run")
        output_dir = self.settings.run_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = RunConfig(
            run_id=run_id,
            workflow_name=workflow.name,
            brief=brief,
            inputs=[str(path.resolve()) for path in inputs],
            urls=urls,
            max_rounds=max_rounds or self.settings.default_max_rounds,
            output_dir=str(output_dir),
        )
        return await self.round_controller.execute(workflow, run_config)

    async def inspect_run(self, run_id: str) -> dict[str, Any]:
        run = await self.storage.get_run(run_id)
        if run is None:
            raise RuntimeError(f"Run '{run_id}' was not found.")
        events = await self.storage.get_events(run_id)
        snapshots = await self.storage.get_snapshots(run_id)
        return {
            "run": run,
            "events": events,
            "snapshots": snapshots,
            "artifacts": {
                "report": str(Path(run["output_dir"]) / "report.md"),
                "trace": str(Path(run["output_dir"]) / "trace.json"),
            },
        }

    async def record_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        event_id = await self.blackboard.publish_event(run_id, event_type, payload)
        await self.storage.record_event(
            EventRecord(
                event_id=str(event_id),
                run_id=run_id,
                event_type=event_type,
                payload=payload,
            )
        )

    async def persist_agent_result(self, run_config: RunConfig, result: AgentResult, round_index: int) -> None:
        dependency_edges: list[tuple[str, str]] = []
        observation_sources: list[DocumentSource] = []
        for observation in result.observations:
            observation_sources.append(
                DocumentSource(
                    source_id=observation.observation_id,
                    uri=f"run://{run_config.run_id}/observation/{observation.observation_id}",
                    source_type="observation",
                    title=f"{result.agent_name} observation",
                    content=observation.content,
                    metadata={
                        "agent_name": result.agent_name,
                        "round_index": round_index,
                        "confidence": observation.confidence,
                        "citations": [citation.model_dump(mode="json") for citation in observation.citations],
                    },
                )
            )
            for citation in observation.citations:
                dependency_edges.append((citation.source_id, observation.observation_id))
        if observation_sources:
            await self.memory.ingest_sources(observation_sources)
        if result.plan_patch is not None:
            for dependency in result.plan_patch.dependencies:
                dependency_edges.append((dependency, result.plan_patch.patch_id))
        if dependency_edges:
            await self.memory.add_dependencies(dependency_edges)
        if result.final_answer:
            cited_records = [citation.source_id for observation in result.observations for citation in observation.citations]
            await self.memory.reward(cited_records, 1.0)

    async def _prime_sources(self, run_config: RunConfig) -> None:
        seed_paths = [path for path in self.settings.seed_corpus_paths if path.exists()]
        input_paths = [Path(path) for path in run_config.inputs]
        if seed_paths or input_paths:
            await self.memory.ingest_paths([*seed_paths, *input_paths])
        if run_config.urls:
            await self.memory.ingest_urls(run_config.urls, self.fetcher)
