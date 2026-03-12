"""Built-in research synthesis workflow."""

from __future__ import annotations

from typing import Any

from blitz_swarm.models import AgentResult, Citation, MemoryRecord, RunConfig


class PlannerAgent:
    name = "planner"
    role = "Creates and revises the shared plan."

    async def run(self, context: Any) -> AgentResult:
        records = await context.query_memory(context.brief, limit=4)
        prompt = self._prompt(context.brief, records, context.snapshot.plan)
        response = await context.runtime.llm_client.complete(
            "You are the planner agent in a research swarm. Produce a concise execution plan.",
            prompt,
        )
        dependencies = [record.record_id for record in records]
        plan_patch = context.plan_patch(
            summary="Research execution plan",
            content=response,
            confidence=0.78,
            dependencies=dependencies,
        )
        tasks = [line.strip("- ").strip() for line in response.splitlines() if line.strip().startswith(("1.", "2.", "3.", "-"))]
        return AgentResult(
            agent_name=self.name,
            plan_patch=plan_patch,
            tasks_added=tasks[:5],
            status=context.status("complete", "Plan updated"),
        )

    def _prompt(self, brief: str, records: list[MemoryRecord], current_plan: str) -> str:
        evidence = "\n\n".join(
            f"[{record.record_id}] {record.content[:500]}"
            for record in records
        )
        return (
            f"Brief:\n{brief}\n\n"
            f"Current plan:\n{current_plan or '(empty)'}\n\n"
            f"Relevant evidence:\n{evidence or '(no evidence)'}"
        )


class LocalCorpusAnalystAgent:
    name = "local_corpus_analyst"
    role = "Extracts evidence from local and seeded documents."

    async def run(self, context: Any) -> AgentResult:
        records = await context.query_memory(context.brief, limit=6, source_type="seed")
        if not records:
            records = await context.query_memory(context.brief, limit=6, source_type="file")
        if not records:
            return AgentResult(
                agent_name=self.name,
                status=context.status("idle", "No local corpus records available"),
            )
        response = await context.runtime.llm_client.complete(
            "You are a local corpus analyst. Summarize the strongest evidence from the supplied documents.",
            self._prompt(context.brief, records),
        )
        observation = context.observation(
            category="local_evidence",
            content=response,
            citations=self._citations(records),
            confidence=0.82,
            dependencies=[record.record_id for record in records],
        )
        return AgentResult(
            agent_name=self.name,
            observations=[observation],
            entities_added=self._entities(records),
            status=context.status("complete", "Local evidence extracted"),
        )

    def _prompt(self, brief: str, records: list[MemoryRecord]) -> str:
        excerpts = "\n\n".join(f"[{record.record_id}] {record.content[:650]}" for record in records)
        return f"Brief:\n{brief}\n\nLocal evidence:\n{excerpts}"

    def _citations(self, records: list[MemoryRecord]) -> list[Citation]:
        return [
            Citation(source_id=record.record_id, uri=record.uri, chunk_id=record.chunk_id, quote=record.content[:160])
            for record in records[:4]
        ]

    def _entities(self, records: list[MemoryRecord]) -> list[str]:
        entities = {"Redis", "SQLite", "LanceDB"}
        for record in records:
            if "networkx" in record.content.lower():
                entities.add("networkx")
            if "pub/sub" in record.content.lower():
                entities.add("Pub/Sub")
        return sorted(entities)


class URLAnalystAgent:
    name = "url_analyst"
    role = "Extracts evidence from explicit URLs supplied to the run."

    async def run(self, context: Any) -> AgentResult:
        if not context.urls:
            return AgentResult(
                agent_name=self.name,
                status=context.status("idle", "No URLs supplied"),
            )
        records = await context.query_memory(context.brief, limit=5, source_type="url")
        if not records:
            for url in context.urls:
                await context.fetch_and_ingest_url(url)
            records = await context.query_memory(context.brief, limit=5, source_type="url")
        if not records:
            return AgentResult(
                agent_name=self.name,
                status=context.status("idle", "URL fetch produced no usable text"),
            )
        response = await context.runtime.llm_client.complete(
            "You are the URL analyst in a research swarm. Extract the strongest web-sourced evidence.",
            self._prompt(context.brief, records),
        )
        observation = context.observation(
            category="url_evidence",
            content=response,
            citations=[
                Citation(source_id=record.record_id, uri=record.uri, chunk_id=record.chunk_id, quote=record.content[:160])
                for record in records[:3]
            ],
            confidence=0.72,
            dependencies=[record.record_id for record in records],
        )
        return AgentResult(
            agent_name=self.name,
            observations=[observation],
            status=context.status("complete", "URL evidence extracted"),
        )

    def _prompt(self, brief: str, records: list[MemoryRecord]) -> str:
        excerpts = "\n\n".join(f"[{record.uri}] {record.content[:650]}" for record in records)
        return f"Brief:\n{brief}\n\nURL evidence:\n{excerpts}"


class SynthesizerAgent:
    name = "synthesizer"
    role = "Combines current findings into a coherent recommendation."

    async def run(self, context: Any) -> AgentResult:
        if not context.snapshot.observations:
            return AgentResult(
                agent_name=self.name,
                status=context.status("waiting", "No observations to synthesize yet"),
            )
        prompt = self._prompt(context.brief, context.snapshot.plan, context.snapshot.observations)
        response = await context.runtime.llm_client.complete(
            "You are the synthesizer. Merge the current observations into a coherent recommendation.",
            prompt,
        )
        citations = [citation for observation in context.snapshot.observations[-4:] for citation in observation.citations[:1]]
        observation = context.observation(
            category="synthesis",
            content=response,
            citations=citations,
            confidence=0.77,
            dependencies=[observation.observation_id for observation in context.snapshot.observations[-4:]],
        )
        return AgentResult(
            agent_name=self.name,
            observations=[observation],
            status=context.status("complete", "Synthesis drafted"),
        )

    def _prompt(self, brief: str, plan: str, observations: list[Any]) -> str:
        latest = "\n\n".join(
            f"[{observation.agent_name}] {observation.content[:500]}"
            for observation in observations[-5:]
        )
        return f"Brief:\n{brief}\n\nPlan:\n{plan}\n\nObservations:\n{latest}"


class CriticAgent:
    name = "critic"
    role = "Calls out weak assumptions, missing evidence, and implementation risks."

    async def run(self, context: Any) -> AgentResult:
        if not context.snapshot.observations:
            return AgentResult(
                agent_name=self.name,
                status=context.status("waiting", "No draft to critique"),
            )
        response = await context.runtime.llm_client.complete(
            "You are the critic agent. Identify the most important risks, missing evidence, and failure modes.",
            self._prompt(context.brief, context.snapshot.plan, context.snapshot.observations),
        )
        observation = context.observation(
            category="critique",
            content=response,
            citations=[citation for observation in context.snapshot.observations[-3:] for citation in observation.citations[:1]],
            confidence=0.69,
            dependencies=[observation.observation_id for observation in context.snapshot.observations[-3:]],
        )
        conflicts = []
        if "risk" in response.lower() or "missing" in response.lower():
            conflicts.append(
                context.conflict(
                    "critic_review",
                    [{"summary": response[:400], "round_index": context.round_index}],
                )
            )
        return AgentResult(
            agent_name=self.name,
            observations=[observation],
            conflicts=conflicts,
            status=context.status("complete", "Critique added"),
        )

    def _prompt(self, brief: str, plan: str, observations: list[Any]) -> str:
        latest = "\n\n".join(
            f"[{observation.category}] {observation.content[:450]}"
            for observation in observations[-5:]
        )
        return f"Brief:\n{brief}\n\nPlan:\n{plan}\n\nDraft evidence:\n{latest}"


class ArbiterFinalizerAgent:
    name = "arbiter_finalizer"
    role = "Resolves conflicts and produces the stable final answer."

    async def run(self, context: Any) -> AgentResult:
        if len(context.snapshot.observations) < 2:
            return AgentResult(
                agent_name=self.name,
                status=context.status("waiting", "Need more evidence before finalizing"),
            )
        response = await context.runtime.llm_client.complete(
            "You are the arbiter and finalizer. Resolve conflicts and produce the final answer for the run.",
            self._prompt(context.brief, context.snapshot.plan, context.snapshot.observations, context.snapshot.conflicts),
        )
        should_finalize = context.round_index >= 2 or bool(context.snapshot.conflicts)
        return AgentResult(
            agent_name=self.name,
            final_answer=response if should_finalize else None,
            status=context.status("complete", "Final decision prepared" if should_finalize else "Decision deferred"),
        )

    def _prompt(self, brief: str, plan: str, observations: list[Any], conflicts: list[Any]) -> str:
        evidence = "\n\n".join(
            f"[{observation.agent_name}] {observation.content[:450]}"
            for observation in observations[-6:]
        )
        conflict_text = "\n".join(conflict.key for conflict in conflicts) or "(none)"
        return (
            f"Brief:\n{brief}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Observations:\n{evidence}\n\n"
            f"Conflicts:\n{conflict_text}"
        )


class ResearchWorkflow:
    name = "research"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def setup(self, run_config: RunConfig) -> None:
        del run_config

    def build_agents(self) -> list[Any]:
        return [
            PlannerAgent(),
            LocalCorpusAnalystAgent(),
            URLAnalystAgent(),
            SynthesizerAgent(),
            CriticAgent(),
            ArbiterFinalizerAgent(),
        ]

    async def finalize(self, snapshot: Any, output_dir: str) -> tuple[str, dict[str, Any]]:
        observations = snapshot.observations
        markdown = [
            "# Blitz Swarm Research Report",
            "",
            "## Brief",
            snapshot.brief,
            "",
            "## Final Answer",
            snapshot.final_answer or "No final answer was produced.",
            "",
            "## Shared Plan",
            snapshot.plan or "No shared plan was recorded.",
            "",
            "## Key Observations",
        ]
        for observation in observations[-10:]:
            markdown.append(f"- `{observation.agent_name}` ({observation.category}, confidence {observation.confidence:.2f}): {observation.content}")
            if observation.citations:
                markdown.append(
                    f"  Citations: {', '.join(citation.uri for citation in observation.citations[:3])}"
                )
        markdown.extend(["", "## Conflicts"])
        if snapshot.conflicts:
            markdown.extend(f"- `{conflict.key}`" for conflict in snapshot.conflicts)
        else:
            markdown.append("- None recorded.")
        return (
            "\n".join(markdown) + "\n",
            {
                "output_dir": output_dir,
                "observation_count": len(observations),
                "conflict_count": len(snapshot.conflicts),
                "finalized": bool(snapshot.final_answer),
            },
        )

    async def query_memory(self, brief: str, *, limit: int = 5, source_type: str | None = None) -> list[MemoryRecord]:
        return await self.runtime.memory.query(brief, limit=limit, source_type=source_type)
