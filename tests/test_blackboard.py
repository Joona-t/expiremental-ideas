from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from blitz_swarm.blackboard import RedisBlackboardStore
from blitz_swarm.models import AgentStatus, BlackboardSnapshot, Observation
from blitz_swarm.providers.mock import RuleBasedLLMClient
from blitz_swarm.workflows.research import ArbiterFinalizerAgent


class FakeWatchError(Exception):
    pass


class FakePipeline:
    def __init__(self, client: "FakeRedisClient") -> None:
        self.client = client
        self.pending_mapping: dict[str, object] | None = None

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def watch(self, key: str) -> None:
        self.client.last_watch_key = key

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.client.state)

    async def reset(self) -> None:
        return None

    def multi(self) -> None:
        return None

    def hset(self, key: str, mapping: dict[str, object]) -> None:
        self.pending_mapping = mapping

    async def execute(self) -> None:
        if self.client.fail_once:
            self.client.fail_once = False
            raise FakeWatchError("simulated race")
        if self.pending_mapping is not None:
            self.client.state.update({k: str(v) for k, v in self.pending_mapping.items()})


class FakeRedisClient:
    def __init__(self, *, fail_once: bool = False, plan_version: int = 0) -> None:
        self.fail_once = fail_once
        self.last_watch_key = ""
        self.state = {"plan_version": str(plan_version), "plan": ""}

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        del transaction
        return FakePipeline(self)

    async def hgetall(self, key: str) -> dict[str, str]:
        del key
        return dict(self.state)


class BlackboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutate_plan_retries_after_watch_error(self) -> None:
        store = RedisBlackboardStore("redis://unused")
        store._client = FakeRedisClient(fail_once=True)
        patch_payload = {"base_version": 0, "proposed_version": 1, "content": "Use Redis"}
        with patch("blitz_swarm.blackboard.WatchError", FakeWatchError):
            success, version = await store.mutate_plan("run_1", patch_payload)
        self.assertTrue(success)
        self.assertEqual(version, 1)
        self.assertEqual(store._client.state["plan"], "Use Redis")

    async def test_mutate_plan_rejects_stale_version(self) -> None:
        store = RedisBlackboardStore("redis://unused")
        store._client = FakeRedisClient(plan_version=2)
        success, version = await store.mutate_plan(
            "run_1",
            {"base_version": 0, "proposed_version": 1, "content": "stale"},
        )
        self.assertFalse(success)
        self.assertEqual(version, 2)

    async def test_arbiter_agent_hands_off_final_answer_once_enough_evidence_exists(self) -> None:
        agent = ArbiterFinalizerAgent()
        snapshot = BlackboardSnapshot(
            run_id="run_1",
            round_index=2,
            brief="Summarize the architecture.",
            observations=[
                Observation(
                    observation_id="obs_1",
                    agent_name="planner",
                    category="plan",
                    content="Use layered memory.",
                    confidence=0.7,
                ),
                Observation(
                    observation_id="obs_2",
                    agent_name="critic",
                    category="risk",
                    content="Validate retention behavior.",
                    confidence=0.65,
                ),
            ],
        )
        runtime = SimpleNamespace(llm_client=RuleBasedLLMClient())
        context = SimpleNamespace(
            runtime=runtime,
            brief="Summarize the architecture.",
            round_index=2,
            snapshot=snapshot,
            status=lambda state, detail=None: AgentStatus(
                agent_name="arbiter_finalizer",
                state=state,
                detail=detail,
            ),
        )
        result = await agent.run(context)
        self.assertIsNotNone(result.final_answer)
        self.assertEqual(result.agent_name, "arbiter_finalizer")
