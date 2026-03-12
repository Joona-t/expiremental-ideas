"""Redis-backed shared blackboard store."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from blitz_swarm.models import AgentResult, AgentStatus, BlackboardSnapshot, ConflictBundle, Observation, RunConfig
from blitz_swarm.utils import json_dumps

try:
    from redis import asyncio as redis_async  # type: ignore
    from redis.exceptions import WatchError  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    redis_async = None
    WatchError = RuntimeError


class RedisBlackboardStore:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: Any = None

    async def connect(self) -> None:
        if redis_async is None:
            raise RuntimeError("The 'redis' package is required for the Redis blackboard store.")
        self._client = redis_async.from_url(self.redis_url, decode_responses=True)
        await self._client.ping()

    async def close(self) -> None:
        if self._client is not None:
            if hasattr(self._client, "aclose"):
                await self._client.aclose()
            else:
                await self._client.close()
            self._client = None

    async def initialize_run(self, run_config: RunConfig) -> None:
        state_key = self._state_key(run_config.run_id)
        keys = [
            state_key,
            self._status_key(run_config.run_id),
            self._tasks_key(run_config.run_id),
            self._entities_key(run_config.run_id),
            self._observations_key(run_config.run_id),
            self._events_key(run_config.run_id),
            self._conflicts_key(run_config.run_id),
        ]
        await self._client.delete(*keys)
        await self._client.hset(
            state_key,
            mapping={
                "run_id": run_config.run_id,
                "brief": run_config.brief,
                "plan": "",
                "plan_version": 0,
                "round_index": 0,
                "stable": 0,
                "final_answer": "",
            },
        )
        for stream_key, group_name in [
            (self._events_key(run_config.run_id), "events"),
            (self._observations_key(run_config.run_id), "observations"),
            (self._conflicts_key(run_config.run_id), "conflicts"),
        ]:
            try:
                await self._client.xgroup_create(stream_key, group_name, id="0", mkstream=True)
            except Exception:
                pass

    async def get_snapshot(self, run_id: str) -> BlackboardSnapshot:
        state = await self._client.hgetall(self._state_key(run_id))
        if not state:
            raise RuntimeError(f"Run '{run_id}' does not exist on the blackboard.")
        status_values = await self._client.hgetall(self._status_key(run_id))
        statuses = {
            name: AgentStatus.model_validate_json(payload)
            for name, payload in status_values.items()
        }
        tasks = sorted(await self._client.smembers(self._tasks_key(run_id)))
        entities = sorted(await self._client.smembers(self._entities_key(run_id)))
        observations = [
            Observation.model_validate_json(entry[1]["payload"])
            for entry in await self._client.xrange(self._observations_key(run_id))
        ]
        conflicts = [
            ConflictBundle.model_validate_json(entry[1]["payload"])
            for entry in await self._client.xrange(self._conflicts_key(run_id))
        ]
        return BlackboardSnapshot(
            run_id=run_id,
            round_index=int(state.get("round_index", 0)),
            brief=state["brief"],
            plan=state.get("plan", ""),
            plan_version=int(state.get("plan_version", 0)),
            observations=observations,
            tasks=tasks,
            entities=entities,
            statuses=statuses,
            conflicts=conflicts,
            final_answer=state.get("final_answer") or None,
            stable=bool(int(state.get("stable", 0))),
        )

    async def publish_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> str:
        event_id = await self._client.xadd(
            self._events_key(run_id),
            {"payload": json_dumps({"type": event_type, "payload": payload})},
        )
        await self._client.publish(self._channel_key(run_id), json_dumps({"type": event_type, "payload": payload}))
        return str(event_id)

    async def append_result(self, run_id: str, result: AgentResult, round_index: int) -> None:
        if result.status is not None:
            await self._client.hset(
                self._status_key(run_id),
                result.agent_name,
                result.status.model_dump_json(),
            )
        if result.tasks_added:
            await self._client.sadd(self._tasks_key(run_id), *result.tasks_added)
        if result.entities_added:
            await self._client.sadd(self._entities_key(run_id), *result.entities_added)
        for observation in result.observations:
            await self._client.xadd(
                self._observations_key(run_id),
                {"payload": observation.model_dump_json()},
            )
        for conflict in result.conflicts:
            await self._client.xadd(
                self._conflicts_key(run_id),
                {"payload": conflict.model_dump_json()},
            )
        if result.plan_patch is not None:
            success, version = await self.mutate_plan(run_id, result.plan_patch.model_dump(mode="json"))
            if not success:
                conflict = ConflictBundle(
                    conflict_id=result.plan_patch.patch_id,
                    key="plan_patch",
                    values=[result.plan_patch.model_dump(mode="json"), {"current_version": version}],
                )
                await self._client.xadd(
                    self._conflicts_key(run_id),
                    {"payload": conflict.model_dump_json()},
                )
        if result.final_answer:
            await self.set_final_answer(run_id, result.final_answer)
        await self._client.hset(self._state_key(run_id), mapping={"round_index": round_index})

    async def set_final_answer(self, run_id: str, final_answer: str) -> None:
        await self._client.hset(
            self._state_key(run_id),
            mapping={"final_answer": final_answer, "stable": 1},
        )

    async def mutate_plan(self, run_id: str, patch: dict[str, Any]) -> tuple[bool, int]:
        state_key = self._state_key(run_id)
        delay = 0.05
        for _ in range(5):
            async with self._client.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(state_key)
                    state = await pipe.hgetall(state_key)
                    current_version = int(state.get("plan_version", 0))
                    if patch.get("base_version", 0) < current_version:
                        await pipe.reset()
                        return False, current_version
                    proposed_version = max(int(patch.get("proposed_version", current_version + 1)), current_version + 1)
                    pipe.multi()
                    pipe.hset(
                        state_key,
                        mapping={
                            "plan": patch.get("content", ""),
                            "plan_version": proposed_version,
                        },
                    )
                    await pipe.execute()
                    return True, proposed_version
                except WatchError:
                    await asyncio.sleep(delay)
                    delay *= 2
        state = await self._client.hgetall(state_key)
        return False, int(state.get("plan_version", 0))

    async def cache_memory_records(self, records: list[dict[str, Any]], *, ttl_seconds: int = 3600) -> None:
        for record in records:
            key = self._memory_key(record["record_id"])
            await self._client.set(key, json_dumps(record), ex=ttl_seconds)
            await self._client.zadd(self._memory_hot_key(), {record["record_id"]: record.get("utility_score", 0.0)})

    async def evict_cached_records(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        await self._client.delete(*(self._memory_key(record_id) for record_id in record_ids))
        await self._client.zrem(self._memory_hot_key(), *record_ids)

    def _state_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:state"

    def _status_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:statuses"

    def _tasks_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:tasks"

    def _entities_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:entities"

    def _observations_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:observations"

    def _events_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:events"

    def _conflicts_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:conflicts"

    def _channel_key(self, run_id: str) -> str:
        return f"blitz:run:{run_id}:channel"

    def _memory_key(self, record_id: str) -> str:
        return f"blitz:memory:{record_id}"

    def _memory_hot_key(self) -> str:
        return "blitz:memory:hot"
