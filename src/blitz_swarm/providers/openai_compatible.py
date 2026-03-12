"""OpenAI-compatible chat and embedding clients."""

from __future__ import annotations

import asyncio
import json
from urllib import request

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    httpx = None


class OpenAICompatibleLLMClient:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = await self._post_json("/chat/completions", payload)
        return response["choices"][0]["message"]["content"]

    async def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if httpx is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}{path}", headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        return await asyncio.to_thread(self._post_json_sync, path, payload, headers)

    def _post_json_sync(
        self,
        path: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(f"{self.base_url}{path}", data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleEmbeddingClient:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
        }
        response = await self._post_json("/embeddings", payload)
        data = response["data"]
        return [item["embedding"] for item in data]

    async def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if httpx is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}{path}", headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        return await asyncio.to_thread(self._post_json_sync, path, payload, headers)

    def _post_json_sync(
        self,
        path: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(f"{self.base_url}{path}", data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
