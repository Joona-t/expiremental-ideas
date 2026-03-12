from __future__ import annotations

import os
import unittest

from blitz_swarm.providers.openai_compatible import (
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleLLMClient,
)


@unittest.skipUnless(
    os.getenv("OPENAI_API_BASE") and os.getenv("OPENAI_API_KEY"),
    "OpenAI-compatible credentials not configured",
)
class OpenAICompatibleSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_and_embedding_endpoints_respond(self) -> None:
        llm = OpenAICompatibleLLMClient(
            base_url=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        )
        embedding = OpenAICompatibleEmbeddingClient(
            base_url=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        completion = await llm.complete("You are concise.", "Reply with the word ok.", temperature=0)
        vectors = await embedding.embed_texts(["hello world"])
        self.assertTrue(completion)
        self.assertEqual(len(vectors), 1)
