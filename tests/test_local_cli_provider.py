from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from blitz_swarm.providers.local_cli import (
    LocalCliLLMClient,
    LocalCliLLMError,
    build_cli_env,
)


class BuildCliEnvTests(unittest.TestCase):
    def test_strips_paid_api_and_nesting_vars(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "sk-should-never-be-forwarded",
                "OPENAI_API_KEY": "sk-should-never-be-forwarded",
                "CLAUDECODE": "1",
                "CLAUDE_CODE_ENTRYPOINT": "cli",
                "HOME": os.environ.get("HOME", "/tmp"),
            },
        ):
            env = build_cli_env()
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CLAUDECODE", env)
        self.assertNotIn("CLAUDE_CODE_ENTRYPOINT", env)
        # Unrelated vars pass through untouched.
        self.assertIn("HOME", env)


class LocalCliLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_parses_cli_json_envelope(self) -> None:
        fake_result = mock.Mock(
            returncode=0,
            stdout=json.dumps({"result": "ok", "is_error": False}),
            stderr="",
        )
        client = LocalCliLLMClient(model="sonnet")
        with mock.patch("subprocess.run", return_value=fake_result) as run_mock:
            output = await client.complete("system", "user prompt")
        self.assertEqual(output, "ok")
        called_env = run_mock.call_args.kwargs["env"]
        self.assertNotIn("ANTHROPIC_API_KEY", called_env)
        self.assertNotIn("OPENAI_API_KEY", called_env)

    async def test_complete_raises_on_nonzero_exit(self) -> None:
        fake_result = mock.Mock(returncode=1, stdout="", stderr="auth error")
        client = LocalCliLLMClient(model="sonnet")
        with mock.patch("subprocess.run", return_value=fake_result):
            with self.assertRaises(LocalCliLLMError):
                await client.complete("system", "user prompt")

    async def test_complete_raises_when_cli_missing(self) -> None:
        client = LocalCliLLMClient(model="sonnet")
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(LocalCliLLMError):
                await client.complete("system", "user prompt")


if __name__ == "__main__":
    unittest.main()
