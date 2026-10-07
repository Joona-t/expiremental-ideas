from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from blitz_swarm.config import AppSettings
from blitz_swarm.providers.mock import HashEmbeddingClient, RuleBasedLLMClient
from blitz_swarm.runtime import SwarmRuntime
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


class CodexBackendTests(unittest.IsolatedAsyncioTestCase):
    """BUG-002: the advertised `BLITZ_LLM_CLI_COMMAND=codex` backend."""

    CODEX_JSONL = "\n".join(
        json.dumps(event)
        for event in (
            {"type": "thread.started", "thread_id": "t-1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "i0", "type": "reasoning", "text": "thinking"}},
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "FINAL ANSWER"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        )
    )

    async def test_codex_returns_last_agent_message_not_raw_jsonl(self) -> None:
        fake_result = mock.Mock(returncode=0, stdout=self.CODEX_JSONL, stderr="")
        client = LocalCliLLMClient(command="codex")
        with mock.patch("subprocess.run", return_value=fake_result):
            output = await client.complete("system", "user prompt")
        self.assertEqual(output, "FINAL ANSWER")

    async def test_codex_default_does_not_pass_claude_model_alias(self) -> None:
        fake_result = mock.Mock(returncode=0, stdout=self.CODEX_JSONL, stderr="")
        with mock.patch.dict(os.environ, {"BLITZ_LLM_CLI_COMMAND": "codex"}):
            os.environ.pop("BLITZ_LLM_CLI_MODEL", None)
            settings = AppSettings.from_env()
        client = LocalCliLLMClient(model=settings.llm_cli_model, command=settings.llm_cli_command)
        with mock.patch("subprocess.run", return_value=fake_result) as run_mock:
            await client.complete("system", "user prompt")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[:2], ["codex", "exec"])
        self.assertNotIn("--model", cmd)
        self.assertNotIn("sonnet", cmd)

    async def test_codex_explicit_model_is_forwarded(self) -> None:
        fake_result = mock.Mock(returncode=0, stdout=self.CODEX_JSONL, stderr="")
        client = LocalCliLLMClient(model="gpt-5-codex", command="codex")
        with mock.patch("subprocess.run", return_value=fake_result) as run_mock:
            await client.complete("system", "user prompt")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--model") + 1], "gpt-5-codex")

    async def test_claude_default_model_is_sonnet(self) -> None:
        fake_result = mock.Mock(returncode=0, stdout=json.dumps({"result": "ok"}), stderr="")
        client = LocalCliLLMClient(model=None)
        with mock.patch("subprocess.run", return_value=fake_result) as run_mock:
            await client.complete("system", "user prompt")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--model") + 1], "sonnet")

    async def test_codex_error_event_raises(self) -> None:
        stdout = json.dumps({"type": "error", "message": "not logged in"})
        fake_result = mock.Mock(returncode=0, stdout=stdout, stderr="")
        client = LocalCliLLMClient(command="codex")
        with mock.patch("subprocess.run", return_value=fake_result):
            with self.assertRaises(LocalCliLLMError):
                await client.complete("system", "user prompt")

    async def test_codex_without_agent_message_raises(self) -> None:
        stdout = json.dumps({"type": "turn.completed"})
        fake_result = mock.Mock(returncode=0, stdout=stdout, stderr="")
        client = LocalCliLLMClient(command="codex")
        with mock.patch("subprocess.run", return_value=fake_result):
            with self.assertRaises(LocalCliLLMError):
                await client.complete("system", "user prompt")


class Bug001RegressionTests(unittest.TestCase):
    """BUG-001: paid-API env vars must never activate a paid client."""

    def _runtime_env(self, tmp: str, **extra: str) -> dict:
        env = {
            "BLITZ_STATE_DIR": os.path.join(tmp, "state"),
            "BLITZ_RUN_DIR": os.path.join(tmp, "runs"),
            "OPENAI_API_BASE": "https://api.example.invalid/v1",
            "OPENAI_API_KEY": "sk-should-never-be-used",
            "ANTHROPIC_API_KEY": "sk-ant-should-never-be-used",
        }
        env.update(extra)
        return env

    def test_paid_api_env_vars_still_select_local_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, self._runtime_env(tmp)):
            os.environ.pop("BLITZ_LLM_BACKEND", None)
            runtime = SwarmRuntime.from_env()
        self.assertIsInstance(runtime.llm_client, RuleBasedLLMClient)
        self.assertIsInstance(runtime.embedding_client, HashEmbeddingClient)

    def test_cli_backend_selects_local_cli_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, self._runtime_env(tmp, BLITZ_LLM_BACKEND="CLI")):
            runtime = SwarmRuntime.from_env()
        self.assertIsInstance(runtime.llm_client, LocalCliLLMClient)
        self.assertIsInstance(runtime.embedding_client, HashEmbeddingClient)

    def test_unknown_backend_warns_and_falls_back_to_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, self._runtime_env(tmp, BLITZ_LLM_BACKEND="openai")
        ):
            with self.assertWarns(RuntimeWarning):
                runtime = SwarmRuntime.from_env()
        self.assertIsInstance(runtime.llm_client, RuleBasedLLMClient)

    def test_no_paid_client_module_or_sdk_import_in_src(self) -> None:
        src = Path(__file__).resolve().parents[1] / "src"
        pattern = re.compile(
            r"from openai import|import openai|openai\.OpenAI\(|api\.openai\.com|"
            r"from anthropic import|import anthropic|anthropic\.Anthropic\(|api\.anthropic\.com|"
            r"OPENAI_API_BASE|getenv\([\"'](OPENAI|ANTHROPIC)_API_KEY"
        )
        hits = [
            f"{path}:{lineno}"
            for path in src.rglob("*.py")
            for lineno, line in enumerate(path.read_text().splitlines(), 1)
            if pattern.search(line)
        ]
        self.assertEqual(hits, [])
        self.assertFalse((src / "blitz_swarm" / "providers" / "openai_compatible.py").exists())


if __name__ == "__main__":
    unittest.main()
