"""Local CLI subprocess LLM client.

Replaces the paid OpenAI-compatible HTTP client (BUG-001) with a subprocess
wrapper around the user's own `claude` (or `codex`) CLI. Ported from the
canonical `_build_cli_env` / `call_llm` pattern in
`lovespark-shared-llm/src/lovespark_shared_llm/cli.py` so Blitz Swarm never
bills a paid token API — it rides the user's existing Claude Code / Codex
CLI subscription (CLAUDE.md rule #10).

Embeddings have no equivalent local-CLI primitive (no CLI exposes a raw
embedding endpoint), so `HashEmbeddingClient` in `mock.py` remains the only
embedding backend. That is intentional, not a stub: it is deterministic and
fully local.
"""

from __future__ import annotations

import json
import os
import subprocess


class LocalCliLLMError(Exception):
    """Raised when the local CLI subprocess call fails after retries."""


def build_cli_env() -> dict:
    """Build the subprocess env for a CLI call — strip nesting + paid-API vars.

    * `CLAUDECODE` / `CLAUDE_CODE_ENTRYPOINT`: removed so a `claude -p` call
      made from inside an active Claude Code session doesn't trip the
      "cannot launch Claude Code inside another Claude Code session" guard.
    * `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`: stripped defensively. If either
      is set in the parent shell (for an unrelated tool), we do not want the
      child CLI silently switching from OAuth/subscription billing to a paid
      per-token API. This is the exact failure mode astrospark hit in
      BUG-010.
    """
    env = os.environ.copy()
    for var in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        env.pop(var, None)
    return env


class LocalCliLLMClient:
    """Chat-completion client backed by a local CLI subprocess.

    `command` defaults to `claude` (Claude Code CLI, `claude -p`). Set
    `BLITZ_LLM_CLI_COMMAND=codex` to use the Codex CLI instead (`codex exec`).
    Either way, no API key ever leaves this process — the CLI authenticates
    via the user's own logged-in subscription.

    `model` is optional and per-CLI: when unset, `claude` gets `sonnet` and
    `codex` gets no `--model` flag at all (so it uses the model configured in
    `~/.codex/config.toml`). A Claude alias like `sonnet` is never sent to
    codex by default (BUG-002).
    """

    DEFAULT_MODELS = {"claude": "sonnet", "codex": None}

    def __init__(self, *, model: str | None = None, command: str = "claude", timeout: int = 300) -> None:
        self.command = command
        self.model = model if model else self.DEFAULT_MODELS.get(command)
        self.timeout = timeout

    async def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
        import asyncio

        del temperature  # local CLIs don't expose a temperature knob
        prompt = f"<system-instructions>\n{system_prompt}\n</system-instructions>\n\n<task>\n{user_prompt}\n</task>"
        return await asyncio.to_thread(self._run_sync, prompt)

    def _run_sync(self, prompt: str) -> str:
        env = build_cli_env()
        if self.command == "codex":
            # `-` = read the prompt from stdin; --json emits JSONL events.
            cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]
            if self.model:
                cmd += ["--model", self.model]
            cmd.append("-")
        else:
            cmd = ["claude", "-p", "--output-format", "json", "--max-turns", "1"]
            if self.model:
                cmd += ["--model", self.model]

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalCliLLMError(f"{self.command} CLI timed out after {self.timeout}s") from exc
        except FileNotFoundError as exc:
            raise LocalCliLLMError(
                f"{self.command} CLI not found on PATH. Install it and log in "
                f"(`{self.command} login` for claude, `codex login` for codex)."
            ) from exc

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            raise LocalCliLLMError(f"{self.command} CLI failed (exit {result.returncode}): {stderr or stdout}")
        if not stdout:
            raise LocalCliLLMError(f"{self.command} CLI returned empty stdout. stderr: {stderr or '(none)'}")

        if self.command == "codex":
            return self._parse_codex_jsonl(stdout)

        try:
            cli_output = json.loads(stdout)
        except json.JSONDecodeError:
            # Some CLI configurations emit raw text rather than JSON envelopes.
            return stdout

        if isinstance(cli_output, dict) and cli_output.get("is_error"):
            raise LocalCliLLMError(f"{self.command} CLI error: {cli_output.get('result', 'unknown error')}")
        if isinstance(cli_output, dict):
            return str(cli_output.get("result", stdout))
        return stdout

    def _parse_codex_jsonl(self, stdout: str) -> str:
        """Return the last `agent_message` text from `codex exec --json` output.

        codex emits one JSON event per line (thread.started, turn.started,
        item.completed, turn.completed, ...). Feeding the whole stream to
        `json.loads` fails and used to leak the raw event stream back as the
        "answer" (BUG-002).
        """
        last_message: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            if etype in ("error", "turn.failed"):
                detail = event.get("message") or event.get("error") or event
                raise LocalCliLLMError(f"codex CLI error: {detail}")
            item = event.get("item")
            if etype == "item.completed" and isinstance(item, dict) and item.get("type") == "agent_message":
                last_message = str(item.get("text", ""))
        if last_message is None:
            raise LocalCliLLMError(f"codex CLI produced no agent_message event. stdout: {stdout[:500]}")
        return last_message
