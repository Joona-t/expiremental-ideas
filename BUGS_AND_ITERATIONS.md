# Bugs & Iterations

## BUG-001 — Default LLM path defaulted to a paid OpenAI-compatible API

**Date:** 2026-07-08

**Problem:** `src/blitz_swarm/providers/openai_compatible.py` implemented an
`OpenAICompatibleLLMClient` / `OpenAICompatibleEmbeddingClient` that POSTed to
an arbitrary OpenAI-compatible HTTP endpoint using `OPENAI_API_KEY`. Config
(`AppSettings.from_env`) auto-activated this client whenever
`OPENAI_API_BASE` + `OPENAI_API_KEY` were present in the environment —
including a key set globally for an unrelated tool. This violates CLAUDE.md
rule #10 ("no paid LLM API, ever") and matches the exact incident class
astrospark hit in its BUG-010 (a stray env var silently switching a CLI
subprocess from free/subscription billing to paid per-token billing).

**Root cause:** the runtime treated "OpenAI-compatible HTTP client with a raw
API key" as the real/production LLM backend and "deterministic mock" as the
fallback, inverting the priority the fleet requires (mock/local-CLI first,
paid API never).

**Fix:**
- Deleted `src/blitz_swarm/providers/openai_compatible.py` and its smoke
  test (`tests/test_openai_smoke.py`) — no HTTP client with an API key exists
  in this repo anymore.
- Added `src/blitz_swarm/providers/local_cli.py` (`LocalCliLLMClient`), a
  subprocess wrapper around the user's own `claude -p` (default) or
  `codex exec` CLI, ported from `lovespark-shared-llm`'s canonical
  `_build_cli_env` / `call_llm` pattern. `build_cli_env()` strips
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CLAUDECODE`, and
  `CLAUDE_CODE_ENTRYPOINT` from the subprocess env before every call, so a
  stray key in the parent shell can never silently flip billing.
- `AppSettings` now exposes `llm_backend` (`mock` default / `cli` opt-in via
  `BLITZ_LLM_BACKEND=cli`), `llm_cli_command`, `llm_cli_model` instead of the
  `openai_api_base`/`openai_api_key`/`openai_chat_model` fields.
  `SwarmRuntime.from_env()` defaults to `RuleBasedLLMClient` (deterministic
  mock) unless the caller explicitly opts into `cli`.
- Embeddings always use the existing deterministic `HashEmbeddingClient` —
  there is no local-CLI equivalent for a raw embedding endpoint, so this was
  kept as-is rather than stubbed differently.
- Added `tests/test_local_cli_provider.py` covering env-scrubbing and the
  CLI subprocess success/failure paths (mocked `subprocess.run`, no real
  process spawned).
- Added `scripts/check-no-paid-api.sh` (installable as `.git/hooks/pre-push`)
  that greps tracked source for `anthropic\.Anthropic\(`,
  `api\.anthropic\.com`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and blocks the
  push if any are found outside the documented, defensive exceptions
  (this guard script, the env-scrubbing code, and the docs that describe it).
  Installed locally for this clone via
  `ln -sf ../../scripts/check-no-paid-api.sh .git/hooks/pre-push`.

**Verification:** `PYTHONPATH=src python3 -m unittest discover -s tests -v`
→ 20 tests, 19 passed, 1 skipped (Redis not running locally — pre-existing,
environment-only, not caused by this change).

## BUG-002 — Advertised `codex` CLI backend was broken; BUG-001 had no regression test

**Date:** 2026-10-07

**Problem:** `BLITZ_LLM_BACKEND=cli BLITZ_LLM_CLI_COMMAND=codex` (documented
in README) did not work. (a) `BLITZ_LLM_CLI_MODEL` defaulted to `sonnet` for
both CLIs, so codex was invoked as `codex exec --model sonnet --json`, and
`sonnet` is not a codex model. (b) `codex exec --json` prints JSONL events.
`json.loads` on the multi-line stream raised, and the fallback returned the
whole raw event stream (`{"type": "thread.started", ...}`) as the "answer".
Also: (c) no unittest failed if the BUG-001 paid-client auto-activation was
restored. (d) The guard script let `from openai import OpenAI`,
`api.openai.com` and `OPENAI_API_BASE` through. (e) An unknown
`BLITZ_LLM_BACKEND` (for example `CLI`) silently fell back to mock.

**Root cause:** the codex path was written by analogy to `claude -p
--output-format json` (one JSON envelope) and was never exercised against
real `codex exec --json` output.

**Fix:**
- `LocalCliLLMClient(model=None)` uses a per-CLI default: `sonnet` for
  claude, and no `--model` flag for codex (it uses `~/.codex/config.toml`).
  `AppSettings.llm_cli_model` now defaults to `None`.
- codex is called as `codex exec --json --skip-git-repo-check [--model M] -`
  (prompt on stdin). Its JSONL is parsed for the last `item.completed`
  `agent_message`. `error`/`turn.failed` events, or no agent message at all,
  raise `LocalCliLLMError`.
- `BLITZ_LLM_BACKEND` / `BLITZ_LLM_CLI_COMMAND` are trimmed and lowercased.
  An unknown backend emits a `RuntimeWarning` before it falls back to mock.
- The guard pattern now also catches openai/anthropic SDK imports,
  `openai.OpenAI(`, `api.openai.com` and `OPENAI_API_BASE`.
- New tests in `tests/test_local_cli_provider.py`: `CodexBackendTests` (6)
  and `Bug001RegressionTests` (4). With main's `runtime.py`, `config.py` and
  `openai_compatible.py` restored, 4 of these fail and 1 errors.

**Verification:** `PYTHONPATH=src python -m unittest discover -s tests`
gives 30 tests, OK (skipped=1, Redis integration). `scripts/check-no-paid-api.sh`
prints clean. A tracked file containing `from openai import OpenAI` now exits 1.
