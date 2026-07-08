# Blitz Swarm

Blitz Swarm is an experimental, CLI-first multi-agent runtime for research
synthesis. It runs a small specialist swarm over a Redis blackboard, persists
durable state in SQLite WAL, stores semantic memory in LanceDB, and emits both
human-readable reports and machine-readable traces for every run.

## What it does

- Coordinates a six-agent research workflow: planner, local corpus analyst, URL analyst, synthesizer, critic, and arbiter/finalizer.
- Uses append-only event and observation streams for most writes, with optimistic concurrency for shared plan updates.
- Seeds the runtime with the included research artifact and supports additional local files plus explicit URL ingestion.
- Produces `report.md` and `trace.json` artifacts under `.blitz/runs/<run_id>/`.

## Architecture

- Redis: blackboard state, streams, pub/sub notifications, optimistic plan mutation, and hot-memory caching.
- SQLite + WAL: durable run metadata, event history, source records, memory metadata, dependency edges, and snapshot history.
- LanceDB: semantic retrieval for source chunks and derived observations, with a JSON fallback path for local development.
- Retention layer: composite utility scoring, dependency-aware eviction checks, and cold-storage demotion instead of destructive deletion.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
docker run --rm -p 6379:6379 redis:7
blitz ingest compass_artifact_wf-f6ac8f4e-2c3f-47d2-8611-8dc2eb1032cc_text_markdown.md
blitz run research --brief "Summarize the optimal memory architecture for an agent swarm."
```

## CLI

```bash
blitz ingest <paths...> [--url <url> ...]
blitz run research --brief "<prompt>" [--input <path> ...] [--url <url> ...]
blitz inspect run <run_id>
```

## Environment

- `BLITZ_REDIS_URL` defaults to `redis://127.0.0.1:6379/0`
- `BLITZ_STATE_DIR` overrides the local SQLite and vector-store state root
- `BLITZ_RUN_DIR` overrides where `report.md` and `trace.json` are written
- `BLITZ_LLM_BACKEND` selects the chat-completion backend: `mock` (default,
  deterministic, fully offline) or `cli` (shells out to your own `claude`/
  `codex` CLI subscription — see below)
- `BLITZ_LLM_CLI_COMMAND` selects which CLI to invoke when
  `BLITZ_LLM_BACKEND=cli`: `claude` (default) or `codex`
- `BLITZ_LLM_CLI_MODEL` selects the model alias passed to the CLI (default
  `sonnet`)

Blitz Swarm never talks to a paid token API. By default it uses the
deterministic mock LLM/embedding providers so the runtime and tests run fully
offline with no credentials of any kind. Setting `BLITZ_LLM_BACKEND=cli`
routes chat completions through your locally installed and logged-in
`claude -p` (or `codex exec`) subprocess — the same CLI subscription you use
interactively, never an `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` HTTP call.
Embeddings always use the local deterministic hash embedder
(`providers/mock.py::HashEmbeddingClient`), since no CLI exposes a raw
embedding endpoint.

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The integration suite expects a local Redis server and the `redis` Python
package to be installed.
