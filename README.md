# Blitz Swarm

Blitz Swarm is a CLI-first research synthesis runtime that coordinates a small
parallel agent swarm over a Redis-backed blackboard with durable SQLite state
and semantic memory.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
docker run --rm -p 6379:6379 redis:7
blitz ingest compass_artifact_wf-f6ac8f4e-2c3f-47d2-8611-8dc2eb1032cc_text_markdown.md
blitz run research --brief "Summarize the optimal memory architecture for an agent swarm."
```

## Environment

- `BLITZ_REDIS_URL` defaults to `redis://127.0.0.1:6379/0`
- `OPENAI_API_BASE` points to an OpenAI-compatible API
- `OPENAI_API_KEY` provides the API credential
- `OPENAI_CHAT_MODEL` defaults to `gpt-4.1-mini`
- `OPENAI_EMBEDDING_MODEL` defaults to `text-embedding-3-small`

Run artifacts are written to `.blitz/runs/<run_id>/`.
