from __future__ import annotations

import http.server
import importlib.util
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from blitz_swarm.config import AppSettings
from blitz_swarm.providers.mock import HashEmbeddingClient, RuleBasedLLMClient
from blitz_swarm.runtime import SwarmRuntime
from blitz_swarm.workflows.research import ResearchWorkflow


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        del format, args


class LocalHTTPServer:
    def __init__(self, root: Path) -> None:
        handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(root), **kwargs)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "LocalHTTPServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.server.shutdown()
        self.thread.join(timeout=2)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/source.html"


@unittest.skipUnless(importlib.util.find_spec("redis"), "redis package not installed")
class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        try:
            socket.create_connection(("127.0.0.1", 6379), timeout=0.5).close()
        except OSError as exc:
            self.skipTest(f"Redis is not available locally: {exc}")

    async def test_research_workflow_writes_report_trace_and_persisted_state(self) -> None:
        workspace_root = Path(__file__).resolve().parents[1]
        artifact = workspace_root / "compass_artifact_wf-f6ac8f4e-2c3f-47d2-8611-8dc2eb1032cc_text_markdown.md"
        self.assertTrue(artifact.exists(), "Seed artifact is missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            source_html = temp_root / "source.html"
            source_html.write_text(
                "<html><head><title>Local URL</title></head><body><h1>Agent Swarm</h1><p>Redis coordinates the swarm.</p></body></html>",
                encoding="utf-8",
            )
            settings = AppSettings(
                workspace_root=workspace_root,
                state_dir=temp_root / "state",
                run_dir=temp_root / "runs",
                redis_url=os.getenv("BLITZ_REDIS_URL", "redis://127.0.0.1:6379/0"),
                sqlite_path=temp_root / "state" / "blitz.db",
                vector_dir=temp_root / "state" / "vectors",
                seed_corpus_paths=[artifact],
            )
            runtime = SwarmRuntime(
                settings=settings,
                llm_client=RuleBasedLLMClient(),
                embedding_client=HashEmbeddingClient(),
            )
            async with runtime:
                with LocalHTTPServer(temp_root) as server:
                    workflow = ResearchWorkflow(runtime)
                    trace = await runtime.run_workflow(
                        workflow,
                        brief="Recommend the best architecture for a parallel agent swarm.",
                        inputs=[],
                        urls=[server.url],
                        max_rounds=3,
                    )
                    report_path = Path(trace["artifacts"]["report"])
                    trace_path = Path(trace["artifacts"]["trace"])
                    self.assertTrue(report_path.exists())
                    self.assertTrue(trace_path.exists())
                    inspected = await runtime.inspect_run(trace["run_id"])
                    self.assertGreaterEqual(len(inspected["events"]), 1)
                    self.assertGreaterEqual(len(inspected["snapshots"]), 1)
                    results = await runtime.memory.query("Redis blackboard", limit=3)
                    self.assertTrue(results)
