"""Typer CLI entrypoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from blitz_swarm.runtime import SwarmRuntime
from blitz_swarm.workflows.research import ResearchWorkflow

app = typer.Typer(help="Blitz Swarm CLI")
run_app = typer.Typer(help="Run built-in workflows")
inspect_app = typer.Typer(help="Inspect stored runs")
app.add_typer(run_app, name="run")
app.add_typer(inspect_app, name="inspect")


@app.command()
def ingest(
    paths: list[Path] | None = typer.Argument(None, exists=True, readable=True),
    url: list[str] = typer.Option(None, "--url"),
) -> None:
    """Ingest local files and explicit URLs into semantic memory."""

    async def _run() -> None:
        async with SwarmRuntime.from_env() as runtime:
            result = await runtime.ingest(paths=paths or [], urls=url or [])
            typer.echo(json.dumps(result, indent=2, ensure_ascii=True))

    asyncio.run(_run())


@run_app.command("research")
def run_research(
    brief: str = typer.Option(..., "--brief", help="Research question or task brief."),
    input: list[Path] | None = typer.Option(None, "--input", exists=True, readable=True),
    url: list[str] | None = typer.Option(None, "--url"),
    max_rounds: int | None = typer.Option(None, "--max-rounds"),
) -> None:
    """Execute the built-in research synthesis workflow."""

    async def _run() -> None:
        async with SwarmRuntime.from_env() as runtime:
            workflow = ResearchWorkflow(runtime)
            trace = await runtime.run_workflow(
                workflow,
                brief=brief,
                inputs=input or [],
                urls=url or [],
                max_rounds=max_rounds,
            )
            typer.echo(json.dumps(trace["artifacts"], indent=2, ensure_ascii=True))

    asyncio.run(_run())


@inspect_app.command("run")
def inspect_run(run_id: str) -> None:
    """Inspect a stored run by ID."""

    async def _run() -> None:
        async with SwarmRuntime.from_env() as runtime:
            result = await runtime.inspect_run(run_id)
            typer.echo(json.dumps(result, indent=2, ensure_ascii=True, default=str))

    asyncio.run(_run())


def main() -> None:
    app()
