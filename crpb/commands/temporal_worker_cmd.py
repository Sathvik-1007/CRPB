from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Annotated

import typer
from rich.console import Console

from ..core.env import load_env

# Ensure env is loaded (and blank vars are filled from .env) before importing workflow code.
load_env()

# Lazy import to avoid requiring temporalio for all commands
try:
    from ..temporal.workflow import run_worker
except ImportError:
    run_worker = None  # type: ignore

app = typer.Typer(help="Run Temporal worker for CRPB activities")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    server_address: Annotated[
        str, typer.Option("--server", help="Temporal server address")
    ] = "localhost:7233",
    namespace: Annotated[str, typer.Option("--namespace", help="Temporal namespace")] = "default",
    task_queue: Annotated[
        str, typer.Option("--task-queue", help="Temporal task queue")
    ] = "crpb-task-queue",
    connect_timeout_seconds: Annotated[
        float,
        typer.Option(
            "--connect-timeout-seconds",
            help="Timeout (seconds) for each Temporal server connection attempt",
        ),
    ] = 10.0,
    connect_retries: Annotated[
        int,
        typer.Option(
            "--connect-retries",
            help="Number of retries if the Temporal server is not reachable (0 = fail fast)",
        ),
    ] = 0,
    connect_retry_backoff_seconds: Annotated[
        float,
        typer.Option(
            "--connect-retry-backoff-seconds",
            help="Base backoff (seconds) between retries (exponential backoff)",
        ),
    ] = 1.0,
):
    """Start the CRPB Temporal worker (requires temporalio and running Temporal server)."""
    if run_worker is None:
        console.print(
            "[yellow]temporalio package not installed.[/yellow] Install with: pip install temporalio>=1.0"
        )
        raise typer.Exit(code=2)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        console.print(
            f"[cyan]Starting CRPB Temporal worker on queue '{task_queue}' (ns={namespace})...[/cyan]"
        )
        asyncio.run(
            run_worker(
                task_queue=task_queue,
                namespace=namespace,
                server_address=server_address,
                connect_timeout_seconds=connect_timeout_seconds,
                connect_retries=connect_retries,
                connect_retry_backoff_seconds=connect_retry_backoff_seconds,
            )
        )
    except Exception as e:
        console.print(f"[red]Failed to start Temporal worker: {e}[/red]")
        console.print("[red]Traceback (most recent call last):[/red]")
        console.print(traceback.format_exc())
        console.print(
            "[yellow]Note:[/yellow] A Temporal Worker requires the Temporal Service to be running. "
            "For local dev: temporal server start-dev (or temporal.exe server start-dev on Windows)."
        )
        raise typer.Exit(code=1)
