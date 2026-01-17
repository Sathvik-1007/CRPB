from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from ..core.config import make_paths, resolve_run_dir
from ..core.eventbus import EventBus
from ..utils.fs import atomic_write_json
from ..utils.ui import sep

try:
    from ..temporal.workflow import start_workflow
except ImportError:
    start_workflow = None  # type: ignore

app = typer.Typer(help="Plan using Temporal workflow (produces plan.json + codespec.json)")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    idea: Annotated[Optional[str], typer.Option("--idea", help="High-level idea to build")] = None,
    constraints: Annotated[
        Optional[str], typer.Option("--constraints", help="JSON string of constraints")
    ] = None,
    constraints_file: Annotated[
        Optional[str],
        typer.Option("--constraints-file", help="Path to JSON file with constraints"),
    ] = None,
    run_dir: Annotated[Optional[str], typer.Option("--run-dir", help="Base runs folder")] = None,
    run: Annotated[str, typer.Option("--run", help="run_<ts> | latest | new | name")] = "new",
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
    if start_workflow is None:
        console.print(
            "[red]Temporal workflow entrypoint unavailable.[/red] Ensure temporal support is installed and importable."
        )
        raise typer.Exit(code=2)

    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    sep("TEMPORAL PLAN START")

    constraints_obj: dict = {}
    if constraints_file:
        try:
            constraints_obj = json.loads(Path(constraints_file).read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[red]Failed to read constraints file: {e}[/red]")
            raise typer.Exit(code=2)
    elif isinstance(constraints, str):
        try:
            constraints_obj = json.loads(constraints)
        except Exception as e:
            console.print(f"[red]Invalid constraints JSON: {e}[/red]")
            raise typer.Exit(code=2)
    else:
        cf = paths.plan / "constraints.json"
        if cf.exists():
            try:
                constraints_obj = json.loads(cf.read_text(encoding="utf-8"))
            except Exception as e:
                console.print(f"[red]Failed to read constraints file: {e}[/red]")
                raise typer.Exit(code=2)

    if idea is None or not str(idea).strip():
        console.print("[red]--idea is required[/red]")
        raise typer.Exit(code=2)

    constraints_str = json.dumps(constraints_obj, sort_keys=True)
    run_id_input = f"{idea}:{constraints_str}"
    run_id_hash = hashlib.sha256(run_id_input.encode("utf-8")).hexdigest()
    run_id = run_id_hash

    console.print(f"Idea: [bold]{idea}[/bold]")
    console.print(f"[cyan]run_id:[/cyan] {run_id}")

    bus = EventBus(paths.logs / "events.jsonl")
    try:
        atomic_write_json(paths.plan / "idea.json", {"idea": idea, "run_id": run_id})
        atomic_write_json(paths.plan / "constraints.json", constraints_obj)
    except Exception as e:
        console.print(f"[red]Failed to persist plan inputs:[/red] {e}")
        bus.emit("TEMPORAL_INPUT_PERSIST_FAILED", run_id=run_id, error=str(e))
        raise typer.Exit(code=1)

    try:
        result = asyncio.run(
            start_workflow(
                idea=idea,
                constraints=constraints_obj,
                run_id=run_id,
                run_dir=str(run_path),
                stop_after="plan",
                server_address=server_address,
                namespace=namespace,
                task_queue=task_queue,
                connect_timeout_seconds=connect_timeout_seconds,
                connect_retries=connect_retries,
                connect_retry_backoff_seconds=connect_retry_backoff_seconds,
            )
        )
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error running Temporal workflow:[/red] {e}")
        bus.emit("TEMPORAL_PLAN_ERROR", run_id=run_id, error=str(e))
        raise typer.Exit(code=1)

    if not isinstance(result, dict) or not result.get("ok", False):
        err = (result or {}).get("error", "unknown") if isinstance(result, dict) else "unknown"
        issues = (result or {}).get("issues", []) if isinstance(result, dict) else []
        console.print(f"[red]Temporal plan failed:[/red] {err}")
        if issues:
            console.print(f"[yellow]Issues:[/yellow] {', '.join([str(x) for x in issues])}")
        raise typer.Exit(code=1)

    sep("TEMPORAL PLAN DONE")
    console.print(f"[green]Wrote:[/green] {paths.plan / 'plan.json'}")
    console.print(f"[green]Wrote:[/green] {paths.plan / 'codespec.json'}")
