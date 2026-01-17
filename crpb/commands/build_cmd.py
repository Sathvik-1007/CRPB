"""CLI command for running CRPB build via Temporal workflow.

This command implements the `build` subcommand, which starts the Temporal workflow
orchestrating the full CRPB build pipeline with deterministic run_id.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from ..core.config import make_paths, resolve_run_dir
from ..core.eventbus import EventBus
from ..utils.fs import atomic_write_json
from ..utils.ui import sep

# Lazy import to avoid requiring temporalio for all commands
try:
    from ..temporal.workflow import start_workflow
except ImportError:
    start_workflow = None  # type: ignore

app = typer.Typer(help="Run CRPB build using Temporal workflow orchestration")
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
    """Run CRPB build via Temporal workflow."""

    run_id: Optional[str] = None

    if start_workflow is None:
        console.print(
            "[red]Temporal workflow entrypoint unavailable.[/red] Ensure temporal support is installed and importable."
        )
        raise typer.Exit(code=2)

    # Resolve run directory
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    sep("TEMPORAL BUILD START")

    constraints_obj: dict = {}
    constraints_source = "default"
    if constraints_file:
        try:
            constraints_obj = json.loads(Path(constraints_file).read_text(encoding="utf-8"))
            constraints_source = "file"
        except Exception as e:
            console.print(f"[red]Failed to read constraints file: {e}[/red]")
            raise typer.Exit(code=2)
    elif isinstance(constraints, str):
        try:
            constraints_obj = json.loads(constraints)
            constraints_source = "arg"
        except Exception as e:
            console.print(f"[red]Invalid constraints JSON: {e}[/red]")
            raise typer.Exit(code=2)
    else:
        cf = paths.plan / "constraints.json"
        if cf.exists():
            try:
                constraints_obj = json.loads(cf.read_text(encoding="utf-8"))
                constraints_source = "runfile"
            except Exception as e:
                console.print(f"[red]Failed to read constraints file: {e}[/red]")
                raise typer.Exit(code=2)

    # Resolve idea
    if idea is None or not str(idea).strip():
        idea_file = paths.plan / "idea.json"
        if idea_file.exists():
            try:
                idea_data = json.loads(idea_file.read_text(encoding="utf-8"))
                idea = idea_data.get("idea", "")
                run_id = idea_data.get("run_id")
            except Exception:
                idea = None
        if not idea:
            console.print("[red]--idea is required and idea.json not found[/red]")
            raise typer.Exit(code=2)

    console.print(f"Idea: [bold]{idea}[/bold]")

    # Workflow code must not read environment variables (Temporal sandbox). If callers want
    # environment-driven toggles, map them into constraints here.
    try:
        # Default behavior: do not fail the build on project validation issues.
        # Strict/fatal mode can be enabled explicitly via constraints or env.
        if "project_validation_nonfatal" not in constraints_obj:
            strict_env = str(os.environ.get("CRPB_PROJECT_VALIDATION_STRICT", "")).strip().lower()
            strict = strict_env in ("1", "true", "yes", "y", "on")
            if not strict:
                constraints_obj["project_validation_nonfatal"] = True
            v = str(os.environ.get("CRPB_PROJECT_VALIDATE_NONFATAL", "")).strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                constraints_obj["project_validation_nonfatal"] = True
    except Exception:
        pass

    try:
        if "project_validation_strict" not in constraints_obj:
            v = str(os.environ.get("CRPB_PROJECT_VALIDATION_STRICT", "")).strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                constraints_obj["project_validation_strict"] = True
    except Exception:
        pass

    try:
        if "project_jury_passes" not in constraints_obj:
            v = str(os.environ.get("CRPB_PROJECT_JURY_PASSES", "")).strip()
            if v:
                constraints_obj["project_jury_passes"] = int(v)
    except Exception:
        pass

    # Generate deterministic run_id if not already provided
    if not run_id:
        constraints_str = json.dumps(constraints_obj, sort_keys=True)
        run_id_input = f"{idea}:{constraints_str}"
        run_id_hash = hashlib.sha256(run_id_input.encode("utf-8")).hexdigest()
        run_id = run_id_hash
    console.print(f"[cyan]run_id:[/cyan] {run_id}")

    bus = EventBus(paths.logs / "events.jsonl")
    try:
        atomic_write_json(paths.plan / "idea.json", {"idea": idea, "run_id": run_id})
        cpath = paths.plan / "constraints.json"
        if constraints_source in {"arg", "file"} or (not cpath.exists()):
            atomic_write_json(cpath, constraints_obj)
    except Exception as e:
        console.print(f"[red]Failed to persist plan inputs:[/red] {e}")
        bus.emit("TEMPORAL_INPUT_PERSIST_FAILED", run_id=run_id, error=str(e))
        raise typer.Exit(code=1)

    # Start Temporal workflow
    try:
        console.print(f"[cyan]Starting Temporal workflow at {server_address}...[/cyan]")
        result = asyncio.run(
            start_workflow(
                idea=idea,
                constraints=constraints_obj,
                run_id=run_id,
                run_dir=str(run_path),
                server_address=server_address,
                namespace=namespace,
                task_queue=task_queue,
                connect_timeout_seconds=connect_timeout_seconds,
                connect_retries=connect_retries,
                connect_retry_backoff_seconds=connect_retry_backoff_seconds,
            )
        )

        if result.get("ok", False):
            console.print("[green]Temporal workflow completed successfully![/green]")
            console.print(f"[cyan]Generated files:[/cyan] {len(result.get('generated_files', []))}")
        else:
            error = result.get("error", "unknown")
            console.print(f"[red]Temporal workflow failed: {error}[/red]")
            pv = result.get("project_validation") if isinstance(result, dict) else None
            if isinstance(pv, dict):
                try:
                    issues = list(pv.get("issues") or [])
                    warnings = list(pv.get("warnings") or [])
                    suggestions = list(pv.get("suggestions") or [])
                    console.print(
                        f"[yellow]Project validation:[/yellow] issues={len(issues)} warnings={len(warnings)} suggestions={len(suggestions)}"
                    )
                    report_path = paths.validations / "project_validation.json"
                    if report_path.exists():
                        console.print(f"[cyan]Report:[/cyan] {report_path}")
                except Exception:
                    pass
            # Keep raw lists visible for quick copy/paste.
            if isinstance(result, dict):
                if result.get("issues"):
                    console.print(
                        f"[yellow]Issues:[/yellow] {', '.join([str(x) for x in (result.get('issues') or [])])}"
                    )
                if result.get("warnings"):
                    console.print(
                        f"[yellow]Warnings:[/yellow] {', '.join([str(x) for x in (result.get('warnings') or [])])}"
                    )
                if result.get("suggestions"):
                    console.print(
                        f"[yellow]Suggestions:[/yellow] {', '.join([str(x) for x in (result.get('suggestions') or [])])}"
                    )
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error running Temporal workflow: {e}[/red]")
        console.print(
            "[yellow]Note:[/yellow] Temporal must be running and reachable. "
            "For local dev: temporal server start-dev (or temporal.exe server start-dev on Windows)."
        )
        bus.emit("TEMPORAL_BUILD_ERROR", run_id=run_id, error=str(e))
        raise typer.Exit(code=1)

    sep("TEMPORAL BUILD DONE")
