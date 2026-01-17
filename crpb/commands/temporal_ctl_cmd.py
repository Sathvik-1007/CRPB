from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from ..utils.ui import sep

try:
    from ..temporal.workflow import connect_client
except ImportError:
    connect_client = None  # type: ignore

app = typer.Typer(help="Control a running CRPB Temporal workflow (pause/unpause/status/cancel)")
console = Console()


def _workflow_id_from_mode(run_id: str, mode: str) -> str:
    m = str(mode or "build").strip().lower()
    if m == "plan":
        return f"{run_id}::plan"
    return run_id


def _resolve_run_id(
    *,
    run_id: Optional[str],
    run_dir: Optional[str],
    run: str,
) -> str:
    if run_id and str(run_id).strip():
        return str(run_id).strip()

    base = Path(run_dir) if run_dir else (Path.cwd() / "runs")
    if run in (None, "", "new"):
        raise typer.BadParameter(
            "--run-id is required when --run is 'new' (no existing run directory to inspect)"
        )
    if run == "latest":
        runs = sorted([p for p in base.glob("run_*") if p.is_dir()])
        if not runs:
            raise typer.BadParameter(
                f"no existing runs found under {base}; pass --run-id explicitly"
            )
        run_path = runs[-1]
    else:
        run_path = base / run
        if not run_path.exists():
            raise typer.BadParameter(
                f"run directory does not exist: {run_path}; pass --run-id explicitly"
            )

    idea_file = run_path / "plan" / "idea.json"
    if not idea_file.exists():
        raise typer.BadParameter("--run-id is required unless run/plan/idea.json exists")

    try:
        obj = json.loads(idea_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise typer.BadParameter(f"failed to parse {idea_file}: {e}")

    rid = obj.get("run_id")
    if not isinstance(rid, str) or not rid.strip():
        raise typer.BadParameter("run/plan/idea.json missing run_id; pass --run-id")
    return rid.strip()


async def _handle(
    *,
    server_address: str,
    namespace: str,
    connect_timeout_seconds: float,
    connect_retries: int,
    connect_retry_backoff_seconds: float,
    workflow_id: str,
):
    if connect_client is None:
        raise RuntimeError(
            "temporalio package not installed. Install with: pip install temporalio>=1.0"
        )

    client = await connect_client(
        server_address=server_address,
        namespace=namespace,
        connect_timeout_seconds=connect_timeout_seconds,
        connect_retries=connect_retries,
        connect_retry_backoff_seconds=connect_retry_backoff_seconds,
    )
    return client.get_workflow_handle(workflow_id)


@app.command("status")
def status(
    mode: Annotated[str, typer.Option("--mode", help="Workflow mode: build | plan")] = "build",
    run_id: Annotated[
        Optional[str], typer.Option("--run-id", help="Temporal workflow id (CRPB run_id)")
    ] = None,
    run_dir: Annotated[
        Optional[str], typer.Option("--run-dir", help="Base runs folder (optional)")
    ] = None,
    run: Annotated[str, typer.Option("--run", help="run_<ts> | latest | new | name")] = "latest",
    server_address: Annotated[
        str, typer.Option("--server", help="Temporal server address")
    ] = "localhost:7233",
    namespace: Annotated[str, typer.Option("--namespace", help="Temporal namespace")] = "default",
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
    rid = _resolve_run_id(run_id=run_id, run_dir=run_dir, run=run)
    wid = _workflow_id_from_mode(rid, mode)

    async def _run() -> None:
        h = await _handle(
            server_address=server_address,
            namespace=namespace,
            connect_timeout_seconds=connect_timeout_seconds,
            connect_retries=connect_retries,
            connect_retry_backoff_seconds=connect_retry_backoff_seconds,
            workflow_id=wid,
        )
        s = await h.query("status")
        console.print_json(data=s)

    sep("TEMPORAL STATUS")
    try:
        asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Failed to query workflow status:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("pause")
def pause(
    mode: Annotated[str, typer.Option("--mode", help="Workflow mode: build | plan")] = "build",
    run_id: Annotated[
        Optional[str], typer.Option("--run-id", help="Temporal workflow id (CRPB run_id)")
    ] = None,
    run_dir: Annotated[
        Optional[str], typer.Option("--run-dir", help="Base runs folder (optional)")
    ] = None,
    run: Annotated[str, typer.Option("--run", help="run_<ts> | latest | new | name")] = "latest",
    server_address: Annotated[
        str, typer.Option("--server", help="Temporal server address")
    ] = "localhost:7233",
    namespace: Annotated[str, typer.Option("--namespace", help="Temporal namespace")] = "default",
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
    rid = _resolve_run_id(run_id=run_id, run_dir=run_dir, run=run)
    wid = _workflow_id_from_mode(rid, mode)

    async def _run() -> None:
        h = await _handle(
            server_address=server_address,
            namespace=namespace,
            connect_timeout_seconds=connect_timeout_seconds,
            connect_retries=connect_retries,
            connect_retry_backoff_seconds=connect_retry_backoff_seconds,
            workflow_id=wid,
        )
        await h.signal("pause")

    sep("TEMPORAL PAUSE")
    try:
        asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Failed to pause workflow:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("unpause")
def unpause(
    mode: Annotated[str, typer.Option("--mode", help="Workflow mode: build | plan")] = "build",
    run_id: Annotated[
        Optional[str], typer.Option("--run-id", help="Temporal workflow id (CRPB run_id)")
    ] = None,
    run_dir: Annotated[
        Optional[str], typer.Option("--run-dir", help="Base runs folder (optional)")
    ] = None,
    run: Annotated[str, typer.Option("--run", help="run_<ts> | latest | new | name")] = "latest",
    server_address: Annotated[
        str, typer.Option("--server", help="Temporal server address")
    ] = "localhost:7233",
    namespace: Annotated[str, typer.Option("--namespace", help="Temporal namespace")] = "default",
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
    rid = _resolve_run_id(run_id=run_id, run_dir=run_dir, run=run)
    wid = _workflow_id_from_mode(rid, mode)

    async def _run() -> None:
        h = await _handle(
            server_address=server_address,
            namespace=namespace,
            connect_timeout_seconds=connect_timeout_seconds,
            connect_retries=connect_retries,
            connect_retry_backoff_seconds=connect_retry_backoff_seconds,
            workflow_id=wid,
        )
        await h.signal("unpause")

    sep("TEMPORAL UNPAUSE")
    try:
        asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Failed to unpause workflow:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("cancel")
def cancel(
    mode: Annotated[str, typer.Option("--mode", help="Workflow mode: build | plan")] = "build",
    run_id: Annotated[
        Optional[str], typer.Option("--run-id", help="Temporal workflow id (CRPB run_id)")
    ] = None,
    run_dir: Annotated[
        Optional[str], typer.Option("--run-dir", help="Base runs folder (optional)")
    ] = None,
    run: Annotated[str, typer.Option("--run", help="run_<ts> | latest | new | name")] = "latest",
    server_address: Annotated[
        str, typer.Option("--server", help="Temporal server address")
    ] = "localhost:7233",
    namespace: Annotated[str, typer.Option("--namespace", help="Temporal namespace")] = "default",
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
    rid = _resolve_run_id(run_id=run_id, run_dir=run_dir, run=run)
    wid = _workflow_id_from_mode(rid, mode)

    async def _run() -> None:
        h = await _handle(
            server_address=server_address,
            namespace=namespace,
            connect_timeout_seconds=connect_timeout_seconds,
            connect_retries=connect_retries,
            connect_retry_backoff_seconds=connect_retry_backoff_seconds,
            workflow_id=wid,
        )
        await h.signal("cancel")

    sep("TEMPORAL CANCEL")
    try:
        asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Failed to cancel workflow:[/red] {e}")
        raise typer.Exit(code=1)
