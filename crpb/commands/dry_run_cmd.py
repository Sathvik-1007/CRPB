from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
from ..config import resolve_run_dir, make_paths
from ..eventbus import EventBus
from ..status import NodeStatus
from ..leases import Leases
from ..utils.fs import atomic_write_json, ensure_parent
from ..utils.ui import sep

app = typer.Typer(help="Create a deterministic scaffold without LLM calls")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("new", "--run", help="run_<ts> | latest | new | name"),
    node_id: str = typer.Option(None, "--node-id", help="Deterministic node id for this dry-run"),
    parent_id: str = typer.Option(None, "--parent-id", help="Parent node id for orchestration tracking"),
):
    sep("DRY-RUN START")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    # seed core files
    sep("SEED")
    # Create empty jsonl files (append-only semantics); avoid writing JSON arrays
    ensure_parent(paths.graph / "nodes.jsonl")
    ensure_parent(paths.graph / "edges.jsonl")
    (paths.graph / "nodes.jsonl").touch(exist_ok=True)
    (paths.graph / "edges.jsonl").touch(exist_ok=True)
    atomic_write_json(paths.graph / "leases.json", {})
    atomic_write_json(paths.registry / "registry.json", {"version": 0, "files": {}})
    # align with plan.md: create a stubs directory under registry/
    (paths.registry / "stubs").mkdir(parents=True, exist_ok=True)
    # validations/repair_plans/ is a directory, created by make_paths(); no file creation here

    bus = EventBus(paths.logs / "events.jsonl")
    status = NodeStatus(paths.graph / "node_status.jsonl")
    leases = Leases(paths.graph / "leases.json")

    # simple demo lifecycle
    sep("LIFECYCLE")
    node_id = node_id or f"dryrun::{run_path.name}"
    parent_id = parent_id or f"root::{run_path.name}"
    status.write(node_id, "CREATED", parent_id=parent_id)
    lease_id = leases.grant(node_id, ttl=120)
    status.write(node_id, "LEASED", prev_state="CREATED", lease_id=lease_id, parent_id=parent_id)
    bus.emit("NODE_CREATED", node_id=node_id, parent_id=parent_id)
    bus.emit("LEASE_GRANTED", node_id=node_id, lease_id=lease_id, parent_id=parent_id)

    status.write(node_id, "DONE", prev_state="LEASED", parent_id=parent_id)
    bus.emit("NODE_DONE", node_id=node_id, parent_id=parent_id)

    sep("DRY-RUN DONE")
    console.print(f"[green]Dry-run scaffold created[/green] at: {run_path}")
