from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from ..config import resolve_run_dir, make_paths
from ..utils.ui import sep

app = typer.Typer(help="Show current node status rollups")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("latest", "--run", help="run_<ts> | latest | name"),
):
    sep("STATUS")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)
    status_file = paths.graph / "node_status.jsonl"

    rows = []
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    import json
                    rec = json.loads(line)
                    rows.append(rec)
                except Exception:
                    pass

    table = Table(title=f"Node Status — {run_path.name}")
    table.add_column("at")
    table.add_column("node_id")
    table.add_column("state")
    table.add_column("prev")
    for r in rows[-50:]:
        table.add_row(str(r.get("at")), r.get("node_id", ""), r.get("state", ""), str(r.get("prev_state")))

    console.print(table)
