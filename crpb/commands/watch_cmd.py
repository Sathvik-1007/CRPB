from __future__ import annotations
from pathlib import Path
import time
import json
import typer
from rich.console import Console
from ..config import resolve_run_dir, make_paths
from ..utils.ui import sep

app = typer.Typer(help="Tail the event log")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("latest", "--run", help="run_<ts> | latest | name"),
):
    sep("WATCH")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    events = paths.logs / "events.jsonl"
    console.print(f"Watching: {events}")
    pos = 0
    while True:
        try:
            with open(events, "r", encoding="utf-8") as f:
                f.seek(pos)
                for line in f:
                    try:
                        rec = json.loads(line)
                        console.print(rec)
                    except Exception:
                        pass
                pos = f.tell()
        except FileNotFoundError:
            pass
        time.sleep(1)
