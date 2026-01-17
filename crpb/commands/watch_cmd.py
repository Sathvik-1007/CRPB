from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console

from ..core.config import make_paths, resolve_run_dir
from ..utils.ui import sep

app = typer.Typer(help="Tail the event log")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("latest", "--run", help="run_<ts> | latest | name"),
    raw: bool = typer.Option(
        False, "--raw/--compact", help="Print raw JSON events instead of a compact summary"
    ),
    interval: float = typer.Option(1.0, "--interval", min=0.1, help="Polling interval in seconds"),
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
            with open(events, encoding="utf-8") as f:
                f.seek(pos)
                for line in f:
                    try:
                        rec = json.loads(line)
                        if raw:
                            console.print(rec)
                        else:
                            et = rec.get("type")
                            at = rec.get("at")
                            payload = rec.get("payload", {}) or {}
                            nid = (
                                payload.get("node_id")
                                or payload.get("child")
                                or payload.get("parent")
                            )
                            tid = payload.get("task_id")
                            parent = payload.get("parent_id")
                            # Minimal compact rendering
                            console.print(f"[{at}] {et} nid={nid} tid={tid} parent={parent}")
                    except Exception:
                        pass
                pos = f.tell()
        except FileNotFoundError:
            pass
        time.sleep(interval)
