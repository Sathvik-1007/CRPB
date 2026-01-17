from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..core.config import make_paths, resolve_run_dir
from ..utils.payloads import externalize_json
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
        with open(status_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    rows.append(rec)
                except Exception:
                    pass

    # Build last state per node and state counts
    last_by_node = {}
    for r in rows:
        nid = r.get("node_id")
        if not nid:
            continue
        last_by_node[nid] = r
    state_counts = {}
    for r in last_by_node.values():
        st = r.get("state", "")
        state_counts[st] = state_counts.get(st, 0) + 1

    # Summary
    summary = Table(title=f"Status Summary — {run_path.name}")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Nodes", str(len(last_by_node)))
    for st, cnt in sorted(state_counts.items(), key=lambda x: x[0]):
        summary.add_row(st, str(cnt))
    console.print(summary)

    # Recent entries (tail)
    try:
        tail_limit = int(os.environ.get("CRPB_STATUS_TAIL", "50"))
    except Exception:
        tail_limit = 50
    if tail_limit < 0:
        tail_limit = 0

    table = Table(title=f"Recent Node Status Entries — {run_path.name}")
    table.add_column("at")
    table.add_column("node_id")
    table.add_column("state")
    table.add_column("prev")
    start_idx = max(0, len(rows) - tail_limit)
    for i in range(start_idx, len(rows)):
        r = rows[i]
        table.add_row(
            str(r.get("at")), r.get("node_id", ""), r.get("state", ""), str(r.get("prev_state"))
        )

    console.print(table)

    # Filtered: task:: nodes (last state per node)
    task_nodes = [
        (nid, rec.get("state", ""))
        for nid, rec in last_by_node.items()
        if isinstance(nid, str) and nid.startswith("task::")
    ]
    if task_nodes:
        try:
            max_tasks = int(os.environ.get("CRPB_STATUS_MAX_TASKS", "100"))
        except Exception:
            max_tasks = 100
        if max_tasks <= 0:
            max_tasks = 100

        sorted_task_nodes = sorted(task_nodes)

        ttable = Table(title="Task Nodes (last state)")
        ttable.add_column("node_id")
        ttable.add_column("state")
        for i, (nid, st) in enumerate(sorted_task_nodes):
            if i >= max_tasks:
                break
            ttable.add_row(nid, st)
        console.print(ttable)

        if len(sorted_task_nodes) > max_tasks:
            ref = externalize_json(
                base_dir=paths.artifacts,
                category="status",
                obj=[{"node_id": nid, "state": st} for nid, st in sorted_task_nodes],
                ext=".json",
            )
            console.print(
                f"[yellow]Task nodes truncated in display[/yellow]: showing={max_tasks} total={len(sorted_task_nodes)} ref={ref.get('path')}"
            )
