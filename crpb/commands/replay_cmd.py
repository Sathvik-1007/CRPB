from __future__ import annotations
from pathlib import Path
import json
import typer
from rich.console import Console
from rich.table import Table
from ..config import resolve_run_dir, make_paths
from ..utils.ui import sep

app = typer.Typer(help="Replay events.jsonl to produce a summary rollup")
console = Console()


def _iter_events(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@app.callback(invoke_without_command=True)
def main(
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("latest", "--run", help="run_<ts> | latest | name"),
    write: bool = typer.Option(True, "--write/--no-write", help="Write rollup to outputs/rollups/status.json"),
):
    sep("REPLAY")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    events_path = paths.logs / "events.jsonl"
    status = {
        "nodes": {},  # node_id -> last_event_type
        "counts": {},  # event_type -> count
        "first_event_at": None,
        "last_event_at": None,
        "functions": {},  # key path::name -> {last_event, first_at, last_at, future_wait, future_ready, timeline}
        "tasks": {},      # task_id -> {kind, last_event, first_at, last_at, assigned, split, done, failed, timeline}
    }

    for evt in _iter_events(events_path):
        et = evt.get("type")
        at = evt.get("at")
        payload = evt.get("payload", {})
        status["counts"][et] = status["counts"].get(et, 0) + 1
        if at is not None:
            status["first_event_at"] = at if status["first_event_at"] is None else min(status["first_event_at"], at)
            status["last_event_at"] = at if status["last_event_at"] is None else max(status["last_event_at"], at)
        node_id = payload.get("node_id")
        if node_id:
            status["nodes"][node_id] = et
        # per-function rollup (classic build flow)
        fpath = payload.get("path")
        fname = payload.get("name")
        if fpath and fname:
            key = f"{fpath}::{fname}"
            rec = status["functions"].setdefault(
                key,
                {"last_event": None, "first_at": None, "last_at": None, "future_wait": 0, "future_ready": 0, "timeline": []},
            )
            rec["last_event"] = et
            if at is not None:
                rec["first_at"] = at if rec["first_at"] is None else min(rec["first_at"], at)
                rec["last_at"] = at if rec["last_at"] is None else max(rec["last_at"], at)
                rec["timeline"].append({"at": at, "type": et})
            if et == "FUTURE_WAIT":
                rec["future_wait"] += 1
            elif et == "FUTURE_READY":
                rec["future_ready"] += 1

        # per-task rollup (TaskPlan recursive flow)
        norm = et
        if isinstance(et, str) and et.startswith("CHILD_"):
            norm = "TASK_" + et[len("CHILD_"):]
        if norm in ("TASK_ASSIGNED", "TASK_SPLIT", "TASK_DONE", "TASK_FAILED"):
            tid = payload.get("task_id")
            if tid:
                trec = status["tasks"].setdefault(
                    tid,
                    {
                        "kind": payload.get("kind"),
                        "last_event": None,
                        "first_at": None,
                        "last_at": None,
                        "assigned": 0,
                        "split": 0,
                        "done": 0,
                        "failed": 0,
                        "timeline": [],
                    },
                )
                trec["kind"] = trec.get("kind") or payload.get("kind")
                trec["last_event"] = norm
                if at is not None:
                    trec["first_at"] = at if trec["first_at"] is None else min(trec["first_at"], at)
                    trec["last_at"] = at if trec["last_at"] is None else max(trec["last_at"], at)
                    trec["timeline"].append({"at": at, "type": norm})
                if norm == "TASK_ASSIGNED":
                    trec["assigned"] += 1
                elif norm == "TASK_SPLIT":
                    trec["split"] += 1
                elif norm == "TASK_DONE":
                    trec["done"] += 1
                elif norm == "TASK_FAILED":
                    trec["failed"] += 1

    table = Table(title="Replay Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Nodes", str(len(status["nodes"])) )
    table.add_row("Events", str(sum(status["counts"].values())))
    table.add_row("First At", str(status["first_event_at"]))
    table.add_row("Last At", str(status["last_event_at"]))
    console.print(table)

    if status["nodes"]:
        t2 = Table(title="Nodes (last observed event)")
        t2.add_column("Node ID")
        t2.add_column("Last Event")
        for nid, last in status["nodes"].items():
            t2.add_row(nid, last)
        console.print(t2)

    if status["tasks"]:
        t_tasks = Table(title="Tasks (TaskPlan rollup)")
        t_tasks.add_column("Task ID")
        t_tasks.add_column("Kind")
        t_tasks.add_column("Last Event")
        t_tasks.add_column("ASSIGNED")
        t_tasks.add_column("SPLIT")
        t_tasks.add_column("DONE")
        t_tasks.add_column("FAILED")
        # show up to 100 tasks
        for tid, trec in list(status["tasks"].items())[:100]:
            t_tasks.add_row(
                tid,
                str(trec.get("kind")),
                str(trec.get("last_event")),
                str(trec.get("assigned")),
                str(trec.get("split")),
                str(trec.get("done")),
                str(trec.get("failed")),
            )
        console.print(t_tasks)

    if status["functions"]:
        t3 = Table(title="Functions (timeline + futures)")
        t3.add_column("Function")
        t3.add_column("Last Event")
        t3.add_column("FUTURE_WAIT")
        t3.add_column("FUTURE_READY")
        for key, rec in status["functions"].items():
            t3.add_row(key, str(rec["last_event"]), str(rec["future_wait"]), str(rec["future_ready"]))
        console.print(t3)

    if write:
        out = paths.outputs / "rollups" / "status.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(status, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote rollup[/green]: {out}")
