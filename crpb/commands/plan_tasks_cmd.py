from __future__ import annotations
import json
from pathlib import Path
import os
import typer
from rich.console import Console
from ..config import resolve_run_dir, make_paths
from ..planner import generate_task_plan
from ..utils.fs import atomic_write_json
from ..utils.ui import sep

app = typer.Typer(help="Plan a hierarchical TaskPlan and save under run/plan/")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    idea: str = typer.Option(..., "--idea", help="High-level idea to build"),
    constraints: str = typer.Option("{}", "--constraints", help="JSON string of constraints"),
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("new", "--run", help="run_<ts> | latest | new | name"),
):
    sep("PLAN TASKS START")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    # Fail fast if no LLM key
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]LLM required for task planning: set OPENAI_API_KEY[/red]")
        raise typer.Exit(code=2)

    try:
        constraints_obj = json.loads(constraints)
    except Exception as e:
        console.print(f"[red]Invalid constraints JSON: {e}")
        raise typer.Exit(code=2)

    task_plan = generate_task_plan(idea, constraints_obj, use_llm=True)
    atomic_write_json(paths.plan / "idea.json", {"idea": idea})
    atomic_write_json(paths.plan / "constraints.json", constraints_obj)
    atomic_write_json(paths.plan / "task_plan.json", task_plan.model_dump())
    atomic_write_json(paths.outputs / "manifest.json", {"run": str(run_path)})
    sep("PLAN TASKS DONE")
    console.print(f"[green]Planned TaskPlan[/green] at: {run_path}. Tasks: {len(task_plan.tasks)}")
