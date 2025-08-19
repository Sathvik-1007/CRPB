from __future__ import annotations
import json
from pathlib import Path
import typer
from rich.console import Console
from ..config import resolve_run_dir, make_paths
from ..specs import Plan
from ..planner import generate_plan
from ..utils.fs import atomic_write_json
from ..utils.ui import sep

app = typer.Typer(help="Plan a project and save idea/constraints under run/plan/")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    idea: str = typer.Option(..., "--idea", help="High-level idea to build"),
    constraints: str = typer.Option("{}", "--constraints", help="JSON string of constraints"),
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("new", "--run", help="run_<ts> | latest | new | name"),
    require_llm: bool = typer.Option(True, "--require-llm/--allow-fallback", help="Require LLM for planning; fail if unavailable"),
):
    sep("PLAN START")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    try:
        constraints_obj = json.loads(constraints)
    except Exception as e:
        console.print(f"[red]Invalid constraints JSON: {e}")
        raise typer.Exit(code=2)

    # Generate a plan (LLM-backed). By default require LLM; fallback only if explicitly allowed.
    plan_model, file_specs = generate_plan(idea, constraints_obj, use_llm=True, require_llm=require_llm)
    atomic_write_json(paths.plan / "idea.json", {"idea": idea})
    atomic_write_json(paths.plan / "constraints.json", constraints_obj)
    atomic_write_json(paths.plan / "plan.json", plan_model.model_dump())
    # Also write individual file specs for traceability
    import hashlib
    for fs in file_specs:
        fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
        atomic_write_json(paths.specs / f"file_{fid}.json", fs.model_dump())
    atomic_write_json(paths.outputs / "manifest.json", {"run": str(run_path)})
    sep("PLAN DONE")
    console.print(f"[green]Planned[/green] run at: {run_path}. Files: {[fs.path for fs in file_specs]}")
