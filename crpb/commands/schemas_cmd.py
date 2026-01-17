from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from ..core.config import make_paths, resolve_run_dir
from ..core.specs import CodeSpec, CodeSpecFile, TaskPlan, TaskSpec
from ..utils.ui import sep

app = typer.Typer(help="Generate JSON Schemas for core specs into outputs/schemas/")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("latest", "--run", help="run_<ts> | latest | name"),
):
    sep("SCHEMAS")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    out_dir = paths.outputs / "schemas"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "TaskSpec": TaskSpec,
        "TaskPlan": TaskPlan,
        "CodeSpecFile": CodeSpecFile,
        "CodeSpec": CodeSpec,
    }

    for name, model in models.items():
        schema = model.model_json_schema()
        (out_dir / f"{name}.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    console.print(f"[green]Wrote schemas[/green] to: {out_dir}")
