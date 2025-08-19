import os
import time
from pathlib import Path
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("CRPB_OPENAI_MODEL", "gpt-4o-mini")

@dataclass
class RunPaths:
    root: Path
    plan: Path
    graph: Path
    registry: Path
    specs: Path
    artifacts: Path
    validations: Path
    repair_plans: Path
    logs: Path
    outputs: Path


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(base: Path | None, name: str | None) -> Path:
    base = Path(base or Path.cwd() / "runs")
    base.mkdir(parents=True, exist_ok=True)
    if name in (None, "new"):
        run = base / f"run_{now_ts()}"
    elif name == "latest":
        runs = sorted([p for p in base.glob("run_*") if p.is_dir()])
        run = runs[-1] if runs else base / f"run_{now_ts()}"
    else:
        run = base / name
    run.mkdir(parents=True, exist_ok=True)
    return run


def make_paths(run: Path) -> RunPaths:
    plan = run / "plan"
    graph = run / "graph"
    registry = run / "registry"
    specs = run / "specs"
    artifacts = run / "artifacts"
    validations = run / "validations"
    repair_plans = run / "repair_plans"
    logs = run / "logs"
    outputs = run / "outputs"
    for p in (plan, graph, registry, specs, artifacts, validations, repair_plans, logs, outputs):
        p.mkdir(parents=True, exist_ok=True)
    return RunPaths(run, plan, graph, registry, specs, artifacts, validations, repair_plans, logs, outputs)
