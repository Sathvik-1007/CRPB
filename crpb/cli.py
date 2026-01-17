import importlib
import sys

import typer
from rich.console import Console

from . import __version__
from .core.env import load_env

console = Console()
# Load environment variables from a .env file if present (e.g., provider keys).
load_env()

EPILOG = "\n".join(
    [
        "Use `python -m crpb <command> --help` for detailed options.",
        "Try `python -m crpb llm --help` to manage LLM providers.",
        "Use `python -m crpb doctor` to verify CRPB's local scaffolding (no LLM/provider calls).",
    ]
)

app = typer.Typer(
    help="CRPB - Context Recursive Project Builder",
    epilog=EPILOG,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

# Lazy-load only the requested subcommand to avoid importing modules with optional/heavy dependencies unnecessarily
_SUBAPPS = {
    "plan": "crpb.commands.plan_cmd",
    "doctor": "crpb.commands.doctor_cmd",
    "status": "crpb.commands.status_cmd",
    "watch": "crpb.commands.watch_cmd",
    "build": "crpb.commands.build_cmd",
    "replay": "crpb.commands.replay_cmd",
    "schemas": "crpb.commands.schemas_cmd",
    "tasks": "crpb.commands.tasks_cmd",
    "llm": "crpb.commands.llm_cmd",
    "embeddings": "crpb.commands.embeddings_cmd",
    "temporal-worker": "crpb.commands.temporal_worker_cmd",
    "temporal-ctl": "crpb.commands.temporal_ctl_cmd",
}


def _detect_requested() -> str | None:
    for arg in sys.argv[1:]:
        if arg in _SUBAPPS:
            return arg
    return None


def _add_subapp(name: str) -> None:
    try:
        mod_path = _SUBAPPS[name]
        mod = importlib.import_module(mod_path)
        sub_app = getattr(mod, "app", None)
        if sub_app is not None:
            app.add_typer(sub_app, name=name)
    except Exception as _:
        # On import failure, register a stub that explains the issue on invocation
        @app.command(name)
        def _stub():
            console.print(
                f"[red]Subcommand '{name}' is currently unavailable due to missing dependencies or configuration.[/red]"
            )
            raise typer.Exit(code=2)


# If a specific subcommand was requested, load only that; otherwise load a minimal safe set
_requested = _detect_requested()
if _requested:
    _add_subapp(_requested)
else:
    # Register all known subcommands so `--help` reflects project capabilities.
    # Modules are still imported lazily (and fall back to a stub on failure).
    for nm in sorted(_SUBAPPS.keys()):
        _add_subapp(nm)


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit.",
        is_eager=True,
    ),
):
    """Top-level entrypoint. Use subcommand --help for details."""
    if version:
        console.print(f"CRPB {__version__}")
        raise typer.Exit()
