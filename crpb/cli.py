import typer
from rich.console import Console
from dotenv import load_dotenv
from .commands.plan_cmd import app as plan_app
from .commands.plan_tasks_cmd import app as plan_tasks_app
from .commands.dry_run_cmd import app as dry_run_app
from .commands.status_cmd import app as status_app
from .commands.watch_cmd import app as watch_app
from .commands.build_cmd import app as build_app
from .commands.replay_cmd import app as replay_app
from .commands.schemas_cmd import app as schemas_app
from .commands.tasks_cmd import app as tasks_app

console = Console()

# Load environment variables from a .env file if present (e.g., OPENAI_API_KEY)
load_dotenv()

app = typer.Typer(help="CRPB — Context Recursive Project Builder")
app.add_typer(plan_app, name="plan")
app.add_typer(plan_tasks_app, name="plan-tasks")
app.add_typer(dry_run_app, name="dry-run")
app.add_typer(status_app, name="status")
app.add_typer(watch_app, name="watch")
app.add_typer(build_app, name="build")
app.add_typer(replay_app, name="replay")
app.add_typer(schemas_app, name="schemas")
app.add_typer(tasks_app, name="tasks")
