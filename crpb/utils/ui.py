from __future__ import annotations

from rich.console import Console
from rich.rule import Rule

console = Console()


def sep(title: str = "") -> None:
    console.print(Rule(title or ""))
