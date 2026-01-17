from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class IssueSeverity(str, Enum):
    fatal = "fatal"
    warning = "warning"


@dataclass(frozen=True)
class Issue:
    """A structured issue derived from deterministic and/or LM validation.

    Invariants:
    - `code` is a stable machine key (e.g., "missing_script_ref").
    - `path` is a project-relative file path when applicable.
    - `ref` is the referenced token/path/symbol when applicable.
    - `message` is a human-readable, lossless description.
    """

    code: str
    severity: IssueSeverity
    message: str
    path: Optional[str] = None
    ref: Optional[str] = None
    data: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RepairEdit:
    """A concrete edit to apply.

    `new_text` is the entire file contents after the edit.
    This avoids patch-format ambiguity and keeps application deterministic.
    """

    path: str
    new_text: str
    rationale: str = ""


@dataclass
class RepairReport:
    ok: bool
    rounds: int
    applied_edits: int
    initial_issue_count: int
    final_issue_count: int
    notes: list[str]
    details: Dict[str, Any]
