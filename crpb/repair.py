from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import json
from typing import Any, Dict, List


@dataclass
class RepairIssue:
    kind: str
    message: str
    file: str | None = None
    function: str | None = None


@dataclass
class RepairPlan:
    created_at: float
    node_id: str
    idea: str
    issues: List[RepairIssue]
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "node_id": self.node_id,
            "idea": self.idea,
            "issues": [asdict(i) for i in self.issues],
            "suggestions": self.suggestions,
        }


def generate_repair_plan(node_id: str, idea: str, file_path: str, failures: List[RepairIssue]) -> RepairPlan:
    # Minimal heuristic suggestions
    suggestions: List[str] = []
    for iss in failures:
        if iss.kind.startswith("schema"):
            suggestions.append("Update the spec to conform to JSON Schema or adjust generator prompts.")
        elif iss.kind.startswith("SyntaxError"):
            suggestions.append("Fix syntax in generated code; re-run build for the affected function.")
        elif iss.kind.startswith("disallowed_imports"):
            suggestions.append("Remove or replace disallowed imports in implementation.")
        elif iss.kind.startswith("style"):
            suggestions.append("Wrap long lines to the configured maximum line length.")
        else:
            suggestions.append("Review validator output and adjust implementation or spec accordingly.")
    if not suggestions:
        suggestions.append("Re-run build with increased logging and inspect events via replay.")
    return RepairPlan(
        created_at=time.time(),
        node_id=node_id,
        idea=idea,
        issues=failures,
        suggestions=list(dict.fromkeys(suggestions)),  # de-dup while preserving order
    )


def write_repair_plan(path: Path, plan: RepairPlan) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return path
