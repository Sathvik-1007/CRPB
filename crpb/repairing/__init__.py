"""Language-agnostic validate→repair→revalidate system.

This package provides deterministic orchestration (issue parsing, grouping, budgeting,
scoring, rollback) plus a pluggable repair provider (typically LLM-backed).
"""

from .models import Issue, IssueSeverity, RepairEdit, RepairReport
from .spider import repair_outputs_until_ok

__all__ = [
    "Issue",
    "IssueSeverity",
    "RepairEdit",
    "RepairReport",
    "repair_outputs_until_ok",
]
