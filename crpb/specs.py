"""Data model specifications.

The canonical models live in :mod:`crpb.core.specs`.
This shim preserves the import path used by tests and external callers.
"""

from __future__ import annotations

from .core.specs import CodeSpecFile, TaskPlan, TaskSpec

__all__ = ["TaskSpec", "TaskPlan", "CodeSpecFile"]
