"""Temporal workflow surface module.

The canonical implementation lives in :mod:`crpb.temporal.workflow`.
This shim preserves a stable import path for consumers/tests.
"""

from __future__ import annotations

from .temporal.workflow import (
    CRPBWorkflow,
    generate_file_activity,
    merge_activity,
    plan_activity,
    verify_exports_activity,
)

__all__ = [
    "CRPBWorkflow",
    "plan_activity",
    "generate_file_activity",
    "verify_exports_activity",
    "merge_activity",
]
