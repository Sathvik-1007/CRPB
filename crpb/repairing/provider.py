from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from .models import Issue


class RepairProvider(Protocol):
    def propose_repairs(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        issues: List[Issue],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a dict with at least `edits: RepairEdit[]`.

        The provider may also return `notes: str[]` and `expected_fixes: str[]`.
        """


@dataclass
class DspyRepairProvider:
    """LLM-backed provider implemented via DspyEngine signatures."""

    engine: Any

    def propose_repairs(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        issues: List[Issue],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        c = dict(constraints or {})

        # Default: patch-based repairs only (safer; prevents whole-file rewrites and indentation drift).
        # Callers may explicitly disable patch mode (bool) or explicitly allow full rewrites.
        patch_mode = True
        if "repair_patch_mode" in c and isinstance(c.get("repair_patch_mode"), bool):
            patch_mode = bool(c.get("repair_patch_mode"))

        allow_full_rewrites = False
        if "repair_allow_full_file_rewrites" in c and isinstance(
            c.get("repair_allow_full_file_rewrites"), bool
        ):
            allow_full_rewrites = bool(c.get("repair_allow_full_file_rewrites"))

        if patch_mode and hasattr(self.engine, "repair_project_patches"):
            proposal = self.engine.repair_project_patches(
                idea=idea or "",
                constraints=c,
                issues=[i.__dict__ for i in (issues or [])],
                files=dict(files or {}),
                file_specs=dict(file_specs or {}),
                context=dict(context or {}),
            )
            if isinstance(proposal, dict) and isinstance(proposal.get("patches"), list):
                return proposal

        if allow_full_rewrites and hasattr(self.engine, "repair_project"):
            return self.engine.repair_project(
                idea=idea or "",
                constraints=c,
                issues=[i.__dict__ for i in (issues or [])],
                files=dict(files or {}),
                file_specs=dict(file_specs or {}),
                context=dict(context or {}),
            )

        return {"patches": [], "notes": ["repair_disabled:no_patch_proposal"]}
