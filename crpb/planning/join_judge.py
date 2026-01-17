from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class JoinJudgeCapsule:
    """Node-scoped validation capsule for join-judging parent-child decompositions.

    This capsule provides authoritative context for the JoinJudgeSignature LLM operation.
    When passed via constraints.side_context.join_judge, it enables the LLM to validate:
    - Decomposition completeness: Are all parent responsibilities covered by children?
    - Interface clarity: Are inputs/outputs well-defined with proper artifact contracts?
    - Artifact wiring: Do artifact producers/consumers match expectations?
    - Structural integrity: Are there deterministic issues that block progress?

    Fields align with JoinJudgeSignature input expectations:
    - target: The task node being judged (includes id, kind, title, description, deps, inputs, outputs, node_plan)
    - parent: Parent node for decomposition context (same structure as target)
    - siblings: Sibling nodes at same level for boundary validation
    - children: Child nodes to validate decomposition completeness
    - artifacts: Slice of artifact registry showing only artifacts touched by target (id -> {producers[], consumers[]})
    - deterministic_issues: Pre-computed structural issues from validators (e.g., missing node_plan keys, broken deps)

    The capsule is language-agnostic and includes only minimal, focused context to avoid token bloat.
    It deliberately excludes global plan state, full file contents, and unrelated artifacts.
    """
    target: Dict[str, Any]
    parent: Dict[str, Any]
    siblings: List[Dict[str, Any]]
    children: List[Dict[str, Any]]
    artifacts: Dict[str, Any]
    deterministic_issues: List[str]


def _node_view(n: Dict[str, Any]) -> Dict[str, Any]:
    meta = n.get("meta") if isinstance(n.get("meta"), dict) else {}
    return {
        "id": n.get("id"),
        "kind": n.get("kind"),
        "title": n.get("title") or "",
        "description": n.get("description") or "",
        "deps": list(n.get("deps") or []),
        "meta": {"path": (meta or {}).get("path")},
        "inputs": n.get("inputs") if isinstance(n.get("inputs"), dict) else {},
        "outputs": n.get("outputs") if isinstance(n.get("outputs"), dict) else {},
        "node_plan": n.get("node_plan") if isinstance(n.get("node_plan"), dict) else {},
    }


def build_join_judge_capsule(
    *,
    target: Dict[str, Any],
    parent: Optional[Dict[str, Any]],
    siblings: List[Dict[str, Any]],
    artifacts_index: Dict[str, Any],
    deterministic_issues: List[str],
) -> JoinJudgeCapsule:
    """Build a node-scoped validation capsule for join-judging decomposition quality.

    This function creates an authoritative context bundle that will be passed to the LLM via
    constraints.side_context.join_judge. The capsule structure is designed to align precisely
    with the JoinJudgeSignature input field expectations.

    Args:
        target: Task node being judged; will be exposed as capsule.target
        parent: Parent task for decomposition context; will be exposed as capsule.parent
        siblings: Sibling tasks at same level; will be exposed as capsule.siblings
        artifacts_index: Global artifact registry with producer/consumer mappings; only relevant artifacts are included in capsule
        deterministic_issues: Pre-computed structural problems (e.g., from taskplan validators); will be exposed as capsule.deterministic_issues

    Returns:
        JoinJudgeCapsule with minimal, focused context for LLM validation. Node views include:
        - id, kind, title, description, deps, meta.path, inputs, outputs, node_plan
        - Children are extracted from target.children
        - Artifact slice includes only artifacts referenced in target's inputs.consumes or outputs.produces

    This is language-agnostic and avoids pulling in unrelated global context (no full files, no distant nodes).
    The capsule is frozen (immutable) to prevent accidental modification during validation.
    """

    t_view = _node_view(target)
    p_view = _node_view(parent) if isinstance(parent, dict) else {}
    sib_views = [_node_view(s) for s in siblings if isinstance(s, dict)]

    child_views: List[Dict[str, Any]] = []
    for c in target.get("children") or []:
        if isinstance(c, dict):
            child_views.append(_node_view(c))

    artifacts_touched: List[str] = []
    try:
        outs = t_view.get("outputs") if isinstance(t_view.get("outputs"), dict) else {}
        for it in (outs or {}).get("produces") or []:
            if isinstance(it, dict) and isinstance(it.get("id"), str):
                artifacts_touched.append(it.get("id"))
    except Exception:
        pass
    try:
        ins = t_view.get("inputs") if isinstance(t_view.get("inputs"), dict) else {}
        for it in (ins or {}).get("consumes") or []:
            if isinstance(it, dict) and isinstance(it.get("id"), str):
                artifacts_touched.append(it.get("id"))
    except Exception:
        pass

    slice_ids = sorted({str(x) for x in artifacts_touched if x})
    art_slice: Dict[str, Any] = {}
    for rid in slice_ids:
        ent = artifacts_index.get(rid)
        if isinstance(ent, dict):
            art_slice[rid] = {
                "producers": list(ent.get("producers") or []),
                "consumers": list(ent.get("consumers") or []),
            }

    return JoinJudgeCapsule(
        target=t_view,
        parent=p_view,
        siblings=sorted(sib_views, key=lambda x: str(x.get("id") or x.get("title") or "")),
        children=sorted(child_views, key=lambda x: str(x.get("id") or x.get("title") or "")),
        artifacts=art_slice,
        deterministic_issues=[str(x) for x in (deterministic_issues or []) if str(x)],
    )
