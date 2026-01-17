from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _norm_ws(s: str) -> str:
    # Collapse all whitespace runs to a single space without regex.
    return " ".join(str(s or "").split()).strip()


def _strip_bullets(s: str) -> str:
    s = str(s or "").strip()
    s = s.lstrip("-*•\t ")
    return s.strip()


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()


def stable_obligation_id(
    *, node_id: str, scope: str, statement: str, dod: str, deps: Sequence[str]
) -> str:
    """Deterministic obligation id.

    This is designed to remain stable across runs for the *same semantic obligation*,
    while still being deterministic and local-only (no LLM calls).

    The ID is a hash of a stable tuple; caller should keep `statement` and `dod` canonical.
    """

    payload = {
        "node_id": str(node_id or ""),
        "scope": str(scope or ""),
        "statement": _norm_ws(statement),
        "dod": _norm_ws(dod),
        "deps": sorted([_norm_ws(d) for d in (deps or []) if _norm_ws(d)]),
    }
    return f"obl:{_sha1_hex(_stable_json(payload))}"


@dataclass(frozen=True)
class Obligation:
    id: str
    node_id: str
    scope: str  # "node" | "project"
    statement: str
    dod: str
    deps: List[str]
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "scope": self.scope,
            "statement": self.statement,
            "dod": self.dod,
            "deps": list(self.deps),
            "tags": list(self.tags),
        }


def _lines_from_node_plan(np: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    def add_block(kind: str, text: str) -> None:
        for raw in str(text or "").splitlines():
            s = _strip_bullets(raw)
            if not s:
                continue
            out.append({"kind": kind, "text": s})

    def add_list(kind: str, val: Any) -> None:
        if val is None:
            return
        if isinstance(val, (list, tuple)):
            for it in val:
                add_block(kind, str(it))
            return
        add_block(kind, str(val))

    add_block("test_plan", str(np.get("test_plan") or ""))
    add_block("acceptance_criteria", str(np.get("acceptance_criteria") or ""))

    # Broaden obligation surface (still deterministic + conservative):
    # These fields are frequently where integration/run/verification requirements hide.
    add_list("deliverables", np.get("deliverables"))
    add_block("completion_definition", str(np.get("completion_definition") or ""))
    add_block("verification", str(np.get("verification") or ""))
    return out


def extract_obligations_from_node(
    *,
    node: Dict[str, Any],
    inherited_deps: Optional[Sequence[str]] = None,
) -> List[Obligation]:
    """Deterministically extract obligations from a node.

    Source of truth is `node.node_plan.{acceptance_criteria,test_plan}`.

    This is intentionally conservative and LLM-free: it does not attempt semantic rewriting.
    """

    if not isinstance(node, dict):
        return []

    node_id = str(node.get("id") or "")
    np = node.get("node_plan")
    if not isinstance(np, dict):
        np = {}

    deps = sorted([_norm_ws(d) for d in (inherited_deps or []) if _norm_ws(d)])

    obligations: List[Obligation] = []
    for item in _lines_from_node_plan(np):
        kind = item.get("kind") or ""
        txt = _norm_ws(item.get("text") or "")
        if not txt:
            continue

        # Scope heuristic: acceptance criteria tends to be project-facing; test plan is node-local by default.
        scope = "project" if kind == "acceptance_criteria" else "node"

        statement = txt
        dod = f"Evidence exists in run artifacts that: {txt}"

        oid = stable_obligation_id(
            node_id=node_id,
            scope=scope,
            statement=statement,
            dod=dod,
            deps=deps,
        )

        obligations.append(
            Obligation(
                id=oid,
                node_id=node_id,
                scope=scope,
                statement=statement,
                dod=dod,
                deps=list(deps),
                tags=[kind],
            )
        )

    # Stable ordering
    obligations.sort(key=lambda o: o.id)
    return obligations


def stable_todo_id_from_obligation(obligation_id: str) -> str:
    # Keep TODO IDs separate from obligation IDs but stable and derived.
    oid = _norm_ws(obligation_id)
    return f"todo:{_sha1_hex(oid)}"


def format_structured_todo_text(*, obligation: Obligation) -> str:
    """Create a structured TODO body that retains context.

    This is intended to be stored in the ledger as a single string, but with
    strong internal structure for later compilation and search.
    """

    deps = ", ".join(obligation.deps) if obligation.deps else ""
    tags = ", ".join(obligation.tags) if obligation.tags else ""

    parts = [
        f"Trigger: derived from node_plan ({tags})" if tags else "Trigger: derived from node_plan",
        f"Work: {obligation.statement}",
        f"DoD: {obligation.dod}",
        f"Deps: {deps}" if deps else "Deps: (none)",
        f"Scope: {obligation.scope}",
        f"ObligationId: {obligation.id}",
    ]
    return "\n".join(parts).strip() + "\n"
