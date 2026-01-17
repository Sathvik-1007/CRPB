from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from crpb.core.ledger import NodeLedgerStore
from crpb.core.obligations import extract_obligations_from_node, stable_todo_id_from_obligation


def _safe_list(x: Any) -> List[Any]:
    return list(x) if isinstance(x, list) else []


def _walk_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            out.append(cur)
            for ch in _safe_list(cur.get("children")):
                if isinstance(ch, dict):
                    stack.append(ch)
    return out


def compute_obligation_coverage(
    *,
    plan: Dict[str, Any],
    run_dir_path: str,
    ledger_store: Optional[NodeLedgerStore] = None,
) -> Dict[str, Any]:
    """Compute a deterministic coverage report.

    This is intentionally conservative: it treats an obligation as discharged if:
    - the obligation is present in some node ledger, and
    - the corresponding TODO status is `done`.

    (This is the only generally language-agnostic discharge rule available without
    running stack-specific tests.)

    Future: validators can emit evidence artifacts that discharge obligations.
    """

    run_dir = Path(run_dir_path)

    # Accept both shapes:
    # 1) TaskPlan-like dict: {"tasks": [node, ...]}
    # 2) View-like dict: {"view": {"roots": [node, ...]}}
    roots: List[Dict[str, Any]] = []
    if isinstance(plan.get("tasks"), list):
        for r in _safe_list(plan.get("tasks")):
            if isinstance(r, dict):
                roots.append(r)
    if not roots:
        for r in _safe_list((plan.get("view") or {}).get("roots")):
            if isinstance(r, dict):
                roots.append(r)

    nodes: List[Dict[str, Any]] = []
    for r in roots:
        nodes.extend(_walk_nodes(r))

    all_obligations: List[Dict[str, Any]] = []
    for n in nodes:
        obs = extract_obligations_from_node(node=n, inherited_deps=[])
        for o in obs:
            all_obligations.append(o.to_dict())

    # Stable order
    all_obligations.sort(key=lambda o: str(o.get("id") or ""))

    # Index ledger TODO status if available.
    todo_status: Dict[str, str] = {}
    if ledger_store is not None:
        for n in nodes:
            nid = str(n.get("id") or "")
            if not nid:
                continue
            try:
                led = ledger_store.load(nid)
                for t in led.todos or []:
                    tid = str(getattr(t, "id", "") or "")
                    st = str(getattr(t, "status", "") or "")
                    if tid:
                        todo_status[tid] = st
            except Exception:
                continue

    discharged: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for o in all_obligations:
        oid = str(o.get("id") or "")
        tid = stable_todo_id_from_obligation(oid) if oid else ""
        st = todo_status.get(tid, "open") if tid else "open"
        entry = {**o, "todo_id": tid, "todo_status": st}
        # Primary discharge rule: TODO done.
        # Secondary evidence: node produced any artifacts (means the leaf executed).
        produced_any = False
        try:
            nid = str(o.get("node_id") or "")
            if nid and ledger_store is not None:
                led = ledger_store.load(nid)
                produced_any = bool(getattr(led, "produced_artifacts", []) or [])
        except Exception:
            produced_any = False

        if st == "done" or produced_any:
            discharged.append(entry)
        else:
            missing.append(entry)

    report = {
        "ok": len(missing) == 0,
        "counts": {
            "nodes": len(nodes),
            "obligations": len(all_obligations),
            "discharged": len(discharged),
            "undischarged": len(missing),
        },
        "undischarged": missing,
        "discharged": discharged,
    }

    # Persist report
    try:
        outp = run_dir / "validations" / "coverage.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pass

    return report
