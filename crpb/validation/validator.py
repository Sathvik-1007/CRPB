from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.specs import TaskPlan, TaskSpec
from ..utils.artifacts import ArtifactRegistry

logger = logging.getLogger(__name__)


def is_deterministic_leaf(task: TaskSpec) -> bool:
    """Determine if a task is a deterministic leaf.

    A deterministic leaf is a code:function task that has no child tasks.
    This function provides a language‑agnostic guardrail for leaf detection.
    """
    if not isinstance(task, TaskSpec):
        return False
    if getattr(task, "kind", None) != "code:function":
        return False
    # Ensure no children are present
    if getattr(task, "children", []):
        return False
    return True


def basic_file_validation(fs: Any) -> Tuple[bool, str]:
    """Language-agnostic, minimal file validation.
    - Validates that exports, when present, is a list of non-empty strings.
    - If an entrypoint is specified, it must be present in exports (discipline for runnable modules).
    Does NOT assume that exports map only to functions (they may refer to classes/constants/unknown).
    """
    try:
        exports = getattr(fs, "exports", [])
        if exports is None:
            exports = []
        if not isinstance(exports, list):
            return False, "exports_invalid_shape: not a list"
        for exp in exports:
            if not isinstance(exp, str) or not exp.strip():
                return False, "exports_invalid_shape: items must be non-empty strings"

        entry = getattr(fs, "entrypoint", None)
        if isinstance(entry, str) and entry.strip():
            if entry not in exports:
                return False, f"entrypoint_not_exported:{entry}"
    except Exception as e:
        return False, f"basic_file_validation_error:{e}"
    return True, "ok"


def jsonschema_validate(data: dict, schema: dict) -> Tuple[bool, str]:
    """
    Validate data against a JSON Schema, if jsonschema is available.
    Returns (ok, message). If jsonschema is not installed, returns (True, 'skipped').
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        return True, "skipped: jsonschema not installed"
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, "ok"
    except Exception as e:
        return False, f"schema_error: {e}"


# --------------------------- Additional Validators / Gates -------------------------------

def _norm_token(s: str) -> str:
    """Normalize a token for cheap equality checks without regex."""
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def validate_leaf_readiness(task: TaskSpec) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate that a leaf `code:function` task has enough information to build.
    Enforces non-empty path and name, plus export discipline.
    Language is best-effort only (LLM may omit it and decide via context or file path).
    Returns (ok, message, meta) where meta includes effective fields for build.
    """
    meta: Dict[str, Any] = {}
    if task.kind != "code:function":
        return False, "not_a_code_function", meta
    if task.children:
        return False, "not_a_leaf", meta

    inputs = task.inputs or {}
    name = inputs.get("name") or ""
    path = inputs.get("path") or ""
    lang = inputs.get("language")
    exports = list(inputs.get("exports", [name]))
    imports = list(inputs.get("allowed_imports", []))
    entry = inputs.get("entrypoint")
    sig = inputs.get("signature")

    if not isinstance(path, str) or not path.strip():
        return False, "missing_input:path", meta
    if not isinstance(name, str) or not name.strip():
        return False, "missing_input:name", meta

    if isinstance(entry, str) and entry and entry not in exports:
        exports = exports + [entry]

    meta = {
        "path": path,
        "name": name,
        "language": lang if isinstance(lang, str) and lang.strip() else None,
        "exports": exports,
        "imports": imports,
        "entrypoint": entry if isinstance(entry, str) else None,
        "signature": sig if isinstance(sig, str) else None,
    }
    return True, "ok", meta


def validate_artifact_gating(
    *,
    task: TaskSpec,
    registry: ArtifactRegistry,
    base_dir: Path,
    require_valid: bool = True,
) -> Tuple[bool, str]:
    """Gate a task on consumed artifacts: they must exist (and be valid when required)."""
    try:
        raw = []
        if isinstance(task.inputs, dict):
            c = task.inputs.get("consumes")
            if isinstance(c, list):
                raw = c
            elif c is None:
                raw = []
            else:
                raw = [c]
        # Normalize to dict refs
        refs: List[Dict[str, Any]] = []
        for r in raw:
            if isinstance(r, str):
                refs.append({"id": r})
            elif isinstance(r, dict):
                refs.append(r)
        # Check gating
        for ref in refs:
            if not registry.exists(ref, base_dir=base_dir):
                return False, "artifact_missing"
            if require_valid:
                v = registry.get_validation(ref)
                if not v or not v.get("ok", False):
                    return False, "artifact_not_validated"
        return True, "ok"
    except Exception as e:
        return False, f"artifact_gate_error:{e}"


def validate_taskplan_structure(tp: TaskPlan) -> Tuple[bool, List[str]]:
    """Lightweight structure check: unique ids and kind sanity. No LLM calls."""
    issues: List[str] = []
    seen: set[str] = set()

    def walk(ts: List[TaskSpec]) -> None:
        for t in ts:
            tid = str(getattr(t, "id", ""))
            if not tid:
                issues.append("task_missing_id")
            elif tid in seen:
                issues.append(f"duplicate_id:{tid}")
            else:
                seen.add(tid)
            k = getattr(t, "kind", "")
            if k not in ("composite", "code:function"):
                issues.append(f"unknown_kind:{k}")
            walk(getattr(t, "children", []))

    walk(list(getattr(tp, "tasks", [])))
    return (len(issues) == 0), issues


def validate_taskplan_general(tp: TaskPlan) -> Dict[str, Any]:
    """
    General, logical validation of a TaskPlan (language-agnostic, world-class guardrails):
    - Node structure: non-empty title/kind, atomic vs. children coherence.
    - Node plan presence: expect a dict with core fields (intent, acceptance_criteria, test_plan).
    - Deps: uniqueness of ids, valid references, acyclic DAG.
    - Artifact shape: inputs.consumes / outputs.produces arrays of dicts with at least 'id'.

    Returns a dict report: {"ok": bool, "issues": string[], "nodes": [{"id":str,"path":str,"issues":string[]}]}
    Path values prefer task.meta.path when present, otherwise computed 1-based dotted index.
    """
    logger.info("[validator] Starting validate_taskplan_general")
    issues: List[str] = []
    node_reports: List[Dict[str, Any]] = []

    # Flatten with paths and id index
    id_index: Dict[str, TaskSpec] = {}

    def _walk(ts: List[TaskSpec], parent_path: str = "") -> List[str]:
        paths: List[str] = []
        for idx, t in enumerate(ts or [], start=1):
            path = f"{parent_path}.{idx}" if parent_path else str(idx)
            # Prefer precomputed meta.path if available
            try:
                p2 = (
                    getattr(t, "meta", {}).get("path")
                    if isinstance(getattr(t, "meta", {}), dict)
                    else None
                )  # type: ignore[call-arg]
            except Exception:
                p2 = None
            path_use = p2 or path
            tid = str(getattr(t, "id", "") or "")
            if tid:
                if tid in id_index:
                    issues.append(f"duplicate_id:{tid}")
                else:
                    id_index[tid] = t
            # Node-level checks (general)
            n_issues: List[str] = []
            k = getattr(t, "kind", "")
            if not isinstance(k, str) or not k.strip():
                n_issues.append("missing_kind")
            title = getattr(t, "title", "")
            if not isinstance(title, str) or not title.strip():
                n_issues.append("missing_title")
            # Atomic vs children coherence
            children = list(getattr(t, "children", []) or [])
            is_leaf = not bool(children)
            atomic_flag = False
            try:
                atomic_flag = bool((getattr(t, "meta", {}) or {}).get("atomic"))
            except Exception:
                atomic_flag = False
            if atomic_flag and children:
                n_issues.append("atomic_has_children")
            if (not atomic_flag) and is_leaf and k == "composite":
                n_issues.append("composite_without_children")

            # Kind/leaf coherence: code:function nodes must be leaves
            if k == "code:function" and children:
                n_issues.append("code_function_has_children")

            # Leaf readiness: enforce that code:function leaves are implementable (language-agnostic)
            if k == "code:function" and is_leaf:
                ok_lr, msg_lr, _meta_lr = validate_leaf_readiness(t)
                if not ok_lr:
                    n_issues.append(f"leaf_not_ready:{msg_lr}")
            # Ensure atomic_has_children is reported first per test expectations
            if "atomic_has_children" in n_issues:
                n_issues.remove("atomic_has_children")
                n_issues.insert(0, "atomic_has_children")
            # Node plan presence (core fields)
            np = getattr(t, "node_plan", {}) or {}
            if not isinstance(np, dict):
                n_issues.append("node_plan_missing_or_invalid")
            else:
                for core in ("intent", "acceptance_criteria", "test_plan"):
                    if core not in np:
                        n_issues.append(f"node_plan_missing:{core}")

            # Artifact shape check
            def _chk_art(
                shape: Dict[str, Any] | None,
                key: str,
                *,
                n_issues: List[str] = n_issues,
            ) -> None:
                if not isinstance(shape, dict):
                    return
                arr = shape.get(key)
                if arr is None:
                    return
                if not isinstance(arr, list):
                    n_issues.append(f"{key}_not_list")
                    return
                for it in arr:
                    if not (
                        isinstance(it, dict)
                        and isinstance(it.get("id"), str)
                        and it.get("id").strip()
                    ):
                        n_issues.append(f"{key}_item_invalid")
                        break

            _chk_art(getattr(t, "inputs", {}) or {}, "consumes")
            _chk_art(getattr(t, "outputs", {}) or {}, "produces")

            if n_issues:
                node_reports.append({"id": tid, "path": path_use, "issues": n_issues})
            paths.append(path_use)
            # Recurse
            if children:
                # Deterministic sibling duplicate-title detection (cheap redundancy signal)
                try:
                    norm_map: Dict[str, List[str]] = {}
                    for ch in children:
                        ctitle = getattr(ch, "title", "")
                        cid = str(getattr(ch, "id", "") or "")
                        if isinstance(ctitle, str) and ctitle.strip():
                            key = _norm_token(ctitle)
                            if key:
                                norm_map.setdefault(key, []).append(cid or "<no-id>")
                    for key, ids in norm_map.items():
                        if len(ids) >= 2:
                            issues.append("duplicate_sibling_titles")
                            # Add node-level marker to each sibling (first few only)
                            for ch in children:
                                ctitle = getattr(ch, "title", "")
                                if _norm_token(ctitle) != key:
                                    continue
                                chid = str(getattr(ch, "id", "") or "")
                                node_reports.append(
                                    {
                                        "id": chid,
                                        "path": path_use,
                                        "issues": [f"possible_duplicate_sibling_title:{key}"],
                                    }
                                )
                except Exception:
                    pass
                _walk(children, path_use)
        return paths

    _walk(list(getattr(tp, "tasks", []) or []))

    # Deps validity and cycle detection
    # Build adjacency
    adj: Dict[str, List[str]] = {}
    for tid, t in id_index.items():
        try:
            adj[tid] = [d for d in (getattr(t, "deps", []) or []) if isinstance(d, str) and d]
        except Exception:
            adj[tid] = []
    # Unknown deps
    for tid, deps in adj.items():
        for d in deps:
            if d not in id_index:
                issues.append(f"unknown_dep:{tid}->{d}")

    # Cycle detection via DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = dict.fromkeys(adj.keys(), WHITE)

    def _dfs(u: str, stack: List[str]) -> None:
        color[u] = GRAY
        stack.append(u)
        for v in adj.get(u, []):
            if color.get(v, WHITE) == WHITE:
                _dfs(v, stack)
            elif color.get(v) == GRAY:
                cycle = stack[stack.index(v) :] + [v]
                issues.append("cycle:" + "->".join(cycle))
        color[u] = BLACK
        stack.pop()

    for tid in list(adj.keys()):
        if color[tid] == WHITE:
            _dfs(tid, [])

    # Artifact coverage index (producers/consumers) + coverage issues
    art_index: Dict[str, Dict[str, Any]] = {}

    def _norm(val: Any) -> List[Dict[str, Any]]:
        if isinstance(val, list):
            arr = val
        elif val is None:
            arr = []
        else:
            arr = [val]
        out: List[Dict[str, Any]] = []
        for r in arr:
            if isinstance(r, str):
                rid = r.strip()
                if rid:
                    out.append({"id": rid})
            elif isinstance(r, dict):
                rid = r.get("id")
                if isinstance(rid, str) and rid.strip():
                    out.append(r)
        return out

    task_consumes: Dict[str, List[str]] = {}
    task_produces: Dict[str, List[str]] = {}

    def _collect(ts: List[TaskSpec], parent_path: str = "") -> None:
        for idx, t in enumerate(ts or [], start=1):
            # Path
            try:
                meta = getattr(t, "meta", {}) or {}
            except Exception:
                meta = {}
            path = meta.get("path") or (f"{parent_path}.{idx}" if parent_path else str(idx))
            tid = str(getattr(t, "id", "") or "")
            outs = getattr(t, "outputs", {}) or {}
            ins = getattr(t, "inputs", {}) or {}
            prods = _norm(outs.get("produces")) if isinstance(outs, dict) else []
            cons = _norm(ins.get("consumes")) if isinstance(ins, dict) else []
            for ref in prods:
                rid = ref.get("id")
                if not rid:
                    continue
                ent = art_index.setdefault(
                    rid, {"producers": [], "consumers": [], "producer_ids": [], "consumer_ids": []}
                )
                if path not in ent["producers"]:
                    ent["producers"].append(path)
                if tid and tid not in ent["producer_ids"]:
                    ent["producer_ids"].append(tid)
                if tid:
                    task_produces.setdefault(tid, []).append(rid)
            for ref in cons:
                rid = ref.get("id")
                if not rid:
                    continue
                ent = art_index.setdefault(
                    rid, {"producers": [], "consumers": [], "producer_ids": [], "consumer_ids": []}
                )
                if path not in ent["consumers"]:
                    ent["consumers"].append(path)
                if tid and tid not in ent["consumer_ids"]:
                    ent["consumer_ids"].append(tid)
                if tid:
                    task_consumes.setdefault(tid, []).append(rid)
            _collect(getattr(t, "children", []) or [], path)

    _collect(list(getattr(tp, "tasks", []) or []))

    # Coverage issues (general, neutral)
    for rid, ent in art_index.items():
        if not ent.get("producers"):
            issues.append(f"artifact_no_producer:{rid}")
        if not ent.get("consumers"):
            issues.append(f"artifact_no_consumer:{rid}")

    # Communication wiring: if a task consumes an artifact, it should depend on at least one producer task.
    # This forces explicit ordering/contract awareness and prevents implicit sibling coupling.
    try:
        deps_by_id: Dict[str, List[str]] = {}
        for tid, t in id_index.items():
            deps_by_id[tid] = [
                d for d in (getattr(t, "deps", []) or []) if isinstance(d, str) and d
            ]

        for consumer_id, rids in task_consumes.items():
            deps = deps_by_id.get(consumer_id, [])
            for rid in rids or []:
                ent = art_index.get(rid) if isinstance(art_index, dict) else None
                if not isinstance(ent, dict):
                    continue
                producer_ids = [
                    str(x) for x in (ent.get("producer_ids") or []) if isinstance(x, str) and x
                ]
                if not producer_ids:
                    continue
                if not any(p in deps for p in producer_ids):
                    issues.append(f"artifact_consumer_missing_dep:{consumer_id}:{rid}")
                    try:
                        # Prefer meta.path if present
                        pth = ""
                        t = id_index.get(consumer_id)
                        if t is not None:
                            meta = getattr(t, "meta", {}) or {}
                            if isinstance(meta, dict):
                                pth = str(meta.get("path") or "")
                        node_reports.append(
                            {
                                "id": consumer_id,
                                "path": pth,
                                "issues": [f"artifact_consumer_missing_dep:{rid}"],
                            }
                        )
                    except Exception:
                        pass
    except Exception:
        pass

    # Coverage enforcement (artifact-delegation axiom): composites orchestrate; children own concrete I/O.
    # If a composite parent directly produces/consumes an artifact, at least one child subtree must also
    # produce/consume it (or an explicit integration child must exist). This prevents "parents implementing".
    # Also detect sibling multi-producer collisions for the same artifact (redundancy / broken interfaces).
    try:
        subtree_produces: Dict[str, Set[str]] = {}
        subtree_consumes: Dict[str, Set[str]] = {}

        def _direct_artifacts(t: TaskSpec) -> Tuple[Set[str], Set[str]]:
            tid = str(getattr(t, "id", "") or "")
            prods = {str(x) for x in (task_produces.get(tid, []) or []) if isinstance(x, str) and x}
            cons = {str(x) for x in (task_consumes.get(tid, []) or []) if isinstance(x, str) and x}
            return prods, cons

        def _subtree(t: TaskSpec) -> Tuple[Set[str], Set[str]]:
            tid = str(getattr(t, "id", "") or "")
            if tid and tid in subtree_produces and tid in subtree_consumes:
                return subtree_produces[tid], subtree_consumes[tid]
            prods, cons = _direct_artifacts(t)
            for c in list(getattr(t, "children", []) or []):
                p2, c2 = _subtree(c)
                prods |= p2
                cons |= c2
            if tid:
                subtree_produces[tid] = set(prods)
                subtree_consumes[tid] = set(cons)
            return prods, cons

        for rt in list(getattr(tp, "tasks", []) or []):
            _subtree(rt)

        def _path_for(tid: str) -> str:
            t = id_index.get(tid)
            if t is None:
                return ""
            try:
                meta = getattr(t, "meta", {}) or {}
            except Exception:
                meta = {}
            if isinstance(meta, dict) and isinstance(meta.get("path"), str):
                return str(meta.get("path") or "")
            return ""

        for parent_id, parent in id_index.items():
            if getattr(parent, "kind", "") != "composite":
                continue
            children = list(getattr(parent, "children", []) or [])
            if not children:
                continue

            # Parent delegation check
            p_prods, p_cons = _direct_artifacts(parent)
            if p_prods or p_cons:
                child_prod_union: Set[str] = set()
                child_cons_union: Set[str] = set()
                for ch in children:
                    cid = str(getattr(ch, "id", "") or "")
                    if not cid:
                        continue
                    child_prod_union |= subtree_produces.get(cid, set())
                    child_cons_union |= subtree_consumes.get(cid, set())

                for rid in sorted(p_prods):
                    if rid not in child_prod_union:
                        issues.append(f"parent_artifact_not_delegated:{parent_id}:produces:{rid}")
                        node_reports.append(
                            {
                                "id": parent_id,
                                "path": _path_for(parent_id),
                                "issues": [f"parent_artifact_not_delegated:produces:{rid}"],
                            }
                        )
                for rid in sorted(p_cons):
                    if rid not in child_cons_union:
                        issues.append(f"parent_artifact_not_delegated:{parent_id}:consumes:{rid}")
                        node_reports.append(
                            {
                                "id": parent_id,
                                "path": _path_for(parent_id),
                                "issues": [f"parent_artifact_not_delegated:consumes:{rid}"],
                            }
                        )

            # Sibling multi-producer collision check (immediate children)
            try:
                by_art: Dict[str, List[str]] = {}
                for ch in children:
                    cid = str(getattr(ch, "id", "") or "")
                    if not cid:
                        continue
                    for rid in task_produces.get(cid, []) or []:
                        if isinstance(rid, str) and rid:
                            by_art.setdefault(rid, []).append(cid)
                for rid, cids in by_art.items():
                    if len(cids) >= 2:
                        issues.append(
                            f"sibling_artifact_multi_producer:{parent_id}:{rid}:{','.join(sorted(cids))}"
                        )
                        for cid in sorted(cids):
                            node_reports.append(
                                {
                                    "id": cid,
                                    "path": _path_for(cid),
                                    "issues": [f"sibling_artifact_multi_producer:{rid}"],
                                }
                            )
            except Exception:
                pass
    except Exception:
        pass

    ok = not issues and not any(n.get("issues") for n in node_reports)
    return {"ok": ok, "issues": issues, "nodes": node_reports, "artifacts": {"index": art_index}}


def validate_taskplan_tree_spider(
    *,
    tp: TaskPlan,
    run_dir_path: str | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    enabled = str((tp.constraints or {}).get("tree_validation_enable") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not enabled:
        try:
            enabled = str(
                __import__("os").environ.get("CRPB_TREE_VALIDATION_ENABLE", "0")
            ).strip().lower() in ("1", "true", "yes")
        except Exception:
            enabled = False
    if not enabled:
        return {"ok": True, "skipped": True, "reason": "tree_validation_disabled"}

    try:
        import os

        from crpb.core.config import make_paths
        from crpb.utils.fs import atomic_write_json
    except Exception:
        os = None  # type: ignore
        make_paths = None  # type: ignore
        atomic_write_json = None  # type: ignore

    try:
        max_nodes = int(os.environ.get("CRPB_TREE_VALIDATION_MAX_NODES", "30") if os else "30")
    except Exception:
        max_nodes = 30
    try:
        max_children = int(
            os.environ.get("CRPB_TREE_VALIDATION_MAX_CHILDREN", "12") if os else "12"
        )
    except Exception:
        max_children = 12
    try:
        max_siblings = int(os.environ.get("CRPB_TREE_VALIDATION_MAX_SIBLINGS", "8") if os else "8")
    except Exception:
        max_siblings = 8
    try:
        sim_top_k = int(os.environ.get("CRPB_TREE_VALIDATION_SIM_TOP_K", "6") if os else "6")
    except Exception:
        sim_top_k = 6
    try:
        sim_min = float(
            os.environ.get("CRPB_TREE_VALIDATION_SIM_MIN_SCORE", "0.82") if os else "0.82"
        )
    except Exception:
        sim_min = 0.82

    gen_report = validate_taskplan_general(tp)
    artifacts_index = (
        (gen_report.get("artifacts") or {}).get("index")
        if isinstance(gen_report.get("artifacts"), dict)
        else {}
    )
    if not isinstance(artifacts_index, dict):
        artifacts_index = {}

    id_index: Dict[str, TaskSpec] = {}
    parent_by_id: Dict[str, str] = {}
    path_by_id: Dict[str, str] = {}
    children_by_id: Dict[str, List[str]] = {}
    order: List[str] = []

    def _walk(ts: List[TaskSpec], parent_id: str | None = None, parent_path: str = "") -> None:
        for idx, t in enumerate(ts or [], start=1):
            try:
                meta = getattr(t, "meta", {}) or {}
            except Exception:
                meta = {}
            path = (meta.get("path") if isinstance(meta, dict) else None) or (
                f"{parent_path}.{idx}" if parent_path else str(idx)
            )
            tid = str(getattr(t, "id", "") or "")
            if not tid:
                continue
            id_index[tid] = t
            path_by_id[tid] = str(path)
            if parent_id:
                parent_by_id[tid] = parent_id
                children_by_id.setdefault(parent_id, []).append(tid)
            order.append(tid)
            ch = list(getattr(t, "children", []) or [])
            if ch:
                _walk(ch, tid, str(path))

    _walk(list(getattr(tp, "tasks", []) or []))

    def _node_view(t: TaskSpec) -> Dict[str, Any]:
        try:
            meta = getattr(t, "meta", {}) or {}
        except Exception:
            meta = {}
        outs = getattr(t, "outputs", {}) or {}
        ins = getattr(t, "inputs", {}) or {}
        produces = []
        consumes = []
        try:
            if isinstance(outs, dict) and isinstance(outs.get("produces"), list):
                produces = [
                    str(x.get("id"))
                    for x in outs.get("produces")
                    if isinstance(x, dict) and isinstance(x.get("id"), str)
                ]
        except Exception:
            produces = []
        try:
            if isinstance(ins, dict) and isinstance(ins.get("consumes"), list):
                consumes = [
                    str(x.get("id"))
                    for x in ins.get("consumes")
                    if isinstance(x, dict) and isinstance(x.get("id"), str)
                ]
        except Exception:
            consumes = []
        return {
            "id": str(getattr(t, "id", "") or ""),
            "path": str((meta.get("path") if isinstance(meta, dict) else None) or ""),
            "kind": str(getattr(t, "kind", "") or ""),
            "title": str(getattr(t, "title", "") or ""),
            "description": str(getattr(t, "description", "") or ""),
            "deps": list(getattr(t, "deps", []) or []),
            "produces": produces,
            "consumes": consumes,
        }

    def _slice_artifacts(ids: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for rid in ids:
            ent = artifacts_index.get(rid) if isinstance(artifacts_index, dict) else None
            if isinstance(ent, dict):
                out[rid] = {
                    "producers": list(ent.get("producers") or []),
                    "consumers": list(ent.get("consumers") or []),
                }
        return out

    def _prune_task_dict(obj: Dict[str, Any], depth: int) -> Dict[str, Any]:
        out = dict(obj or {})
        if depth <= 0:
            out.pop("children", None)
            return out
        ch = out.get("children")
        if isinstance(ch, list):
            out["children"] = [_prune_task_dict(c, depth - 1) for c in ch if isinstance(c, dict)]
        return out

    def _stub_task(task_id: str) -> Dict[str, Any]:
        return {
            "id": task_id,
            "kind": "composite",
            "title": f"dep:{task_id}",
            "description": "",
            "deps": [],
            "inputs": {},
            "outputs": {},
            "node_plan": {
                "intent": "",
                "in_scope": "",
                "out_of_scope": "",
                "constraints": "",
                "assumptions": "",
                "deliverables": "",
                "acceptance_criteria": "",
                "preconditions": "",
                "postconditions": "",
                "interfaces": "",
                "data_contracts": "",
                "integration_points": "",
                "dependencies": "",
                "sequencing": "",
                "risks": "",
                "mitigations": "",
                "open_questions": "",
                "test_plan": "",
                "verification": "",
                "non_functional": "",
                "step_outline": "",
                "completion_definition": "",
                "success_metrics": "",
                "ownership": "",
                "handoffs": "",
            },
            "meta": {"spider_stub": True},
        }

    # Load embeddings index (optional; no new embedding calls)
    embed_index = None
    embed_items_by_id: Dict[str, Any] = {}
    embed_vec_by_id: Dict[str, List[float]] = {}
    provider_pref: List[str] = []
    embed_feature_on = False
    try:
        from crpb.core.llm_config import embeddings_enabled as _embeddings_enabled

        embed_feature_on = bool(_embeddings_enabled())
    except Exception:
        embed_feature_on = False

    if embed_feature_on:
        try:
            pp = os.environ.get("CRPB_TASK_EMBED_PROVIDER_PREFERENCE")
            if isinstance(pp, str) and pp.strip():
                provider_pref = [p.strip() for p in pp.split(",") if p.strip()]
        except Exception:
            provider_pref = []

        # Opt-in guard: embeddings are only used when a provider preference is explicit.
        if not provider_pref:
            embed_feature_on = False
        if embed_feature_on:
            try:
                from crpb.utils.embeddings import TaskEmbeddingIndex

                if run_dir_path:
                    idx_path = Path(run_dir_path) / "registry" / "task_embeddings.json"
                    if idx_path.exists():
                        embed_index = TaskEmbeddingIndex(path=idx_path)
                        embed_index.load()
                        for it in getattr(embed_index, "_items", []) or []:
                            try:
                                embed_items_by_id[str(it.id)] = it
                                v = embed_index.get_vector(it, provider_pref)
                                if isinstance(v, list) and v:
                                    embed_vec_by_id[str(it.id)] = v
                            except Exception:
                                continue
            except Exception:
                embed_index = None

    def _avg_vec(vs: List[List[float]]) -> Optional[List[float]]:
        if not vs:
            return None
        base = vs[0]
        if not base:
            return None
        dim = len(base)
        acc = [0.0] * dim
        cnt = 0
        for v in vs:
            if not isinstance(v, list) or len(v) != dim:
                continue
            for i in range(dim):
                acc[i] += float(v[i])
            cnt += 1
        if cnt <= 0:
            return None
        return [float(x) / float(cnt) for x in acc]

    def _node_vector(tid: str) -> Optional[List[float]]:
        v = embed_vec_by_id.get(tid)
        if isinstance(v, list) and v:
            return v
        ch_ids = children_by_id.get(tid, []) or []
        child_vs: List[List[float]] = []
        for cid in ch_ids:
            cv = _node_vector(cid)
            if isinstance(cv, list) and cv:
                child_vs.append(cv)
        return _avg_vec(child_vs)

    def _similar_for(tid: str) -> List[Dict[str, Any]]:
        if embed_index is None:
            return []
        v = _node_vector(tid)
        if not isinstance(v, list) or not v:
            return []
        try:
            scored = embed_index.query_similar(
                vector=v,
                top_k=max(0, int(sim_top_k) + 1),
                min_score=float(sim_min),
                provider_preference=provider_pref,
            )
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for it, s in scored:
            if str(getattr(it, "id", "")) == tid:
                continue
            out.append(
                {
                    "id": str(getattr(it, "id", "")),
                    "path": str(getattr(it, "path", "")),
                    "title": str(getattr(it, "title", "")),
                    "score": float(s),
                }
            )
            if len(out) >= int(sim_top_k):
                break
        return out

    # Determine node validation order (composites first, then leaves)
    composites: List[str] = []
    leaves: List[str] = []
    for tid in order:
        t = id_index.get(tid)
        if not t:
            continue
        ch = list(getattr(t, "children", []) or [])
        if ch:
            composites.append(tid)
        else:
            leaves.append(tid)
    validate_ids = (composites + leaves)[: max(0, int(max_nodes))]

    logger.info(
        "[tree_spider] Starting node validation: total_nodes=%d validate=%d max_nodes=%d",
        len(order),
        len(validate_ids),
        int(max_nodes),
    )

    nodes_out: List[Dict[str, Any]] = []
    out_dir: Optional[Path] = None
    if run_dir_path and make_paths is not None:
        try:
            rp = make_paths(Path(run_dir_path))
            out_dir = rp.validations / "tree"
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            out_dir = None

    # LLM validation per node (best-effort)
    engine = None
    try:
        from crpb.agents.dspy_engine import DspyEngine

        engine = DspyEngine()
    except Exception:
        engine = None

    for tid in validate_ids:
        t = id_index.get(tid)
        if not t:
            continue
        path = path_by_id.get(tid, "")
        logger.debug("[tree_spider] Validating node id=%s path=%s", tid, path)
        pid = parent_by_id.get(tid)
        parent_view = _node_view(id_index[pid]) if pid and pid in id_index else {}
        sib_views: List[Dict[str, Any]] = []
        if pid and pid in children_by_id:
            for sid in children_by_id.get(pid, []) or []:
                if sid == tid:
                    continue
                st = id_index.get(sid)
                if st:
                    sib_views.append(_node_view(st))
                if len(sib_views) >= int(max_siblings):
                    break

        child_views: List[Dict[str, Any]] = []
        for cid in children_by_id.get(tid, []) or []:
            ct = id_index.get(cid)
            if ct:
                child_views.append(_node_view(ct))
            if len(child_views) >= int(max_children):
                break

        artifacts_touched: List[str] = []
        try:
            outs = getattr(t, "outputs", {}) or {}
            if isinstance(outs, dict) and isinstance(outs.get("produces"), list):
                for it2 in outs.get("produces") or []:
                    if isinstance(it2, dict) and isinstance(it2.get("id"), str):
                        artifacts_touched.append(it2.get("id"))
        except Exception:
            pass
        try:
            ins = getattr(t, "inputs", {}) or {}
            if isinstance(ins, dict) and isinstance(ins.get("consumes"), list):
                for it2 in ins.get("consumes") or []:
                    if isinstance(it2, dict) and isinstance(it2.get("id"), str):
                        artifacts_touched.append(it2.get("id"))
        except Exception:
            pass
        artifacts_slice = _slice_artifacts(sorted({str(x) for x in artifacts_touched if x}))

        similar = _similar_for(tid)

        det_issues: List[str] = []
        # Embedding-based sibling redundancy signal (deterministic)
        try:
            if embed_index is not None and child_views and len(child_views) >= 2:
                child_ids = [cv.get("id") for cv in child_views if isinstance(cv.get("id"), str)]
                child_vecs = [(cid, _node_vector(cid)) for cid in child_ids]
                for i in range(len(child_vecs)):
                    for j in range(i + 1, len(child_vecs)):
                        a_id, a_v = child_vecs[i]
                        b_id, b_v = child_vecs[j]
                        if not (isinstance(a_v, list) and isinstance(b_v, list)):
                            continue
                        # reuse cosine from embeddings module if available
                        from crpb.utils.embeddings import _cosine_similarity as _cs  # type: ignore

                        sc = float(_cs(a_v, b_v))
                        if sc >= float(sim_min):
                            det_issues.append(f"sibling_similarity_high:{a_id}~{b_id}:{sc:.3f}")
        except Exception:
            pass

        capsule = {
            "target": {
                **_prune_task_dict(t.model_dump(exclude_none=True), 1),
                "meta": {
                    **(
                        (getattr(t, "meta", {}) or {})
                        if isinstance(getattr(t, "meta", {}), dict)
                        else {}
                    ),
                    "path": path,
                },
            },
            "parent": parent_view,
            "siblings": sib_views,
            "children": child_views,
            "artifacts": artifacts_slice,
            "similar": similar,
            "deterministic_issues": det_issues,
        }

        llm_result: Dict[str, Any] = {"ok": True, "issues": [], "suggestions": [], "skipped": True}
        if engine is not None:
            try:
                base_plan = {"tasks": [_prune_task_dict(t.model_dump(exclude_none=True), 1)]}
                # Add stubs for external deps referenced by this node + its immediate children
                included_ids: set[str] = set()
                try:

                    def _collect_ids(
                        n: Dict[str, Any], *, included_ids: set[str] = included_ids
                    ) -> None:
                        nid = n.get("id")
                        if isinstance(nid, str) and nid:
                            included_ids.add(nid)
                        for c in n.get("children") or []:
                            if isinstance(c, dict):
                                _collect_ids(c)

                    for rt in base_plan.get("tasks", []):
                        if isinstance(rt, dict):
                            _collect_ids(rt)
                except Exception:
                    included_ids = set()

                dep_ids: set[str] = set()
                try:

                    def _collect_deps(n: Dict[str, Any], *, dep_ids: set[str] = dep_ids) -> None:
                        for d in n.get("deps") or []:
                            if isinstance(d, str) and d:
                                dep_ids.add(d)
                        for c in n.get("children") or []:
                            if isinstance(c, dict):
                                _collect_deps(c)

                    for rt in base_plan.get("tasks", []):
                        if isinstance(rt, dict):
                            _collect_deps(rt)
                except Exception:
                    dep_ids = set()

                stubs: List[Dict[str, Any]] = []
                for did in sorted(dep_ids):
                    if did in included_ids:
                        continue
                    stubs.append(_stub_task(did))
                base_plan["tasks"] = list(base_plan.get("tasks") or []) + stubs

                c2 = dict(tp.constraints or {})
                c2.setdefault("side_context", {})
                c2["side_context"].update({"tree_spider": capsule})
                add_blocks = c2.get("rule_blocks_add")
                if not isinstance(add_blocks, list):
                    add_blocks = []
                if "tree_spider_validate" not in [str(x) for x in add_blocks]:
                    add_blocks.append("tree_spider_validate")
                c2["rule_blocks_add"] = add_blocks
                llm_result = engine.validate_task_plan(
                    idea=str(getattr(tp, "idea", "") or ""),
                    constraints=c2,
                    plan=base_plan,
                )
                llm_result["skipped"] = False
            except Exception as e:
                llm_result = {
                    "ok": True,
                    "issues": [f"tree_spider_llm_error:{e}"],
                    "suggestions": [],
                    "skipped": True,
                }

        node_out = {
            "id": tid,
            "path": path,
            "ok": bool(llm_result.get("ok", True)) and not bool(det_issues),
            "issues": [str(x) for x in (llm_result.get("issues") or [])] + det_issues,
            "suggestions": [str(x) for x in (llm_result.get("suggestions") or [])],
            "similar": similar,
            "skipped": bool(llm_result.get("skipped", False)),
        }
        nodes_out.append(node_out)

        if out_dir is not None and atomic_write_json is not None:
            try:
                raw = f"{path}__{tid}"
                safe = "".join(ch if (ch.isalnum() or ch in "_-" ) else "_" for ch in raw)
                atomic_write_json(
                    out_dir / f"node_{safe}.json", {"node": node_out, "capsule": capsule}
                )
            except Exception:
                pass

    all_issues: List[str] = []
    for n in nodes_out:
        for it in n.get("issues") or []:
            s = str(it)
            if s and s not in all_issues:
                all_issues.append(s)

    summary = {
        "ok": len(all_issues) == 0,
        "run_id": str(run_id or ""),
        "skipped": False,
        "general": gen_report,
        "issues": all_issues,
        "nodes": nodes_out,
    }
    if out_dir is not None and atomic_write_json is not None:
        try:
            atomic_write_json(out_dir / "index.json", summary)
            atomic_write_json(out_dir.parent / "tree_validation.json", summary)
        except Exception:
            pass
    return summary
