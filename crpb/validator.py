from __future__ import annotations
from typing import Tuple, Dict, Any, List, Set
from pathlib import Path
from .specs import TaskSpec, TaskPlan
from .utils.artifacts import ArtifactRegistry


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

# Local language inference to avoid coupling to commands/* modules
_LANG_BY_EXT = {
    "py": "python",
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "go": "go",
    "html": "html",
    "css": "css",
    "json": "json",
    "md": "markdown",
    "toml": "toml",
    "yaml": "yaml",
    "yml": "yaml",
}


def infer_language_from_path(path: str | None) -> str | None:
    if not path:
        return None
    ext = (Path(path).suffix or "").lower().lstrip(".")
    return _LANG_BY_EXT.get(ext)


def validate_leaf_readiness(task: TaskSpec) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate that a leaf `code:function` task has enough information to build.
    Enforces explicit or inferable language, non-empty path and name, and export discipline.
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
    lang = inputs.get("language") or infer_language_from_path(path)
    exports = list(inputs.get("exports", [name]))
    imports = list(inputs.get("allowed_imports", []))
    entry = inputs.get("entrypoint")
    sig = inputs.get("signature")

    if not isinstance(path, str) or not path.strip():
        return False, "missing_input:path", meta
    if not isinstance(name, str) or not name.strip():
        return False, "missing_input:name", meta
    if not (isinstance(lang, str) and lang.strip()):
        return False, "language_required_or_inferable", meta

    if isinstance(entry, str) and entry and entry not in exports:
        exports = exports + [entry]

    meta = {
        "path": path,
        "name": name,
        "language": lang,
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
                p2 = getattr(t, "meta", {}).get("path") if isinstance(getattr(t, "meta", {}), dict) else None  # type: ignore[call-arg]
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
            atomic_flag = False
            try:
                atomic_flag = bool((getattr(t, "meta", {}) or {}).get("atomic"))
            except Exception:
                atomic_flag = False
            if atomic_flag and children:
                n_issues.append("atomic_has_children")
            if (not atomic_flag) and (not children) and k == "composite":
                n_issues.append("composite_without_children")
            # Node plan presence (core fields)
            np = getattr(t, "node_plan", {}) or {}
            if not isinstance(np, dict):
                n_issues.append("node_plan_missing_or_invalid")
            else:
                for core in ("intent", "acceptance_criteria", "test_plan"):
                    if core not in np:
                        n_issues.append(f"node_plan_missing:{core}")
            # Artifact shape check
            def _chk_art(shape: Dict[str, Any] | None, key: str) -> None:
                if not isinstance(shape, dict):
                    return
                arr = shape.get(key)
                if arr is None:
                    return
                if not isinstance(arr, list):
                    n_issues.append(f"{key}_not_list")
                    return
                for it in arr:
                    if not (isinstance(it, dict) and isinstance(it.get("id"), str) and it.get("id").strip()):
                        n_issues.append(f"{key}_item_invalid")
                        break
            _chk_art(getattr(t, "inputs", {}) or {}, "consumes")
            _chk_art(getattr(t, "outputs", {}) or {}, "produces")

            if n_issues:
                node_reports.append({"id": tid, "path": path_use, "issues": n_issues})
            paths.append(path_use)
            # Recurse
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
    color: Dict[str, int] = {tid: WHITE for tid in adj.keys()}
    def _dfs(u: str, stack: List[str]) -> None:
        color[u] = GRAY
        stack.append(u)
        for v in adj.get(u, []):
            if color.get(v, WHITE) == WHITE:
                _dfs(v, stack)
            elif color.get(v) == GRAY:
                cycle = stack[stack.index(v):] + [v]
                issues.append("cycle:" + "->".join(cycle))
        color[u] = BLACK
        stack.pop()
    for tid in list(adj.keys()):
        if color[tid] == WHITE:
            _dfs(tid, [])

    # Artifact coverage index (producers/consumers) + coverage issues
    art_index: Dict[str, Dict[str, List[str]]] = {}
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
    def _collect(ts: List[TaskSpec], parent_path: str = "") -> None:
        for idx, t in enumerate(ts or [], start=1):
            # Path
            try:
                meta = getattr(t, "meta", {}) or {}
            except Exception:
                meta = {}
            path = meta.get("path") or (f"{parent_path}.{idx}" if parent_path else str(idx))
            outs = getattr(t, "outputs", {}) or {}
            ins = getattr(t, "inputs", {}) or {}
            prods = _norm(outs.get("produces")) if isinstance(outs, dict) else []
            cons = _norm(ins.get("consumes")) if isinstance(ins, dict) else []
            for ref in prods:
                rid = ref.get("id")
                if not rid:
                    continue
                ent = art_index.setdefault(rid, {"producers": [], "consumers": []})
                if path not in ent["producers"]:
                    ent["producers"].append(path)
            for ref in cons:
                rid = ref.get("id")
                if not rid:
                    continue
                ent = art_index.setdefault(rid, {"producers": [], "consumers": []})
                if path not in ent["consumers"]:
                    ent["consumers"].append(path)
            _collect(getattr(t, "children", []) or [], path)
    _collect(list(getattr(tp, "tasks", []) or []))

    # Coverage issues (general, neutral)
    for rid, ent in art_index.items():
        if not ent.get("producers"):
            issues.append(f"artifact_no_producer:{rid}")
        if not ent.get("consumers"):
            issues.append(f"artifact_no_consumer:{rid}")

    ok = not issues and not any(n.get("issues") for n in node_reports)
    return {"ok": ok, "issues": issues, "nodes": node_reports, "artifacts": {"index": art_index}}
