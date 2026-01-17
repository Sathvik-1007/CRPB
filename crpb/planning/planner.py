from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.specs import TaskPlan, TaskSpec
from ..validation.validator import (
    is_deterministic_leaf,
    validate_leaf_readiness,
    validate_taskplan_general,
)
from .join_judge import build_join_judge_capsule
from .parallel import stable_parallel_map

logger = logging.getLogger(__name__)


def generate_task_plan(
    idea: str,
    constraints: dict,
    use_llm: bool = True,
    enforce_deterministic_leaves: bool = True,
    run_dir_path: str | None = None,
) -> TaskPlan:
    """
    Produce a generic hierarchical TaskPlan using the LLM. No code, only structure and metadata.
    LLM is required; if unavailable or fails, this function raises an error.
    """
    if use_llm:
        try:
            from ..agents.dspy_engine import DspyEngine

            engine = DspyEngine()

            # Optional run-scoped ledger + context packing for planning.
            ledger_store = None
            context_compiler = None
            if run_dir_path:
                try:
                    from ..core.config import make_paths
                    from ..core.context_compiler import ContextCompiler
                    from ..core.ledger import NodeLedgerStore

                    paths = make_paths(Path(str(run_dir_path)))
                    ledger_store = NodeLedgerStore(base_dir=paths.artifacts)
                    context_compiler = ContextCompiler()
                except Exception:
                    ledger_store = None
                    context_compiler = None

            ledger_lock: threading.Lock | None = None
            if ledger_store is not None and context_compiler is not None:
                ledger_lock = threading.Lock()
            # Initial proposal
            obj: Dict[str, Any] = engine.task_plan(idea, constraints)

            # Helpers
            def _normalize_task_dict(t: Dict[str, Any]) -> Dict[str, Any]:
                d = dict(t or {})
                d.setdefault("kind", d.get("type") or "composite")
                d.setdefault("title", "")
                d.setdefault("description", "")
                if not isinstance(d.get("deps"), list):
                    d["deps"] = []
                if not isinstance(d.get("inputs"), dict):
                    d["inputs"] = {}
                if not isinstance(d.get("outputs"), dict):
                    d["outputs"] = {}
                if not isinstance(d.get("children"), list):
                    d["children"] = []
                return d

            def _walk(tasks: List[Dict[str, Any]]):
                for t in tasks:
                    yield t
                    for c in t.get("children", []) or []:
                        for x in _walk([c]):
                            yield x

            def _collect_ids(tasks: List[Dict[str, Any]]) -> Set[str]:
                ids: Set[str] = set()
                for t in _walk(tasks):
                    tid = t.get("id")
                    if isinstance(tid, str) and tid:
                        ids.add(tid)
                return ids

            def _validate_taskplan(plan_obj: Dict[str, Any]) -> Dict[str, Any]:
                issues: List[str] = []
                suggestions: List[str] = []
                tasks = plan_obj.get("tasks") if isinstance(plan_obj, dict) else None
                if not isinstance(tasks, list) or not tasks:
                    issues.append("no_tasks")
                    return {"ok": False, "issues": issues, "suggestions": suggestions}

                # Normalize minimal structure and collect ids
                for i in range(len(tasks)):
                    tasks[i] = _normalize_task_dict(tasks[i])
                ids = _collect_ids(tasks)

                # Validate deps reference existing ids when present
                for t in _walk(tasks):
                    for dref in t.get("deps") or []:
                        if isinstance(dref, str) and dref and dref not in ids:
                            issues.append(f"dep_missing:{dref}")

                # Composite nodes should not be empty leaves
                for t in _walk(tasks):
                    if t.get("kind") == "composite" and not (t.get("children") or []):
                        suggestions.append(f"split_needed:{t.get('id') or t.get('title')}")

                # Artifact contract surface check (shape only, language-agnostic)
                def _is_art_list(x: Any) -> bool:
                    if not isinstance(x, list):
                        return False
                    for it in x:
                        if not isinstance(it, dict):
                            return False
                    return True

                for t in _walk(tasks):
                    ins = t.get("inputs") or {}
                    outs = t.get("outputs") or {}
                    if "consumes" in ins and not _is_art_list(ins.get("consumes")):
                        issues.append("inputs.consumes_not_list")
                    if "produces" in outs and not _is_art_list(outs.get("produces")):
                        issues.append("outputs.produces_not_list")

                ok = not issues and (len(suggestions) == 0)
                return {"ok": ok, "issues": issues, "suggestions": suggestions}

            # Iterative validate/refine loop for TaskPlan
            def _clamp(n: int) -> int:
                return max(0, min(5, n))

            max_rounds = 5
            try:
                env_val = int(os.environ.get("CRPB_TASKPLAN_REFINE_MAX_ROUNDS", str(max_rounds)))
                max_rounds = _clamp(env_val)
            except Exception:
                max_rounds = _clamp(max_rounds)
            if isinstance(constraints, dict):
                try:
                    mr = int(constraints.get("taskplan_refine_max_rounds", max_rounds))
                    max_rounds = _clamp(mr)
                except Exception:
                    max_rounds = _clamp(max_rounds)

            rounds = 0
            # Ensure minimal normalization before entering the loop
            if not isinstance(obj.get("tasks"), list):
                obj["tasks"] = []
            obj["tasks"] = [_normalize_task_dict(t) for t in obj.get("tasks", [])]

            while rounds < max_rounds:
                rounds += 1
                val = _validate_taskplan(obj)
                ok = bool(val.get("ok", False))
                # suggestions are consumed indirectly via ok and downstream deterministic validation

                # Do not stop early if deterministic validator still reports coverage/communication issues
                try:
                    tp_det = TaskPlan.model_validate(obj)
                    det_rep = validate_taskplan_general(tp_det)
                    if isinstance(det_rep, dict) and not bool(det_rep.get("ok", True)):
                        ok = False
                except Exception:
                    pass

                if ok:
                    break

                prev = json.dumps(obj, sort_keys=True, separators=(",", ":"))

                # 1) Split empty composites deterministically using engine guardrails
                changed = False
                tasks_list: List[Dict[str, Any]] = obj.get("tasks", [])

                def _constraints_with_pack(
                    *,
                    node: Dict[str, Any],
                    parent: Optional[Dict[str, Any]],
                    siblings: List[Dict[str, Any]],
                ) -> Dict[str, Any]:
                    c = dict(constraints or {})
                    sc = c.get("side_context")
                    if not isinstance(sc, dict):
                        sc = {}
                    # Attach side_context before compiling so ContextCompiler runs in strict mode
                    # (no silent truncation) for LLM-bound calls.
                    c["side_context"] = sc
                    if ledger_store is not None and context_compiler is not None:
                        tid = node.get("id")
                        led = None
                        try:
                            if isinstance(tid, str) and tid:
                                if ledger_lock is not None:
                                    with ledger_lock:
                                        led = ledger_store.load(
                                            tid,
                                            parent_id=(parent or {}).get("id")
                                            if isinstance(parent, dict)
                                            else None,
                                        )
                                else:
                                    led = ledger_store.load(
                                        tid,
                                        parent_id=(parent or {}).get("id")
                                        if isinstance(parent, dict)
                                        else None,
                                    )
                        except Exception:
                            led = None
                        try:
                            pack = context_compiler.compile(
                                idea=idea or "",
                                constraints=c,
                                node=node,
                                parent=parent,
                                siblings=siblings,
                                ledger=led,
                                artifacts=[],
                                files={},
                                file_specs={},
                                signals={},
                            )
                            sc["context_pack"] = pack
                            if led is not None:
                                try:
                                    led.context_pack_digests.append(str(pack.get("digest") or ""))
                                    with ledger_lock:
                                        ledger_store.save(led)
                                except Exception:
                                    pass
                        except Exception as e:
                            # Do not silently proceed without context_pack when side_context is present;
                            # strict compilation failures should be actionable.
                            sc["context_pack_error"] = {
                                "type": type(e).__name__,
                                "message": str(e),
                            }
                            raise
                    return c

                def _boolish(v: Any) -> bool:
                    if v is None:
                        return False
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    return s in ("1", "true", "yes", "y", "on")

                def _planner_parallel_enabled() -> bool:
                    if isinstance(constraints, dict) and _boolish(
                        constraints.get("planner_parallel_enable")
                    ):
                        return True
                    try:
                        return _boolish(os.environ.get("CRPB_PLANNER_PARALLEL_ENABLE", "0"))
                    except Exception:
                        return False

                def _planner_parallel_workers() -> int:
                    v = None
                    if isinstance(constraints, dict):
                        v = constraints.get("planner_parallel_workers")
                    if v is None:
                        v = os.environ.get("CRPB_PLANNER_PARALLEL_WORKERS", "4")
                    try:
                        n = int(str(v))
                    except Exception:
                        n = 4
                    return max(1, min(32, n))

                def _planner_refine_passes_per_round() -> int:
                    v = None
                    if isinstance(constraints, dict):
                        v = constraints.get("planner_refine_passes_per_round")
                    if v is None:
                        v = os.environ.get("CRPB_PLANNER_REFINE_PASSES_PER_ROUND", "2")
                    try:
                        n = int(str(v))
                    except Exception:
                        n = 2
                    return max(1, min(6, n))

                parallel_on = _planner_parallel_enabled()
                max_workers = _planner_parallel_workers() if parallel_on else 1
                max_passes = _planner_refine_passes_per_round()

                processed: set[int] = set()

                def _collect_refine_work(
                    nodes: List[Dict[str, Any]],
                    parent: Optional[Dict[str, Any]],
                    _processed: set[int] = processed,
                ) -> List[Dict[str, Any]]:
                    work: List[Dict[str, Any]] = []
                    for idx, n in enumerate(nodes or []):
                        if not isinstance(n, dict):
                            continue
                        ident = id(n)
                        if ident in _processed:
                            continue
                        sibs = [s for j, s in enumerate(nodes) if j != idx and isinstance(s, dict)]
                        work.append({"node_ref": n, "parent": parent, "siblings": sibs})
                    return work

                def _traverse_collect(root_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    out: List[Dict[str, Any]] = []

                    def rec(nodes: List[Dict[str, Any]], parent: Optional[Dict[str, Any]]) -> None:
                        out.extend(_collect_refine_work(nodes, parent))
                        for n in nodes or []:
                            if not isinstance(n, dict):
                                continue
                            ch = n.get("children") or []
                            if isinstance(ch, list) and ch:
                                rec([c for c in ch if isinstance(c, dict)], n)

                    rec([t for t in root_nodes if isinstance(t, dict)], None)
                    return out

                def _refine_one(item: Dict[str, Any]) -> Dict[str, Any]:
                    node_ref = item.get("node_ref")
                    parent_ref = item.get("parent")
                    siblings_ref = item.get("siblings") or []
                    node = _normalize_task_dict(copy.deepcopy(node_ref or {}))
                    parent_snap = copy.deepcopy(parent_ref) if isinstance(parent_ref, dict) else None
                    sibs_snap = [copy.deepcopy(s) for s in siblings_ref if isinstance(s, dict)]

                    local_changed = False
                    should_consider_split = not (node.get("children") or [])
                    if should_consider_split:
                        decision = engine.decide_split(
                            task=node,
                            idea=idea,
                            constraints=_constraints_with_pack(
                                node=node, parent=parent_snap, siblings=sibs_snap
                            ),
                        )
                        action = decision.get("action") if isinstance(decision, dict) else None
                        children = (
                            decision.get("children", []) if isinstance(decision, dict) else []
                        )
                        if action == "split" and children:
                            node["children"] = [_normalize_task_dict(c) for c in children]
                            local_changed = True

                    clarified = engine.clarify_task(
                        task=node,
                        parent=parent_snap or {},
                        siblings=sibs_snap,
                        artifacts=[],
                        files={},
                        idea=idea,
                        constraints=_constraints_with_pack(
                            node=node, parent=parent_snap, siblings=sibs_snap
                        ),
                    )
                    if isinstance(clarified, dict) and clarified:
                        merged = _normalize_task_dict({**node, **clarified})
                        merged["children"] = node.get("children", [])
                        node = merged
                        local_changed = True

                    return {"node": node, "changed": bool(local_changed)}

                for _pass in range(max_passes):
                    work = _traverse_collect(tasks_list)
                    if not work:
                        break

                    for it in work:
                        nr = it.get("node_ref")
                        if isinstance(nr, dict):
                            processed.add(id(nr))

                    results = stable_parallel_map(_refine_one, work, max_workers=max_workers)
                    for it, res in zip(work, results, strict=True):
                        node_ref = it.get("node_ref")
                        if not isinstance(node_ref, dict):
                            continue
                        new_node = res.get("node") if isinstance(res, dict) else None
                        if isinstance(new_node, dict) and new_node:
                            node_ref.clear()
                            node_ref.update(_normalize_task_dict(new_node))
                        if bool(res.get("changed")):
                            changed = True

                # 2) Non-destructive amend pass from the LM (optional)
                # Embeddings-backed dedup/coverage assist: compute leaf summaries and provide only top-K similar ones
                dedup_context: Dict[str, Any] = {}
                det_validation: Dict[str, Any] = {}
                try:
                    # Collect leaf summaries (title + description) for indexing
                    def _collect_leaf_items(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                        out: List[Dict[str, Any]] = []
                        for node in _walk(tasks):
                            ch = node.get("children") or []
                            if isinstance(ch, list) and len(ch) > 0:
                                continue
                            tid = node.get("id")
                            if not isinstance(tid, str) or not tid:
                                continue
                            title = str(node.get("title") or "")
                            desc = str(node.get("description") or "")
                            try:
                                np = (
                                    node.get("node_plan")
                                    if isinstance(node.get("node_plan"), dict)
                                    else {}
                                )
                                intent = (
                                    str(np.get("intent") or "").strip()
                                    if isinstance(np, dict)
                                    else ""
                                )
                            except Exception:
                                intent = ""
                            summary = intent or desc
                            out.append(
                                {
                                    "id": tid,
                                    "path": str(
                                        (
                                            (node.get("meta") or {})
                                            if isinstance(node.get("meta"), dict)
                                            else {}
                                        ).get("path")
                                        or ""
                                    ),
                                    "title": title,
                                    "summary": summary,
                                }
                            )
                        return out

                    leaf_items = _collect_leaf_items(obj.get("tasks", []) or [])
                    # Compute a minimal run-scoped index path if a run dir is provided
                    run_dir = str(run_dir_path or "").strip()
                    provider_pref: List[str] = []
                    try:
                        pp = os.environ.get("CRPB_TASK_EMBED_PROVIDER_PREFERENCE")
                        if isinstance(pp, str) and pp.strip():
                            provider_pref = [p.strip() for p in pp.split(",") if p.strip()]
                    except Exception:
                        provider_pref = []
                    try:
                        from ..core.llm_config import embeddings_enabled as _embeddings_enabled
                    except Exception:
                        _embeddings_enabled = None  # type: ignore

                    # Embeddings are strictly opt-in: require the feature flag AND an explicit provider preference.
                    if (
                        run_dir
                        and provider_pref
                        and (callable(_embeddings_enabled) and bool(_embeddings_enabled()))
                    ):
                        from ..core.config import make_paths
                        from ..utils.embeddings import (
                            TaskEmbeddingIndex,
                            build_task_embedding_index,
                            make_embedder_from_env,
                            normalize_task_embedding_fields,
                        )

                        paths = make_paths(Path(run_dir))
                        index_path = paths.registry / "task_embeddings.json"
                        build_task_embedding_index(
                            index_path=index_path,
                            items=leaf_items,
                            provider_preference=provider_pref,
                        )

                        # Query similar leaves for each leaf (top-K) to detect redundancy without dumping all text
                        # Only query if we have at least one provider configured.
                        idx = TaskEmbeddingIndex(path=index_path)
                        try:
                            idx.load()
                        except Exception:
                            idx = None  # type: ignore

                        # Build embedder for queries
                        q_embedder = None
                        if provider_pref:
                            for prov in provider_pref:
                                q_embedder = make_embedder_from_env(
                                    provider=prov,
                                    input_type="query",
                                )
                                if q_embedder is not None:
                                    break

                        if idx is not None and q_embedder is not None and leaf_items and provider_pref:
                            # Query only for the first N leaves to bound cost
                            try:
                                max_leaf_queries = int(
                                    os.environ.get("CRPB_TASK_DEDUP_MAX_LEAF_QUERIES", "12")
                                )
                            except Exception:
                                max_leaf_queries = 12
                            max_leaf_queries = max(0, min(50, max_leaf_queries))
                            sim_results: List[Dict[str, Any]] = []
                            for li in leaf_items[:max_leaf_queries]:
                                _, _, text = normalize_task_embedding_fields(
                                    title=str(li.get("title") or ""),
                                    summary=str(li.get("summary") or ""),
                                )
                                if not text:
                                    continue
                                try:
                                    qv = q_embedder.embed_texts([text])[0]
                                except Exception:
                                    continue
                                hits = idx.query_similar(
                                    vector=qv,
                                    top_k=6,
                                    min_score=float(
                                        os.environ.get("CRPB_TASK_DEDUP_MIN_SIM", "0.84")
                                    ),
                                    provider_preference=provider_pref,
                                )
                                # Exclude self
                                filtered = [
                                    {
                                        "id": h[0].id,
                                        "path": h[0].path,
                                        "title": h[0].title,
                                        "summary": h[0].summary,
                                        "score": h[1],
                                    }
                                    for h in hits
                                    if h[0].id != li.get("id")
                                ]
                                if filtered:
                                    sim_results.append(
                                        {
                                            "leaf": {
                                                "id": li.get("id"),
                                                "title": li.get("title"),
                                                "path": li.get("path"),
                                            },
                                            "similar": filtered,
                                        }
                                    )
                            if sim_results:
                                dedup_context["similar_leaves"] = sim_results
                                dedup_context["policy"] = {
                                    "goal": "prevent redundant subtasks; maximize independence; ensure child union covers parent",
                                    "min_similarity": os.environ.get(
                                        "CRPB_TASK_DEDUP_MIN_SIM", "0.84"
                                    ),
                                }
                except Exception:
                    dedup_context = {}

                # Deterministic validation findings (coverage/communication) as a compact side_context signal
                try:
                    tp_tmp = TaskPlan.model_validate(obj)
                    det_validation = validate_taskplan_general(tp_tmp)
                    if not isinstance(det_validation, dict):
                        det_validation = {}
                    # Reduce payload deterministically without slicing strings/lists: include whole items until a char budget is met.
                    if det_validation:
                        max_chars = None
                        try:
                            max_chars = int(
                                (constraints or {}).get("deterministic_validation_max_chars")
                                or os.environ.get("CRPB_DET_VALIDATION_MAX_CHARS", "6000")
                            )
                        except Exception:
                            max_chars = 6000

                        def _stable_json(obj: Any) -> str:
                            return json.dumps(
                                obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                            )

                        def _fit_list(*, items: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
                            out: List[Dict[str, Any]] = []
                            used = 0
                            for it in items:
                                try:
                                    n = len(_stable_json(it))
                                except Exception:
                                    n = 10**9
                                if budget > 0 and (used + n) > budget:
                                    break
                                out.append(it)
                                used += n
                            return out

                        base = {
                            "ok": bool(det_validation.get("ok", False)),
                            "issues": [],
                            "nodes": [],
                            "omissions": {
                                "issues_dropped": 0,
                                "nodes_dropped": 0,
                                "node_issues_dropped": 0,
                            },
                        }

                        # Allocate budget across issues/nodes, leaving room for keys/metadata.
                        overhead = len(_stable_json(base))
                        remaining = max(0, int(max_chars) - overhead)
                        issues_budget = int(remaining * 0.55)
                        nodes_budget = int(remaining * 0.45)

                        issues_all = [{"text": str(x)} for x in (det_validation.get("issues") or [])]
                        issues_fit = _fit_list(items=issues_all, budget=issues_budget)
                        base["issues"] = [it.get("text") for it in issues_fit if isinstance(it, dict)]
                        base["omissions"]["issues_dropped"] = max(0, len(issues_all) - len(issues_fit))

                        nodes_all_raw = [n for n in (det_validation.get("nodes") or []) if isinstance(n, dict)]
                        # For each node, fit issues as whole strings without slicing.
                        nodes_items: List[Dict[str, Any]] = []
                        node_issues_dropped = 0
                        for n in nodes_all_raw:
                            node_entry = {
                                "id": str(n.get("id") or ""),
                                "path": str(n.get("path") or ""),
                                "issues": [],
                            }
                            node_issue_all = [{"text": str(x)} for x in (n.get("issues") or [])]
                            # A small per-node budget derived from total nodes budget.
                            per_node_budget = max(200, int(nodes_budget * 0.10))
                            fit = _fit_list(items=node_issue_all, budget=per_node_budget)
                            node_entry["issues"] = [it.get("text") for it in fit if isinstance(it, dict)]
                            node_issues_dropped += max(0, len(node_issue_all) - len(fit))
                            nodes_items.append(node_entry)

                        nodes_fit = _fit_list(items=nodes_items, budget=nodes_budget)
                        base["nodes"] = nodes_fit
                        base["omissions"]["nodes_dropped"] = max(0, len(nodes_items) - len(nodes_fit))
                        base["omissions"]["node_issues_dropped"] = int(node_issues_dropped)

                        det_validation = base
                except Exception:
                    det_validation = {}

                amend_constraints = dict(constraints or {})
                amend_constraints.setdefault("side_context", {})
                try:
                    if isinstance(amend_constraints.get("side_context"), dict) and dedup_context:
                        amend_constraints["side_context"]["task_dedup"] = dedup_context
                except Exception:
                    pass

                try:
                    if isinstance(amend_constraints.get("side_context"), dict) and det_validation:
                        amend_constraints["side_context"]["deterministic_taskplan_validation"] = (
                            det_validation
                        )
                except Exception:
                    pass

                # Ask the LM to explicitly use embeddings-derived redundancy signals when available
                try:
                    add_blocks = amend_constraints.get("rule_blocks_add")
                    if not isinstance(add_blocks, list):
                        add_blocks = []
                    if "task_dedup_policy" not in [str(x) for x in add_blocks]:
                        add_blocks.append("task_dedup_policy")
                    if det_validation and "deterministic_taskplan_fixes" not in [
                        str(x) for x in add_blocks
                    ]:
                        add_blocks.append("deterministic_taskplan_fixes")
                    amend_constraints["rule_blocks_add"] = add_blocks
                except Exception:
                    pass

                edits_obj = engine.amend_task_plan(
                    current_plan=obj,
                    statuses={},
                    artifacts=[],
                    idea=idea,
                    constraints=amend_constraints,
                )
                try:
                    edits = edits_obj.get("edits", []) if isinstance(edits_obj, dict) else []
                except Exception:
                    edits = []

                if edits:
                    # Normalize edits across signature variants and apply safely
                    try:
                        from ..utils.plan_edits import normalize_amend_edits

                        edits = normalize_amend_edits({"edits": edits})
                    except Exception:
                        pass
                    # Apply a minimal subset of safe edits
                    id_map: Dict[str, Dict[str, Any]] = {}
                    for t in _walk(obj.get("tasks", [])):
                        tid = t.get("id")
                        if isinstance(tid, str) and tid:
                            id_map[tid] = t

                    def _apply_edit(e: Dict[str, Any], *, id_map: Dict[str, Dict[str, Any]] = id_map):
                        nonlocal changed
                        op = e.get("op")
                        if op == "edit_task":
                            tid = e.get("id")
                            if tid in id_map and isinstance(e.get("fields"), dict):
                                t = id_map[tid]
                                new_t = _normalize_task_dict({**t, **e.get("fields", {})})
                                # preserve children if not explicitly provided
                                if "children" not in e.get("fields", {}):
                                    new_t["children"] = t.get("children", [])
                                t.clear()
                                t.update(new_t)
                                changed = True
                        elif op == "add_child":
                            pid = e.get("parent_id")
                            child = _normalize_task_dict(e.get("child") or {})
                            if pid in id_map:
                                p = id_map[pid]
                                ch = p.get("children") or []
                                ch.append(child)
                                p["children"] = ch
                                changed = True
                        elif op == "add_dep":
                            tid = e.get("id")
                            dep = e.get("dep_id")
                            if tid in id_map and isinstance(dep, str) and dep:
                                t = id_map[tid]
                                deps = t.get("deps") or []
                                if dep not in deps:
                                    deps.append(dep)
                                    t["deps"] = deps
                                    changed = True
                        elif op == "rewire_artifacts":
                            tid = e.get("id")
                            if tid in id_map:
                                t = id_map[tid]
                                if isinstance(e.get("consumes"), list):
                                    ins = t.get("inputs") or {}
                                    ins["consumes"] = e.get("consumes")
                                    t["inputs"] = ins
                                    changed = True
                                if isinstance(e.get("produces"), list):
                                    outs = t.get("outputs") or {}
                                    outs["produces"] = e.get("produces")
                                    t["outputs"] = outs
                                    changed = True

                    for ed in edits:
                        if isinstance(ed, dict):
                            _apply_edit(ed)

                # 3) Join-judge pass (optional): validate coverage/overlap/interfaces at parent nodes.
                def _join_judge_enabled() -> bool:
                    if isinstance(constraints, dict) and _boolish(
                        constraints.get("join_judge_enable")
                    ):
                        return True
                    try:
                        return _boolish(os.environ.get("CRPB_JOIN_JUDGE_ENABLE", "0"))
                    except Exception:
                        return False

                def _join_judge_max_nodes() -> int:
                    v = None
                    if isinstance(constraints, dict):
                        v = constraints.get("join_judge_max_nodes")
                    if v is None:
                        v = os.environ.get("CRPB_JOIN_JUDGE_MAX_NODES", "20")
                    try:
                        n = int(str(v))
                    except Exception:
                        n = 20
                    return max(0, min(200, n))

                if _join_judge_enabled():
                    logger.info("[planner] Join-judge enabled")
                    artifacts_index: Dict[str, Any] = {}
                    node_issue_map: Dict[str, List[str]] = {}
                    try:
                        tp_tmp = TaskPlan.model_validate(obj)
                        det_rep2 = validate_taskplan_general(tp_tmp)
                        if isinstance(det_rep2, dict):
                            a2 = det_rep2.get("artifacts")
                            if isinstance(a2, dict):
                                idx2 = a2.get("index")
                                if isinstance(idx2, dict):
                                    artifacts_index = idx2
                            for nr in det_rep2.get("nodes") or []:
                                if isinstance(nr, dict) and isinstance(nr.get("id"), str):
                                    issues_list = [str(x) for x in (nr.get("issues") or []) if str(x)]
                                    node_issue_map[str(nr.get("id"))] = issues_list
                    except Exception:
                        artifacts_index = {}
                        node_issue_map = {}

                    # Collect composite nodes (parents) with minimal local neighborhood.
                    jj_work: List[Dict[str, Any]] = []

                    def _collect(
                        nodes: List[Dict[str, Any]],
                        parent: Optional[Dict[str, Any]],
                        *,
                        _jj_work: List[Dict[str, Any]] = jj_work,
                    ) -> None:
                        for idx, n in enumerate(nodes or []):
                            if not isinstance(n, dict):
                                continue
                            ch = n.get("children") or []
                            if isinstance(ch, list) and len([c for c in ch if isinstance(c, dict)]) > 0:
                                sibs = [s for j, s in enumerate(nodes) if j != idx and isinstance(s, dict)]
                                _jj_work.append({"node_ref": n, "parent": parent, "siblings": sibs})
                            if isinstance(ch, list) and ch:
                                _collect([c for c in ch if isinstance(c, dict)], n)

                    _collect([t for t in tasks_list if isinstance(t, dict)], None)
                    jj_work = jj_work[: _join_judge_max_nodes()]

                    logger.info(
                        "[planner] Join-judge worklist prepared: nodes=%d (max=%d)",
                        len(jj_work),
                        _join_judge_max_nodes(),
                    )

                    def _jj_one(
                        item: Dict[str, Any],
                        *,
                        _artifacts_index: Dict[str, Any] = artifacts_index,
                        _node_issue_map: Dict[str, List[str]] = node_issue_map,
                    ) -> Dict[str, Any]:
                        node_ref = item.get("node_ref")
                        parent_ref = item.get("parent")
                        siblings_ref = item.get("siblings") or []
                        node_id = str((node_ref or {}).get("id") or "")
                        capsule = build_join_judge_capsule(
                            target=node_ref or {},
                            parent=parent_ref if isinstance(parent_ref, dict) else None,
                            siblings=[s for s in siblings_ref if isinstance(s, dict)],
                            artifacts_index=_artifacts_index,
                            deterministic_issues=_node_issue_map.get(node_id, []),
                        )
                        c3 = dict(constraints or {})
                        c3.setdefault("side_context", {})
                        if isinstance(c3.get("side_context"), dict):
                            c3["side_context"]["join_judge"] = {
                                "target": capsule.target,
                                "parent": capsule.parent,
                                "siblings": capsule.siblings,
                                "children": capsule.children,
                                "artifacts": capsule.artifacts,
                                "deterministic_issues": capsule.deterministic_issues,
                            }
                        logger.debug(
                            "[planner][join_judge] Invoking for node_id=%s parent_id=%s siblings=%d children=%d artifacts=%d det_issues=%d",
                            (capsule.target or {}).get("id"),
                            (capsule.parent or {}).get("id"),
                            len(capsule.siblings),
                            len(capsule.children),
                            len(capsule.artifacts),
                            len(capsule.deterministic_issues),
                        )
                        return engine.join_judge_node(node=capsule.target, idea=idea, constraints=c3)

                    jj_results = stable_parallel_map(_jj_one, jj_work, max_workers=max_workers)
                    for item, jr in zip(jj_work, jj_results, strict=True):
                        node_ref = item.get("node_ref")
                        if not isinstance(node_ref, dict) or not isinstance(jr, dict):
                            continue

                        try:
                            node_id = str(node_ref.get("id") or "")
                            ok = bool(jr.get("ok", True))
                            actions = jr.get("actions") if isinstance(jr.get("actions"), list) else []
                            questions = jr.get("questions") if isinstance(jr.get("questions"), list) else []
                            notes = jr.get("notes") if isinstance(jr.get("notes"), list) else []
                            rubric = jr.get("rubric") if isinstance(jr.get("rubric"), dict) else {}
                            logger.info(
                                "[planner][join_judge] node_id=%s ok=%s actions=%d questions=%d notes=%d",
                                node_id,
                                ok,
                                len(actions),
                                len(questions),
                                len(notes),
                            )
                            # Log rubric details for debugging
                            if rubric:
                                dimensions = rubric.get("dimensions") if isinstance(rubric.get("dimensions"), list) else []
                                for dim in dimensions[:3]:  # Log first 3 dimensions
                                    if isinstance(dim, dict):
                                        logger.debug(
                                            "[planner][join_judge] Rubric dimension: name=%s score=%s",
                                            dim.get("name"), dim.get("score")
                                        )
                            # Log actions details
                            for act in actions[:3]:  # Log first 3 actions
                                if isinstance(act, dict):
                                    logger.debug(
                                        "[planner][join_judge] Action: op=%s target=%s",
                                        act.get("op"), act.get("target_id") or act.get("id")
                                    )
                            # Log questions for visibility
                            for q in questions[:2]:  # Log first 2 questions
                                if isinstance(q, dict):
                                    logger.debug(
                                        "[planner][join_judge] Question: %s",
                                        str(q.get("question") or "")[:100]
                                    )
                        except Exception:
                            pass
                        meta = node_ref.get("meta")
                        if not isinstance(meta, dict):
                            meta = {}
                        meta["join_judge"] = {
                            "ok": bool(jr.get("ok", True)),
                            "rubric": jr.get("rubric") if isinstance(jr.get("rubric"), dict) else {},
                            "questions": jr.get("questions") if isinstance(jr.get("questions"), list) else [],
                            "notes": jr.get("notes") if isinstance(jr.get("notes"), list) else [],
                        }
                        node_ref["meta"] = meta

                        actions = jr.get("actions") if isinstance(jr.get("actions"), list) else []
                        for act in actions:
                            if isinstance(act, dict):
                                before = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                                logger.debug("[planner][join_judge] Applying action: %s", act)
                                _apply_edit(act)
                                after = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                                if after != before:
                                    logger.info("[planner][join_judge] Action applied successfully: %s", act.get("op"))
                                    changed = True

                # If nothing changed this round, stop early
                cur = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                if cur == prev or not changed:
                    break

            # Final validation gate
            final_val = _validate_taskplan(obj)
            if not bool(final_val.get("ok", False)):
                issues = ",".join(final_val.get("issues", []))
                raise RuntimeError(f"Task planning failed validation: {issues}")

            if bool(enforce_deterministic_leaves):
                # -------------------------------------------------
                # Deterministic leaf guardrails
                # Ensure every leaf node (no children) satisfies the deterministic leaf predicate.
                def _enforce_deterministic_leaves(tasks: List[Dict[str, Any]]) -> None:
                    for t in _walk(tasks):
                        # Leaf = no children
                        if not (t.get("children") or []):
                            # Convert dict to TaskSpec for the validator check
                            task_spec = TaskSpec(**t)
                            if not is_deterministic_leaf(task_spec):
                                leaf_id = t.get("id") or t.get("title") or "<unknown>"
                                raise RuntimeError(f"Non-deterministic leaf detected: {leaf_id}")
                            ok_lr, msg_lr, _meta_lr = validate_leaf_readiness(task_spec)
                            if not ok_lr:
                                leaf_id = t.get("id") or t.get("title") or "<unknown>"
                                raise RuntimeError(f"Leaf not ready: {leaf_id}: {msg_lr}")

                _enforce_deterministic_leaves(obj.get("tasks", []))

            # -------------------------------------------------
            # Deterministic artifact wiring
            # If a task consumes an artifact, it must depend on at least one producer.
            # This is a spec-level invariant enforced by validate_taskplan_general.
            def _auto_wire_artifact_deps(tasks: List[Dict[str, Any]]) -> int:
                # Collect all nodes by id and artifact producers
                id_to_task: Dict[str, Dict[str, Any]] = {}
                art_to_producers: Dict[str, List[str]] = {}

                def _norm_refs(val: Any) -> List[Dict[str, Any]]:
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

                for t in _walk(tasks):
                    tid = t.get("id")
                    if isinstance(tid, str) and tid.strip():
                        id_to_task[tid] = t
                    outs = t.get("outputs") if isinstance(t.get("outputs"), dict) else {}
                    for ref in _norm_refs((outs or {}).get("produces")):
                        rid = ref.get("id")
                        if isinstance(rid, str) and rid.strip() and isinstance(tid, str) and tid.strip():
                            art_to_producers.setdefault(rid, []).append(tid)

                # Normalize producer lists deterministically
                for rid in list(art_to_producers.keys()):
                    art_to_producers[rid] = sorted({p for p in art_to_producers[rid] if p})

                # Build adjacency for cycle checks (edges: task -> deps)
                def _deps_of(task_dict: Dict[str, Any]) -> List[str]:
                    raw = task_dict.get("deps")
                    if not isinstance(raw, list):
                        return []
                    out: List[str] = []
                    for d in raw:
                        if isinstance(d, str) and d.strip():
                            out.append(d.strip())
                        elif isinstance(d, dict) and isinstance(d.get("id"), str) and d.get("id").strip():
                            out.append(str(d.get("id")).strip())
                        elif d is not None:
                            ds = str(d).strip()
                            if ds:
                                out.append(ds)
                    return out

                adj: Dict[str, List[str]] = {}
                for tid, t in id_to_task.items():
                    adj[tid] = [d for d in _deps_of(t) if d in id_to_task]

                def _reachable(src: str, dst: str) -> bool:
                    # Is dst reachable from src following deps edges?
                    if src == dst:
                        return True
                    seen: set[str] = set()
                    stack = [src]
                    while stack:
                        u = stack.pop()
                        if u in seen:
                            continue
                        seen.add(u)
                        for v in adj.get(u, []):
                            if v == dst:
                                return True
                            if v not in seen:
                                stack.append(v)
                    return False

                changed = 0
                for consumer_id, t in id_to_task.items():
                    ins = t.get("inputs") if isinstance(t.get("inputs"), dict) else {}
                    consumes = _norm_refs((ins or {}).get("consumes"))
                    if not consumes:
                        continue

                    existing = _deps_of(t)
                    dep_set = set(existing)

                    for ref in consumes:
                        rid = ref.get("id")
                        if not isinstance(rid, str) or not rid.strip():
                            continue
                        producer_ids = art_to_producers.get(rid, [])
                        if not producer_ids:
                            continue
                        if any(p in dep_set for p in producer_ids):
                            continue

                        # Choose a producer deterministically, avoiding self-dep and obvious cycles.
                        chosen: str | None = None
                        for pid in producer_ids:
                            if pid == consumer_id:
                                continue
                            # Adding edge consumer->pid creates a cycle if pid can reach consumer.
                            if _reachable(pid, consumer_id):
                                continue
                            chosen = pid
                            break
                        if chosen is None:
                            continue
                        # Apply dep
                        raw_deps = t.get("deps")
                        if not isinstance(raw_deps, list):
                            raw_deps = []
                        raw_deps.append(chosen)
                        t["deps"] = raw_deps
                        dep_set.add(chosen)
                        adj.setdefault(consumer_id, []).append(chosen)
                        changed += 1
                return changed

            try:
                _auto_wire_artifact_deps(obj.get("tasks", []))
            except Exception:
                pass

            # Build typed TaskPlan
            def build_task(t: dict) -> TaskSpec:
                children = [build_task(c) for c in t.get("children", [])]
                # Normalize deps to strings only (LLM sometimes generates objects)
                raw_deps = t.get("deps", [])
                normalized_deps = []
                if isinstance(raw_deps, list):
                    for dep in raw_deps:
                        if isinstance(dep, str):
                            normalized_deps.append(dep)
                        elif isinstance(dep, dict) and "id" in dep:
                            normalized_deps.append(str(dep["id"]))
                        elif dep is not None:
                            normalized_deps.append(str(dep))
                return TaskSpec(
                    id=t.get("id"),
                    kind=t.get("kind", "composite"),
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    deps=normalized_deps,
                    inputs=t.get("inputs", {}),
                    outputs=t.get("outputs", {}),
                    children=children,
                    node_plan=t.get("node_plan", {}),
                    meta=t.get("meta", {}),
                )

            tasks = [build_task(x) for x in obj.get("tasks", [])]
            if not tasks:
                raise ValueError("empty-task-plan")

            # Spec-first: TaskPlan remains language-agnostic. File specifications live in CodeSpec.
            tp = TaskPlan(idea=idea, constraints=constraints, tasks=tasks)

            # Run-scoped NodeLedger initialization (best-effort).
            try:
                if run_dir_path:
                    from ..core.config import make_paths
                    from ..core.ledger import NodeLedgerStore

                    paths = make_paths(Path(str(run_dir_path)))
                    store = NodeLedgerStore(base_dir=paths.artifacts)

                    def _walk(ts: List[TaskSpec], parent_id: str | None = None) -> None:
                        for node in ts:
                            nid = str(getattr(node, "id", "") or "").strip()
                            if nid:
                                led = store.load(nid, parent_id=parent_id)
                                # Seed obligations from constraints and node_plan text.
                                store.save(led)
                            _walk(
                                list(getattr(node, "children", []) or []),
                                parent_id=nid or parent_id,
                            )

                    _walk(list(tp.tasks or []), parent_id=None)
            except Exception:
                pass

            return tp
        except Exception as e:
            raise RuntimeError(f"Task planning failed: {e}")
    # If use_llm is False or anything else, we still enforce LLM usage to keep behavior strict.
    raise RuntimeError(
        "LLM task planning disabled by configuration, but fallbacks are removed. Enable LLM."
    )


# Spec-first architecture: all file-level specifications are produced as CodeSpec and consumed by build.
