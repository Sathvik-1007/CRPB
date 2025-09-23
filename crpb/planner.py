from __future__ import annotations
from typing import List, Tuple, Dict, Any, Set, Optional
import os
import json
import hashlib
import re
import logging
from .specs import TaskPlan, TaskSpec

logger = logging.getLogger(__name__)

def generate_task_plan(idea: str, constraints: dict, use_llm: bool = True) -> TaskPlan:
    """
    Produce a generic hierarchical TaskPlan using the LLM. No code, only structure and metadata.
    LLM is required; if unavailable or fails, this function raises an error.
    """
    if use_llm:
        try:
            from .agents.dspy_engine import DspyEngine
            engine = DspyEngine()
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
                    for dref in (t.get("deps") or []):
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
                suggestions = val.get("suggestions", []) or []
                if ok:
                    break

                prev = json.dumps(obj, sort_keys=True, separators=(",", ":"))

                # 1) Split empty composites deterministically using engine guardrails
                changed = False
                tasks_list: List[Dict[str, Any]] = obj.get("tasks", [])
                def _refine_task_in_place(t: Dict[str, Any], parent: Optional[Dict[str, Any]] = None, siblings: Optional[List[Dict[str, Any]]] = None):
                    nonlocal changed
                    t = _normalize_task_dict(t)
                    # Always consider splitting nodes without children to deepen the tree when warranted
                    should_consider_split = not (t.get("children") or [])
                    if should_consider_split:
                        decision = engine.decide_split(task=t, idea=idea, constraints=constraints)
                        action = decision.get("action")
                        children = decision.get("children", []) if isinstance(decision, dict) else []
                        if action == "split" and children:
                            t["children"] = [_normalize_task_dict(c) for c in children]
                            changed = True
                    # Clarify to enrich fields; always preserve any children
                    clarified = engine.clarify_task(task=t, parent=parent or {}, siblings=siblings or [], artifacts=[], files={}, idea=idea, constraints=constraints)
                    if isinstance(clarified, dict) and clarified:
                        new_t = _normalize_task_dict({**t, **clarified})
                        new_t["children"] = t.get("children", [])
                        t.clear(); t.update(new_t)
                        changed = True
                    # Recurse on children
                    ch = t.get("children", []) or []
                    for idx in range(len(ch)):
                        _refine_task_in_place(ch[idx], parent=t, siblings=[c for j, c in enumerate(ch) if j != idx])

                for i in range(len(tasks_list)):
                    _refine_task_in_place(tasks_list[i], parent=None, siblings=[tasks_list[j] for j in range(len(tasks_list)) if j != i])

                # 2) Non-destructive amend pass from the LM (optional)
                edits_obj = engine.amend_task_plan(current_plan=obj, statuses={}, artifacts=[], idea=idea, constraints=constraints)
                try:
                    edits = edits_obj.get("edits", []) if isinstance(edits_obj, dict) else []
                except Exception:
                    edits = []

                if edits:
                    # Apply a minimal subset of safe edits
                    id_map: Dict[str, Dict[str, Any]] = {}
                    for t in _walk(obj.get("tasks", [])):
                        tid = t.get("id")
                        if isinstance(tid, str) and tid:
                            id_map[tid] = t

                    def _apply_edit(e: Dict[str, Any]):
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
                                t.clear(); t.update(new_t)
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
                            tid = e.get("id"); dep = e.get("dep_id")
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

                # If nothing changed this round, stop early
                cur = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                if cur == prev or not changed:
                    break

            # Final validation gate
            final_val = _validate_taskplan(obj)
            if not bool(final_val.get("ok", False)):
                issues = ",".join(final_val.get("issues", []))
                raise RuntimeError(f"Task planning failed validation: {issues}")

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

            return TaskPlan(idea=idea, constraints=constraints, tasks=tasks)
        except Exception as e:
            raise RuntimeError(f"Task planning failed: {e}")
    # If use_llm is False or anything else, we still enforce LLM usage to keep behavior strict.
    raise RuntimeError("LLM task planning disabled by configuration, but fallbacks are removed. Enable LLM.")


# Spec-first architecture: all file-level specifications are produced as CodeSpec and consumed by build.

