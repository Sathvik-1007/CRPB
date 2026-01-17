from __future__ import annotations

from typing import Any, Dict, List


def apply_plan_edits(*, plan: Dict[str, Any], edits: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return plan
    if not isinstance(edits, list) or not edits:
        return plan

    def _walk(tasks: List[Dict[str, Any]]):
        for t in tasks or []:
            if not isinstance(t, dict):
                continue
            yield t
            ch = t.get("children") or []
            if isinstance(ch, list):
                for c in _walk([x for x in ch if isinstance(x, dict)]):
                    yield c

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
        if not isinstance(d.get("node_plan"), dict):
            d["node_plan"] = {}
        if not isinstance(d.get("meta"), dict):
            d["meta"] = {}
        return d

    tasks_list = plan.get("tasks")
    if not isinstance(tasks_list, list):
        return plan

    plan["tasks"] = [x for x in tasks_list if isinstance(x, dict)]

    id_map: Dict[str, Dict[str, Any]] = {}
    for t in _walk(plan.get("tasks", []) or []):
        tid = t.get("id")
        if isinstance(tid, str) and tid:
            id_map[tid] = t

    def _apply_edit(e: Dict[str, Any]) -> None:
        op = e.get("op")
        if op == "edit_task":
            tid = e.get("id")
            fields = e.get("fields")
            if isinstance(tid, str) and tid in id_map and isinstance(fields, dict):
                t = id_map[tid]
                cur_children = t.get("children", []) if isinstance(t.get("children"), list) else []
                merged = _normalize_task_dict({**t, **fields})
                if "children" not in fields:
                    merged["children"] = cur_children
                t.clear()
                t.update(merged)
        elif op == "add_child":
            pid = e.get("parent_id")
            child = e.get("child")
            if isinstance(pid, str) and pid in id_map and isinstance(child, dict):
                p = id_map[pid]
                ch = p.get("children") if isinstance(p.get("children"), list) else []
                ch.append(_normalize_task_dict(child))
                p["children"] = ch
        elif op == "add_dep":
            tid = e.get("id")
            dep = e.get("dep_id")
            if isinstance(tid, str) and tid in id_map and isinstance(dep, str) and dep:
                t = id_map[tid]
                deps = t.get("deps") if isinstance(t.get("deps"), list) else []
                if dep not in deps:
                    deps.append(dep)
                    t["deps"] = deps
        elif op == "rewire_artifacts":
            tid = e.get("id")
            if isinstance(tid, str) and tid in id_map:
                t = id_map[tid]
                if isinstance(e.get("consumes"), list):
                    ins = t.get("inputs") if isinstance(t.get("inputs"), dict) else {}
                    ins["consumes"] = e.get("consumes")
                    t["inputs"] = ins
                if isinstance(e.get("produces"), list):
                    outs = t.get("outputs") if isinstance(t.get("outputs"), dict) else {}
                    outs["produces"] = e.get("produces")
                    t["outputs"] = outs

    for ed in edits:
        if isinstance(ed, dict):
            _apply_edit(ed)

    return plan


def normalize_amend_edits(edits_obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(edits_obj, dict):
        return []
    raw = edits_obj.get("edits")
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        op = e.get("op")
        if op == "update_task":
            tid = e.get("id")
            fields = e.get("set") if isinstance(e.get("set"), dict) else e.get("fields")
            out.append({"op": "edit_task", "id": tid, "fields": fields or {}})
            continue
        if op == "add_child":
            if "child" not in e and isinstance(e.get("task"), dict):
                e = dict(e)
                e["child"] = e.get("task")
            out.append(e)
            continue
        out.append(e)

    return [x for x in out if isinstance(x, dict) and isinstance(x.get("op"), str)]
