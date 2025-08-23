from __future__ import annotations
import hashlib
import re
import ast
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple
import typer
from rich.console import Console

from ..config import resolve_run_dir, make_paths
from ..eventbus import EventBus
from ..status import NodeStatus
from ..leases import Leases
from ..utils.fs import ensure_parent, atomic_write_json
from ..utils.ui import sep
from ..scheduler import Scheduler
from ..planner import generate_task_plan
from ..specs import TaskPlan, TaskSpec, FileSpec, FunctionSpec
from ..workers.code_function import implement_code_function
from ..aggregator import assemble_python_file, write_code_file, write_file_meta
from ..validator import (
    jsonschema_validate,
    basic_file_validation,
    validate_assembled_python_file,
    simple_style_check,
    import_safety_check,
    integrity_check,
)
from ..agents.dspy_engine import DspyEngine

app = typer.Typer(help="Execute a hierarchical TaskPlan with recursive LLM-driven split-or-implement orchestration")
console = Console()


def infer_language_from_path(path: str | None) -> str | None:
    if not path:
        return None
    ext = (Path(path).suffix or "").lower().lstrip(".")
    mapping = {
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
    return mapping.get(ext)


def _normalize_ids(tasks: List[TaskSpec]) -> None:
    counter = 0

    def walk(ts: List[TaskSpec]):
        nonlocal counter
        for t in ts:
            if not t.id:
                t.id = f"t_{counter}"
                counter += 1
            walk(t.children)

    walk(tasks)


@app.callback(invoke_without_command=True)
def main(
    idea: str = typer.Option(None, "--idea", help="Idea; if omitted uses run/plan/idea.json or task_plan.json"),
    constraints: str = typer.Option(None, "--constraints", help="Optional JSON string of constraints to steer splitting/implementation"),
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("new", "--run", help="run_<ts> | latest | new | name"),
    model: str = typer.Option(None, "--model", help="Override model for LLM steps"),
    max_children: int = typer.Option(2, "--max-children", min=1, help="Max concurrent child tasks the scheduler will select"),
    node_id: str = typer.Option(None, "--node-id", help="Deterministic node id for this executor"),
    parent_id: str = typer.Option(None, "--parent-id", help="Parent node id for orchestration tracking"),
    lease_ttl: int = typer.Option(180, "--lease-ttl", min=30, help="TTL seconds for executor lease"),
    child_ttl: int = typer.Option(120, "--child-ttl", min=30, help="TTL seconds for child leases"),
    keep_going: bool = typer.Option(True, "--keep-going/--fail-fast", help="On task failures, continue other tasks instead of aborting"),
):
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    sep("TASKS START")

    # Fail fast if no LLM key to avoid slow loops or silent stubs
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]LLM required for tasks: set OPENAI_API_KEY[/red]")
        raise typer.Exit(code=2)

    # Constraints load
    constraints_obj: Dict = {}
    if constraints is not None:
        try:
            constraints_obj = json.loads(constraints)
        except Exception as e:
            console.print(f"[red]Invalid constraints JSON: {e}")
            raise typer.Exit(code=2)
    else:
        cfile = paths.plan / "constraints.json"
        if cfile.exists():
            try:
                constraints_obj = json.loads(cfile.read_text(encoding="utf-8"))
            except Exception:
                constraints_obj = {}

    # Idea resolution
    if idea is None:
        idea_file = paths.plan / "idea.json"
        if idea_file.exists():
            try:
                idea = json.loads(idea_file.read_text(encoding="utf-8")).get("idea", "")
            except Exception:
                idea = ""
        else:
            idea = ""

    bus = EventBus(paths.logs / "events.jsonl")
    status = NodeStatus(paths.graph / "node_status.jsonl")
    leases = Leases(paths.graph / "leases.json")
    scheduler = Scheduler(max_parallel_children=max_children)
    # Initialize DSPy engine once for all LLM-backed decisions
    engine = DspyEngine(model=model)

    # Deterministic defaults
    node_id = node_id or f"tasks::{run_path.name}"
    parent_id = parent_id or f"root::{run_path.name}"
    status.write(node_id, "CREATED", parent_id=parent_id)
    lease_id = leases.grant(node_id, ttl=lease_ttl)
    status.write(node_id, "LEASED", prev_state="CREATED", lease_id=lease_id, parent_id=parent_id)
    bus.emit("NODE_CREATED", node_id=node_id, parent_id=parent_id)
    bus.emit("LEASE_GRANTED", node_id=node_id, lease_id=lease_id, parent_id=parent_id)

    # Load TaskPlan or generate if missing and idea provided
    task_plan_path = paths.plan / "task_plan.json"
    if task_plan_path.exists():
        tp = TaskPlan.model_validate(json.loads(task_plan_path.read_text(encoding="utf-8")))
    else:
        if not idea:
            console.print("[red]No task_plan.json and no idea provided[/red]")
            raise typer.Exit(code=2)
        tp = generate_task_plan(idea, constraints_obj, use_llm=True)
        atomic_write_json(task_plan_path, tp.model_dump())

    _normalize_ids(tp.tasks)
    # Immediately persist normalized ids
    atomic_write_json(task_plan_path, tp.model_dump())

    # Orchestration state
    # Index all tasks to avoid repeated flattening
    def _index(ts: List[TaskSpec], acc: Dict[str, TaskSpec]):
        for t in ts:
            acc[t.id] = t  # type: ignore[index]
            _index(t.children, acc)

    all_tasks: Dict[str, TaskSpec] = {}
    _index(tp.tasks, all_tasks)

    statuses: Dict[str, str] = {tid: "pending" for tid in list(all_tasks.keys())}
    deps_map: Dict[str, List[str]] = {tid: list(t.deps) for tid, t in all_tasks.items()}
    parents: Dict[str, str | None] = {}

    def set_parents(ts: List[TaskSpec], parent: str | None = None):
        for t in ts:
            parents[t.id] = parent  # type: ignore[index]
            set_parents(t.children, t.id)

    set_parents(tp.tasks)

    chunks_dir = paths.artifacts / "chunks"
    ensure_parent(chunks_dir / "dummy.txt")

    # Aggregation for code leaves
    code_impls_by_path: Dict[str, Dict[str, str]] = {}
    file_meta_by_path: Dict[str, Dict] = {}

    def ready_tasks() -> List[TaskSpec]:
        # select tasks that are pending and all deps done
        candidate: List[TaskSpec] = []
        for tid, t in all_tasks.items():
            # A task is ready if pending, deps done, and if it has children, all of them are done
            if (
                statuses.get(tid) == "pending"
                and all(statuses.get(d) == "done" for d in deps_map.get(tid, []))
                and (not t.children or all(statuses.get(getattr(c, "id", None)) == "done" for c in t.children))
            ):
                candidate.append(t)
        # sort by priority then stable by id
        prio_order = {"high": 0, "medium": 1, "low": 2}
        candidate.sort(key=lambda x: (prio_order.get(getattr(x, "priority", "medium"), 1), str(getattr(x, "id", ""))))
        # limit via scheduler.ready_set stub
        return scheduler.ready_set(candidate)

    def split_or_implement(task: TaskSpec) -> Tuple[bool, List[TaskSpec]]:
        """
        Ask LLM whether to split or implement. If split, return (True, children).
        If implement, return (False, []).
        """
        # If task already has children declared, do NOT split again here.
        # The parent will be considered ready only after its children are done (see ready_tasks).
        if task.children:
            return False, []
        # Use LLM to decide
        decided_split = False
        new_children: List[TaskSpec] = []
        try:
            obj = engine.decide_split(task=task.model_dump(), idea=idea, constraints=constraints_obj)
            action = obj.get("action", "implement")
            if action == "split":
                def build_t(td: dict) -> TaskSpec:
                    return TaskSpec(
                        id=td.get("id"),
                        kind=td.get("kind", "composite"),
                        title=td.get("title", ""),
                        description=td.get("description", ""),
                        priority=td.get("priority", "medium"),
                        deps=td.get("deps", []),
                        inputs=td.get("inputs", {}),
                        outputs=td.get("outputs", {}),
                        children=[build_t(c) for c in td.get("children", [])],
                    )
                new_children = [build_t(x) for x in obj.get("children", [])]
                decided_split = True
        except Exception as e:
            raise RuntimeError(f"DSPy split/implement decision failed: {e}")
        return decided_split, new_children

    def _subtree_files(task: TaskSpec) -> List[str]:
        acc: set[str] = set()
        def walk(t: TaskSpec):
            if t.kind == "code:function":
                p = t.inputs.get("path") if isinstance(t.inputs, dict) else None
                if isinstance(p, str):
                    acc.add(p)
            for c in t.children:
                walk(c)
        walk(task)
        return sorted(acc)

    def _review_composite(task: TaskSpec) -> Tuple[bool, str]:
        # Validate assembled files touched by this composite's subtree; minimal auto-repair for imports
        touched = _subtree_files(task)
        for fpath in touched:
            impls = code_impls_by_path.get(fpath)
            if not impls:
                # Nothing to assemble yet
                continue
            meta = file_meta_by_path.get(fpath, {})
            exports = list(meta.get("exports", list(impls.keys())))
            language = meta.get("language") or infer_language_from_path(fpath)
            if language != "python":
                continue
            imports = list(meta.get("imports", []))
            entry = meta.get("entrypoint")
            signatures = meta.get("signatures", {})
            functions = {name: FunctionSpec(name=name, signature=signatures.get(name, f"def {name}() -> None"), returns="None", description="", examples=[], tests=[], deps=[]) for name in impls.keys()}
            fs = FileSpec(path=fpath, language=language, functions=functions, exports=exports, imports=imports, entrypoint=entry)
            assembled = assemble_python_file(fs, impls)
            out_file = paths.outputs / fpath
            write_code_file(out_file, assembled)

            # validations (subset; mirror end-of-run with import auto-repair)
            fs_schema = FileSpec.model_json_schema()
            ok_schema, msg_schema = jsonschema_validate(fs.model_dump(), fs_schema)
            if not ok_schema:
                return False, f"schema: {msg_schema}"
            ok, msg = basic_file_validation(fs)
            if not ok:
                return False, f"basic: {msg}"
            ok_code, msg_code = validate_assembled_python_file(fs, assembled)
            if not ok_code:
                return False, f"ast: {msg_code}"
            ok_style, msg_style = simple_style_check(assembled)
            ok_imports, msg_imports = import_safety_check(assembled, allowed=set(getattr(fs, "imports", []) or []))
            if not ok_imports:
                # try one-shot import auto-repair as in build_cmd
                violations: list[str] = []
                try:
                    m = re.search(r"\[(.*?)\]", msg_imports)
                    if m:
                        violations = ast.literal_eval("[" + m.group(1) + "]")
                except Exception:
                    violations = []
                additions = sorted({v.split(".")[0] for v in violations if isinstance(v, str) and v and v != "__future__"})
                if additions:
                    fs.imports = sorted(set((getattr(fs, "imports", []) or []) + additions))
                    # persist in meta so later stages are consistent
                    file_meta_by_path[fpath] = {
                        **meta,
                        "imports": list(fs.imports),
                        "exports": exports,
                        "language": language,
                        "entrypoint": entry,
                        "signatures": signatures,
                    }
                    # reassemble and revalidate once
                    assembled = assemble_python_file(fs, impls)
                    write_code_file(out_file, assembled)
                    ok_style, msg_style = simple_style_check(assembled)
                    ok_imports, msg_imports = import_safety_check(assembled, allowed=set(getattr(fs, "imports", []) or []))
            if not ok_imports:
                return False, f"imports: {msg_imports}; style={msg_style}"
            ok_integrity, msg_integrity = integrity_check(fs, assembled)
            if not ok_integrity:
                return False, f"integrity: {msg_integrity}"
        return True, "ok"

    progressed = True
    iters = 0
    max_iters = 10000
    while progressed and iters < max_iters:
        iters += 1
        progressed = False
        # renew lease heartbeat
        try:
            leases.renew(node_id, lease_id, ttl=lease_ttl)
        except Exception:
            pass
        rtasks = ready_tasks()
        if not rtasks:
            break
        # limit to max_children
        rtasks = rtasks[:max_children]

        for t in rtasks:
            tid = t.id  # type: ignore[assignment]
            statuses[tid] = "running"
            child_exec_id = f"task::{tid}"
            status.write(child_exec_id, "CREATED", parent_id=node_id)
            tlease = leases.grant(child_exec_id, ttl=child_ttl)
            status.write(child_exec_id, "LEASED", prev_state="CREATED", lease_id=tlease, parent_id=node_id)
            bus.emit("TASK_ASSIGNED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
            # alias for watchers expecting CHILD_* events
            bus.emit("CHILD_ASSIGNED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)

            # Decide split vs implement
            do_split, children = split_or_implement(t)
            if do_split and children:
                # Attach only truly new children and normalize ids
                pre_ids = {getattr(c, "id", None) for c in t.children}
                t.children.extend(children)
                _normalize_ids(t.children)
                # deduplicate by id while preserving order
                seen: set[str | None] = set()
                deduped: List[TaskSpec] = []
                for c in t.children:
                    cid = getattr(c, "id", None)
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(c)
                t.children = deduped
                # find newly added children by id
                new_children = [c for c in t.children if getattr(c, "id", None) not in pre_ids]
                # index newly added subtree
                _index(new_children, all_tasks)
                # Update deps/status maps
                for c in new_children:
                    statuses[c.id] = "pending"  # type: ignore[index]
                    deps_map[c.id] = list(c.deps)  # type: ignore[index]
                    parents[c.id] = tid  # type: ignore[index]
                # Persist updated plan
                atomic_write_json(task_plan_path, tp.model_dump())
                status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                added_count = len(new_children)
                bus.emit("TASK_SPLIT", parent=node_id, child=child_exec_id, task_id=tid, added=added_count, parent_id=parent_id)
                bus.emit("CHILD_SPLIT", parent=node_id, child=child_exec_id, task_id=tid, added=added_count, parent_id=parent_id)
                # Put parent back to pending so it can complete after children
                statuses[tid] = "pending"
                progressed = True
                continue

            # Implement leaf task
            try:
                if t.kind == "code:function":
                    # Determine effective language from inputs or file path
                    fname = t.inputs.get("name", "func")
                    fpath = t.inputs.get("path", "src/main.py")
                    lang_eff = t.inputs.get("language") or infer_language_from_path(fpath)
                    if lang_eff == "python":
                        # Implement as Python function snippet and aggregate for later assembly
                        code, meta = implement_code_function(t.inputs, constraints_obj, model=model)
                        cstem = f"{Path(fpath).stem}_{fname}_{hashlib.sha1(str(meta.get('signature','')).encode('utf-8')).hexdigest()[:10]}"
                        cpath = chunks_dir / f"{cstem}.py"
                        cpath.write_text(code, encoding="utf-8")
                        code_impls_by_path.setdefault(fpath, {})[fname] = code
                        file_meta_by_path.setdefault(fpath, {
                            "language": lang_eff,
                            "exports": list(meta.get("exports", [fname])),
                            "imports": list(t.inputs.get("allowed_imports", [])),
                            "entrypoint": t.inputs.get("entrypoint"),
                            "signatures": {fname: meta.get("signature")},
                        })
                        # merge metadata
                        fmeta = file_meta_by_path[fpath]
                        if fname not in fmeta["exports"]:
                            fmeta["exports"].append(fname)
                        fmeta["signatures"][fname] = meta.get("signature")
                        status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                        bus.emit("TASK_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                        bus.emit("CHILD_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                        statuses[tid] = "done"
                    else:
                        # Non-Python: treat as file-level generation using LLM, guided by task inputs
                        exports = list(t.inputs.get("exports", [fname]))
                        imports = list(t.inputs.get("allowed_imports", []))
                        entry = t.inputs.get("entrypoint")
                        # Build a minimal functions contract when provided
                        sig = t.inputs.get("signature")
                        functions = {fname: sig} if sig else {}
                        file_text = engine.generate_full_file(
                            idea=idea or "",
                            constraints=constraints_obj,
                            file=fpath,
                            language=lang_eff or "text",
                            exports=exports,
                            imports=imports,
                            entrypoint=entry,
                            functions=functions,
                        )
                        out_file = paths.outputs / fpath
                        write_code_file(out_file, file_text)
                        # record a generic chunk for traceability
                        cstem = f"{Path(fpath).stem}_{fname}_{hashlib.sha1((sig or fname).encode('utf-8')).hexdigest()[:10]}"
                        cpath = chunks_dir / f"{cstem}.txt"
                        cpath.write_text(file_text, encoding="utf-8")
                        # update meta for later validations/merges
                        file_meta_by_path[fpath] = {
                            "language": lang_eff,
                            "exports": exports,
                            "imports": imports,
                            "entrypoint": entry,
                            "signatures": functions,
                        }
                        status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                        bus.emit("FILE_GENERATED", parent=node_id, child=child_exec_id, path=fpath, language=(lang_eff or ""), parent_id=parent_id)
                        bus.emit("TASK_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                        bus.emit("CHILD_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                        statuses[tid] = "done"
                elif t.kind == "composite":
                    # If leaf-composite (no children), mark done. Otherwise, children are all done (due to readiness),
                    # so perform parent-level review and finalize based on validation.
                    if not t.children:
                        status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                        bus.emit("TASK_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                        bus.emit("CHILD_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                        statuses[tid] = "done"
                    else:
                        ok, reason = _review_composite(t)
                        if ok:
                            status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                            bus.emit("COMPOSITE_REVIEW_PASSED", parent=node_id, child=child_exec_id, task_id=tid, parent_id=parent_id)
                            bus.emit("TASK_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                            bus.emit("CHILD_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                            statuses[tid] = "done"
                        else:
                            statuses[tid] = "failed"
                            status.write(child_exec_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=reason)
                            bus.emit("COMPOSITE_REVIEW_FAILED", parent=node_id, child=child_exec_id, task_id=tid, reason=reason, parent_id=parent_id)
                            bus.emit("TASK_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=reason, parent_id=parent_id)
                            bus.emit("CHILD_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=reason, parent_id=parent_id)
                            if not keep_going:
                                status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason=reason, parent_id=parent_id)
                                bus.emit("NODE_FAILED", node_id=node_id, reason=reason, parent_id=parent_id)
                                raise typer.Exit(code=1)
                else:
                    # Unknown kinds: mark as done (extensible via future workers)
                    status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                    bus.emit("TASK_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                    bus.emit("CHILD_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                    statuses[tid] = "done"
            except Exception as e:
                statuses[tid] = "failed"
                status.write(child_exec_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=str(e))
                bus.emit("TASK_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=str(e), parent_id=parent_id)
                bus.emit("CHILD_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=str(e), parent_id=parent_id)
                if not keep_going:
                    status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason=str(e), parent_id=parent_id)
                    bus.emit("NODE_FAILED", node_id=node_id, reason=str(e), parent_id=parent_id)
                    raise typer.Exit(code=1)

            progressed = True
            # lease heartbeat
            try:
                leases.renew(node_id, lease_id, ttl=lease_ttl)
                time.sleep(0.01)
            except Exception:
                pass

    # After tasks, check if any pending remain
    remaining = [tid for tid, st in statuses.items() if st not in ("done",)]
    if remaining:
        console.print(f"[yellow]Tasks left pending or blocked[/yellow]: {remaining[:10]} ...")

    # Assemble aggregated code files and validate (python only)
    any_failed = False
    out_files: List[str] = []
    for fpath, impls in code_impls_by_path.items():
        meta = file_meta_by_path.get(fpath, {})
        exports = list(meta.get("exports", list(impls.keys())))
        language = meta.get("language") or infer_language_from_path(fpath)
        if language != "python":
            # Skip non-Python assembly; tasks executor currently assembles Python only
            continue
        imports = list(meta.get("imports", []))
        entry = meta.get("entrypoint")
        signatures = meta.get("signatures", {})
        functions = {name: FunctionSpec(name=name, signature=signatures.get(name, f"def {name}() -> None"), returns="None", description="", examples=[], tests=[], deps=[]) for name in impls.keys()}
        fs = FileSpec(path=fpath, language=language, functions=functions, exports=exports, imports=imports, entrypoint=entry)
        assembled = assemble_python_file(fs, impls)
        out_file = paths.outputs / fpath
        write_code_file(out_file, assembled)
        out_files.append(str(out_file.relative_to(paths.outputs)))

        # validations
        fs_schema = FileSpec.model_json_schema()
        ok_schema, msg_schema = jsonschema_validate(fs.model_dump(), fs_schema)
        if not ok_schema:
            console.print(f"[red]Spec schema validation failed:[/red] {msg_schema}")
            any_failed = True
            continue
        ok, msg = basic_file_validation(fs)
        if not ok:
            console.print(f"[red]Validation failed:[/red] {msg}")
            any_failed = True
            continue
        ok_code, msg_code = validate_assembled_python_file(fs, assembled)
        if not ok_code:
            console.print(f"[red]Code validation failed:[/red] {msg_code}")
            any_failed = True
            continue
        ok_style, msg_style = simple_style_check(assembled)
        ok_imports, msg_imports = import_safety_check(assembled, allowed=set(getattr(fs, "imports", []) or []))
        if not ok_imports:
            console.print(f"[red]Import safety failed[/red]: imports={msg_imports}; style={msg_style}")
            any_failed = True
            continue
        ok_integrity, msg_integrity = integrity_check(fs, assembled)
        if not ok_integrity:
            console.print(f"[red]Integrity failed[/red]: {msg_integrity}")
            any_failed = True
            continue
        # meta & report
        _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
        write_file_meta(paths.artifacts / f"file_{_fid}.json", fs.model_dump())
        validations_report = {
            "file": str((paths.outputs / fs.path)),
            "schema": {"ok": ok_schema, "msg": msg_schema},
            "basic": {"ok": ok, "msg": msg},
            "ast": {"ok": ok_code, "msg": msg_code},
            "style": {"ok": ok_style, "msg": msg_style},
            "imports": {"ok": ok_imports, "msg": msg_imports},
            "integrity": {"ok": ok_integrity, "msg": msg_integrity},
        }
        vpath = paths.validations / f"report_{_fid}.json"
        vpath.write_text(json.dumps(validations_report, indent=2), encoding="utf-8")

    # Validate non-Python outputs (language-agnostic checks similar to build_cmd)
    for fpath, meta in file_meta_by_path.items():
        language = meta.get("language") or infer_language_from_path(fpath)
        if language == "python":
            continue
        out_path = paths.outputs / fpath
        assembled = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if not out_path.exists() or not assembled.strip():
            console.print(f"[red]Non-Python output missing or empty[/red]: {fpath}")
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="non_python_output", message="output_missing_or_empty", file=fpath, parent_id=parent_id)
            any_failed = True
            continue

        exports = list(meta.get("exports", []) or [])
        if exports:
            try:
                verify = engine.verify_exports_in_text(
                    file=fpath,
                    language=(language or ""),
                    exports=exports,
                    text=assembled,
                )
            except Exception:
                verify = {"ok": False, "missing": exports}
            if not verify.get("ok", False):
                missing = verify.get("missing", exports)
                console.print(f"[red]Non-Python exports missing[/red] in {fpath}: {missing}")
                bus.emit("VALIDATION_FAILED", node_id=node_id, kind="exports", message=f"missing_exports: {missing}", file=fpath, parent_id=parent_id)
                any_failed = True
                continue

        # Track in project structure on success
        try:
            out_files.append(str(out_path.relative_to(paths.outputs)))
            bus.emit("VALIDATION_PASSED", node_id=node_id, file=fpath, report=None, parent_id=parent_id)
        except Exception:
            # Non-fatal
            pass

    atomic_write_json(paths.outputs / "project_structure.json", {"files": out_files})

    # If any task failed or validations failed, mark node failed and exit non-zero
    any_task_failed = any(st == "failed" for st in statuses.values())
    if any_failed or any_task_failed:
        reason = "validation failed" if any_failed else "one or more tasks failed"
        status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason=reason, parent_id=parent_id)
        bus.emit("NODE_FAILED", node_id=node_id, reason=reason, parent_id=parent_id)
        raise typer.Exit(code=1)

    status.write(node_id, "DONE", prev_state="LEASED", parent_id=parent_id)
    bus.emit("NODE_DONE", node_id=node_id, parent_id=parent_id)
    sep("TASKS DONE")
    if out_files:
        console.print(f"[green]Tasks build complete[/green]. Outputs: {out_files}")
    else:
        console.print("[yellow]Tasks execution complete with no assembled outputs yet (no code:function leaves).[/yellow]")
