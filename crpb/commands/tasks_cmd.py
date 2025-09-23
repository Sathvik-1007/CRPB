from __future__ import annotations
import hashlib
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Annotated, Any
import typer
from rich.console import Console

from ..config import resolve_run_dir, make_paths
from ..eventbus import EventBus
from ..status import NodeStatus
from ..leases import Leases
from ..utils.fs import ensure_parent, atomic_write_json, lock_file
from ..utils.ui import sep
from ..scheduler import Scheduler
from ..planner import generate_task_plan
from ..specs import TaskSpec, TaskPlan, CodeSpec, CodeSpecFile
from ..agents.dspy_engine import DspyEngine
from ..utils.artifacts import ArtifactRegistry
from ..validator import (
    validate_leaf_readiness,
    validate_artifact_gating,
    infer_language_from_path,
)
from ..llm_config import load_selection, require_env_vars

app = typer.Typer(help="Execute a hierarchical TaskPlan with recursive LLM-driven split-or-implement orchestration")
console = Console()


# --- Local helper replacing deprecated aggregator function ---
def write_code_file(path: Path, text: str) -> None:
    """Safely write text content to a file, ensuring parent directories exist."""
    ensure_parent(path)
    # Use a sidecar lock file to coordinate concurrent writers across processes
    lock_path = path.parent / (path.name + ".lock")
    with lock_file(lock_path):
        path.write_text(text, encoding="utf-8")


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


# ---------------------- BuildFromTaskPlan Adapter ----------------------
class BuildFromTaskPlan:
    """Adapter to materialize artifacts from TaskPlan leaves.
    Enforces leaf readiness (explicit or inferable language, path, entry/exports),
    generates full files via DSPy, registers and validates produced artifacts.
    """

    def __init__(
        self,
        *,
        engine: DspyEngine,
        registry: ArtifactRegistry,
        paths,
        idea: str,
        constraints: dict,
        file_meta_by_path: Dict[str, Dict],
        codespec: Optional[CodeSpec] = None,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.paths = paths
        self.idea = idea
        self.constraints = constraints
        self.file_meta_by_path = file_meta_by_path
        self.codespec = codespec or CodeSpec()
        self._file_specs_by_path: Dict[str, CodeSpecFile] = {}

    def _normalize_artifact_ref(self, x):
        if isinstance(x, str):
            return {"id": x}
        return x if isinstance(x, dict) else None

    def _produces(self, t: TaskSpec) -> List[dict]:
        try:
            raw = t.outputs.get("produces") if isinstance(t.outputs, dict) else []
            if not isinstance(raw, list):
                raw = [] if raw is None else [raw]
            refs = []
            for r in raw:
                nr = self._normalize_artifact_ref(r)
                if isinstance(nr, dict):
                    refs.append(nr)
            return refs
        except Exception:
            return []

    def _validate_artifact_ref(self, ref: dict) -> Tuple[bool, dict]:
        try:
            art_id = str(ref.get("id") or "")
            kind = ref.get("kind")
            # Resolve path and load content
            p = self.registry.resolve_path(ref, base_dir=self.paths.outputs)
            text = ""
            jdata = None
            if p is not None and p.exists():
                # Always try JSON first; fall back to text
                raw = p.read_text(encoding="utf-8")
                try:
                    jdata = json.loads(raw)
                except Exception:
                    if p.suffix.lower() == ".json" or str(kind).lower() in {"json", "openapi", "schema"}:
                        try:
                            with p.open("r", encoding="utf-8") as fh:
                                jdata = json.load(fh)
                        except Exception:
                            jdata = None
                    if jdata is None:
                        text = raw
            result = self.engine.validate_artifact(
                artifact_id=art_id or None,
                kind=kind if isinstance(kind, str) else None,
                path=str(p) if p is not None else (ref.get("path") or None),
                text=text,
                json_content=jdata,
                idea=self.idea or "",
                constraints=self.constraints,
            )
            ok = bool(result.get("ok", False))
            if art_id:
                self.registry.set_validation(art_id, ok=ok, report=result)
            else:
                if p is not None:
                    self.registry.set_validation_for_path(p, ok=ok, report=result)
            return ok, result
        except Exception as e:
            return False, {"error": str(e), "issues": [f"validator_error: {e}"]}

    def build_code_function(self, t: TaskSpec) -> Tuple[bool, str, Dict[str, str]]:
        # Contract readiness
        ok_r, msg_r, meta = validate_leaf_readiness(t)
        if not ok_r:
            return False, msg_r, {}

        fpath = str(meta.get("path"))
        fname = str(meta.get("name"))
        lang_eff = str(meta.get("language") or "")
        exports = list(meta.get("exports") or [])
        imports = list(meta.get("imports") or [])
        entry = meta.get("entrypoint")
        sig = meta.get("signature")

        # Accumulate known function signatures for this file
        prev = self.file_meta_by_path.get(fpath, {})
        prev_sigs: Dict[str, str] = dict(prev.get("signatures", {}))
        if isinstance(sig, str) and sig:
            prev_sigs[fname] = sig
        functions = prev_sigs

        # Build node-scoped context from CodeSpec only (Spec-first, language-agnostic)
        c_ext = dict(self.constraints or {})
        c_ext.setdefault("side_context", {})
        cs_file = self._file_specs_by_path.get(fpath)
        cs_raw = None
        purpose = ""
        description = ""
        classes_meta: Dict[str, Any] = {}
        constants_meta: Dict[str, Any] = {}
        content_meta: Optional[str] = None
        if cs_file is not None:
            try:
                cs_raw = cs_file.model_dump(exclude_none=True)
            except Exception:
                cs_raw = None
            purpose = getattr(cs_file, "purpose", "") or ""
            description = getattr(cs_file, "description", "") or ""
            classes_meta = dict(getattr(cs_file, "classes", {}) or {})
            constants_meta = dict(getattr(cs_file, "constants", {}) or {})
            content_meta = getattr(cs_file, "content", None)
        c_ext["side_context"].update({
            "codespec_file": cs_raw or {},
        })

        file_text = self.engine.generate_full_file(
            idea=self.idea or "",
            constraints=c_ext,
            file=fpath,
            language=lang_eff,
            exports=exports,
            imports=imports,
            entrypoint=entry,
            functions=functions,
            # Pass rich Codespec metadata for better generation
            purpose=purpose,
            description=description,
            classes=classes_meta,
            constants=constants_meta,
            content=content_meta,
        )
        out_file = self.paths.outputs / fpath
        write_code_file(out_file, file_text)

        # Trace chunk for provenance
        chunks_dir = self.paths.artifacts / "chunks"
        ensure_parent(chunks_dir / "_.txt")
        cstem = f"{Path(fpath).stem}_{fname}_{hashlib.sha1((str(sig) or fname).encode('utf-8')).hexdigest()[:10]}"
        cfile = chunks_dir / f"{cstem}.txt"
        # Lock chunk file to avoid interleaved writes across concurrent builders
        lock_path = cfile.parent / (cfile.name + ".lock")
        with lock_file(lock_path):
            cfile.write_text(file_text, encoding="utf-8")

        # Update CodeSpec with generated file information
        self._update_codespec_file(fpath, lang_eff, exports, imports, entry, functions)

        # Update meta for later validations/reviews (merge with previous)
        prev_meta = self.file_meta_by_path.get(fpath, {})
        merged_exports = sorted(set(list(prev_meta.get("exports", [])) + exports))
        merged_imports = sorted(set(list(prev_meta.get("imports", [])) + imports))
        merged_sigs: Dict[str, str] = dict(prev_meta.get("signatures", {}))
        merged_sigs.update(functions)
        self.file_meta_by_path[fpath] = {
            "language": lang_eff,
            "exports": merged_exports,
            "imports": merged_imports,
            "entrypoint": entry or prev_meta.get("entrypoint"),
            "signatures": merged_sigs,
        }

        # Register produced artifacts and validate
        prods = self._produces(t)
        if prods:
            fixed = []
            for r in prods:
                r = dict(r)
                r.setdefault("path", fpath)
                fixed.append(r)
            self.registry.register(fixed, base_dir=self.paths.outputs)
            for r in fixed:
                self._validate_artifact_ref(r)

        return True, "ok", {"path": fpath, "language": lang_eff}

        
    def _update_codespec_file(self, path: str, language: str, exports: List[str], 
                             imports: List[str], entrypoint: Optional[str], 
                             functions: Dict[str, str]) -> None:
        """Update CodeSpec file with generated information."""
        if path in self._file_specs_by_path:
            # Update existing file spec
            file_spec = self._file_specs_by_path[path]
            if language:
                file_spec.language = language
            if exports:
                file_spec.exports = exports
            if imports:
                file_spec.imports = imports
            if entrypoint:
                file_spec.entrypoint = entrypoint
            
            # Update functions with signatures
            if functions:
                if not file_spec.functions:
                    file_spec.functions = {}
                for name, signature in functions.items():
                    if name not in file_spec.functions:
                        file_spec.functions[name] = {}
                    file_spec.functions[name]['signature'] = signature
        else:
            # Create new file spec
            file_spec = CodeSpecFile(
                path=path,
                language=language,
                imports=imports,
                exports=exports,
                entrypoint=entrypoint,
                functions={name: {'signature': sig} for name, sig in functions.items()}
            )
            self.codespec.files.append(file_spec)
            self._file_specs_by_path[path] = file_spec
            
    def save_codespec(self) -> None:
        """Save the progressive CodeSpec to file."""
        codespec_path = self.paths.plan / "codespec.json"
        ensure_parent(codespec_path)
        
        # Convert to dict for JSON serialization
        codespec_dict = {
            'files': []
        }
        
        for file_spec in self.codespec.files:
            file_dict = {
                'path': file_spec.path,
                'purpose': file_spec.purpose or '',
                'language': file_spec.language,
                'imports': file_spec.imports or [],
                'exports': file_spec.exports or [],
                'functions': file_spec.functions or {},
                'classes': file_spec.classes or {},
                'constants': file_spec.constants or {},
                'entrypoint': file_spec.entrypoint,
                'content': file_spec.content or '',
                'description': file_spec.description or '',
                'exports_detail': file_spec.exports_detail or []
            }
            codespec_dict['files'].append(file_dict)
        
        atomic_write_json(codespec_path, codespec_dict)


@app.callback(invoke_without_command=True)
def main(
    idea: Annotated[Optional[str], typer.Option("--idea", help="Idea; if omitted uses run/plan/idea.json or unified_plan.json")] = None,
    constraints: Annotated[Optional[str], typer.Option("--constraints", help="Optional JSON string of constraints to steer splitting/implementation")] = None,
    run_dir: Annotated[Optional[str], typer.Option("--run-dir", help="Base runs folder")] = None,
    run: Annotated[str, typer.Option("--run", help="run_<ts> | latest | new | name")] = "new",
    model: Annotated[Optional[str], typer.Option("--model", help="Override model for LLM steps")] = None,
    max_children: Annotated[int, typer.Option("--max-children", min=1, help="Max concurrent child tasks the scheduler will select")] = 2,
    node_id: Annotated[Optional[str], typer.Option("--node-id", help="Deterministic node id for this executor")] = None,
    parent_id: Annotated[Optional[str], typer.Option("--parent-id", help="Parent node id for orchestration tracking")] = None,
    lease_ttl: Annotated[int, typer.Option("--lease-ttl", min=30, help="TTL seconds for executor lease")] = 180,
    child_ttl: Annotated[int, typer.Option("--child-ttl", min=30, help="TTL seconds for child leases")] = 120,
    keep_going: Annotated[bool, typer.Option("--keep-going/--fail-fast", help="On task failures, continue other tasks instead of aborting")] = True,
    allow_amend: Annotated[bool, typer.Option("--amend/--no-amend", help="Allow dynamic LLM-driven plan amendments when progress stalls")] = True,
    amend_max_rounds: Annotated[int, typer.Option("--amend-rounds", min=0, help="Max amendment rounds when blocked")] = 3,
    require_artifact_valid: Annotated[bool, typer.Option("--require-artifact-valid/--allow-unknown-artifact", help="Gate tasks on consumed artifacts being validated OK")] = True,
    max_iters: Annotated[int, typer.Option("--max-iters", min=1, help="Max scheduler loop iterations before aborting")] = 10000,
    max_seconds: Annotated[Optional[int], typer.Option("--max-seconds", min=1, help="Optional wall-clock timeout in seconds (fail when exceeded)")] = None,
):
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    sep("TASKS START")
    # Initialize bus early so we can emit diagnostics during early-stage failures
    bus = EventBus(paths.logs / "events.jsonl")

    # Fail fast if provider-specific requirements are not satisfied (no hardcoding)
    sel = load_selection()
    if not sel:
        console.print("[red]No LLM selection found.[/red] Use `python -m crpb llm choose` to select a provider and model before running tasks.")
        raise typer.Exit(code=2)
    ok, missing = require_env_vars(sel)
    if not ok:
        miss = ", ".join(missing)
        console.print(f"[red]LLM configuration incomplete for tasks: provider={sel.provider} missing: {miss}[/red]")
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
            except Exception as e:
                bus.emit("CONSTRAINTS_PARSE_ERROR", node_id="tasks", file=str(cfile), error=str(e), parent_id=parent_id)
                constraints_obj = {}

    # Idea resolution
    if idea is None:
        idea_file = paths.plan / "idea.json"
        if idea_file.exists():
            try:
                idea = json.loads(idea_file.read_text(encoding="utf-8")).get("idea", "")
            except Exception as e:
                bus.emit("IDEA_PARSE_FAILED", node_id="tasks", file=str(idea_file), error=str(e), parent_id=parent_id)
                idea = ""
        else:
            idea = ""

    status = NodeStatus(paths.graph / "node_status.jsonl")
    leases = Leases(paths.graph / "leases.json")
    scheduler = Scheduler(max_parallel_children=max_children)
    # Initialize DSPy engine once for all LLM-backed decisions
    engine = DspyEngine(model=model)
    # Initialize Artifact Registry
    registry = ArtifactRegistry(paths.artifacts / "index.json")

    # Wall-clock timeout tracking
    start_wall = time.time()
    deadline = (start_wall + max_seconds) if isinstance(max_seconds, int) else None

    # TaskPlan node-level refine rounds (clarification loops)
    # Precedence: constraints.taskplan_refine_max_rounds > env CRPB_TASKPLAN_REFINE_MAX_ROUNDS > default 2
    def _clamp(n: int) -> int:
        return max(0, min(5, n))

    taskplan_refine_max_rounds = 2
    try:
        env_val = int(os.environ.get("CRPB_TASKPLAN_REFINE_MAX_ROUNDS", str(taskplan_refine_max_rounds)))
        taskplan_refine_max_rounds = _clamp(env_val)
    except Exception as e:
        bus.emit("ENV_PARSE_INVALID", node_id=node_id, var="CRPB_TASKPLAN_REFINE_MAX_ROUNDS", error=str(e), parent_id=parent_id)
        taskplan_refine_max_rounds = _clamp(taskplan_refine_max_rounds)
    if isinstance(constraints_obj, dict):
        try:
            mr = int(constraints_obj.get("taskplan_refine_max_rounds", taskplan_refine_max_rounds))
            taskplan_refine_max_rounds = _clamp(mr)
        except Exception as e:
            bus.emit("CONSTRAINTS_PARSE_INVALID", node_id=node_id, field="taskplan_refine_max_rounds", error=str(e), parent_id=parent_id)
            taskplan_refine_max_rounds = _clamp(taskplan_refine_max_rounds)

    # Deterministic defaults
    node_id = node_id or f"tasks::{run_path.name}"
    parent_id = parent_id or f"root::{run_path.name}"
    status.write(node_id, "CREATED", parent_id=parent_id)
    lease_id = leases.grant(node_id, ttl=lease_ttl)
    status.write(node_id, "LEASED", prev_state="CREATED", lease_id=lease_id, parent_id=parent_id)
    bus.emit("NODE_CREATED", node_id=node_id, parent_id=parent_id)
    bus.emit("LEASE_GRANTED", node_id=node_id, lease_id=lease_id, parent_id=parent_id)

    # Load TaskPlan from plan directory (prefer plan.json; fallback to legacy task_plan.json)
    plan_path = paths.plan / "plan.json"
    legacy_task_plan_path = paths.plan / "task_plan.json"
    if plan_path.exists():
        try:
            tp_obj = json.loads(plan_path.read_text(encoding="utf-8"))
            tp = TaskPlan.model_validate(tp_obj)
        except Exception as e:
            console.print(f"[red]Failed to load plan.json: {e}[/red]")
            bus.emit("PLAN_READ_FAILED", node_id=node_id, file=str(plan_path), error=str(e), parent_id=parent_id)
            raise typer.Exit(code=2)
    elif legacy_task_plan_path.exists():
        try:
            tp_obj = json.loads(legacy_task_plan_path.read_text(encoding="utf-8"))
            tp = TaskPlan.model_validate(tp_obj)
            bus.emit("PLAN_LEGACY_USED", node_id=node_id, file=str(legacy_task_plan_path), parent_id=parent_id)
        except Exception as e:
            console.print(f"[red]Failed to load task_plan.json: {e}[/red]")
            bus.emit("PLAN_LEGACY_READ_FAILED", node_id=node_id, file=str(legacy_task_plan_path), error=str(e), parent_id=parent_id)
            raise typer.Exit(code=2)
    else:
        if not idea:
            console.print("[red]No plan.json found and no idea provided[/red]")
            raise typer.Exit(code=2)
        tp = generate_task_plan(idea, constraints_obj, use_llm=True)
        atomic_write_json(plan_path, tp.model_dump(exclude_none=True))

    _normalize_ids(tp.tasks)
    # Immediately persist normalized ids back to plan.json to keep a single source of truth
    try:
        atomic_write_json(plan_path, tp.model_dump(exclude_none=True))
    except Exception as e:
        bus.emit("PLAN_WRITE_FAILED", node_id=node_id, file=str(plan_path), error=str(e), parent_id=parent_id)

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

    # File metadata cache for code leaves (language-agnostic)
    file_meta_by_path: Dict[str, Dict] = {}

    # Build adapter with CodeSpec support
    adapter = BuildFromTaskPlan(
        engine=engine,
        registry=registry,
        paths=paths,
        idea=idea or "",
        constraints=constraints_obj,
        file_meta_by_path=file_meta_by_path,
        codespec=CodeSpec(),
    )
    
    # Seed CodeSpec from plan/codespec.json when present (Spec-first)
    try:
        cs_path = paths.plan / "codespec.json"
        if cs_path.exists():
            cs_obj = json.loads(cs_path.read_text(encoding="utf-8"))
            files = cs_obj.get("files", []) if isinstance(cs_obj, dict) else []
            seeded: List[CodeSpecFile] = []
            for f in files:
                if not isinstance(f, dict):
                    continue
                p = f.get("path")
                if not isinstance(p, str) or not p.strip():
                    continue
                # Normalize functions map to dict[name]->dict
                fns = {}
                rawf = f.get("functions") or {}
                if isinstance(rawf, dict):
                    for k, v in rawf.items():
                        if not isinstance(k, str) or not k.strip():
                            continue
                        if isinstance(v, str):
                            fns[k.strip()] = {"signature": v}
                        elif isinstance(v, dict):
                            fns[k.strip()] = dict(v)
                # Normalize classes/constants as dicts
                cls = dict(f.get("classes") or {}) if isinstance(f.get("classes"), dict) else {}
                consts = dict(f.get("constants") or {}) if isinstance(f.get("constants"), dict) else {}
                # Normalize exports_detail to list of dicts
                ed = f.get("exports_detail") or []
                exports_detail: List[Dict[str, Any]] = []
                if isinstance(ed, list):
                    for item in ed:
                        if isinstance(item, dict):
                            exports_detail.append(item)
                cs_file = CodeSpecFile(
                    path=p.strip(),
                    purpose=str(f.get("purpose") or ""),
                    description=str(f.get("description") or ""),
                    language=f.get("language"),
                    imports=list(f.get("imports") or []),
                    exports=list(f.get("exports") or []),
                    functions=fns,
                    classes=cls,
                    constants=consts,
                    entrypoint=f.get("entrypoint"),
                    content=f.get("content"),
                    exports_detail=exports_detail,
                )
                seeded.append(cs_file)
                adapter._file_specs_by_path[cs_file.path] = cs_file
                adapter.codespec = CodeSpec(files=seeded)
    except Exception as e:
        # Non-fatal: continue without initial CodeSpec but record diagnostic
        bus.emit("CODESPEC_SEED_FAILED", node_id=node_id, file=str(paths.plan / "codespec.json"), error=str(e), parent_id=parent_id)

    # Artifact validation helpers (Phase 4)
    def _validate_artifact_ref(ref: dict) -> Tuple[bool, dict]:
        """Validate a single artifact ref using JSON parsing when applicable and LLM for generality.
        Returns (ok, report). Also persists validation into the registry when id present.
        """
        try:
            art_id = str(ref.get("id") or "")
            kind = ref.get("kind")
            # Resolve path and load content
            p = registry.resolve_path(ref, base_dir=paths.outputs)
            text = ""
            jdata = None
            if p is not None and p.exists():
                # Always try JSON first; fall back to text
                raw = p.read_text(encoding="utf-8")
                try:
                    jdata = json.loads(raw)
                except Exception:
                    # If suffix/kind strongly indicate JSON, try json.load as double-check
                    if p.suffix.lower() == ".json" or str(kind).lower() in {"json", "openapi", "schema"}:
                        try:
                            with p.open("r", encoding="utf-8") as fh:
                                jdata = json.load(fh)
                        except Exception:
                            jdata = None
                    if jdata is None:
                        text = raw
            # LLM validation (language-neutral)
            result = engine.validate_artifact(
                artifact_id=art_id or None,
                kind=kind if isinstance(kind, str) else None,
                path=str(p) if p is not None else (ref.get("path") or None),
                text=text,
                json_content=jdata,
                idea=idea or "",
                constraints=constraints_obj,
            )
            ok = bool(result.get("ok", False))
            if art_id:
                registry.set_validation(art_id, ok=ok, report=result)
            else:
                # Persist validation keyed by path when no id is available
                if p is not None:
                    registry.set_validation_for_path(p, ok=ok, report=result)
            return ok, result
        except Exception as e:
            bus.emit("VALIDATOR_ERROR", node_id=node_id, error=str(e), ref=str(ref), parent_id=parent_id)
            return False, {"ok": False, "issues": [f"validator_error: {e}"]}

    def _ensure_consumes_ready(consumes: List[dict]) -> bool:
        if not consumes:
            return True
        base = paths.outputs
        for ref in consumes:
            if not registry.exists(ref, base_dir=base):
                return False
            if require_artifact_valid:
                v = registry.get_validation(ref)
                if not v or not v.get("ok", False):
                    ok, _rep = _validate_artifact_ref(ref)
                    if not ok:
                        return False
        return True

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
                # Artifact gating: all consumed artifacts must exist
                consumes = _consumes(t)
                if consumes and not _ensure_consumes_ready(consumes):
                    # gated; skip until artifacts available and valid
                    continue
                candidate.append(t)
        # stable sort by id only (no priority usage)
        candidate.sort(key=lambda x: str(getattr(x, "id", "")))
        # limit via scheduler.ready_set stub
        return scheduler.ready_set(candidate)

    def _idle_breakdown() -> Dict[str, int]:
        """Classify why pending tasks are not ready without triggering validations."""
        totals = {
            "total_pending": 0,
            "deps_blocked": 0,
            "children_waiting": 0,
            "artifact_gated": 0,
        }
        for tid, t in all_tasks.items():
            if statuses.get(tid) != "pending":
                continue
            totals["total_pending"] += 1
            # deps blocked
            if any(statuses.get(d) != "done" for d in deps_map.get(tid, [])):
                totals["deps_blocked"] += 1
            # children waiting
            if t.children and any(statuses.get(getattr(c, "id", None)) != "done" for c in t.children):
                totals["children_waiting"] += 1
            # artifact gating (cheap check; do not validate here)
            cons = _consumes(t)
            if cons:
                gated = False
                for r in cons:
                    try:
                        if not registry.exists(r, base_dir=paths.outputs):
                            gated = True
                            break
                        if require_artifact_valid:
                            v = registry.get_validation(r)
                            if not v or not v.get("ok", False):
                                gated = True
                                break
                    except Exception as e:
                        bus.emit("ARTIFACT_GATE_CHECK_FAILED", node_id=node_id, ref=str(r), error=str(e), parent_id=parent_id)
                        gated = True
                        break
                if gated:
                    totals["artifact_gated"] += 1
        return totals

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
            obj = engine.decide_split(task=task.model_dump(exclude_none=True), idea=idea, constraints=constraints_obj)
            action = obj.get("action", "implement")
            if action == "split":
                def build_t(td: dict) -> TaskSpec:
                    return TaskSpec(
                        id=td.get("id"),
                        kind=td.get("kind", "composite"),
                        title=td.get("title", ""),
                        description=td.get("description", ""),
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

    # Amend/replan (Phase 5)
    def amend_once() -> int:
        try:
            edits_obj = engine.amend_task_plan(
                # Provide current TaskPlan object directly
                current_plan=tp.model_dump(exclude_none=True),
                statuses=statuses,
                artifacts=registry.list(),
                idea=idea or "",
                constraints=constraints_obj,
            )
        except Exception as e:
            bus.emit("AMEND_ENGINE_FAILED", node_id=node_id, error=str(e), parent_id=parent_id)
            return 0
        edits = edits_obj.get("edits", []) if isinstance(edits_obj, dict) else []
        applied = 0

        def find_task(tid: str) -> TaskSpec | None:
            return all_tasks.get(tid)

        def build_task(td: dict) -> TaskSpec:
            return TaskSpec(
                id=td.get("id"),
                kind=td.get("kind", "composite"),
                title=td.get("title", ""),
                description=td.get("description", ""),
                deps=td.get("deps", []),
                inputs=td.get("inputs", {}),
                outputs=td.get("outputs", {}),
                children=[build_task(c) for c in td.get("children", [])],
            )

        for ed in edits:
            if not isinstance(ed, dict):
                continue
            op = ed.get("op")
            if op == "add_child":
                parent_id = ed.get("parent_id")
                td = ed.get("task") or {}
                parent = find_task(parent_id) if isinstance(parent_id, str) else None
                if parent is None:
                    continue
                child = build_task(td)
                parent.children.append(child)
                _normalize_ids(parent.children)
                _index([child], all_tasks)
                statuses[child.id] = "pending"  # type: ignore[index]
                deps_map[child.id] = list(child.deps)  # type: ignore[index]
                parents[child.id] = parent.id  # type: ignore[index]
                applied += 1
            elif op == "update_task":
                tid = ed.get("id")
                setv = ed.get("set") or {}
                t = find_task(tid) if isinstance(tid, str) else None
                if t is None:
                    continue
                # Update selected fields
                for key in ["kind", "title", "description"]:
                    if key in setv:
                        setattr(t, key, setv[key])
                if "deps" in setv and isinstance(setv["deps"], list):
                    t.deps = list(setv["deps"])  # type: ignore[assignment]
                    deps_map[t.id] = list(t.deps)  # type: ignore[index]
                if "inputs" in setv and isinstance(setv["inputs"], dict):
                    t.inputs = dict(setv["inputs"])  # type: ignore[assignment]
                if "outputs" in setv and isinstance(setv["outputs"], dict):
                    t.outputs = dict(setv["outputs"])  # type: ignore[assignment]
                applied += 1
            elif op == "add_dep":
                tid = ed.get("id")
                dep_id = ed.get("dep_id")
                t = find_task(tid) if isinstance(tid, str) else None
                if t is None or not isinstance(dep_id, str):
                    continue
                if dep_id not in t.deps:
                    t.deps.append(dep_id)
                    deps_map[t.id] = list(t.deps)  # type: ignore[index]
                    applied += 1
            elif op == "rewire_artifacts":
                tid = ed.get("id")
                t = find_task(tid) if isinstance(tid, str) else None
                if t is None:
                    continue
                cons = ed.get("consumes")
                prods = ed.get("produces")
                if isinstance(cons, list):
                    t.inputs["consumes"] = cons  # type: ignore[index]
                if isinstance(prods, list):
                    t.outputs["produces"] = prods  # type: ignore[index]
                applied += 1
        if applied:
            atomic_write_json(task_plan_path, tp.model_dump(exclude_none=True))
        return applied

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

    def _files_map(paths_obj, file_list: List[str]) -> Dict[str, str]:
        fm: Dict[str, str] = {}
        base = paths_obj.outputs
        for fp in file_list:
            out = base / fp
            if out.exists():
                try:
                    fm[fp] = out.read_text(encoding="utf-8")
                except Exception as e:
                    bus.emit("READ_OUTPUT_FAILED", node_id=node_id, file=fp, error=str(e), parent_id=parent_id)
                    fm[fp] = ""
        return fm

    def _load_plan_obj() -> dict:
        """Return a compatibility plan object derived from codespec.json.
        Shape: {"modules":[{"name":"codespec","files":[{path,language,exports,imports,entrypoint,functions}]}]}
        """
        try:
            cs_path = paths.plan / "codespec.json"
            if cs_path.exists():
                cs_obj = json.loads(cs_path.read_text(encoding="utf-8"))
                files = []
                for item in (cs_obj.get("files", []) or []):
                    if not isinstance(item, dict):
                        continue
                    p = item.get("path")
                    if isinstance(p, str) and p.strip():
                        files.append({
                            "path": p,
                            "language": item.get("language"),
                            "exports": item.get("exports", []),
                            "imports": item.get("imports", []),
                            "entrypoint": item.get("entrypoint"),
                            "functions": item.get("functions", {}),
                        })
                return {"modules": [{"name": "codespec", "files": files}]}
        except Exception as e:
            bus.emit("CODESPEC_COMPAT_PLAN_FAILED", node_id=node_id, error=str(e), parent_id=parent_id)
        return {}

    def _collect_file_specs(plan_obj: dict, files: List[str]) -> Dict[str, dict]:
        specs: Dict[str, dict] = {}
        try:
            wanted = set(files)
            for m in plan_obj.get("modules", []) or []:
                for f in m.get("files", []) or []:
                    p = f.get("path")
                    if isinstance(p, str) and p in wanted:
                        specs[p] = {
                            "language": f.get("language"),
                            "exports": f.get("exports", []),
                            "imports": f.get("imports", []),
                            "entrypoint": f.get("entrypoint"),
                            "functions": f.get("functions", {}),
                        }
        except Exception as e:
            bus.emit("FILE_SPECS_COLLECT_FAILED", node_id=node_id, error=str(e), parent_id=parent_id)
        return specs

    def _project_validate_gate(task: TaskSpec) -> Tuple[bool, str]:
        """Run project-level validator over the touched subtree and gate completion."""
        touched = _subtree_files(task)
        fmap = _files_map(paths, touched)
        plan_obj = _load_plan_obj()
        fspecs = _collect_file_specs(plan_obj, list(fmap.keys()))
        try:
            report = engine.project_validate(
                idea=idea or "",
                constraints=constraints_obj,
                plan=plan_obj,
                files=fmap,
                file_specs=fspecs,
            )
        except Exception as e:
            bus.emit("PROJECT_VALIDATE_ERROR", node_id=node_id, error=str(e), parent_id=parent_id)
            report = {"ok": False, "issues": [f"validator_error:{e}"], "warnings": [], "suggestions": []}
        # Persist report per-task
        rep_path = paths.validations / f"project_validation_{getattr(task, 'id', 'unknown')}.json"
        ensure_parent(rep_path)
        rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        ok = bool(report.get("ok", False))
        msg = "; ".join([str(x) for x in (report.get("issues", []) or [])]) or "ok"
        bus.emit(
            "PROJECT_VALIDATED",
            node_id=node_id,
            ok=ok,
            report=str(rep_path),
            task_id=getattr(task, "id", None),
            parent_id=parent_id,
        )
        if not ok:
            bus.emit(
                "VALIDATION_FAILED",
                node_id=node_id,
                kind="project",
                message=msg[:500],
                task_id=getattr(task, "id", None),
                parent_id=parent_id,
            )
        return ok, msg

    def _review_composite(task: TaskSpec) -> Tuple[bool, str]:
        """Language-agnostic review: verify that generated files contain declared exports.
        Uses DSPy-based export verification across languages; no Python-specific logic.
        """
        touched = _subtree_files(task)
        for fpath in touched:
            out_file = paths.outputs / fpath
            if not out_file.exists():
                # File not generated yet; let pipeline continue
                continue
            meta = file_meta_by_path.get(fpath, {})
            exports = list(meta.get("exports", []))
            entry = meta.get("entrypoint")
            if entry and entry not in exports:
                exports = exports + [entry]
            language = meta.get("language") or infer_language_from_path(fpath) or ""
            text = out_file.read_text(encoding="utf-8")
            v = engine.verify_exports_in_text(file=fpath, language=str(language or ""), exports=exports, text=text)
            if not v.get("ok", False):
                missing = v.get("missing", [])
                return False, f"missing_exports: {missing}"
        return True, "ok"

    progressed = True
    iters = 0
    timed_out = False
    while progressed and iters < max_iters:
        # enforce wall-clock timeout early in the loop
        if deadline is not None and time.time() > deadline:
            timed_out = True
            bus.emit("TIMEOUT", node_id=node_id, parent_id=parent_id, seconds=max_seconds)
            break
        iters += 1
        progressed = False
        tick_started = time.perf_counter()
        # renew lease heartbeat
        try:
            leases.renew(node_id, lease_id, ttl=lease_ttl)
        except Exception as e:
            bus.emit("LEASE_RENEW_FAILED", node_id=node_id, lease_id=lease_id, error=str(e), parent_id=parent_id)
        rtasks = ready_tasks()
        # loop tick + ready set metrics
        pending_ct = sum(1 for s in statuses.values() if s == "pending")
        running_ct = sum(1 for s in statuses.values() if s == "running")
        done_ct = sum(1 for s in statuses.values() if s == "done")
        failed_ct = sum(1 for s in statuses.values() if s == "failed")
        bus.emit(
            "LOOP_TICK",
            node_id=node_id,
            tick=iters,
            ready=len(rtasks),
            pending=pending_ct,
            running=running_ct,
            done=done_ct,
            failed=failed_ct,
            elapsed=int(time.time() - start_wall),
            parent_id=parent_id,
        )
        bus.emit(
            "READY_SET",
            node_id=node_id,
            count=len(rtasks),
            sample=[getattr(t, "id", None) for t in rtasks[:5]],
            parent_id=parent_id,
        )
        if not rtasks:
            # attempt amend if allowed and pending tasks remain
            if allow_amend and any(st == "pending" for st in statuses.values()):
                rounds = 0
                breakdown = _idle_breakdown()
                bus.emit(
                    "SCHEDULER_IDLE",
                    node_id=node_id,
                    parent_id=parent_id,
                    breakdown=breakdown,
                    will_amend=True,
                    amend_limit=amend_max_rounds,
                )
                total_applied = 0
                while rounds < amend_max_rounds:
                    bus.emit("AMEND_ATTEMPT", node_id=node_id, parent_id=parent_id, round=rounds + 1)
                    made = amend_once()
                    if made:
                        total_applied += int(made)
                        bus.emit("AMEND_APPLIED", node_id=node_id, parent_id=parent_id, round=rounds + 1, edits=made)
                        progressed = True
                        rtasks = ready_tasks()
                        if rtasks:
                            break
                    rounds += 1
                bus.emit("AMEND_RESULT", node_id=node_id, parent_id=parent_id, rounds=rounds, total_applied=total_applied)
                if not rtasks:
                    break
            else:
                breakdown = _idle_breakdown()
                bus.emit(
                    "SCHEDULER_IDLE",
                    node_id=node_id,
                    parent_id=parent_id,
                    breakdown=breakdown,
                    will_amend=False,
                    amend_limit=0,
                )
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

            # Clarify tasks at every level using DSPy with a bounded refine loop.
            # Parent provides context; siblings provide boundaries.
            try:
                rounds = 0
                last_clarified: Dict | None = None
                clarify_start = time.perf_counter()
                clarify_changed_any = False
                prev_snapshot = json.dumps(t.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))
                while rounds < taskplan_refine_max_rounds:
                    rounds += 1
                    p_id = parents.get(tid)
                    p_task = all_tasks.get(p_id) if isinstance(p_id, str) else None
                    siblings_specs: List[TaskSpec] = []
                    if p_task is not None:
                        for s in p_task.children:
                            if getattr(s, "id", None) != tid:
                                siblings_specs.append(s)
                    touched = _subtree_files(p_task) if p_task is not None else _subtree_files(t)
                    fmap = _files_map(paths, touched)
                    clarified = engine.clarify_task(
                        task=t.model_dump(exclude_none=True),
                        parent=(p_task.model_dump(exclude_none=True) if p_task is not None else None),
                        siblings=[s.model_dump(exclude_none=True) for s in siblings_specs],
                        artifacts=registry.list(),
                        files=fmap,
                        idea=idea or "",
                        constraints=constraints_obj,
                    )
                    last_clarified = clarified if isinstance(clarified, dict) else None
                    if isinstance(clarified, dict):
                        # Apply safe subset of fields; ignore children field if present
                        for key in ["kind", "title", "description"]:
                            if key in clarified:
                                setattr(t, key, clarified[key])
                        if "deps" in clarified and isinstance(clarified["deps"], list):
                            t.deps = list(clarified["deps"])  # type: ignore[assignment]
                            deps_map[tid] = list(t.deps)
                        if "inputs" in clarified and isinstance(clarified["inputs"], dict):
                            t.inputs = dict(clarified["inputs"])  # type: ignore[assignment]
                        if "outputs" in clarified and isinstance(clarified["outputs"], dict):
                            t.outputs = dict(clarified["outputs"])  # type: ignore[assignment]
                    cur_snapshot = json.dumps(t.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))
                    if cur_snapshot == prev_snapshot:
                        break
                    prev_snapshot = cur_snapshot
                    clarify_changed_any = True
                # Persist once after loop and emit event
                if last_clarified is not None:
                    atomic_write_json(task_plan_path, tp.model_dump(exclude_none=True))
                    elab_dir = paths.validations / "tasks" / "elaborations"
                    ensure_parent(elab_dir / "_.json")
                    (elab_dir / f"{tid}.json").write_text(json.dumps(last_clarified, indent=2), encoding="utf-8")
                    bus.emit("TASK_CLARIFIED", parent=node_id, child=child_exec_id, task_id=tid, parent_id=parent_id)
                bus.emit(
                    "TASK_CLARIFY_SUMMARY",
                    parent=node_id,
                    child=child_exec_id,
                    task_id=tid,
                    rounds=rounds,
                    changed=clarify_changed_any,
                    duration_ms=int((time.perf_counter() - clarify_start) * 1000),
                    parent_id=parent_id,
                )
            except Exception as _e:
                # Non-fatal: proceed without clarification but log
                bus.emit("TASK_CLARIFY_FAILED", parent=node_id, child=child_exec_id, task_id=tid, error=str(_e), parent_id=parent_id)

            # Decide split vs implement
            sd_start = time.perf_counter()
            do_split, children = split_or_implement(t)
            bus.emit(
                "SPLIT_DECISION",
                parent=node_id,
                child=child_exec_id,
                task_id=tid,
                action=("split" if do_split else "implement"),
                children_count=len(children),
                duration_ms=int((time.perf_counter() - sd_start) * 1000),
                parent_id=parent_id,
            )
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
                # Elaborate each newly created child from the mother node's context with a bounded refine loop
                for c in new_children:
                    try:
                        rounds = 0
                        last_child_clarified: Dict | None = None
                        c_clarify_start = time.perf_counter()
                        c_changed_any = False
                        prev_snapshot = json.dumps(c.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))
                        while rounds < taskplan_refine_max_rounds:
                            rounds += 1
                            siblings_specs = [s for s in t.children if getattr(s, "id", None) != getattr(c, "id", None)]
                            fmap = _files_map(paths, _subtree_files(t))
                            clarified_child = engine.clarify_task(
                                task=c.model_dump(exclude_none=True),
                                parent=t.model_dump(exclude_none=True),
                                siblings=[s.model_dump(exclude_none=True) for s in siblings_specs],
                                artifacts=registry.list(),
                                files=fmap,
                                idea=idea or "",
                                constraints=constraints_obj,
                            )
                            last_child_clarified = clarified_child if isinstance(clarified_child, dict) else None
                            if isinstance(clarified_child, dict):
                                # apply safe fields
                                for key in ["kind", "title", "description"]:
                                    if key in clarified_child:
                                        setattr(c, key, clarified_child[key])
                                if "deps" in clarified_child and isinstance(clarified_child["deps"], list):
                                    c.deps = list(clarified_child["deps"])  # type: ignore[assignment]
                                if "inputs" in clarified_child and isinstance(clarified_child["inputs"], dict):
                                    c.inputs = dict(clarified_child["inputs"])  # type: ignore[assignment]
                                if "outputs" in clarified_child and isinstance(clarified_child["outputs"], dict):
                                    c.outputs = dict(clarified_child["outputs"])  # type: ignore[assignment]
                            cur_snapshot = json.dumps(c.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))
                            if cur_snapshot == prev_snapshot:
                                break
                            prev_snapshot = cur_snapshot
                            c_changed_any = True
                    # Persist once after loop and emit event for the child
                    if last_child_clarified is not None:
                        atomic_write_json(task_plan_path, tp.model_dump(exclude_none=True))
                        elab_dir = paths.validations / "tasks" / "elaborations"
                        ensure_parent(elab_dir / "_.json")
                        (elab_dir / f"{getattr(c, 'id', 'child')}.json").write_text(json.dumps(last_child_clarified, indent=2), encoding="utf-8")
                    bus.emit(
                        "TASK_CHILD_CLARIFY_SUMMARY",
                        parent=node_id,
                        child=child_exec_id,
                        task_id=getattr(c, "id", None),
                        rounds=rounds,
                        changed=c_changed_any,
                        duration_ms=int((time.perf_counter() - c_clarify_start) * 1000),
                        parent_id=parent_id,
                    )
                except Exception as _e:
                    # Non-fatal: proceed without clarification but log
                    bus.emit("TASK_CHILD_CLARIFY_FAILED", parent=node_id, child=child_exec_id, task_id=getattr(c, "id", None), error=str(_e), parent_id=parent_id)
                # index newly added subtree
                _index(new_children, all_tasks)
                # Update deps/status maps
                for c in new_children:
                    statuses[c.id] = "pending"  # type: ignore[index]
                    deps_map[c.id] = list(c.deps)  # type: ignore[index]
                    parents[c.id] = tid  # type: ignore[index]
                # Persist updated plan
                atomic_write_json(task_plan_path, tp.model_dump(exclude_none=True))
                status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                added_count = len(new_children)
                bus.emit("TASK_SPLIT", parent=node_id, child=child_exec_id, task_id=tid, added=added_count, parent_id=parent_id)
                bus.emit("CHILD_SPLIT", parent=node_id, child=child_exec_id, task_id=tid, added=added_count, parent_id=parent_id)
                # Guard against no-op split decisions
                if added_count == 0:
                    reason = "no_op_split: decision returned no new actionable children"
                    statuses[tid] = "failed"
                    status.write(child_exec_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=reason)
                    bus.emit("TASK_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=reason, parent_id=parent_id)
                    bus.emit("CHILD_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=reason, parent_id=parent_id)
                    if not keep_going:
                        status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason=reason, parent_id=parent_id)
                        bus.emit("NODE_FAILED", node_id=node_id, reason=reason, parent_id=parent_id)
                        raise typer.Exit(code=1)
                    # continue scheduling other tasks
                    progressed = True
                    continue
                # Put parent back to pending so it can complete after children
                statuses[tid] = "pending"
                progressed = True
                continue

            # Implement leaf task
            try:
                if t.kind == "code:function":
                    ok_b, msg_b, info_b = adapter.build_code_function(t)
                    if not ok_b:
                        raise ValueError(msg_b)
                    fpath = info_b.get("path", "")
                    lang_eff = info_b.get("language", "")
                    # Persist progressive CodeSpec update after this leaf build
                    try:
                        adapter.save_codespec()
                    except Exception as e:
                        bus.emit("CODESPEC_SAVE_FAILED", node_id=node_id, file=str(paths.plan / "codespec.json"), error=str(e), parent_id=parent_id)
                    status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                    bus.emit("FILE_GENERATED", parent=node_id, child=child_exec_id, path=fpath, language=(lang_eff or ""), parent_id=parent_id)
                    bus.emit("TASK_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                    bus.emit("CHILD_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                    statuses[tid] = "done"
                elif t.kind == "composite":
                    # If leaf-composite (no children), mark done. Otherwise, children are all done (due to readiness),
                    # so perform parent-level review and finalize based on validation.
                    if not t.children:
                        # Enforce strict parent discipline: composite tasks must delegate; they cannot self-complete
                        reason = "composite_leaf_forbidden: parent must delegate by splitting; cannot mark done"
                        statuses[tid] = "failed"
                        status.write(child_exec_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=reason)
                        bus.emit("TASK_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=reason, parent_id=parent_id)
                        bus.emit("CHILD_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=reason, parent_id=parent_id)
                        if not keep_going:
                            status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason=reason, parent_id=parent_id)
                            bus.emit("NODE_FAILED", node_id=node_id, reason=reason, parent_id=parent_id)
                            raise typer.Exit(code=1)
                    else:
                        # Merge children's outputs into parent-level artifacts/files using DSPy (language-agnostic)
                        try:
                            parent_payload = t.model_dump(exclude_none=True)
                            children_payload = [c.model_dump(exclude_none=True) for c in t.children]
                            touched = _subtree_files(t)
                            fmap = _files_map(paths, touched)
                            merge_plan = engine.merge_subtasks(
                                parent=parent_payload,
                                children=children_payload,
                                artifacts=registry.list(),
                                files=fmap,
                                idea=idea or "",
                                constraints=constraints_obj,
                            )
                            if isinstance(merge_plan, dict):
                                # Apply writes
                                for w in (merge_plan.get("writes") or []):
                                    if isinstance(w, dict):
                                        wpath = w.get("path")
                                        wtext = w.get("text", "")
                                        if isinstance(wpath, str):
                                            out_path = paths.outputs / wpath
                                            write_code_file(out_path, str(wtext))
                                # Register artifacts
                                marts = merge_plan.get("artifacts") or []
                                if isinstance(marts, list) and marts:
                                    registry.register(marts, base_dir=paths.outputs)
                                    for r in marts:
                                        if isinstance(r, dict):
                                            _validate_artifact_ref(r)
                                bus.emit("TASK_MERGED", parent=node_id, child=child_exec_id, task_id=tid, parent_id=parent_id)
                        except Exception as _e:
                            # Non-fatal: continue to review; failures will surface there if critical
                            pass
                        ok, reason = _review_composite(t)
                        if ok:
                            # Gate composite completion on project-level validation for its subtree
                            pv_ok, pv_msg = _project_validate_gate(t)
                            if not pv_ok:
                                statuses[tid] = "failed"
                                status.write(child_exec_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=pv_msg)
                                bus.emit("TASK_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=pv_msg, parent_id=parent_id)
                                bus.emit("CHILD_FAILED", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, reason=pv_msg, parent_id=parent_id)
                                if not keep_going:
                                    status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason=pv_msg, parent_id=parent_id)
                                    bus.emit("NODE_FAILED", node_id=node_id, reason=pv_msg, parent_id=parent_id)
                                    raise typer.Exit(code=1)
                                continue
                            status.write(child_exec_id, "DONE", prev_state="LEASED", parent_id=node_id)
                            bus.emit("COMPOSITE_REVIEW_PASSED", parent=node_id, child=child_exec_id, task_id=tid, parent_id=parent_id)
                            bus.emit("TASK_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                            bus.emit("CHILD_DONE", parent=node_id, child=child_exec_id, task_id=tid, kind=t.kind, parent_id=parent_id)
                            # Register any declared produced artifacts at the composite boundary and validate
                            prods = _produces(t)
                            if prods:
                                registry.register(prods, base_dir=paths.outputs)
                                for r in prods:
                                    _validate_artifact_ref(r)
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
            except Exception as e:
                bus.emit("LEASE_RENEW_FAILED", node_id=node_id, lease_id=lease_id, error=str(e), parent_id=parent_id)

    # Enforce wall-clock timeout after loop
    if timed_out:
        reason = f"wall_clock_timeout:{max_seconds}s"
        status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason=reason, parent_id=parent_id)
        bus.emit("NODE_FAILED", node_id=node_id, reason=reason, parent_id=parent_id)
        raise typer.Exit(code=1)

    # After tasks, check if any pending remain
    remaining = [tid for tid, st in statuses.items() if st not in ("done",)]
    if remaining:
        console.print(f"[yellow]Tasks left pending or blocked[/yellow]: {remaining[:10]} ...")

    # Unified validation across all generated files
    any_failed = False
    out_files: List[str] = []
    for fpath, meta in file_meta_by_path.items():
        language = meta.get("language") or infer_language_from_path(fpath) or ""
        out_path = paths.outputs / fpath
        assembled = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if not out_path.exists() or not assembled.strip():
            console.print(f"[red]Output missing or empty[/red]: {fpath}")
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="output", message="output_missing_or_empty", file=fpath, parent_id=parent_id)
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
                console.print(f"[red]Exports missing[/red] in {fpath}: {missing}")
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

    # Save progressive CodeSpec after all tasks complete
    adapter.save_codespec()
    
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
