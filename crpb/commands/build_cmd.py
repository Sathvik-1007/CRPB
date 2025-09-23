from __future__ import annotations
import hashlib
import os
import json
import time
from pathlib import Path
import typer
from rich.console import Console
from ..config import resolve_run_dir, make_paths
from ..eventbus import EventBus
from ..status import NodeStatus
from ..leases import Leases
from ..registry import Registry, ConflictError
from ..specs import CodeSpec, CodeSpecFile
from ..validator import (
    basic_file_validation,
    jsonschema_validate,
    infer_language_from_path,
    validate_leaf_readiness,
)
from ..utils.fs import atomic_write_json, ensure_parent, lock_file
from ..utils.ui import sep
from ..repair import RepairIssue, generate_repair_plan, write_repair_plan
from ..agents.dspy_engine import DspyEngine
from ..llm_config import load_selection, require_env_vars
from ..communication import Comms

app = typer.Typer(help="Build end-to-end from idea + codespec.json with registry, events, leases, validator, and LLM-backed implementation")
console = Console()


# --- Local helpers for file operations (safe, locked writes and metadata) ---
def write_code_file(path: Path, text: str) -> None:
    """Safely write text content to a file, ensuring parent directories exist."""
    ensure_parent(path)
    # Use a sidecar lock file to coordinate concurrent writers across processes
    lock_path = path.parent / (path.name + ".lock")
    with lock_file(lock_path):
        path.write_text(text, encoding="utf-8")


def write_file_meta(path: Path, data: dict) -> None:
    """Atomically write JSON metadata to a file."""
    atomic_write_json(path, data)


def validate_file_content(text: str, path: Path) -> bool:
    """Basic pre-write validation for generated content.
    Currently checks for non-empty, non-whitespace content only. Detailed validations
    are performed later via schema/basic/export/project validators.
    """
    try:
        return bool(text and text.strip())
    except Exception:
        return False

# removed: _coerce_example_io (unused)

def effective_language(fs: CodeSpecFile) -> str | None:
    return fs.language or infer_language_from_path(getattr(fs, "path", None))


def func_id(signature: str) -> str:
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]


@app.callback(invoke_without_command=True)
def main(
    idea: str = typer.Option(None, "--idea", help="High-level idea; if missing, tries run/plan/idea.json"),
    run_dir: str = typer.Option(None, "--run-dir", help="Base runs folder"),
    run: str = typer.Option("new", "--run", help="run_<ts> | latest | new | name"),
    model: str = typer.Option(None, "--model", help="Override model for LLM steps"),
    max_children: int = typer.Option(2, "--max-children", min=1, help="Max concurrent child tasks parent will schedule"),
    node_id: str = typer.Option(None, "--node-id", help="Deterministic node id for this builder"),
    parent_id: str = typer.Option(None, "--parent-id", help="Parent node id for orchestration tracking"),
    lease_ttl: int = typer.Option(180, "--lease-ttl", min=30, help="TTL seconds for builder lease"),
    child_ttl: int = typer.Option(120, "--child-ttl", min=30, help="TTL seconds for child leases"),
):
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    sep("BUILD START")

    # LLM is mandatory; provider configuration will be validated by the engine.

    # Resolve idea
    if idea is None:
        idea_file = paths.plan / "idea.json"
        if idea_file.exists():
            import json
            idea = json.loads(idea_file.read_text(encoding="utf-8")).get("idea", "")
        else:
            console.print("[red]No idea provided and plan/idea.json not found[/red]")
            raise typer.Exit(code=2)

    console.print(f"Idea: [bold]{idea}[/bold]")

    bus = EventBus(paths.logs / "events.jsonl")
    status = NodeStatus(paths.graph / "node_status.jsonl")
    leases = Leases(paths.graph / "leases.json")
    registry = Registry(paths.registry / "registry.json")
    comms = Comms(paths.graph / "comms.jsonl")

    # Deterministic defaults if not provided
    node_id = node_id or f"builder::{run_path.name}"
    parent_id = parent_id or f"root::{run_path.name}"
    status.write(node_id, "CREATED", parent_id=parent_id)
    lease_id = leases.grant(node_id, ttl=lease_ttl)
    status.write(node_id, "LEASED", prev_state="CREATED", lease_id=lease_id, parent_id=parent_id)
    bus.emit("NODE_CREATED", node_id=node_id, parent_id=parent_id)
    bus.emit("LEASE_GRANTED", node_id=node_id, lease_id=lease_id, parent_id=parent_id)

    # LLM env preflight (fail-fast with diagnostics, no hardcoded defaults)
    sel = load_selection()
    if not sel:
        console.print("[red]No LLM selection found.[/red] Use `python -m crpb llm choose` to select a provider and model before build.")
        bus.emit("LLM_SELECTION_MISSING", node_id=node_id, parent_id=parent_id)
        status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason="llm_selection_missing", parent_id=parent_id)
        raise typer.Exit(code=2)
    ok_env, missing_env = require_env_vars(sel)
    if not ok_env:
        miss = ", ".join(missing_env)
        console.print(f"[red]LLM configuration incomplete for provider={sel.provider}: {miss}[/red]")
        bus.emit("ENV_MISSING", node_id=node_id, provider=sel.provider, missing=missing_env, parent_id=parent_id)
        status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason="llm_env_missing", parent_id=parent_id)
        raise typer.Exit(code=2)

    sep("SPEC & STUB")

    # Load Plan and build per-file index for context/validation
    plan_obj: dict = {}
    plan_file = paths.plan / "plan.json"
    if plan_file.exists():
        try:
            plan_obj = json.loads(plan_file.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to parse plan.json: {e}. Proceeding with codespec-only context[/yellow]")
            plan_obj = {}
    else:
        console.print("[yellow]Warning: plan/plan.json not found; proceeding with codespec-only context[/yellow]")

    def _iter_tasks_dict(ts):
        if isinstance(ts, list):
            for t in ts:
                if isinstance(t, dict):
                    yield t
                    for c in _iter_tasks_dict(t.get("children", []) or []):
                        yield c

    # Flatten plan nodes for node-level context (path/title/kind/atomic/socratic)
    plan_nodes: list[dict] = []
    try:
        def _flatten_nodes(ts_list: list, parent_path: str = ""):
            for idx, t in enumerate(ts_list or [], start=1):
                if not isinstance(t, dict):
                    continue
                meta = t.get("meta") or {}
                path = meta.get("path") or (f"{parent_path}.{idx}" if parent_path else str(idx))
                atomic = bool(meta.get("atomic"))
                soc = meta.get("socratic") or {}
                questions = [q for q in (soc.get("questions") or []) if isinstance(q, str)] if isinstance(soc, dict) else []
                monologue = soc.get("monologue") if isinstance(soc, dict) else ""
                plan_nodes.append({
                    "path": path,
                    "title": t.get("title", ""),
                    "kind": t.get("kind", ""),
                    "atomic": atomic,
                    "socratic": {"questions": questions, "monologue": monologue or ""},
                })
                _flatten_nodes(t.get("children", []) or [], path)
        _flatten_nodes(plan_obj.get("tasks", []) or [])
    except Exception:
        plan_nodes = []

    # Load plan-level validation report if present
    plan_validation_obj: dict = {}
    try:
        pvf = paths.plan / "plan_validation.json"
        if pvf.exists():
            plan_validation_obj = json.loads(pvf.read_text(encoding="utf-8"))
    except Exception:
        plan_validation_obj = {}

    # Build index: file path -> list of codespec entries (per-file) to provide side_context to the generator
    plan_file_entries_by_path: dict[str, list] = {}

    # Plan artifacts index (producers/consumers)
    plan_artifacts: dict = {}
    try:
        art = plan_obj.get("artifacts", {})
        if isinstance(art, dict):
            plan_artifacts = art
    except Exception:
        plan_artifacts = {}

    # Strictly load CodeSpec from plan/codespec.json and build file specs.
    specs_by_path: dict[str, CodeSpecFile] = {}
    codespec_raw_by_path: dict[str, dict] = {}
    cs_file = paths.plan / "codespec.json"
    if not cs_file.exists():
        console.print("[red]Missing plan/codespec.json. Run planning to generate codespec.json.[/red]")
        raise typer.Exit(code=2)
    try:
        cs_obj = json.loads(cs_file.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[red]Failed to parse codespec.json: {e}[/red]")
        bus.emit("CODESPEC_PARSE_ERROR", node_id=node_id, file=str(cs_file), error=str(e), parent_id=parent_id)
        status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason="codespec_parse_error", parent_id=parent_id)
        raise typer.Exit(code=2)

    # Validate CodeSpec root schema (fail-fast)
    ok_cs_root, msg_cs_root = jsonschema_validate(cs_obj, CodeSpec.model_json_schema())
    if not ok_cs_root:
        console.print(f"[red]codespec.json schema invalid:[/red] {msg_cs_root}")
        bus.emit("CODESPEC_SCHEMA_FAILED", node_id=node_id, file=str(cs_file), message=msg_cs_root, parent_id=parent_id)
        status.write(node_id, "FAILED_FINAL", prev_state="LEASED", reason="codespec_schema_invalid", parent_id=parent_id)
        raise typer.Exit(code=2)
    for item in (cs_obj.get("files", []) or []):
        if not isinstance(item, dict):
            continue
        p = item.get("path")
        if not isinstance(p, str) or not p:
            continue
        codespec_raw_by_path[p] = item
        # Side-context entries sourced from codespec
        try:
            plan_file_entries_by_path.setdefault(p, []).append({k: v for k, v in item.items() if k != "path"})
        except Exception as e:
            bus.emit("SIDECONTEXT_BUILD_FAILED", node_id=node_id, path=p, error=str(e), parent_id=parent_id)
    specs_from_codespec: dict[str, CodeSpecFile] = {}
    for p, it in codespec_raw_by_path.items():
        guess = infer_language_from_path(p)
        lang = it.get("language") or guess
        # Build functions mapping for CodeSpecFile (name -> dict metadata). Accepts either
        # string signatures or rich dicts with a 'signature' field in codespec.json.
        funcs_meta = it.get("functions", {}) or {}
        funcs: dict[str, dict] = {}
        if isinstance(funcs_meta, dict):
            for fname, fmeta in funcs_meta.items():
                if isinstance(fmeta, str):
                    funcs[fname] = {"signature": fmeta}
                elif isinstance(fmeta, dict):
                    # keep as-is but ensure signature string when present
                    mm = dict(fmeta)
                    sig = mm.get("signature")
                    if isinstance(sig, str):
                        mm["signature"] = sig
                    else:
                        mm["signature"] = str(sig or "")
                    funcs[fname] = mm
        
        # Extract all rich metadata from progressive CodeSpec
        classes = dict(it.get("classes", {}) or {})
        constants = dict(it.get("constants", {}) or {})
        # exports_detail is a list of dicts
        ed_raw = it.get("exports_detail", []) or []
        exports_detail = [e for e in ed_raw if isinstance(e, dict)]
        metadata = dict(it.get("metadata", {}) or {})
        
        fs = CodeSpecFile(
            path=p,
            language=lang,
            purpose=it.get("purpose", ""),
            description=it.get("description", ""),
            content=it.get("content"),
            functions=funcs,
            exports=list(it.get("exports", list(funcs.keys())) or []),
            imports=list(it.get("imports", []) or []),
            imports_description=it.get("imports_description", ""),
            exports_description=it.get("exports_description", ""),
            functions_description=it.get("functions_description", ""),
            classes_description=it.get("classes_description", ""),
            constants_description=it.get("constants_description", ""),
            exports_detail=exports_detail,
            classes=classes,
            constants=constants,
            entrypoint=it.get("entrypoint"),
            metadata=metadata,
        )
        specs_from_codespec[p] = fs
    specs_by_path = specs_from_codespec

    # Deterministic file processing order across all phases
    file_order = sorted(specs_by_path.keys())

    # Spec-first: use CodeSpec-derived context only (language-agnostic)

    if not specs_by_path:
        console.print("[red]CodeSpec has no files; nothing to build[/red]")
        raise typer.Exit(code=2)

    # Save/refresh file-level spec artifacts to ensure consistency
    for _p in file_order:
        fs = specs_by_path[_p]
        _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
        atomic_write_json(paths.specs / f"file_{_fid}.json", fs.model_dump(exclude_none=True))

    # Publish to registry as stubs for all files
    ver, _data = registry.load()
    def mutate_stub_all(data: dict):
        files = data.setdefault("files", {})
        for fpath in file_order:
            fs = specs_by_path[fpath]
            f = files.setdefault(fpath, {"functions": {}, "exports": [], "timestamp": None})
            for fname, meta in fs.functions.items():
                # meta may be a dict; extract signature
                sig = meta.get("signature") if isinstance(meta, dict) else str(meta)
                f["functions"][fname] = {
                    "signature": sig,
                    "published_by": node_id,
                    "status": "stub",
                    "checksum": None,
                    "deps": [],
                }
            f["exports"] = fs.exports
            f["timestamp"] = time.time()
        return data

    try:
        registry.update(ver, mutate_stub_all)
    except ConflictError:
        console.print(f"[yellow]Registry conflict; reload & retry once[/yellow]")
        ver, _ = registry.load()
        registry.update(ver, mutate_stub_all)

    # Publish events for all stubbed functions
    for fpath in file_order:
        fs = specs_by_path[fpath]
        for fname in fs.functions.keys():
            bus.emit("FUNCTION_PUBLISHED", path=fpath, name=fname, status="stub", parent_id=parent_id)
    status.write(node_id, "PRODUCED_STUB", prev_state="LEASED", parent_id=parent_id)

    sep("GENERATE FILES")
    
    chunks_dir = paths.artifacts / "chunks"
    staging_dir = paths.outputs / "_staging"
    ensure_parent(chunks_dir / "dummy.txt")
    ensure_parent(staging_dir / "dummy.txt")
    out_files_staging: list[str] = []
    # Load constraints to inform LLM prompts
    constraints: dict = {}
    constraints_file = paths.plan / "constraints.json"
    if constraints_file.exists():
        try:
            constraints = json.loads(constraints_file.read_text(encoding="utf-8"))
        except Exception as e:
            constraints = {}
            bus.emit("CONSTRAINTS_PARSE_ERROR", node_id=node_id, file=str(constraints_file), error=str(e), parent_id=parent_id)

    # Initialize DSPy engine for all LLM-backed steps
    engine = DspyEngine(model=model)

    # Generate each file in a language-agnostic way via DSPy using both specs as context
    for fpath in file_order:
        fs = specs_by_path[fpath]
        child_id = f"gen::{Path(fpath).stem}::{hashlib.sha1(fpath.encode('utf-8')).hexdigest()[:8]}"
        status.write(child_id, "CREATED", parent_id=node_id)
        child_lease = leases.grant(child_id, ttl=child_ttl)
        status.write(child_id, "LEASED", prev_state="CREATED", lease_id=child_lease, parent_id=node_id)
        bus.emit("CHILD_ASSIGNED", parent=node_id, child=child_id, path=fpath, kind="generation", parent_id=parent_id)
        try:
            # Send side-context to the child via Comms for traceability
            comms.send(
                from_id=node_id,
                to_id=child_id,
                kind="CONTEXT",
                path=fpath,
                payload={
                    "codespec_file": codespec_raw_by_path.get(fpath, {}),
                    "plan_file_entries": plan_file_entries_by_path.get(fpath, []),
                },
            )
        except Exception as e:
            bus.emit("COMMS_SEND_FAILED", node_id=node_id, to=child_id, path=fpath, kind="CONTEXT", error=str(e), parent_id=parent_id)
        try:
            lang_eff = effective_language(fs) or ""
            # Per-file context from progressive CodeSpec
            cs_raw = codespec_raw_by_path.get(fpath, {})
            c_ext = dict(constraints or {})
            c_ext.setdefault("side_context", {})
            c_ext["side_context"].update({
                "codespec_file": cs_raw,
                # Provide any CodeSpec-derived entries associated with this file (neutral, language-agnostic context)
                "plan_file_entries": plan_file_entries_by_path.get(fpath, []),
                # Node-level plan context (general, language-neutral)
                "plan_nodes": plan_nodes,
                # Plan structural validation summary (general, language-neutral)
                "plan_validation": plan_validation_obj,
                # Artifact index for producers/consumers mapping
                "plan_artifacts": plan_artifacts,
            })

            if not isinstance(lang_eff, str) or not lang_eff.strip():
                console.print(f"[yellow]Skipping {fpath}: language required for generation (explicit in spec or inferable from path)[/yellow]")
                status.write(child_id, "SKIPPED", prev_state="LEASED", parent_id=node_id, reason="no_language")
                bus.emit("CHILD_SKIPPED", parent=node_id, child=child_id, path=fpath, kind="generation", reason="no_language", parent_id=parent_id)
                continue
            # Pass all available CodeSpecFile data to the generator
            # Extract simple name->signature map for generation
            fn_sigs = {}
            try:
                for n, meta in getattr(fs, "functions", {}).items():
                    if isinstance(meta, dict):
                        s = meta.get("signature")
                    else:
                        s = str(meta)
                    if isinstance(s, str) and s:
                        fn_sigs[n] = s
            except Exception as e:
                fn_sigs = {}
                bus.emit("COLLECT_FN_SIGS_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)

            file_text = engine.generate_full_file(
                idea=idea or "",
                constraints=c_ext,
                file=fpath,
                language=lang_eff,
                exports=list(getattr(fs, "exports", [])),
                imports=list(getattr(fs, "imports", [])),
                entrypoint=getattr(fs, "entrypoint", None),
                functions=fn_sigs,
                # Pass rich metadata for better generation
                purpose=getattr(fs, "purpose", ""),
                description=getattr(fs, "description", ""),
                classes=dict(getattr(fs, "classes", {}) or {}),
                constants=dict(getattr(fs, "constants", {}) or {}),
                content=getattr(fs, "content", None),
            )
            # Micro-adjustment pass for minor fixes without semantic changes
            try:
                madj = engine.micro_adjust(file=fpath, language=lang_eff, text=file_text, idea=idea or "", constraints=constraints)
                if isinstance(madj, dict) and isinstance(madj.get("text"), str):
                    if madj.get("notes"):
                        try:
                            comms.send(from_id=child_id, to_id=node_id, kind="MICRO_ADJUSTMENT", path=fpath, payload={"notes": madj.get("notes", [])})
                        except Exception as e:
                            bus.emit("COMMS_SEND_FAILED", node_id=child_id, to=node_id, path=fpath, kind="MICRO_ADJUSTMENT", error=str(e), parent_id=parent_id)
                    file_text = madj.get("text", file_text)
            except Exception as e:
                bus.emit("MICROADJUST_FAILED", node_id=child_id, path=fpath, error=str(e), parent_id=parent_id)
            out_file = staging_dir / fpath
            # Validate content before writing
            if not validate_file_content(file_text, out_file):
                console.print(f"[yellow]Warning: Generated content for {fpath} failed basic validation[/yellow]")
            write_code_file(out_file, file_text)
            out_files_staging.append(str(out_file.relative_to(staging_dir)))
            # write chunk trace (locked)
            cstem = f"{Path(fpath).stem}_{hashlib.sha1(file_text.encode('utf-8')).hexdigest()[:10]}"
            cpath = chunks_dir / f"{cstem}.txt"
            lock_path = cpath.parent / (cpath.name + ".lock")
            with lock_file(lock_path):
                cpath.write_text(file_text, encoding="utf-8")
            # Defer registry updates until merge; emit generation event now
            bus.emit("FILE_GENERATED", path=fpath, language=lang_eff, parent_id=parent_id)
            status.write(child_id, "DONE", prev_state="LEASED", parent_id=node_id)
            bus.emit("CHILD_DONE", parent=node_id, child=child_id, path=fpath, kind="generation", parent_id=parent_id)
            # lease heartbeat
            try:
                leases.renew(node_id, lease_id, ttl=lease_ttl)
            except Exception as e:
                bus.emit("LEASE_RENEW_FAILED", node_id=node_id, lease_id=lease_id, error=str(e), parent_id=parent_id)
        except Exception as e:
            status.write(child_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=str(e))
            try:
                comms.send(from_id=child_id, to_id=node_id, kind="ERROR", path=fpath, payload={"error": str(e)})
            except Exception:
                pass
            bus.emit("CHILD_FAILED", parent=node_id, child=child_id, path=fpath, kind="generation", reason=str(e), parent_id=parent_id)
            status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_STUB", reason=str(e), parent_id=parent_id)
            bus.emit("NODE_FAILED", node_id=node_id, reason=str(e), parent_id=parent_id)
            raise typer.Exit(code=1)

    status.write(node_id, "PRODUCED_IMPL", prev_state="PRODUCED_STUB", parent_id=parent_id)

    sep("VALIDATE & MERGE")

    # Validate and write metadata per file (language-agnostic)
    any_failed = False
    failed_files: set[str] = set()
    for fpath in file_order:
        fs = specs_by_path[fpath]
        # create a validation child node per file
        vchild_id = f"validator::{Path(fpath).stem}::{hashlib.sha1(fpath.encode('utf-8')).hexdigest()[:8]}"
        status.write(vchild_id, "CREATED", parent_id=node_id)
        vlease_id = leases.grant(vchild_id, ttl=child_ttl)
        status.write(vchild_id, "LEASED", prev_state="CREATED", lease_id=vlease_id, parent_id=node_id)
        bus.emit("CHILD_ASSIGNED", parent=node_id, child=vchild_id, path=fpath, kind="validation", parent_id=parent_id)
        try:
            comms.send(from_id=node_id, to_id=vchild_id, kind="CONTEXT", path=fpath, payload={"codespec_file": codespec_raw_by_path.get(fpath, {})})
        except Exception as e:
            bus.emit("COMMS_SEND_FAILED", node_id=node_id, to=vchild_id, path=fpath, kind="CONTEXT", error=str(e), parent_id=parent_id)

        out_path = staging_dir / fpath
        assembled = out_path.read_text(encoding="utf-8") if out_path.exists() else ""

        # JSON Schema validation for CodeSpecFile (all languages)
        fs_schema = CodeSpecFile.model_json_schema()
        ok_schema, msg_schema = jsonschema_validate(fs.model_dump(), fs_schema)
        if not ok_schema:
            console.print(f"[red]Spec schema validation failed:[/red] {msg_schema}")
            issues = [RepairIssue(kind="schema_file_spec", message=msg_schema, file=fs.path)]
            _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
            plan_path = paths.repair_plans / f"repair_{_fid}.json"
            write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
            bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="schema", message=msg_schema, file=fs.path, parent_id=parent_id)
            status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=msg_schema)
            bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason=msg_schema, parent_id=parent_id)
            any_failed = True
            failed_files.add(fpath)
            try:
                comms.send(from_id=vchild_id, to_id=node_id, kind="VALIDATION_FAILED", path=fpath, payload={"schema": msg_schema})
            except Exception as e:
                bus.emit("COMMS_SEND_FAILED", node_id=vchild_id, to=node_id, path=fpath, kind="VALIDATION_FAILED", error=str(e), parent_id=parent_id)
            continue

        # Basic logical validation (language-agnostic)
        ok, msg = basic_file_validation(fs)
        if not ok:
            console.print(f"[red]Validation failed:[/red] {msg}")
            issues = [RepairIssue(kind="basic_file_validation", message=msg, file=fs.path)]
            _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
            plan_path = paths.repair_plans / f"repair_{_fid}.json"
            write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
            bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="basic", message=msg, file=fs.path, parent_id=parent_id)
            status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=msg)
            bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason=msg, parent_id=parent_id)
            any_failed = True
            failed_files.add(fpath)
            continue

        # Unified file checks: existence and declared exports via DSPy
        if not out_path.exists() or not assembled.strip():
            msg_np = "output_missing_or_empty"
            issues = [RepairIssue(kind="output", message=msg_np, file=fs.path)]
            _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
            plan_path = paths.repair_plans / f"repair_{_fid}.json"
            write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
            bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="output", message=msg_np, file=fs.path, parent_id=parent_id)
            status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=msg_np)
            bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason=msg_np, parent_id=parent_id)
            any_failed = True
            failed_files.add(fpath)
            continue

        lang_eff_all = effective_language(fs) or ""
        exports_list = list(getattr(fs, "exports", []) or [])
        ok_exports = True
        msg_exports = "ok"
        if exports_list:
            try:
                verify = engine.verify_exports_in_text(
                    file=fpath,
                    language=lang_eff_all,
                    exports=exports_list,
                    text=assembled,
                )
            except Exception as e:
                verify = {"ok": False, "missing": exports_list}
                bus.emit("EXPORT_VERIFY_ERROR", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)
            if not verify.get("ok", False):
                missing = verify.get("missing", exports_list)
                msg_exports = f"missing_exports: {missing}"
                issues = [RepairIssue(kind="exports", message=msg_exports, file=fs.path)]
                _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
                plan_path = paths.repair_plans / f"repair_{_fid}.json"
                write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
                bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
                bus.emit("VALIDATION_FAILED", node_id=node_id, kind="exports", message=msg_exports, file=fs.path, parent_id=parent_id)
                status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=msg_exports)
                bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason=msg_exports, parent_id=parent_id)
                any_failed = True
                failed_files.add(fpath)
                try:
                    comms.send(from_id=vchild_id, to_id=node_id, kind="VALIDATION_FAILED", path=fpath, payload={"exports": missing})
                except Exception:
                    pass
                continue

        # Passed unified checks
        status.write(vchild_id, "DONE", prev_state="LEASED", parent_id=node_id)
        bus.emit("CHILD_DONE", parent=node_id, child=vchild_id, path=fpath, kind="validation", parent_id=parent_id)

        # Write file meta and validations report
        _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
        write_file_meta(paths.artifacts / f"file_{_fid}.json", fs.model_dump(exclude_none=True))
        validations_report = {
            "file": str((staging_dir / fs.path)),
            "schema": {"ok": ok_schema, "msg": msg_schema},
            "basic": {"ok": ok, "msg": msg},
            "exports": {"ok": ok_exports, "msg": msg_exports},
        }
        vpath = paths.validations / f"report_{_fid}.json"
        vpath.write_text(__import__("json").dumps(validations_report, indent=2), encoding="utf-8")
        bus.emit("VALIDATION_PASSED", node_id=node_id, file=fs.path, report=str(vpath), parent_id=parent_id)

    # Attempt a simple repair cycle for failed files (leaf self-correction)
    if any_failed and failed_files:
        sep("REPAIR & REVALIDATE (FILES)")
        # Bounds: allow up to N attempts via env or default 1
        try:
            max_attempts = int(os.environ.get("CRPB_REPAIR_MAX_ATTEMPTS", "1"))
        except Exception as e:
            max_attempts = 1
            bus.emit("ENV_PARSE_INVALID", node_id=node_id, var="CRPB_REPAIR_MAX_ATTEMPTS", error=str(e), parent_id=parent_id)
        attempts = 0
        while attempts < max_attempts and failed_files:
            attempts += 1
            current_failed = list(failed_files)
            failed_files = set()
            for fpath in current_failed:
                fs = specs_by_path.get(fpath)
                if not fs:
                    continue
                try:
                    lang_eff = effective_language(fs) or ""
                    # Rebuild context for repair from progressive CodeSpec
                    cs_raw = codespec_raw_by_path.get(fpath, {})
                    c_ext = dict(constraints or {})
                    c_ext.setdefault("side_context", {})
                    c_ext["side_context"].update({
                        "codespec_file": cs_raw,
                        "plan_nodes": plan_nodes,
                        "plan_validation": plan_validation_obj,
                        "plan_artifacts": plan_artifacts,
                    })

                    # Pass all available CodeSpecFile data to the generator (repair cycle)
                    # Extract signatures again for repair cycle
                    fn_sigs = {}
                    try:
                        for n, meta in getattr(fs, "functions", {}).items():
                            if isinstance(meta, dict):
                                s = meta.get("signature")
                            else:
                                s = str(meta)
                            if isinstance(s, str) and s:
                                fn_sigs[n] = s
                    except Exception as e:
                        fn_sigs = {}
                        bus.emit("COLLECT_FN_SIGS_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)

                    file_text = engine.generate_full_file(
                        idea=idea or "",
                        constraints=c_ext,
                        file=fpath,
                        language=lang_eff,
                        exports=list(getattr(fs, "exports", [])),
                        imports=list(getattr(fs, "imports", [])),
                        entrypoint=getattr(fs, "entrypoint", None),
                        functions=fn_sigs,
                        # Pass rich metadata for better generation
                        purpose=getattr(fs, "purpose", ""),
                        description=getattr(fs, "description", ""),
                        classes=dict(getattr(fs, "classes", {}) or {}),
                        constants=dict(getattr(fs, "constants", {}) or {}),
                        content=getattr(fs, "content", None),
                    )
                    out_path = staging_dir / fpath
                    # Validate content before writing (repair cycle)
                    if not validate_file_content(file_text, out_path):
                        console.print(f"[yellow]Warning: Repaired content for {fpath} failed basic validation[/yellow]")
                    write_code_file(out_path, file_text)
                except Exception as e:
                    failed_files.add(fpath)
                    bus.emit("REPAIR_GENERATE_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)
                    continue
                # Re-run validations (schema/basic/output/exports)
                # JSON Schema
                fs_schema = CodeSpecFile.model_json_schema()
                ok_schema, msg_schema = jsonschema_validate(fs.model_dump(), fs_schema)
                if not ok_schema:
                    failed_files.add(fpath)
                    continue
                # Basic validation
                ok, msg = basic_file_validation(fs)
                if not ok:
                    failed_files.add(fpath)
                    continue
                # Output & exports checks
                assembled = (staging_dir / fpath).read_text(encoding="utf-8") if (staging_dir / fpath).exists() else ""
                if not assembled.strip():
                    failed_files.add(fpath)
                    continue
                exports_list = list(getattr(fs, "exports", []) or [])
                if exports_list:
                    try:
                        verify = engine.verify_exports_in_text(
                            file=fpath,
                            language=lang_eff,
                            exports=exports_list,
                            text=assembled,
                        )
                    except Exception:
                        verify = {"ok": False, "missing": exports_list}
                    if not verify.get("ok", False):
                        failed_files.add(fpath)
                        continue
                # Passed on retry: write meta and report
                _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
                write_file_meta(paths.artifacts / f"file_{_fid}.json", fs.model_dump(exclude_none=True))
                # Compute exports ok flag for report
                ok_exports = True if not exports_list else bool(verify.get("ok", False))
                validations_report = {
                    "file": str((staging_dir / fs.path)),
                    "schema": {"ok": True, "msg": "ok"},
                    "basic": {"ok": True, "msg": "ok"},
                    "exports": {"ok": ok_exports, "msg": "ok"},
                }
                vpath = paths.validations / f"report_{_fid}.json"
                vpath.write_text(__import__("json").dumps(validations_report, indent=2), encoding="utf-8")
                bus.emit("VALIDATION_PASSED", node_id=node_id, file=fs.path, report=str(vpath), parent_id=parent_id)
        any_failed = bool(failed_files)

    # Pre-merge validations above operate exclusively on the staging directory

    # Project-level holistic validation using DSPy (language-agnostic)
    proj_failed = False
    files_map: dict[str, str] = {}
    file_specs_map: dict[str, dict] = {}
    for fpath, fs in specs_by_path.items():
        op = staging_dir / fpath
        try:
            files_map[fpath] = op.read_text(encoding="utf-8") if op.exists() else ""
        except Exception as e:
            files_map[fpath] = ""
            bus.emit("READ_STAGING_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)
        try:
            file_specs_map[fpath] = fs.model_dump(exclude_none=True)
        except Exception as e:
            file_specs_map[fpath] = {}
            bus.emit("SPEC_DUMP_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)

    def _project_validate(cur_files: dict[str, str]) -> dict:
        try:
            # Enrich constraints with plan-level context for general, logical validation
            c_proj_ext = dict(constraints or {})
            try:
                c_proj_ext.setdefault("side_context", {})
                c_proj_ext["side_context"].update({
                    "plan_nodes": plan_nodes,
                    "plan_artifacts": plan_artifacts,
                    "plan_validation": plan_validation_obj,
                })
            except Exception:
                pass
            return engine.project_validate(
                idea=idea or "",
                constraints=c_proj_ext,
                plan=plan_obj,
                files=cur_files,
                file_specs=file_specs_map,
            )
        except Exception as e:
            return {"ok": False, "issues": [f"validator_error:{str(e)}"], "warnings": [], "suggestions": []}

    proj_report = _project_validate(files_map)
    pvpath = paths.validations / "project_validation.json"
    pvpath.write_text(json.dumps(proj_report, indent=2), encoding="utf-8")
    bus.emit("PROJECT_VALIDATED", node_id=node_id, ok=bool(proj_report.get("ok", False)), report=str(pvpath), parent_id=parent_id)

    # Multi-pass project repair loop (language-agnostic, uses micro_adjust with full-context memory)
    if not proj_report.get("ok", False):
        try:
            max_proj_rounds = int(os.environ.get("CRPB_PROJECT_REPAIR_MAX_ROUNDS", "2"))
        except Exception as e:
            max_proj_rounds = 2
            bus.emit("ENV_PARSE_INVALID", node_id=node_id, var="CRPB_PROJECT_REPAIR_MAX_ROUNDS", error=str(e), parent_id=parent_id)
        history_path = paths.validations / "project_history.json"
        history: list[dict] = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception as e:
                history = []
                bus.emit("HISTORY_READ_FAILED", node_id=node_id, file=str(history_path), error=str(e), parent_id=parent_id)
        round_idx = 0
        while round_idx < max_proj_rounds and not proj_report.get("ok", False):
            round_idx += 1
            # Persist report to history
            try:
                hist_entry = {
                    "round": round_idx,
                    "report": proj_report,
                    "timestamp": time.time(),
                }
                history.append(hist_entry)
                atomic_write_json(history_path, history)
            except Exception:
                pass

            # Prepare side_context memory covering all files and issues
            issues = list(proj_report.get("issues", []) or [])
            warnings = list(proj_report.get("warnings", []) or [])
            suggestions = list(proj_report.get("suggestions", []) or [])
            # For each file, attempt a targeted micro-adjust with full project context
            for fpath in file_order:
                fs = specs_by_path.get(fpath)
                if not fs:
                    continue
                lang_eff = effective_language(fs) or ""
                cur_text = files_map.get(fpath, "")
                c_ext = dict(constraints or {})
                c_ext.setdefault("side_context", {})
                c_ext["side_context"].update({
                    "project_issues": issues,
                    "project_warnings": warnings,
                    "project_suggestions": suggestions,
                    "all_files": files_map,
                    "file_specs_map": file_specs_map,
                    "repair_round": round_idx,
                    "plan_nodes": plan_nodes,
                    "plan_validation": plan_validation_obj,
                })
                try:
                    madj = engine.micro_adjust(file=fpath, language=lang_eff, text=cur_text, idea=idea or "", constraints=c_ext)
                    if isinstance(madj, dict) and isinstance(madj.get("text"), str) and madj.get("text").strip():
                        new_text = madj.get("text")
                        out_path = staging_dir / fpath
                        if not validate_file_content(new_text, out_path):
                            # Keep original if invalid
                            continue
                        write_code_file(out_path, new_text)
                        files_map[fpath] = new_text
                        if madj.get("notes"):
                            try:
                                comms.send(from_id=node_id, to_id=node_id, kind="MICROADJUST_ROUND", path=fpath, payload={"round": round_idx, "notes": madj.get("notes", [])})
                            except Exception as e:
                                bus.emit("COMMS_SEND_FAILED", node_id=node_id, to=node_id, path=fpath, kind="MICROADJUST_ROUND", error=str(e), parent_id=parent_id)
                except Exception as e:
                    # Non-fatal: skip this file but log
                    bus.emit("MICROADJUST_FAILED", node_id=node_id, path=fpath, error=str(e), parent_id=parent_id)

            # Re-run holistic validation after adjustments
            proj_report = _project_validate(files_map)
            pvpath = paths.validations / f"project_validation_round_{round_idx}.json"
            pvpath.write_text(json.dumps(proj_report, indent=2), encoding="utf-8")
            bus.emit("PROJECT_VALIDATED", node_id=node_id, ok=bool(proj_report.get("ok", False)), report=str(pvpath), parent_id=parent_id)

    if not proj_report.get("ok", False):
        issues = proj_report.get("issues", [])
        bus.emit("VALIDATION_FAILED", node_id=node_id, kind="project", message="; ".join([str(i) for i in issues])[:500], parent_id=parent_id)
        proj_failed = True

    # Final failure gate after file-level and project-level validation
    if any_failed or proj_failed:
        reason = "one or more files failed validation" if any_failed else "project validation failed"
        status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_IMPL", reason=reason, parent_id=parent_id)
        bus.emit("NODE_FAILED", node_id=node_id, reason=reason, parent_id=parent_id)
        raise typer.Exit(code=1)

    # MERGE: All validations passed. Promote from staging to outputs and update registry atomically per file
    sep("MERGE STAGING -> OUTPUTS")
    merged_files: list[str] = []
    for fpath in file_order:
        fs = specs_by_path[fpath]
        s_path = staging_dir / fpath
        o_path = paths.outputs / fpath
        # Read from staging and write to outputs
        try:
            text = s_path.read_text(encoding="utf-8")
        except Exception as e:
            text = ""
            bus.emit("READ_STAGING_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)
        if not text.strip():
            # Should not happen since pre-merge validation passed, but guard anyway
            status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_IMPL", reason=f"staging_missing_or_empty:{fpath}", parent_id=parent_id)
            bus.emit("NODE_FAILED", node_id=node_id, reason=f"staging_missing_or_empty:{fpath}", parent_id=parent_id)
            raise typer.Exit(code=1)
        # Final validation before writing to outputs
        if not validate_file_content(text, o_path):
            console.print(f"[yellow]Warning: Final content for {fpath} failed basic validation[/yellow]")
        write_code_file(o_path, text)
        merged_files.append(str(o_path.relative_to(paths.outputs)))

        # Update registry now that the file is merged
        def mutate_impl_file(data: dict):
            f = data.setdefault("files", {}).setdefault(fpath, {"functions": {}, "exports": []})
            for fname in (list(fs.exports) or list(fs.functions.keys())):
                f["functions"].setdefault(fname, {})
                f["functions"][fname]["status"] = "implemented"
                f["functions"][fname]["checksum"] = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            f["exports"] = list(fs.exports)
            f["timestamp"] = time.time()
            return data
        try:
            ver_cur, _data_cur = registry.load()
            registry.update(ver_cur, mutate_impl_file)
        except ConflictError:
            ver_retry, _ = registry.load()
            registry.update(ver_retry, mutate_impl_file)

    # Post-merge verification and holistic project validation + self-correction on merged outputs
    sep("POST-MERGE VALIDATION")
    # Quick per-file sanity and (non-fatal) export checks
    for fpath in file_order:
        fs = specs_by_path[fpath]
        opath = paths.outputs / fpath
        assembled = opath.read_text(encoding="utf-8") if opath.exists() else ""
        if not assembled.strip():
            status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_IMPL", reason=f"post_merge_missing:{fpath}", parent_id=parent_id)
            bus.emit("NODE_FAILED", node_id=node_id, reason=f"post_merge_missing:{fpath}", parent_id=parent_id)
            raise typer.Exit(code=1)
        exports_list = list(getattr(fs, "exports", []) or [])
        if exports_list:
            try:
                _ = engine.verify_exports_in_text(
                    file=fpath,
                    language=effective_language(fs) or "",
                    exports=exports_list,
                    text=assembled,
                )
            except Exception as e:
                bus.emit("EXPORT_VERIFY_ERROR", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)

    # Build maps from merged outputs for holistic validation
    out_files_map: dict[str, str] = {}
    out_specs_map: dict[str, dict] = {}
    for fpath, fs in specs_by_path.items():
        op = paths.outputs / fpath
        try:
            out_files_map[fpath] = op.read_text(encoding="utf-8") if op.exists() else ""
        except Exception as e:
            out_files_map[fpath] = ""
            bus.emit("READ_OUTPUT_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)
        try:
            out_specs_map[fpath] = fs.model_dump(exclude_none=True)
        except Exception as e:
            out_specs_map[fpath] = {}
            bus.emit("SPEC_DUMP_FAILED", node_id=node_id, file=fpath, error=str(e), parent_id=parent_id)

    def _project_validate_outputs(cur_files: dict[str, str]) -> dict:
        try:
            c_proj_ext = dict(constraints or {})
            try:
                c_proj_ext.setdefault("side_context", {})
                c_proj_ext["side_context"].update({
                    "plan_nodes": plan_nodes,
                    "plan_artifacts": plan_artifacts,
                    "plan_validation": plan_validation_obj,
                })
            except Exception:
                pass
            return engine.project_validate(
                idea=idea or "",
                constraints=c_proj_ext,
                plan=plan_obj,
                files=cur_files,
                file_specs=out_specs_map,
            )
        except Exception as e:
            return {"ok": False, "issues": [f"validator_error:{str(e)}"], "warnings": [], "suggestions": []}

    out_proj_report = _project_validate_outputs(out_files_map)
    opv = paths.validations / "project_validation_post.json"
    opv.write_text(json.dumps(out_proj_report, indent=2), encoding="utf-8")
    bus.emit("PROJECT_VALIDATED", node_id=node_id, ok=bool(out_proj_report.get("ok", False)), report=str(opv), parent_id=parent_id)

    # Multi-pass holistic repair on outputs using micro_adjust with full project memory
    if not out_proj_report.get("ok", False):
        try:
            max_post_rounds = int(os.environ.get("CRPB_POST_PROJECT_REPAIR_MAX_ROUNDS", "2"))
        except Exception as e:
            max_post_rounds = 2
            bus.emit("ENV_PARSE_INVALID", node_id=node_id, var="CRPB_POST_PROJECT_REPAIR_MAX_ROUNDS", error=str(e), parent_id=parent_id)
        round_idx = 0
        while round_idx < max_post_rounds and not out_proj_report.get("ok", False):
            round_idx += 1
            issues = list(out_proj_report.get("issues", []) or [])
            warnings = list(out_proj_report.get("warnings", []) or [])
            suggestions = list(out_proj_report.get("suggestions", []) or [])
            for fpath in file_order:
                fs = specs_by_path.get(fpath)
                if not fs:
                    continue
                lang_eff = effective_language(fs) or ""
                cur_text = out_files_map.get(fpath, "")
                c_ext = dict(constraints or {})
                c_ext.setdefault("side_context", {})
                c_ext["side_context"].update({
                    "project_issues": issues,
                    "project_warnings": warnings,
                    "project_suggestions": suggestions,
                    "all_files": out_files_map,
                    "file_specs_map": out_specs_map,
                    "repair_round": round_idx,
                    "phase": "post_merge",
                    "plan_nodes": plan_nodes,
                    "plan_validation": plan_validation_obj,
                })
                try:
                    madj = engine.micro_adjust(file=fpath, language=lang_eff, text=cur_text, idea=idea or "", constraints=c_ext)
                    if isinstance(madj, dict) and isinstance(madj.get("text"), str) and madj.get("text").strip():
                        new_text = madj.get("text")
                        opath = paths.outputs / fpath
                        if not validate_file_content(new_text, opath):
                            continue
                        write_code_file(opath, new_text)
                        out_files_map[fpath] = new_text
                        # Update registry checksums/status for adjusted file
                        def _mutate_update_registry(data: dict):
                            f = data.setdefault("files", {}).setdefault(fpath, {"functions": {}, "exports": []})
                            for fname in (list(fs.exports) or list(fs.functions.keys())):
                                f.setdefault("functions", {}).setdefault(fname, {})
                                f["functions"][fname]["status"] = "implemented"
                                f["functions"][fname]["checksum"] = "sha256:" + hashlib.sha256(new_text.encode("utf-8")).hexdigest()
                            f["exports"] = list(fs.exports)
                            f["timestamp"] = time.time()
                            return data
                        try:
                            v1, _d1 = registry.load()
                            registry.update(v1, _mutate_update_registry)
                        except ConflictError:
                            v2, _d2 = registry.load()
                            registry.update(v2, _mutate_update_registry)
                        if madj.get("notes"):
                            try:
                                comms.send(from_id=node_id, to_id=node_id, kind="MICROADJUST_POST", path=fpath, payload={"round": round_idx, "notes": madj.get("notes", [])})
                            except Exception as e:
                                bus.emit("COMMS_SEND_FAILED", node_id=node_id, to=node_id, path=fpath, kind="MICROADJUST_POST", error=str(e), parent_id=parent_id)
                except Exception as e:
                    bus.emit("MICROADJUST_FAILED", node_id=node_id, path=fpath, error=str(e), parent_id=parent_id)

            # Re-run project validation over merged outputs
            out_proj_report = _project_validate_outputs(out_files_map)
            opv_round = paths.validations / f"project_validation_post_round_{round_idx}.json"
            opv_round.write_text(json.dumps(out_proj_report, indent=2), encoding="utf-8")
            bus.emit("PROJECT_VALIDATED", node_id=node_id, ok=bool(out_proj_report.get("ok", False)), report=str(opv_round), parent_id=parent_id)

    if not out_proj_report.get("ok", False):
        status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_IMPL", reason="post_merge_project_validation_failed", parent_id=parent_id)
        bus.emit("NODE_FAILED", node_id=node_id, reason="post_merge_project_validation_failed", parent_id=parent_id)
        raise typer.Exit(code=1)

    # Write final project structure from merged outputs
    atomic_write_json(paths.outputs / "project_structure.json", {"files": merged_files})

    status.write(node_id, "DONE", prev_state="PRODUCED_IMPL", parent_id=parent_id)
    bus.emit("NODE_DONE", node_id=node_id, parent_id=parent_id)

    sep("BUILD DONE")
    if merged_files:
        console.print(f"[green]Build complete[/green]. Outputs: {merged_files}")
    else:
        console.print("[yellow]Build complete with no outputs.[/yellow]")
