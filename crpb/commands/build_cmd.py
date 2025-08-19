from __future__ import annotations
import hashlib
import os
import json
import re
import time
from pathlib import Path
import typer
from rich.console import Console
from ..config import resolve_run_dir, make_paths
from ..eventbus import EventBus
from ..status import NodeStatus
from ..leases import Leases
from ..registry import Registry, ConflictError
from ..specs import FileSpec, FunctionSpec, FunctionExample
from ..validator import (
    basic_file_validation,
    validate_assembled_python_file,
    jsonschema_validate,
    simple_style_check,
    import_safety_check,
    integrity_check,
    runtime_validate_examples,
)
from ..aggregator import assemble_python_file, write_code_file, write_file_meta
from ..utils.fs import atomic_write_json, ensure_parent
from ..utils.ui import sep
from ..scheduler import Scheduler
from ..repair import RepairIssue, generate_repair_plan, write_repair_plan

app = typer.Typer(help="Build minimal end-to-end flow with registry, events, leases, validator, and optional LLM impl")
console = Console()


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
    require_llm: bool = typer.Option(True, "--require-llm/--allow-fallback", help="Require LLM for implementations; fail if unavailable"),
    lease_ttl: int = typer.Option(180, "--lease-ttl", min=30, help="TTL seconds for builder lease"),
    child_ttl: int = typer.Option(120, "--child-ttl", min=30, help="TTL seconds for child leases"),
):
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)

    sep("BUILD START")

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

    # Deterministic defaults if not provided
    node_id = node_id or f"builder::{run_path.name}"
    parent_id = parent_id or f"root::{run_path.name}"
    status.write(node_id, "CREATED", parent_id=parent_id)
    lease_id = leases.grant(node_id, ttl=lease_ttl)
    status.write(node_id, "LEASED", prev_state="CREATED", lease_id=lease_id, parent_id=parent_id)
    bus.emit("NODE_CREATED", node_id=node_id, parent_id=parent_id)
    bus.emit("LEASE_GRANTED", node_id=node_id, lease_id=lease_id, parent_id=parent_id)

    sep("SPEC & STUB")

    # Load plan: prefer plan/plan.json, fallback to specs/file_*.json
    specs_by_path: dict[str, FileSpec] = {}
    plan_file = paths.plan / "plan.json"
    if plan_file.exists():
        obj = json.loads(plan_file.read_text(encoding="utf-8"))
        modules = obj.get("modules", [])
        for m in modules:
            for f in m.get("files", []):
                funcs: dict[str, FunctionSpec] = {}
                for fname, fmeta in f.get("functions", {}).items():
                    funcs[fname] = FunctionSpec(
                        name=fname,
                        signature=fmeta.get("signature", f"def {fname}() -> None"),
                        returns=fmeta.get("returns", "None"),
                        description=fmeta.get("description", ""),
                        examples=[FunctionExample(inp=e.get("in", {}), out=e.get("out", {})) for e in fmeta.get("examples", [])],
                        tests=fmeta.get("tests", []),
                        status=fmeta.get("status", "stub"),
                        deps=fmeta.get("deps", []),
                    )
                fs = FileSpec(
                    path=f.get("path"),
                    language=f.get("language", "python"),
                    functions=funcs,
                    exports=f.get("exports", list(funcs.keys())),
                    imports=f.get("imports", []),
                    entrypoint=f.get("entrypoint"),
                )
                specs_by_path[fs.path] = fs
    else:
        # fallback: read individual file specs from specs/
        for sp in paths.specs.glob("file_*.json"):
            try:
                data = json.loads(sp.read_text(encoding="utf-8"))
                fs = FileSpec.model_validate(data)
                specs_by_path[fs.path] = fs
            except Exception:
                continue

    if not specs_by_path:
        console.print("[red]No plan/plan.json or specs found; nothing to build[/red]")
        raise typer.Exit(code=2)

    # Save/refresh file-level spec artifacts to ensure consistency
    for _p, fs in specs_by_path.items():
        _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
        atomic_write_json(paths.specs / f"file_{_fid}.json", fs.model_dump())

    # Publish to registry as stubs for all files
    ver, _data = registry.load()
    def mutate_stub_all(data: dict):
        files = data.setdefault("files", {})
        for fpath, fs in specs_by_path.items():
            f = files.setdefault(fpath, {"functions": {}, "exports": [], "timestamp": None})
            for fname, fmeta in fs.functions.items():
                f["functions"][fname] = {
                    "signature": fmeta.signature,
                    "returns": fmeta.returns,
                    "published_by": node_id,
                    "status": "stub",
                    "checksum": None,
                    "examples": [{"in": e.inp, "out": e.out} for e in fmeta.examples],
                    "tests": fmeta.tests,
                    "deps": fmeta.deps,
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
    for fpath, fs in specs_by_path.items():
        for fname, fmeta in fs.functions.items():
            bus.emit("FUNCTION_PUBLISHED", path=fpath, name=fname, status="stub", parent_id=parent_id)
            if fmeta.deps:
                bus.emit("FUTURE_WAIT", path=fpath, name=fname, deps=fmeta.deps, parent_id=parent_id)
    status.write(node_id, "PRODUCED_STUB", prev_state="LEASED", parent_id=parent_id)

    sep("IMPLEMENTATION LOOP")

    api_key = os.environ.get("OPENAI_API_KEY")
    chunks_dir = paths.artifacts / "chunks"
    ensure_parent(chunks_dir / "dummy.txt")
    # impls per file
    impls_by_path: dict[str, dict[str, str]] = {p: {} for p in specs_by_path.keys()}
    scheduler = Scheduler(max_parallel_children=max_children)
    bus.emit("SCHEDULER_CONFIG", node_id=node_id, max_children=max_children, parent_id=parent_id)

    # Load constraints to inform LLM prompts
    constraints: dict = {}
    constraints_file = paths.plan / "constraints.json"
    if constraints_file.exists():
        try:
            constraints = json.loads(constraints_file.read_text(encoding="utf-8"))
        except Exception:
            constraints = {}

    def implement(fpath: str, fname: str, signature: str, language: str) -> str:
        code_text: str | None = None
        if api_key and language == "python":
            try:
                from ..agents.llm import LLM
                llm = LLM(model=model)
                # Build a stricter, file-aware, code-only prompt
                fs = specs_by_path[fpath]
                fmeta = fs.functions[fname]
                dep_sig = {n: fs.functions[n].signature for n in fmeta.deps if n in fs.functions}
                all_funcs = {n: m.signature for n, m in fs.functions.items()}
                exports = list(fs.exports)
                allowed_imports = list(getattr(fs, "imports", []) or [])
                sys = (
                    "You are an expert Python engineer. Produce ONLY the code for the single requested function.\n"
                    "Hard rules:\n"
                    "- Match the signature EXACTLY (name, params, returns).\n"
                    "- Do NOT add or change parameters, return extra values, or create globals.\n"
                    "- Use only names from the provided file contract (functions/params) and builtins; do NOT invent new external APIs.\n"
                    "- If you reference helpers, use the declared functions in this file; do NOT invent new functions.\n"
                    "- Do NOT add imports beyond the allowed list; prefer pure-Python logic.\n"
                    "- If uncertain about domain specifics or state keys, prefer minimal, safe logic consistent with the signature.\n"
                    "- Output must be ONLY a single Python def block with its body; no comments, no markdown, no extra text."
                )
                user = (
                    f"File: {fpath}\n"
                    f"Exports: {exports}\n"
                    f"AllFunctions: {all_funcs}\n"
                    f"AllowedImports: {allowed_imports}\n"
                    f"TargetFunction: {fname}\n"
                    f"Signature: {signature}\n"
                    f"Description: {fmeta.description}\n"
                    f"Dependencies: {list(dep_sig.keys())}\n"
                    f"DependencySignatures: {dep_sig}\n"
                    f"Constraints: {constraints}\n"
                    "Return ONLY the function code."
                )
                content = llm.complete(system=sys, messages=[{"role": "user", "content": user}], temperature=0.0, model=model)
                m = re.search(r"```(?:python)?\n([\s\S]*?)```", content)
                code_text = (m.group(1) if m else content).strip()
                if not code_text.startswith(f"def {fname}"):
                    code_text = None
            except Exception as e:
                if require_llm:
                    raise
                console.print(f"[yellow]LLM generation failed: {e}. Falling back.[/yellow]")
                bus.emit("LLM_FALLBACK", node_id=node_id, reason=str(e), target=fname, parent_id=parent_id)
        if code_text is None:
            if require_llm:
                raise RuntimeError("LLM required but no code generated")
            # Generic skeleton fallback matches signature
            if language == "python":
                sig = signature.strip()
                if not sig.endswith(":"):
                    sig = sig + ":"
                code_text = f"{sig}\n    raise NotImplementedError()\n"
            else:
                # Non-Python not yet supported for assembly
                code_text = f"{signature} {{\n    // TODO: implement\n}}\n"
        return code_text

    progressed = True
    # Allow enough passes for larger plans
    total_funcs = sum(len(fs.functions) for fs in specs_by_path.values())
    max_iters = max(100, total_funcs * 3)
    iters = 0
    while progressed and iters < max_iters:
        iters += 1
        progressed = False
        ver_cur, data_cur = registry.load()
        ready = scheduler.ready_from_registry(data_cur)
        # choose up to max_children targets whose spec exists and not implemented yet
        targets: list[tuple[str, str]] = []
        for fp, fn in ready:
            if fp in specs_by_path and fn not in impls_by_path.get(fp, {}):
                targets.append((fp, fn))
                if len(targets) >= max_children:
                    break
        if not targets:
            break
        # implement selected targets deterministically
        for fpath, fname in targets:
            fs = specs_by_path[fpath]
            signature = fs.functions[fname].signature
            # create and track a child node for this implementation
            child_id = f"child::{Path(fpath).stem}::{fname}::{func_id(signature)}"
            status.write(child_id, "CREATED", parent_id=node_id)
            child_lease = leases.grant(child_id, ttl=child_ttl)
            status.write(child_id, "LEASED", prev_state="CREATED", lease_id=child_lease, parent_id=node_id)
            bus.emit("CHILD_ASSIGNED", parent=node_id, child=child_id, path=fpath, name=fname, parent_id=parent_id)
            try:
                code_impl = implement(fpath, fname, signature, fs.language)
            except Exception as e:
                status.write(child_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=str(e))
                bus.emit("CHILD_FAILED", parent=node_id, child=child_id, path=fpath, name=fname, reason=str(e), parent_id=parent_id)
                status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_STUB", reason=str(e), parent_id=parent_id)
                bus.emit("NODE_FAILED", node_id=node_id, reason=str(e), parent_id=parent_id)
                raise typer.Exit(code=1)
            impls_by_path.setdefault(fpath, {})[fname] = code_impl
            # write chunk
            cstem = f"{Path(fpath).stem}_{fname}_{func_id(signature)}"
            cext = ".py" if fs.language == "python" else ".txt"
            cpath = chunks_dir / f"{cstem}{cext}"
            cpath.write_text(code_impl, encoding="utf-8")
            # update registry
            def mutate_impl_once(data: dict):
                f = data.setdefault("files", {}).setdefault(fpath, {"functions": {}, "exports": []})
                f["functions"].setdefault(fname, {})
                f["functions"][fname]["status"] = "implemented"
                f["functions"][fname]["checksum"] = "sha256:" + hashlib.sha256(code_impl.encode("utf-8")).hexdigest()
                return data
            try:
                registry.update(ver_cur, mutate_impl_once)
            except ConflictError:
                ver_retry, _ = registry.load()
                registry.update(ver_retry, mutate_impl_once)
            bus.emit("FUNCTION_IMPLEMENTED", path=fpath, name=fname, status="implemented", parent_id=parent_id)
            # child completion for implementation stage
            status.write(child_id, "DONE", prev_state="LEASED", parent_id=node_id)
            bus.emit("CHILD_DONE", parent=node_id, child=child_id, path=fpath, name=fname, parent_id=parent_id)
            # futures ready for dependents across all files
            for other_fpath, other_fs in specs_by_path.items():
                for other, spec in other_fs.functions.items():
                    if fname in spec.deps:
                        bus.emit("FUTURE_READY", path=other_fpath, name=other, dep=fname, parent_id=parent_id)
            # renew lease heartbeat
            try:
                leases.renew(node_id, lease_id, ttl=lease_ttl)
                time.sleep(0.01)
            except Exception:
                pass
            progressed = True

    # ensure all exports implemented per file
    for fpath, fs in specs_by_path.items():
        missing = [name for name in fs.exports if name not in impls_by_path.get(fpath, {})]
        if missing and fs.language == "python":
            msg = f"Not all functions implemented in {fpath}: {missing}"
            console.print(f"[red]{msg}[/red]")
            status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_STUB", reason=msg, parent_id=parent_id)
            bus.emit("NODE_FAILED", node_id=node_id, reason=msg, parent_id=parent_id)
            raise typer.Exit(code=1)

    # Assemble and write outputs per supported language
    out_files: list[str] = []
    for fpath, fs in specs_by_path.items():
        if fs.language == "python":
            assembled = assemble_python_file(fs, impls_by_path.get(fpath, {}))
            out_file = paths.outputs / fpath
            write_code_file(out_file, assembled)
            out_files.append(str(out_file.relative_to(paths.outputs)))
        else:
            # emit an event that non-python file assembly is skipped
            bus.emit("ASSEMBLY_SKIPPED", path=fpath, language=fs.language, parent_id=parent_id)
    status.write(node_id, "PRODUCED_IMPL", prev_state="PRODUCED_STUB", parent_id=parent_id)

    sep("VALIDATE & MERGE")

    # Validate and write metadata per python file
    any_failed = False
    for fpath, fs in specs_by_path.items():
        if fs.language != "python":
            continue
        # create a validation child node per file
        vchild_id = f"validator::{Path(fpath).stem}::{hashlib.sha1(fpath.encode('utf-8')).hexdigest()[:8]}"
        status.write(vchild_id, "CREATED", parent_id=node_id)
        vlease_id = leases.grant(vchild_id, ttl=child_ttl)
        status.write(vchild_id, "LEASED", prev_state="CREATED", lease_id=vlease_id, parent_id=node_id)
        bus.emit("CHILD_ASSIGNED", parent=node_id, child=vchild_id, path=fpath, kind="validation", parent_id=parent_id)
        assembled = (paths.outputs / fpath).read_text(encoding="utf-8")
        # JSON Schema validation for FileSpec
        fs_schema = FileSpec.model_json_schema()
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
            continue

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
            continue

        ok_code, msg_code = validate_assembled_python_file(fs, assembled)
        if not ok_code:
            console.print(f"[red]Code validation failed:[/red] {msg_code}")
            issues = [RepairIssue(kind="syntax_or_missing", message=msg_code, file=fs.path)]
            _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
            plan_path = paths.repair_plans / f"repair_{_fid}.json"
            write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
            bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="ast", message=msg_code, file=fs.path, parent_id=parent_id)
            status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=msg_code)
            bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason=msg_code, parent_id=parent_id)
            any_failed = True
            continue

        ok_style, msg_style = simple_style_check(assembled)
        ok_imports, msg_imports = import_safety_check(assembled, allowed=set(getattr(fs, "imports", []) or []))
        # Import violations are fatal; style violations are warnings only.
        if not ok_imports:
            issues = [RepairIssue(kind="disallowed_imports", message=msg_imports, file=fs.path)]
            if not ok_style:
                issues.append(RepairIssue(kind="style", message=msg_style, file=fs.path))
            _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
            plan_path = paths.repair_plans / f"repair_{_fid}.json"
            write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
            bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="imports", message=msg_imports, file=fs.path, parent_id=parent_id)
            console.print(f"[red]Import safety failed[/red]: imports={msg_imports}; style={msg_style}")
            status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=msg_imports)
            bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason=msg_imports, parent_id=parent_id)
            any_failed = True
            continue
        elif not ok_style:
            # Warn but do not fail
            bus.emit("VALIDATION_WARNING", node_id=node_id, kind="style", message=msg_style, file=fs.path, parent_id=parent_id)
            console.print(f"[yellow]Style warning[/yellow]: {msg_style}")

        # Integrity check: ensure imports/entrypoint present as specified
        ok_integrity, msg_integrity = integrity_check(fs, assembled)
        if not ok_integrity:
            issues = [RepairIssue(kind="integrity", message=msg_integrity, file=fs.path)]
            _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
            plan_path = paths.repair_plans / f"repair_{_fid}.json"
            write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
            bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="integrity", message=msg_integrity, file=fs.path, parent_id=parent_id)
            console.print(f"[red]Integrity failed[/red]: {msg_integrity}")
            status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason=msg_integrity)
            bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason=msg_integrity, parent_id=parent_id)
            any_failed = True
            continue

        # Runtime example validation per function using provided examples
        ok_examples, results = runtime_validate_examples(fs, str(paths.outputs / fpath))
        if not ok_examples:
            issues = []
            for r in results:
                if not r.get("ok"):
                    msg = r.get("error") or f"mismatch: expected={r.get('expected')} got={r.get('got')}"
                    issues.append(RepairIssue(kind="examples", message=msg, file=fs.path, function=r.get("function")))
            _fid = hashlib.sha1(fs.path.encode("utf-8")).hexdigest()[:10]
            plan_path = paths.repair_plans / f"repair_{_fid}.json"
            write_repair_plan(plan_path, generate_repair_plan(node_id, idea or "", fs.path, issues))
            bus.emit("REPAIR_PLANNED", node_id=node_id, file=fs.path, plan=str(plan_path), parent_id=parent_id)
            bus.emit("VALIDATION_FAILED", node_id=node_id, kind="examples", message="runtime examples failed", file=fs.path, parent_id=parent_id)
            console.print(f"[red]Examples failed[/red] for {fpath}; details written to repair plan")
            status.write(vchild_id, "FAILED", prev_state="LEASED", parent_id=node_id, reason="runtime examples failed")
            bus.emit("CHILD_FAILED", parent=node_id, child=vchild_id, path=fpath, kind="validation", reason="runtime examples failed", parent_id=parent_id)
            any_failed = True
            continue
        else:
            bus.emit("EXAMPLES_PASSED", node_id=node_id, file=fs.path, count=sum(len(v.examples) for v in fs.functions.values()), parent_id=parent_id)
            status.write(vchild_id, "DONE", prev_state="LEASED", parent_id=node_id)
            bus.emit("CHILD_DONE", parent=node_id, child=vchild_id, path=fpath, kind="validation", parent_id=parent_id)

        # Write file meta and validations report
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
        vpath.write_text(__import__("json").dumps(validations_report, indent=2), encoding="utf-8")
        bus.emit("VALIDATION_PASSED", node_id=node_id, file=fs.path, report=str(vpath), parent_id=parent_id)

    atomic_write_json(paths.outputs / "project_structure.json", {"files": out_files})

    if any_failed:
        status.write(node_id, "FAILED_FINAL", prev_state="PRODUCED_IMPL", reason="one or more files failed validation", parent_id=parent_id)
        bus.emit("NODE_FAILED", node_id=node_id, reason="one or more files failed validation", parent_id=parent_id)
        raise typer.Exit(code=1)

    status.write(node_id, "DONE", prev_state="PRODUCED_IMPL", parent_id=parent_id)
    bus.emit("NODE_DONE", node_id=node_id, parent_id=parent_id)

    sep("BUILD DONE")
    if out_files:
        console.print(f"[green]Build complete[/green]. Outputs: {out_files}")
    else:
        console.print("[yellow]Build complete with no assembled Python outputs (non-Python files skipped).[/yellow]")
