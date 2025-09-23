from __future__ import annotations
import json
from pathlib import Path
import os
from typing import Annotated, Optional
import typer
from rich.console import Console
from ..config import resolve_run_dir, make_paths, default_model_for
from ..eventbus import EventBus
from ..llm_config import (
    load_selection,
    require_env_vars,
    effective_model,
)
from ..planner import generate_task_plan
from ..utils.fs import atomic_write_json, ensure_parent
from ..validator import infer_language_from_path, jsonschema_validate, validate_taskplan_general
from ..utils.ui import sep
from ..specs import TaskPlan, CodeSpec
from ..agents.dspy_engine import DspyEngine

app = typer.Typer(help="Plan a project and save idea/constraints plus plan.json (TaskPlan) under run/plan/")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    idea: Annotated[Optional[str], typer.Option("--idea", help="High-level idea to build")] = None,
    constraints: Annotated[str, typer.Option("--constraints", help="JSON string of constraints")] = "{}",
    run_dir: Annotated[Optional[str], typer.Option("--run-dir", help="Base runs folder")] = None,
    run: Annotated[str, typer.Option("--run", help="run_<ts> | latest | new | name")] = "new",
):
    # Pre-flight: require an explicit LLM selection; unsetting removes all implicit inference
    sel = load_selection()
    if not sel:
        console.print("[yellow]No LLM selection found.[/yellow] Use `python -m crpb llm choose` to select a provider and model before planning.")
        raise typer.Exit(code=2)

    # Validate selection requirements neutrally
    ok, missing = require_env_vars(sel)
    if not ok:
        console.print("[yellow]LLM configuration incomplete:[/yellow] " + ", ".join(missing))
        console.print("[yellow]Fix the above before running the planner.[/yellow]")
        raise typer.Exit(code=2)

    eff = effective_model(sel, default_model=default_model_for(sel.provider))
    if not eff:
        console.print(
            "[yellow]A model is required for the current configuration.[/yellow] Set an env default (CRPB_OPENAI_MODEL / CRPB_ANTHROPIC_MODEL / CRPB_HF_MODEL) or persist with `python -m crpb llm choose --model <id>`."
        )
        raise typer.Exit(code=2)

    # Runtime capability check: verify DSPy LM interface is available (no install commentary)
    try:
        import dspy  # type: ignore
    except Exception as e:
        console.print("[yellow]Language model runtime unavailable.[/yellow]")
        raise typer.Exit(code=2)
    if not hasattr(dspy, "LM"):
        console.print("[yellow]Language model interface not available in the runtime.[/yellow]")
        raise typer.Exit(code=2)

    sep("PLAN START")
    base = Path(run_dir) if run_dir else None
    run_path = resolve_run_dir(base, run)
    paths = make_paths(run_path)
    bus = EventBus(paths.logs / "events.jsonl")

    # LLM is required; provider configuration is validated downstream by the engine.

    if idea is None or not str(idea).strip():
        console.print("[red]--idea is required[/red]")
        raise typer.Exit(code=2)

    try:
        constraints_obj = json.loads(constraints)
    except Exception as e:
        bus.emit("CONSTRAINTS_PARSE_ERROR", file="--constraints", error=str(e), parent_id=f"root::{run_path.name}")
        console.print(f"[red]Invalid constraints JSON: {e}")
        raise typer.Exit(code=2)

    # Generate TaskPlan (LLM-backed)
    task_plan: TaskPlan = generate_task_plan(idea, constraints_obj, use_llm=True)
    # Additionally, generate a Spec-first codespec.json upfront so that `build` can run deterministically.

    # Persist artifacts
    atomic_write_json(paths.plan / "idea.json", {"idea": idea})
    atomic_write_json(paths.plan / "constraints.json", constraints_obj)
    # Strip any outputs.files from TaskPlan before saving (TaskPlan must not include file specs)
    try:
        def _strip_files_field(node):
            try:
                outs = getattr(node, "outputs", None)
                if isinstance(outs, dict) and "files" in outs:
                    outs.pop("files", None)
                    setattr(node, "outputs", outs)
            except Exception as e:
                bus.emit("PLAN_STRIP_FILES_FAILED", node_id="planner", error=str(e), parent_id=f"root::{run_path.name}")
            for ch in getattr(node, "children", []) or []:
                _strip_files_field(ch)
        for top in task_plan.tasks:
            _strip_files_field(top)
    except Exception as e:
        bus.emit("PLAN_STRIP_PHASE_FAILED", node_id="planner", error=str(e), parent_id=f"root::{run_path.name}")
    # Embed path/atomic and Socratic QA into TaskSpec.meta; write a single enriched plan.json
    try:
        engine = DspyEngine()

        def _node_view(ts_node) -> dict:
            return {
                "id": getattr(ts_node, "id", None),
                "kind": getattr(ts_node, "kind", None),
                "title": getattr(ts_node, "title", None),
                "description": getattr(ts_node, "description", None),
            }

        def _annotate(ts_list, parent_node: dict, parent_path: str = ""):
            for idx, t in enumerate(ts_list or [], start=1):
                path = f"{parent_path}.{idx}" if parent_path else str(idx)
                siblings_models = [ts_list[j] for j in range(len(ts_list)) if j != (idx - 1)]
                siblings_view = [_node_view(sib) for sib in siblings_models]
                node_view = _node_view(t)
                atomic = len(getattr(t, "children", []) or []) == 0
                try:
                    qa = engine.socratic_interrogate(
                        node=node_view,
                        parent=parent_node,
                        siblings=siblings_view,
                        idea=task_plan.idea,
                        constraints=task_plan.constraints,
                    )
                except Exception as e:
                    qa = {"questions": [], "monologue": ""}
                    bus.emit("SOCRATIC_NODE_FAILED", node_id="planner", path=path, error=str(e), parent_id=f"root::{run_path.name}")
                # Update TaskSpec.meta in place
                try:
                    meta = getattr(t, "meta", {}) or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    meta.update({"path": path, "atomic": atomic, "socratic": qa})
                    setattr(t, "meta", meta)
                except Exception as e:
                    bus.emit("PLAN_META_ENRICH_FAILED", node_id="planner", path=path, error=str(e), parent_id=f"root::{run_path.name}")
                _annotate(getattr(t, "children", []) or [], node_view, path)

        _annotate(task_plan.tasks, parent_node={})
        # Build a clean hierarchical view (no ids/deps) and an artifact index (producers/consumers)
        def _view_node(t) -> dict:
            meta = getattr(t, "meta", {}) or {}
            soc = meta.get("socratic") if isinstance(meta, dict) else {}
            soc_view = {}
            if isinstance(soc, dict):
                soc_view = {"monologue": soc.get("monologue", ""), "questions": list(soc.get("questions", []) or [])[:6]}
            return {
                "title": getattr(t, "title", "") or "",
                "description": getattr(t, "description", "") or "",
                "kind": getattr(t, "kind", "") or "",
                "path": (meta.get("path") if isinstance(meta, dict) else None) or "",
                "atomic": len(getattr(t, "children", []) or []) == 0,
                "socratic": soc_view,
                "children": [_view_node(ch) for ch in (getattr(t, "children", []) or [])],
            }

        plan_view = {
            "idea": task_plan.idea,
            "constraints": task_plan.constraints,
            "roots": [_view_node(t) for t in task_plan.tasks],
        }

        # Artifact index across plan nodes based on inputs.consumes / outputs.produces
        artifacts_index: dict[str, dict] = {}
        def _norm_refs(val):
            if isinstance(val, list):
                arr = val
            elif val is None:
                arr = []
            else:
                arr = [val]
            refs = []
            for r in arr:
                if isinstance(r, str):
                    rid = r.strip()
                    if rid:
                        refs.append({"id": rid})
                elif isinstance(r, dict):
                    rid = r.get("id")
                    if isinstance(rid, str) and rid.strip():
                        refs.append({"id": rid.strip(), **{k: v for k, v in r.items() if k != "id"}})
            return refs
        def _collect_artifacts(ts_list, parent_path: str = ""):
            for idx, t in enumerate(ts_list or [], start=1):
                meta = getattr(t, "meta", {}) or {}
                path = meta.get("path") or (f"{parent_path}.{idx}" if parent_path else str(idx))
                # Producers
                outs = getattr(t, "outputs", {}) or {}
                produces = _norm_refs(outs.get("produces")) if isinstance(outs, dict) else []
                for ref in produces:
                    rid = ref.get("id")
                    if not rid:
                        continue
                    ent = artifacts_index.setdefault(rid, {"producers": [], "consumers": []})
                    if path not in ent["producers"]:
                        ent["producers"].append(path)
                # Consumers
                ins = getattr(t, "inputs", {}) or {}
                consumes = _norm_refs(ins.get("consumes")) if isinstance(ins, dict) else []
                for ref in consumes:
                    rid = ref.get("id")
                    if not rid:
                        continue
                    ent = artifacts_index.setdefault(rid, {"producers": [], "consumers": []})
                    if path not in ent["consumers"]:
                        ent["consumers"].append(path)
                _collect_artifacts(getattr(t, "children", []) or [], path)
        _collect_artifacts(task_plan.tasks)

        # Persist a single enriched plan.json with tasks + view + artifacts
        plan_payload = task_plan.model_dump(exclude_none=True)
        plan_payload["view"] = plan_view
        plan_payload["artifacts"] = {"index": artifacts_index}
        atomic_write_json(paths.plan / "plan.json", plan_payload)
        # General, logical validator over the enriched plan (per-node checks, DAG, artifact shapes)
        try:
            report = validate_taskplan_general(task_plan)
            atomic_write_json(paths.plan / "plan_validation.json", report)
        except Exception as e:
            bus.emit("PLAN_VALIDATION_FAILED", node_id="planner", error=str(e), parent_id=f"root::{run_path.name}")
    except Exception as e:
        bus.emit("PLAN_ENRICH_FAILED", node_id="planner", error=str(e), parent_id=f"root::{run_path.name}")

    # Generate a CodeSpec-first blueprint using DSPy and enrich entries
    try:
        engine = DspyEngine()

        # Lightweight view used by DSPy CodeSpec prompts
        def _to_dict_view(tnode) -> dict:
            return {
                "id": getattr(tnode, "id", None),
                "kind": getattr(tnode, "kind", None),
                "title": getattr(tnode, "title", "") or "",
                "description": getattr(tnode, "description", "") or "",
                "deps": list(getattr(tnode, "deps", []) or []),
                "children": [_to_dict_view(c) for c in (getattr(tnode, "children", []) or [])],
            }

        plan_overview = {"idea": idea, "constraints": constraints_obj}
        tasks_overview = [_to_dict_view(t) for t in task_plan.tasks]

        # Ask LM for initial Codespec root, passing plan view and artifact index as side-context via constraints
        constraints_for_cs = dict(constraints_obj or {})
        try:
            constraints_for_cs.setdefault("side_context", {})
            constraints_for_cs["side_context"].update({
                "plan_view": plan_view,
                "plan_artifacts": {"index": artifacts_index},
            })
        except Exception:
            pass
        root_cs = engine.generate_codespec_root(
            idea=idea,
            constraints=constraints_for_cs,
            plan_overview=plan_overview,
            tasks_overview=tasks_overview,
        )

        files_by_path: dict[str, dict] = {}
        for f in (root_cs or {}).get("files", []) or []:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            if not isinstance(p, str) or not p.strip():
                continue
            fpath = p.strip()
            # Skip directory-only entries
            last = Path(fpath).name
            known_noext = {"README", "LICENSE", "Dockerfile", "Makefile", "Procfile"}
            if ("/" in fpath or "\\" in fpath) and "." not in last and last not in known_noext:
                continue
            entry = {
                "path": fpath,
                "purpose": (f.get("purpose") or "").strip(),
                "description": (f.get("description") or "").strip(),
                "language": (f.get("language") or "").strip() or infer_language_from_path(fpath),
                "imports": list(f.get("imports") or []) or [],
                "exports": list(f.get("exports") or []) or [],
                "functions": dict(f.get("functions") or {}) or {},
                "classes": dict(f.get("classes") or {}) or {},
                "constants": dict(f.get("constants") or {}) or {},
                "entrypoint": f.get("entrypoint") if isinstance(f.get("entrypoint"), str) else None,
                "content": f.get("content") if isinstance(f.get("content"), str) else None,
                "exports_detail": [],
            }
            files_by_path[fpath] = entry

        # Enrich via LM per entry (best-effort)
        for fpath, entry in list(files_by_path.items()):
            try:
                enriched = engine.enrich_codespec_entry(
                    idea=idea,
                    constraints=constraints_for_cs,
                    file_entry=dict(entry),
                )
            except Exception as e:
                bus.emit("CODESPEC_ENRICH_FAILED", path=fpath, error=str(e), parent_id=f"root::{run_path.name}")
                enriched = {}
            if not isinstance(enriched, dict) or not enriched:
                continue
            # Merge conservatively
            if isinstance(enriched.get("functions"), dict):
                fn = entry.setdefault("functions", {})
                for k, v in enriched["functions"].items():
                    if isinstance(k, str) and k.strip():
                        if isinstance(v, str):
                            fn[k.strip()] = {"signature": v}
                        elif isinstance(v, dict):
                            sig = v.get("signature")
                            desc = v.get("description")
                            fd = fn.setdefault(k.strip(), {})
                            if isinstance(sig, str) and sig.strip():
                                fd["signature"] = sig.strip()
                            if isinstance(desc, str) and desc.strip():
                                fd["description"] = desc.strip()
            for key in ("exports", "imports"):
                arr = enriched.get(key)
                if isinstance(arr, list):
                    cur = set(entry.get(key, []))
                    for it in arr:
                        if isinstance(it, str) and it.strip():
                            cur.add(it.strip())
                    entry[key] = sorted(cur)
            if isinstance(enriched.get("entrypoint"), str) and enriched.get("entrypoint").strip():
                entry["entrypoint"] = enriched["entrypoint"].strip()
            if isinstance(enriched.get("classes"), dict):
                cl = entry.setdefault("classes", {})
                for k, v in enriched["classes"].items():
                    if isinstance(k, str) and k.strip():
                        cl.setdefault(k.strip(), v if isinstance(v, dict) else {"description": str(v)})
            if isinstance(enriched.get("constants"), dict):
                cn = entry.setdefault("constants", {})
                for k, v in enriched["constants"].items():
                    if isinstance(k, str) and k.strip():
                        cn.setdefault(k.strip(), v if isinstance(v, dict) else {"value": v})

        # Deterministic description enrichment and exports_detail
        for fpath, entry in files_by_path.items():
            bits: list[str] = []
            if entry.get("language"):
                bits.append(f"Language: {entry['language']}.")
            if entry.get("exports"):
                bits.append("Exports: " + ", ".join(entry["exports"]) + ".")
            if entry.get("imports"):
                bits.append("Imports: " + ", ".join(entry["imports"]) + ".")
            if entry.get("entrypoint"):
                bits.append(f"Entrypoint: {entry['entrypoint']}.")
            fct_names = list(entry.get("functions", {}).keys())
            if fct_names:
                bits.append("Functions: " + ", ".join(fct_names) + ".")
            cls_names = list(entry.get("classes", {}).keys())
            if cls_names:
                bits.append("Classes: " + ", ".join(cls_names) + ".")
            con_names = list(entry.get("constants", {}).keys())
            if con_names:
                bits.append("Constants: " + ", ".join(con_names) + ".")
            base_desc = (entry.get("description") or "").strip()
            if not base_desc:
                base_desc = f"File '{fpath}' for idea: {idea}. "
            entry["description"] = (base_desc + (" " + " ".join(bits) if bits else "")).strip()
            if not entry.get("purpose"):
                entry["purpose"] = "Provide cohesive, testable implementation consistent with declared exports and constraints; keep clear interfaces and separation of concerns."

            exports_detail: list[dict] = []
            for e in entry.get("exports", []) or []:
                if e in entry.get("functions", {}):
                    fobj = entry["functions"][e] or {}
                    exports_detail.append({
                        "name": e,
                        "type": "function",
                        "signature": fobj.get("signature", ""),
                        "description": fobj.get("description", ""),
                    })
                elif e in entry.get("classes", {}):
                    cobj = entry["classes"][e]
                    exports_detail.append({
                        "name": e,
                        "type": "class",
                        "description": (cobj.get("description") if isinstance(cobj, dict) else "") or "",
                    })
                elif e in entry.get("constants", {}):
                    kobj = entry["constants"][e]
                    exports_detail.append({
                        "name": e,
                        "type": "constant",
                        "value": (kobj.get("value") if isinstance(kobj, dict) else kobj),
                        "description": (kobj.get("description") if isinstance(kobj, dict) else "") or "",
                    })
                else:
                    exports_detail.append({"name": e, "type": "unknown"})
            entry["exports_detail"] = exports_detail

        codespec_obj = {"files": sorted(list(files_by_path.values()), key=lambda d: d.get("path", ""))}
        # Validate CodeSpec root schema before writing
        ok_cs, msg_cs = jsonschema_validate(codespec_obj, CodeSpec.model_json_schema())
        if not ok_cs:
            bus.emit("CODESPEC_SCHEMA_FAILED", file=str(paths.plan / "codespec.json"), message=msg_cs, parent_id=f"root::{run_path.name}")
            console.print(f"[red]Generated codespec.json is schema-invalid:[/red] {msg_cs}")
            raise typer.Exit(code=2)
        ensure_parent(paths.plan / "codespec.json")
        atomic_write_json(paths.plan / "codespec.json", codespec_obj)

        # Derive a deterministic file structure view
        try:
            files = [it.get("path", "") for it in codespec_obj.get("files", []) if isinstance(it, dict)]
            files = [p for p in files if isinstance(p, str) and p]
            def _insert(tree: dict, parts: list[str], full: str):
                if not parts:
                    return
                head, *rest = parts
                node = tree.setdefault(head, {"_type": "dir", "_children": {}, "_files": []})
                if rest:
                    _insert(node["_children"], rest, full)
                else:
                    node["_files"].append(full)

            nested: dict = {}
            for p in files:
                norm = str(Path(p).as_posix())
                parts = norm.split("/")
                _insert(nested, parts[:-1], norm)
            lang_by_file = {it.get("path"): it.get("language") for it in codespec_obj.get("files", []) if isinstance(it, dict)}
            def _summarize(node: dict) -> dict:
                out = {"type": "dir", "files": [], "children": {}, "counts": {"files": 0, "dirs": 0}, "languages": {}}
                for f in node.get("_files", []):
                    out["files"].append(f)
                    out["counts"]["files"] += 1
                    lang = lang_by_file.get(f)
                    if lang:
                        out["languages"][lang] = out["languages"].get(lang, 0) + 1
                for name, child in node.get("_children", {}).items():
                    out["children"][name] = _summarize(child)
                    out["counts"]["dirs"] += 1
                    for k, v in out["children"][name]["languages"].items():
                        out["languages"][k] = out["languages"].get(k, 0) + v
                return out
            tree = _summarize({"_children": nested, "_files": []})
            atomic_write_json(paths.plan / "file_structure.json", tree)
        except Exception as e:
            bus.emit("FILE_STRUCTURE_DERIVE_FAILED", error=str(e), parent_id=f"root::{run_path.name}")
    except Exception as e:
        # Non-fatal: proceed without codespec if enrichment fails; build may rely on tasks path
        bus.emit("CODESPEC_GENERATION_FAILED", error=str(e), parent_id=f"root::{run_path.name}")

    sep("PLAN DONE")
    # Brief summary
    try:
        cs_obj = json.loads((paths.plan / "codespec.json").read_text(encoding="utf-8"))
        fcount = len(cs_obj.get("files", []) or [])
    except Exception as e:
        bus.emit("CODESPEC_SUMMARY_READ_FAILED", file=str(paths.plan / "codespec.json"), error=str(e), parent_id=f"root::{run_path.name}")
        fcount = 0
    console.print(f"[green]Planned[/green] run at: {run_path}. Tasks: {len(task_plan.tasks)}. CodeSpec files: {fcount}.")
