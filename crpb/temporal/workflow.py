"""Temporal orchestration layer for CRPB build pipeline.

This module implements a Temporal workflow that orchestrates CRPB build process
in a deterministic, language-agnostic manner. The workflow ensures idempotency
and resilience through proper timeout, retry, and error handling.

Design principles are defined in `docs/design/plan.md` (A1–A6): language-agnostic planning,
artifact-by-reference, deterministic orchestration, and strict no-truncation context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Temporal imports (must be available via pip install temporalio)
try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.worker import Worker
except ImportError:
    activity = None
    workflow = None
    Client = None
    Worker = None
    RetryPolicy = None

logger = logging.getLogger(__name__)


async def _connect_client(
    *,
    server_address: str,
    namespace: str,
    connect_timeout_seconds: float,
    connect_retries: int,
    connect_retry_backoff_seconds: float,
):
    if Client is None:
        raise RuntimeError(
            "temporalio package not installed. Install with: pip install temporalio>=1.0"
        )

    last_err: Exception | None = None
    retries = max(0, int(connect_retries))
    timeout_s = float(connect_timeout_seconds)
    backoff_s = float(connect_retry_backoff_seconds)

    for attempt in range(retries + 1):
        try:
            logger.info(
                f"[temporal_connect] Connecting to Temporal server={server_address} namespace={namespace} "
                f"(attempt {attempt + 1}/{retries + 1}, timeout={timeout_s}s)"
            )
            return await asyncio.wait_for(
                Client.connect(server_address, namespace=namespace), timeout=timeout_s
            )
        except Exception as e:
            last_err = e
            logger.warning(
                f"[temporal_connect] Connection attempt {attempt + 1}/{retries + 1} failed: {e}"
            )
            if attempt >= retries:
                break
            sleep_s = max(0.0, backoff_s * (2**attempt))
            await asyncio.sleep(sleep_s)

    raise RuntimeError(
        "Unable to connect to Temporal server. "
        f"server={server_address} namespace={namespace}. "
        "Temporal Workers require the Temporal Service to be running. "
        "For local dev, start the server with: temporal server start-dev (or temporal.exe server start-dev on Windows). "
        f"Original error: {last_err}"
    )


# Temporal Activity Options (from docs/references/temporal-docs.md)
# LLM-backed activities (planning/validation/repair) can take multiple minutes on real models.
# Too-small timeouts cause activities to time out, then complete late, producing noisy NotFound warnings.
ACTIVITY_TIMEOUT = timedelta(minutes=15)
RETRY_MAX_ATTEMPTS = 3
RETRY_INITIAL_INTERVAL = 5.0
RETRY_BACKOFF_COEFFICIENT = 2.0

if RetryPolicy is not None:
    RETRY_POLICY = RetryPolicy(
        maximum_attempts=RETRY_MAX_ATTEMPTS,
        initial_interval=timedelta(seconds=RETRY_INITIAL_INTERVAL),
        backoff_coefficient=RETRY_BACKOFF_COEFFICIENT,
    )
else:
    RETRY_POLICY = None

# --- Temporal Input/Result Types ---


@dataclass
class BuildInputs:
    """Inputs for CRPB workflow."""

    idea: str
    constraints: Dict[str, Any]
    run_id: str
    run_dir_path: str
    stop_after: str = "build"


@dataclass
class PlanResult:
    """Result from plan generation."""

    ok: bool
    plan: Optional[Dict[str, Any]] = None
    codespec: Optional[Dict[str, Any]] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from plan validation."""

    ok: bool
    issues: List[str] = field(default_factory=list)


# --- Conditional decorators for when temporalio is not installed ---


def _identity_decorator(func):
    """Identity decorator used when temporalio is not installed."""
    return func


# Apply decorators conditionally
_activity_defn = activity.defn if activity is not None else _identity_decorator
_activity_options = (
    getattr(activity, "options", _identity_decorator)
    if activity is not None
    else _identity_decorator
)
_workflow_defn = workflow.defn if workflow is not None else _identity_decorator
_workflow_run = workflow.run if workflow is not None else _identity_decorator
_workflow_signal = workflow.signal if workflow is not None else _identity_decorator
_workflow_query = workflow.query if workflow is not None else _identity_decorator


# Helper to call activities conditionally
async def _execute_activity_async(activity_func, *args, **kwargs):
    """Execute a Temporal activity or call directly if Temporal not available."""
    if _workflow_execute_activity is not None:
        # Temporal SDK expects args to be passed via keyword 'args', not as extra positional params
        call_kwargs = dict(kwargs)
        if "args" not in call_kwargs:
            call_kwargs["args"] = list(args)
        return await _workflow_execute_activity(activity_func, **call_kwargs)
    raise RuntimeError(
        "Temporal execution is required but temporalio is not available (workflow.execute_activity missing). "
        "Install temporalio and run via a Temporal Worker."
    )


_workflow_execute_activity = workflow.execute_activity if workflow is not None else None


def _coerce_build_inputs(inputs: Any) -> BuildInputs:
    if isinstance(inputs, BuildInputs):
        return inputs
    if isinstance(inputs, dict):
        return BuildInputs(
            idea=str(inputs.get("idea") or ""),
            constraints=dict(inputs.get("constraints") or {}),
            run_id=str(inputs.get("run_id") or ""),
            run_dir_path=str(inputs.get("run_dir_path") or ""),
            stop_after=str(inputs.get("stop_after") or "build"),
        )
    raise TypeError("invalid_build_inputs")


# --- Temporal Activities ---


@_activity_defn
async def plan_activity(inputs: Any) -> PlanResult:
    """Generate and validate task plan (T1, T2, T3)."""
    from crpb.agents.dspy_engine import DspyEngine
    from crpb.core.config import make_paths
    from crpb.core.eventbus import EventBus
    from crpb.core.ledger import NodeLedgerStore, TodoItem
    from crpb.core.obligations import (
        extract_obligations_from_node,
        format_structured_todo_text,
        stable_todo_id_from_obligation,
    )
    from crpb.core.specs import CodeSpec
    from crpb.planning.planner import generate_task_plan
    from crpb.utils.fs import atomic_write_json
    from crpb.validation.validator import (
        jsonschema_validate,
        validate_taskplan_general,
    )

    bi = _coerce_build_inputs(inputs)
    logger.info(f"[plan_activity] Starting for run_id: {bi.run_id}")

    run_dir_path = Path(bi.run_dir_path)
    paths = make_paths(run_dir_path)
    bus = EventBus(paths.logs / "events.jsonl")

    bus.emit("TEMPORAL_PLAN_START", run_id=bi.run_id)

    # Generate task plan using CRPB's planner
    try:
        task_plan = generate_task_plan(
            bi.idea,
            bi.constraints,
            use_llm=True,
            enforce_deterministic_leaves=False,
            run_dir_path=str(run_dir_path),
        )
    except Exception as e:
        logger.exception(f"[plan_activity] Task planning failed: {e}")
        bus.emit("TEMPORAL_PLAN_EXCEPTION", run_id=bi.run_id, error=str(e))
        return PlanResult(ok=False, plan=None, codespec=None, issues=[f"planning_failed:{e}"])

    # Validate plan immediately (T1, T2, T3)
    validation_report = validate_taskplan_general(task_plan)
    # Treat artifact coverage issues as warnings for planning; only fail on critical issues
    all_issues = list(validation_report.get("issues", []) or [])
    critical_issues = [
        it
        for it in all_issues
        if not (
            str(it).startswith("artifact_no_consumer:")
            or str(it).startswith("artifact_no_producer:")
        )
    ]

    if critical_issues:
        logger.error(
            f"[plan_activity] Plan validation failed (critical): {', '.join(critical_issues)}"
        )
        bus.emit(
            "TEMPORAL_PLAN_VALIDATION_FAILED",
            run_id=bi.run_id,
            issues=critical_issues,
        )
        return PlanResult(ok=False, plan=None, issues=critical_issues)
    elif all_issues:
        logger.debug(
            f"[plan_activity] Plan validation warnings (non-fatal): {', '.join(all_issues)}"
        )
        bus.emit(
            "TEMPORAL_PLAN_VALIDATION_WARNINGS",
            run_id=bi.run_id,
            issues=all_issues,
        )

    # Persist plan with run_id
    plan_payload = task_plan.model_dump(exclude_none=True)
    plan_payload["run_id"] = bi.run_id
    try:
        plan_payload["_run_dir_path"] = str(bi.run_dir_path)
    except Exception:
        pass

    # Inject missing node_plan keys (T2) – language-agnostic, LLM‑driven
    def inject_missing_keys(node: dict) -> None:
        np = node.setdefault("node_plan", {})
        if not isinstance(np, dict):
            node["node_plan"] = {}
            np = node["node_plan"]
        np.setdefault("intent", "")
        np.setdefault("acceptance_criteria", "")
        np.setdefault("test_plan", "")

    # Recursively inject missing keys into plan structure
    def traverse_and_inject(nodes: List[dict]) -> None:
        for node in nodes:
            inject_missing_keys(node)
            children = node.get("children", [])
            if isinstance(children, list):
                traverse_and_inject(children)

    traverse_and_inject(plan_payload.get("tasks", []))

    # Seed node ledgers with obligation-derived TODOs.
    # This gives every node durable context and allows later context compilation
    # to include structured TODOs without relying on LM memory.
    try:
        ledger_store = NodeLedgerStore(base_dir=paths.artifacts)

        def _walk_seed(nodes: List[dict], parent_id: Optional[str] = None) -> None:
            for n in nodes or []:
                if not isinstance(n, dict):
                    continue
                nid = str(n.get("id") or "").strip()
                if nid:
                    try:
                        led = ledger_store.load(nid, parent_id=parent_id)
                        obs = extract_obligations_from_node(node=n, inherited_deps=list(n.get("deps") or []))

                        # Merge obligations for provenance.
                        led.obligation_items = list(led.obligation_items or [])
                        by_oid = {
                            str(o.get("id") or ""): o
                            for o in (led.obligation_items or [])
                            if isinstance(o, dict)
                        }
                        for o in obs:
                            by_oid[o.id] = o.to_dict()
                        led.obligation_items = list(by_oid.values())

                        # Ensure corresponding TODOs exist and are initially open.
                        todo_by_id = {str(ti.id): ti for ti in (led.todos or [])}
                        for o in obs:
                            tid = stable_todo_id_from_obligation(o.id)
                            if tid in todo_by_id:
                                # Do not override status; just ensure text exists.
                                if not str(todo_by_id[tid].text or "").strip():
                                    todo_by_id[tid].text = format_structured_todo_text(obligation=o)
                            else:
                                led.todos.append(
                                    TodoItem(
                                        id=tid,
                                        text=format_structured_todo_text(obligation=o),
                                        status="open",
                                    )
                                )

                        ledger_store.save(led)
                    except Exception:
                        pass

                children = n.get("children")
                if isinstance(children, list) and children:
                    _walk_seed([c for c in children if isinstance(c, dict)], parent_id=nid or parent_id)

        _walk_seed([t for t in (plan_payload.get("tasks") or []) if isinstance(t, dict)], parent_id=None)
    except Exception:
        pass

    # Enrich with meta.path/atomic and leaf-only socratic summaries (bounded)
    try:
        engine = DspyEngine()
        try:
            max_socratic = int(os.environ.get("CRPB_SOCRATIC_MAX_NODES", "30"))
        except Exception:
            max_socratic = 30
        max_socratic = max(0, min(200, max_socratic))
        soc_used = 0

        def _node_view(n: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": n.get("id"),
                "kind": n.get("kind"),
                "title": n.get("title", "") or "",
                "description": n.get("description", "") or "",
            }

        def _annotate(nodes: List[dict], parent: Dict[str, Any], parent_path: str = "") -> None:
            nonlocal soc_used
            for idx, n in enumerate(nodes or [], start=1):
                if not isinstance(n, dict):
                    continue
                path = f"{parent_path}.{idx}" if parent_path else str(idx)
                meta = n.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                children = n.get("children") or []
                is_leaf = not (isinstance(children, list) and len(children) > 0)
                meta["path"] = str(meta.get("path") or path)
                meta["atomic"] = bool(is_leaf)

                if is_leaf and soc_used < max_socratic:
                    siblings = []
                    if isinstance(nodes, list):
                        for j, s in enumerate(nodes):
                            if j != (idx - 1) and isinstance(s, dict):
                                siblings.append(_node_view(s))
                    try:
                        qa = engine.socratic_interrogate(
                            node=_node_view(n),
                            parent=parent or {},
                            siblings=siblings,
                            idea=bi.idea,
                            constraints=bi.constraints,
                        )
                        if isinstance(qa, dict) and (qa.get("questions") or qa.get("monologue")):
                            meta["socratic"] = qa
                        soc_used += 1
                    except Exception:
                        pass

                n["meta"] = meta
                if isinstance(children, list):
                    _annotate(
                        [c for c in children if isinstance(c, dict)],
                        _node_view(n),
                        meta["path"],
                    )

        _annotate(
            [t for t in (plan_payload.get("tasks") or []) if isinstance(t, dict)],
            parent={},
            parent_path="",
        )
    except Exception:
        pass

    # Add plan view + artifact index (align with plan_cmd conventions)
    plan_view: Dict[str, Any] = {}
    artifacts_index: Dict[str, Dict[str, Any]] = {}
    try:

        def _view_node(n: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": n.get("id"),
                "kind": n.get("kind"),
                "title": n.get("title", "") or "",
                "description": n.get("description", "") or "",
                "deps": list(n.get("deps") or []),
                "children": [
                    _view_node(c) for c in (n.get("children") or []) if isinstance(c, dict)
                ],
            }

        plan_view = {
            "idea": task_plan.idea,
            "constraints": task_plan.constraints,
            "roots": [
                _view_node(t) for t in (plan_payload.get("tasks") or []) if isinstance(t, dict)
            ],
        }

        def _norm_refs(val: Any) -> List[Dict[str, Any]]:
            if isinstance(val, list):
                out: List[Dict[str, Any]] = []
                for it in val:
                    if isinstance(it, dict):
                        out.append(it)
                return out
            return []

        def _walk(nodes: List[Dict[str, Any]], parent_path: str = "") -> None:
            for idx, node in enumerate(nodes or [], start=1):
                if not isinstance(node, dict):
                    continue
                path = f"{parent_path}.{idx}" if parent_path else str(idx)
                ins = node.get("inputs") or {}
                outs = node.get("outputs") or {}
                for ref in _norm_refs(ins.get("consumes")):
                    rid = ref.get("id")
                    if not rid:
                        continue
                    ent = artifacts_index.setdefault(rid, {"producers": [], "consumers": []})
                    if path not in ent["consumers"]:
                        ent["consumers"].append(path)
                for ref in _norm_refs(outs.get("produces")):
                    rid = ref.get("id")
                    if not rid:
                        continue
                    ent = artifacts_index.setdefault(rid, {"producers": [], "consumers": []})
                    if path not in ent["producers"]:
                        ent["producers"].append(path)
                children = node.get("children") or []
                if isinstance(children, list):
                    _walk([c for c in children if isinstance(c, dict)], path)

        _walk([t for t in (plan_payload.get("tasks") or []) if isinstance(t, dict)])
        plan_payload["view"] = plan_view
        plan_payload["artifacts"] = {"index": artifacts_index}
    except Exception:
        plan_view = {}
        artifacts_index = {}

    # Persist validator report for transparency
    try:
        plan_payload["validation"] = validation_report
    except Exception:
        pass

    atomic_write_json(paths.plan / "plan.json", plan_payload)

    # Persist local embeddings index for leaf summaries (optional, explicitly enabled)
    try:
        from crpb.core.llm_config import embeddings_enabled as _embeddings_enabled

        if not bool(_embeddings_enabled()):
            raise RuntimeError("embeddings_disabled")

        provider_pref: list[str] = []
        try:
            pp = os.environ.get("CRPB_TASK_EMBED_PROVIDER_PREFERENCE")
            if isinstance(pp, str) and pp.strip():
                provider_pref = [p.strip() for p in pp.split(",") if p.strip()]
        except Exception:
            provider_pref = []
        if not provider_pref:
            raise RuntimeError("embeddings_provider_preference_unset")

        from crpb.utils.embeddings import build_task_embedding_index

        def _collect_leaf_summaries(nodes: list) -> list[dict]:
            out: list[dict] = []

            def _walk(nlist: list):
                for n in nlist or []:
                    if not isinstance(n, dict):
                        continue
                    meta = n.get("meta") if isinstance(n.get("meta"), dict) else {}
                    ch = n.get("children") or []
                    is_leaf = not (isinstance(ch, list) and len(ch) > 0)
                    if is_leaf:
                        np = n.get("node_plan") if isinstance(n.get("node_plan"), dict) else {}
                        try:
                            intent = (
                                str(np.get("intent") or "").strip() if isinstance(np, dict) else ""
                            )
                        except Exception:
                            intent = ""
                        out.append(
                            {
                                "id": str(n.get("id") or ""),
                                "path": str(meta.get("path") or ""),
                                "title": str(n.get("title") or ""),
                                "summary": str(intent or n.get("description") or ""),
                            }
                        )
                    if isinstance(ch, list):
                        _walk(ch)

            _walk(nodes)
            return [x for x in out if isinstance(x.get("id"), str) and x.get("id").strip()]

        leaf_items = _collect_leaf_summaries(plan_payload.get("tasks", []) or [])
        if leaf_items:
            _ = build_task_embedding_index(
                index_path=paths.registry / "task_embeddings.json",
                items=leaf_items,
                provider_preference=provider_pref,
            )
    except Exception:
        pass

    # Generate CodeSpec deterministically from plan (align with build_cmd expectations)
    codespec_obj: Optional[Dict[str, Any]] = None
    try:
        engine = DspyEngine()

        def _to_dict_view(tnode: Any) -> Dict[str, Any]:
            return {
                "id": getattr(tnode, "id", None),
                "kind": getattr(tnode, "kind", None),
                "title": getattr(tnode, "title", "") or "",
                "description": getattr(tnode, "description", "") or "",
                "deps": list(getattr(tnode, "deps", []) or []),
                "children": [_to_dict_view(c) for c in (getattr(tnode, "children", []) or [])],
            }

        plan_overview = {"idea": bi.idea, "constraints": bi.constraints}
        tasks_overview = [_to_dict_view(t) for t in task_plan.tasks]

        constraints_for_cs = dict(bi.constraints or {})
        constraints_for_cs.setdefault("side_context", {})
        constraints_for_cs["side_context"].update(
            {
                "plan_view": plan_view,
                "plan_artifacts": {"index": artifacts_index},
            }
        )

        root_cs = engine.generate_codespec_root(
            idea=bi.idea,
            constraints=constraints_for_cs,
            plan_overview=plan_overview,
            tasks_overview=tasks_overview,
        )

        def _normalize_obj_map(raw: Any, *, kind: str) -> Dict[str, Dict[str, Any]]:
            """Normalize a map to the CodeSpec schema shape (values must be objects).

            This is a defensive postcondition: LMs sometimes emit primitives for these maps.
            We wrap primitives without introducing stack assumptions.
            """
            out: Dict[str, Dict[str, Any]] = {}
            if not isinstance(raw, dict):
                return out
            for k, v in raw.items():
                if not (isinstance(k, str) and k.strip()):
                    continue
                key = k.strip()
                if isinstance(v, dict):
                    out[key] = v
                    continue
                if kind == "functions":
                    # Allow shorthand string signatures.
                    if isinstance(v, str):
                        out[key] = {"signature": v}
                    else:
                        out[key] = {"signature": str(v)}
                elif kind == "classes":
                    out[key] = {"description": str(v)}
                elif kind == "constants":
                    out[key] = {"value": v}
                else:
                    out[key] = {"value": v}
            return out

        files_by_path: Dict[str, Dict[str, Any]] = {}
        for f in (root_cs or {}).get("files", []) or []:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            if not isinstance(p, str) or not p.strip():
                continue
            fpath = p.strip()

            # Do not special-case particular filenames; extensionless paths are allowed.

            entry: Dict[str, Any] = {
                "path": fpath,
                "purpose": (f.get("purpose") or "").strip(),
                "description": (f.get("description") or "").strip(),
                "language": (f.get("language") or "").strip(),
                "imports": list(f.get("imports") or []) or [],
                "exports": list(f.get("exports") or []) or [],
                "functions": _normalize_obj_map(f.get("functions"), kind="functions"),
                "classes": _normalize_obj_map(f.get("classes"), kind="classes"),
                "constants": _normalize_obj_map(f.get("constants"), kind="constants"),
                "entrypoint": f.get("entrypoint") if isinstance(f.get("entrypoint"), str) else None,
                "content": f.get("content") if isinstance(f.get("content"), str) else None,
                "exports_detail": [],
            }
            files_by_path[fpath] = entry

        for _fpath, entry in list(files_by_path.items()):
            try:
                enriched = engine.enrich_codespec_entry(
                    idea=bi.idea,
                    constraints=constraints_for_cs,
                    file_entry=dict(entry),
                )
            except Exception:
                enriched = {}
            if not isinstance(enriched, dict) or not enriched:
                continue
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
                        cl.setdefault(
                            k.strip(), v if isinstance(v, dict) else {"description": str(v)}
                        )
            if isinstance(enriched.get("constants"), dict):
                cn = entry.setdefault("constants", {})
                for k, v in enriched["constants"].items():
                    if isinstance(k, str) and k.strip():
                        cn.setdefault(k.strip(), v if isinstance(v, dict) else {"value": v})

        for fpath, entry in files_by_path.items():
            bits: List[str] = []
            if entry.get("language"):
                bits.append(f"Language: {entry['language']}.")
            if entry.get("exports"):
                bits.append("Exports: " + ", ".join(entry["exports"]) + ".")
            if entry.get("imports"):
                bits.append("Imports: " + ", ".join(entry["imports"]) + ".")
            if entry.get("entrypoint"):
                bits.append(f"Entrypoint: {entry['entrypoint']}.")
            fct_names = list((entry.get("functions") or {}).keys())
            if fct_names:
                bits.append("Functions: " + ", ".join(fct_names) + ".")
            cls_names = list((entry.get("classes") or {}).keys())
            if cls_names:
                bits.append("Classes: " + ", ".join(cls_names) + ".")
            con_names = list((entry.get("constants") or {}).keys())
            if con_names:
                bits.append("Constants: " + ", ".join(con_names) + ".")
            base_desc = (entry.get("description") or "").strip()
            if not base_desc:
                base_desc = f"File '{fpath}' for idea: {bi.idea}. "
            entry["description"] = (base_desc + (" " + " ".join(bits) if bits else "")).strip()
            if not entry.get("purpose"):
                entry["purpose"] = (
                    "Provide cohesive, testable implementation consistent with declared exports and constraints; keep clear interfaces and separation of concerns."
                )

            exports_detail: List[Dict[str, Any]] = []
            for e in entry.get("exports", []) or []:
                if e in (entry.get("functions") or {}):
                    fobj = (entry.get("functions") or {}).get(e) or {}
                    exports_detail.append(
                        {
                            "name": e,
                            "type": "function",
                            "signature": fobj.get("signature", ""),
                            "description": fobj.get("description", ""),
                        }
                    )
                elif e in (entry.get("classes") or {}):
                    cobj = (entry.get("classes") or {}).get(e)
                    exports_detail.append(
                        {
                            "name": e,
                            "type": "class",
                            "description": (
                                cobj.get("description") if isinstance(cobj, dict) else ""
                            )
                            or "",
                        }
                    )
                elif e in (entry.get("constants") or {}):
                    kobj = (entry.get("constants") or {}).get(e)
                    exports_detail.append(
                        {
                            "name": e,
                            "type": "constant",
                            "value": (kobj.get("value") if isinstance(kobj, dict) else kobj),
                            "description": (
                                kobj.get("description") if isinstance(kobj, dict) else ""
                            )
                            or "",
                        }
                    )
                else:
                    exports_detail.append({"name": e, "type": "unknown"})
            entry["exports_detail"] = exports_detail

        codespec_obj = {
            "files": sorted(files_by_path.values(), key=lambda d: d.get("path", ""))
        }
        ok_cs, msg_cs = jsonschema_validate(codespec_obj, CodeSpec.model_json_schema())
        if not ok_cs:
            logger.error(f"[plan_activity] Generated codespec schema invalid: {msg_cs}")
            bus.emit("TEMPORAL_CODESPEC_SCHEMA_FAILED", run_id=bi.run_id, message=msg_cs)
            return PlanResult(
                ok=False,
                plan=plan_payload,
                codespec=None,
                issues=[f"codespec_schema_invalid:{msg_cs}"],
            )

        atomic_write_json(paths.plan / "codespec.json", codespec_obj)
    except Exception as e:
        logger.exception(f"[plan_activity] CodeSpec generation failed: {e}")
        bus.emit("TEMPORAL_CODESPEC_GENERATION_FAILED", run_id=bi.run_id, error=str(e))
        return PlanResult(
            ok=False, plan=plan_payload, codespec=None, issues=[f"codespec_generation_failed:{e}"]
        )

    bus.emit("TEMPORAL_PLAN_COMPLETE", run_id=bi.run_id)

    return PlanResult(ok=True, plan=plan_payload, codespec=codespec_obj, issues=[])


@_activity_defn
async def validate_plan_activity(plan: Dict[str, Any]) -> ValidationResult:
    """Re-validate plan for consistency (T1, T2, T3)."""
    logger.info("[validate_plan_activity] Re-validating plan")

    from crpb.core.specs import TaskPlan
    from crpb.validation.validator import validate_taskplan_general, validate_taskplan_tree_spider

    # Use CRPB's validator – language‑agnostic
    task_plan = TaskPlan.model_validate(plan)

    validation_report = validate_taskplan_general(task_plan)
    all_issues = list(validation_report.get("issues", []) or [])
    # Filter non-fatal artifact coverage issues for planning
    critical_issues = [
        it
        for it in all_issues
        if not (
            str(it).startswith("artifact_no_consumer:")
            or str(it).startswith("artifact_no_producer:")
        )
    ]

    if critical_issues:
        logger.error(
            f"[validate_plan_activity] Validation critical issues: {', '.join(critical_issues)}"
        )
        return ValidationResult(ok=False, issues=critical_issues)

    # Optional tree-aware spider crawl validator (best-effort; does not block build)
    try:
        run_id = str(plan.get("run_id") or "")
        run_dir_path = None
        try:
            if isinstance(plan.get("_run_dir_path"), str) and plan.get("_run_dir_path"):
                run_dir_path = str(plan.get("_run_dir_path"))
        except Exception:
            run_dir_path = None
        if run_dir_path is None and run_id:
            # Fallback convention: runs/<run_id> (non-authoritative; best-effort only)
            try:
                base = Path.cwd() / "runs"
                cand = base / str(run_id)
                if cand.exists():
                    run_dir_path = str(cand)
            except Exception:
                run_dir_path = None

        if run_dir_path:
            _ = validate_taskplan_tree_spider(
                tp=task_plan, run_dir_path=run_dir_path, run_id=run_id
            )
    except Exception:
        pass

    if all_issues:
        # Keep warnings silent in logs; expose via returned issues only if needed (we return none)
        logger.debug(
            f"[validate_plan_activity] Non-fatal warnings suppressed: {', '.join(all_issues)}"
        )
    return ValidationResult(ok=True, issues=[])


@_activity_defn
async def load_plan_artifacts_activity(run_dir_path: str, run_id: str) -> PlanResult:
    """Load a previously generated plan/codespec from the run directory.

    Build should prefer reusing existing artifacts for the run to avoid introducing
    LLM variance during "build" (which can otherwise fail schema validation even
    when a prior "plan" run succeeded).
    """

    logger.info("[load_plan_artifacts_activity] Loading plan/codespec artifacts")

    try:
        from crpb.core.config import make_paths
        from crpb.core.specs import CodeSpec, TaskPlan
        from crpb.validation.validator import jsonschema_validate
    except Exception as e:
        return PlanResult(ok=False, plan=None, codespec=None, issues=[f"load_plan_import_failed:{e}"])

    try:
        paths = make_paths(Path(str(run_dir_path)))
    except Exception as e:
        return PlanResult(ok=False, plan=None, codespec=None, issues=[f"load_plan_bad_run_dir:{e}"])

    plan_path = paths.plan / "plan.json"
    codespec_path = paths.plan / "codespec.json"
    if not plan_path.exists() or not codespec_path.exists():
        return PlanResult(
            ok=False,
            plan=None,
            codespec=None,
            issues=["load_plan_cache_miss:plan_or_codespec_missing"],
        )

    try:
        plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as e:
        return PlanResult(ok=False, plan=None, codespec=None, issues=[f"load_plan_json_failed:{e}"])

    # Best-effort guard: ensure the loaded plan corresponds to the requested run_id.
    try:
        prun = str((plan_obj or {}).get("run_id") or "")
        if prun and str(run_id) and prun != str(run_id):
            return PlanResult(
                ok=False,
                plan=None,
                codespec=None,
                issues=[f"load_plan_run_id_mismatch:{prun}!={run_id}"],
            )
    except Exception:
        pass

    try:
        _ = TaskPlan.model_validate(plan_obj)
    except Exception as e:
        return PlanResult(ok=False, plan=None, codespec=None, issues=[f"load_plan_schema_invalid:{e}"])

    try:
        codespec_obj = json.loads(codespec_path.read_text(encoding="utf-8"))
    except Exception as e:
        return PlanResult(ok=False, plan=None, codespec=None, issues=[f"load_codespec_json_failed:{e}"])

    try:
        ok_cs, msg_cs = jsonschema_validate(codespec_obj, CodeSpec.model_json_schema())
        if not ok_cs:
            return PlanResult(
                ok=False,
                plan=None,
                codespec=None,
                issues=[f"load_codespec_schema_invalid:{msg_cs}"],
            )
    except Exception as e:
        return PlanResult(ok=False, plan=None, codespec=None, issues=[f"load_codespec_validate_failed:{e}"])

    return PlanResult(ok=True, plan=plan_obj, codespec=codespec_obj, issues=[])


@_activity_defn
async def generate_file_activity(
    file_spec: Dict[str, Any],
    idea: str,
    constraints: Dict[str, Any],
    run_id: str,
    run_dir_path: str,
) -> Dict[str, Any]:
    """Generate a single file using LLM (T5)."""
    from crpb.agents.dspy_engine import DspyEngine
    from crpb.core.config import make_paths
    from crpb.utils.fs import ensure_parent, normalize_project_relpath, safe_join

    logger.info(f"[generate_file_activity] Generating file: {file_spec.get('path', '')}")

    engine = DspyEngine()

    fpath = file_spec.get("path", "")
    language = file_spec.get("language", "") or ""
    exports = file_spec.get("exports", [])
    imports = file_spec.get("imports", [])
    entrypoint = file_spec.get("entrypoint")
    functions = file_spec.get("functions", {})
    classes = file_spec.get("classes", {})
    constants = file_spec.get("constants", {})
    purpose = file_spec.get("purpose", "")
    description = file_spec.get("description", "")
    content = file_spec.get("content")

    # Enrich constraints with file metadata (language‑agnostic side_context)
    enriched_constraints = dict(constraints or {})
    enriched_constraints.setdefault("side_context", {})

    enriched_constraints["side_context"].update(
        {
            "codespec_file": file_spec,
            "run_id": run_id,
        }
    )

    file_text = engine.generate_full_file(
        idea=idea,
        constraints=enriched_constraints,
        file=fpath,
        language=str(language or ""),
        exports=exports,
        imports=imports,
        entrypoint=entrypoint,
        functions=functions,
        purpose=purpose,
        description=description,
        classes=classes,
        constants=constants,
        content=content,
    )

    # Persist to run-scoped staging output to avoid returning large payloads into workflow history.
    try:
        run_dir = Path(str(run_dir_path))
        paths = make_paths(run_dir)
        staging_root = paths.outputs / "_staging"
        fpath_rel = normalize_project_relpath(str(fpath))
        out_path = safe_join(staging_root, fpath_rel)
        ensure_parent(out_path)
        out_path.write_text(str(file_text), encoding="utf-8")
    except Exception as e:
        logger.exception(f"[generate_file_activity] Failed to write staged file: {e}")
        return {"path": str(fpath), "ok": False, "error": f"staging_write_failed:{e}"}

    logger.info(f"[generate_file_activity] Generated file: {fpath} ({len(file_text)} chars)")
    return {"path": str(fpath), "ok": True}


@_activity_defn
async def verify_exports_activity(
    run_dir_path: str,
    file_path: str,
    exports: List[str],
    language: str,
    run_id: str,
) -> Dict[str, Any]:
    """Verify that declared exports exist in file text (T4)."""
    from crpb.agents.dspy_engine import DspyEngine
    from crpb.core.config import make_paths
    from crpb.utils.fs import normalize_project_relpath, safe_join

    logger.info(f"[verify_exports_activity] Verifying {len(exports)} exports in {file_path}")

    engine = DspyEngine()

    lang_eff = str(language or "")
    if not lang_eff.strip():
        lang_eff = ""

    text = ""
    try:
        run_dir = Path(str(run_dir_path))
        paths = make_paths(run_dir)
        staged = safe_join(paths.outputs / "_staging", normalize_project_relpath(str(file_path)))
        if staged.exists():
            text = staged.read_text(encoding="utf-8")
    except Exception:
        text = ""

    result = engine.verify_exports_in_text(
        file=file_path, language=lang_eff, exports=exports, text=text
    )

    ok = result.get("ok", False)
    missing = result.get("missing", exports)

    logger.info(f"[verify_exports_activity] Result: ok={ok}, missing={missing}")

    return {"ok": ok, "missing": missing}


@_activity_defn
async def merge_activity(run_dir_path: str, files: List[str]) -> Dict[str, Any]:
    """Merge generated files into outputs directory (T6)."""
    logger.info(f"[merge_activity] Merging {len(files)} files into run_dir: {run_dir_path}")

    from crpb.core.config import make_paths
    from crpb.utils.fs import ensure_parent, normalize_project_relpath, safe_join

    run_dir = Path(run_dir_path)
    paths = make_paths(run_dir)
    merged: List[str] = []
    staging_root = paths.outputs / "_staging"
    for rel in files:
        if not (isinstance(rel, str) and rel.strip()):
            continue
        try:
            rel_norm = normalize_project_relpath(rel)
        except Exception:
            continue
        src = safe_join(staging_root, rel_norm)
        if not src.exists():
            continue
        out_path = safe_join(paths.outputs, rel_norm)
        ensure_parent(out_path)
        try:
            out_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            merged.append(rel_norm)
        except Exception:
            continue

    return {"ok": True, "count": len(merged), "merged_files": merged}


@_activity_defn
async def project_validate_activity(inputs: Any) -> Dict[str, Any]:
    from crpb.core.config import make_paths
    from crpb.core.eventbus import EventBus
    from crpb.utils.fs import ensure_parent, normalize_project_relpath, safe_join
    from crpb.validation.project_validate import validate_project_outputs
    if not isinstance(inputs, dict):
        raise TypeError("invalid_project_validate_inputs")
    idea = str(inputs.get("idea") or "")
    constraints = dict(inputs.get("constraints") or {})
    plan = dict(inputs.get("plan") or {})
    codespec = dict(inputs.get("codespec") or {})
    files = dict(inputs.get("files") or {})
    file_paths = inputs.get("file_paths") or []
    run_dir_path = str(inputs.get("run_dir_path") or "")
    run_id = str(inputs.get("run_id") or "")

    run_dir = Path(run_dir_path)
    paths = make_paths(run_dir)
    bus = EventBus(paths.logs / "events.jsonl")

    file_specs: Dict[str, Any] = {}
    try:
        for f in codespec.get("files") or []:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            if isinstance(p, str) and p.strip():
                try:
                    p_norm = normalize_project_relpath(p)
                except Exception:
                    continue
                file_specs[p_norm] = f
    except Exception:
        file_specs = {}

    # If file texts are not provided, load from staging (bounded) based on available paths.
    if (not files) and isinstance(file_paths, list) and run_dir_path:
        try:
            run_dir = Path(run_dir_path)
            paths = make_paths(run_dir)
            staging_root = paths.outputs / "_staging"

            available = sorted([p for p in file_paths if isinstance(p, str) and p.strip()])
            # Provide a baseline map for existence checks; actual selection below will bound total.
            files = {}
            for rel in available:
                try:
                    rel_norm = normalize_project_relpath(rel)
                except Exception:
                    continue
                staged = safe_join(staging_root, rel_norm)
                if staged.exists():
                    files[rel_norm] = ""
        except Exception:
            files = {}

    def _select_files_for_project_validation(
        *,
        files_map: Dict[str, str],
        file_specs_map: Dict[str, Any],
        focus_paths: List[str],
        max_total_chars: int,
        max_files: int,
    ) -> Dict[str, str]:
        if not isinstance(files_map, dict) or not files_map:
            return {}
        # If caller provided a path-only placeholder map, hydrate from staging as needed.
        run_dir_local = Path(run_dir_path) if run_dir_path else None
        staging_root_local = None
        try:
            if run_dir_local is not None:
                staging_root_local = make_paths(run_dir_local).outputs / "_staging"
        except Exception:
            staging_root_local = None

        def _read_text(rel_path: str) -> str:
            if staging_root_local is None:
                return str(files_map.get(rel_path, "") or "")
            # If the map already contains non-empty content, prefer it.
            existing = files_map.get(rel_path)
            if isinstance(existing, str) and existing:
                return existing
            try:
                staged = safe_join(staging_root_local, normalize_project_relpath(rel_path))
                if staged.exists():
                    return staged.read_text(encoding="utf-8")
            except Exception:
                return ""
            return ""

        deps: Dict[str, List[str]] = {}
        rev: Dict[str, List[str]] = {}
        for p, spec in (file_specs_map or {}).items():
            if not isinstance(p, str):
                continue
            imps = spec.get("imports") if isinstance(spec, dict) else None
            if not isinstance(imps, list):
                continue
            for imp in imps:
                if not isinstance(imp, str) or not imp.strip():
                    continue
                s = imp.strip().replace("\\", "/")
                if s.startswith("./"):
                    s = s[2:]
                if s in files_map:
                    deps.setdefault(p, []).append(s)
                    rev.setdefault(s, []).append(p)

        seeds: List[str] = []
        for fp in focus_paths or []:
            if isinstance(fp, str) and fp in files_map and fp not in seeds:
                seeds.append(fp)

        if not seeds:
            for p, spec in (file_specs_map or {}).items():
                ep = spec.get("entrypoint") if isinstance(spec, dict) else None
                if isinstance(ep, str) and ep.strip() and p in files_map and p not in seeds:
                    seeds.append(p)

        if not seeds:
            ordered = sorted(
                [p for p in files_map.keys() if isinstance(p, str)],
                key=lambda x: len(str(files_map.get(x, "") or "")),
            )
            seeds = ordered[: max(1, min(int(max_files), 6))]

        queue = list(seeds)
        seen: set[str] = set()
        expanded: List[str] = []
        while queue and len(expanded) < int(max_files):
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            if cur in files_map:
                expanded.append(cur)
            for nxt in deps.get(cur, []) + rev.get(cur, []):
                if nxt not in seen and nxt not in queue:
                    queue.append(nxt)

        out: Dict[str, str] = {}
        total = 0
        for p in expanded:
            txt = _read_text(p)
            if total + len(txt) > int(max_total_chars):
                continue
            out[p] = txt
            total += len(txt)
            if len(out) >= int(max_files):
                break
        return out

    focus_paths: List[str] = []

    try:
        try:
            max_pv_chars = int(os.environ.get("CRPB_PROJECT_VALIDATE_MAX_CHARS", "60000"))
        except Exception:
            max_pv_chars = 60000
        try:
            max_pv_files = int(os.environ.get("CRPB_PROJECT_VALIDATE_MAX_FILES", "20"))
        except Exception:
            max_pv_files = 20
    except Exception:
        max_pv_chars = 60000
        max_pv_files = 20

    # Multi-pass selection: validate all files in bounded chunks of full text.
    # This avoids silently truncating file contents while keeping workflow payload sizes reasonable.
    def _iter_project_validation_chunks() -> List[Dict[str, str]]:
        if not isinstance(files, dict):
            return []

        # Prefer ordering by focus_paths first (connectivity hot spots), then stable remainder.
        all_paths = sorted([p for p in files.keys() if isinstance(p, str) and p.strip()])
        ordered: List[str] = []
        for p in focus_paths or []:
            if isinstance(p, str) and p in all_paths and p not in ordered:
                ordered.append(p)
        for p in all_paths:
            if p not in ordered:
                ordered.append(p)

        # Hydrate from staging if needed (files map may contain placeholders).
        try:
            staging_root = make_paths(Path(run_dir_path)).outputs / "_staging" if run_dir_path else None
        except Exception:
            staging_root = None

        def _read_full_text(rel_path: str) -> str:
            existing = files.get(rel_path)
            if isinstance(existing, str) and existing:
                return existing
            if staging_root is None:
                return str(existing or "")
            try:
                staged = safe_join(staging_root, normalize_project_relpath(rel_path))
                if staged.exists():
                    return staged.read_text(encoding="utf-8")
            except Exception:
                return ""
            return ""

        chunks: List[Dict[str, str]] = []
        cur: Dict[str, str] = {}
        total = 0
        for p in ordered:
            txt = _read_full_text(p)
            # If a single file is huge, give it its own chunk; downstream can choose higher limits.
            if cur and (len(cur) >= int(max_pv_files) or (total + len(txt)) > int(max_pv_chars)):
                chunks.append(cur)
                cur = {}
                total = 0
            cur[p] = txt
            total += len(txt)
            if len(cur) >= int(max_pv_files) or total >= int(max_pv_chars):
                chunks.append(cur)
                cur = {}
                total = 0
        if cur:
            chunks.append(cur)
        return chunks

    # During the workflow, generated files live under outputs/_staging until merge.
    det_root = paths.outputs
    try:
        staging_root = paths.outputs / "_staging"
        if staging_root.exists() and staging_root.is_dir():
            try:
                # Prefer staging if it contains at least one file.
                if any(p.is_file() for p in staging_root.rglob("*")):
                    det_root = staging_root
            except Exception:
                pass
    except Exception:
        pass

    # Run log summary (bounded) is provided as evidence to LLM validation/repair.
    run_logs_summary: Dict[str, Any] = {}
    try:
        from crpb.utils.run_logs import summarize_run_logs

        run_logs_summary = summarize_run_logs(run_dir=Path(run_dir_path))
    except Exception:
        run_logs_summary = {}

    constraints2: Dict[str, Any] = dict(constraints or {})
    sc = constraints2.get("side_context")
    if not isinstance(sc, dict):
        sc = {}
    sc = dict(sc)
    if run_logs_summary:
        sc.setdefault("run_logs", run_logs_summary)
    constraints2["side_context"] = sc

    # Repair budget defaults: keep bounded to reduce token burn.
    # Allow callers to request fewer, but cap to 3 by default.
    try:
        mr_raw = constraints2.get("repair_max_rounds")
        mr = int(mr_raw) if mr_raw is not None else int(os.environ.get("CRPB_REPAIR_MAX_ROUNDS", "3"))
    except Exception:
        mr = 3
    try:
        mf_raw = constraints2.get("repair_max_files_per_round")
        mf = int(mf_raw) if mf_raw is not None else int(os.environ.get("CRPB_REPAIR_MAX_FILES_PER_ROUND", "8"))
    except Exception:
        mf = 8
    constraints2["repair_max_rounds"] = max(0, min(int(mr), 3))
    constraints2["repair_max_files_per_round"] = max(1, min(int(mf), 12))

    # LLM-only, language-neutral project validation (and optional repair loop).
    report = validate_project_outputs(
        outputs_dir=Path(det_root),
        constraints=constraints2,
        idea=idea or "",
        plan=dict(plan or {}),
        file_specs=file_specs,
        validations_dir=paths.validations,
        label="workflow_holistic",
    )

    try:
        pvpath = paths.validations / "project_validation.json"
        ensure_parent(pvpath)
        pvpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pvpath = None

    bus.emit(
        "PROJECT_VALIDATED",
        run_id=run_id,
        ok=bool(report.get("ok", False)),
        report=(str(pvpath) if pvpath is not None else None),
    )
    return report




# --- Workflow Definition ---


@_workflow_defn
class CRPBWorkflow:
    """CRPB build workflow orchestrated by Temporal."""

    def __init__(self) -> None:
        self._paused: bool = False
        self._cancel_requested: bool = False
        self._phase: str = "init"
        self._total_files: int = 0
        self._generated_files: int = 0

    @_workflow_signal
    async def pause(self) -> None:
        self._paused = True

    @_workflow_signal
    async def unpause(self) -> None:
        self._paused = False

    @_workflow_signal
    async def cancel(self) -> None:
        self._cancel_requested = True

    @_workflow_query
    def status(self) -> Dict[str, Any]:
        return {
            "paused": self._paused,
            "cancel_requested": self._cancel_requested,
            "phase": self._phase,
            "generated_files": self._generated_files,
            "total_files": self._total_files,
        }

    @_workflow_run
    async def run(self, inputs: Any) -> Dict[str, Any]:
        """Execute the CRPB build workflow."""
        bi = _coerce_build_inputs(inputs)
        try:
            is_replaying = bool(workflow.unsafe.is_replaying()) if workflow is not None else False
        except Exception:
            is_replaying = False
        if not is_replaying:
            logger.info(f"[CRPBWorkflow] Starting workflow for run_id: {bi.run_id}")

        self._phase = "plan"

        # Step 1: Prefer loading existing plan/codespec artifacts for this run.
        # If absent/unusable, fall back to plan generation.
        plan_result = await _execute_activity_async(
            load_plan_artifacts_activity,
            bi.run_dir_path,
            bi.run_id,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RETRY_POLICY,
        )
        if not plan_result.ok:
            plan_result = await _execute_activity_async(
                plan_activity,
                bi.__dict__,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )

        if not plan_result.ok:
            return {
                "ok": False,
                "error": "Plan/CodeSpec generation failed",
                "issues": plan_result.issues,
            }

        # Step 2: Re-validate plan
        self._phase = "validate_plan"
        validation_result = await _execute_activity_async(
            validate_plan_activity,
            plan_result.plan,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RETRY_POLICY,
        )

        if not validation_result.ok:
            return {
                "ok": False,
                "error": "Plan validation failed",
                "issues": validation_result.issues,
            }

        if str(getattr(bi, "stop_after", "build") or "build").strip() == "plan":
            self._phase = "done"
            return {
                "ok": True,
                "plan": plan_result.plan,
                "codespec": plan_result.codespec,
            }

        # Step 3: Extract files to generate from CodeSpec (authoritative file blueprint)
        self._phase = "extract_files"
        files_to_generate = []
        cs = plan_result.codespec or {}
        # Build a compact project index for cross-file consistency during generation
        project_files_index: List[Dict[str, Any]] = []
        try:
            for f in cs.get("files", []) or []:
                if not isinstance(f, dict):
                    continue
                p = f.get("path")
                if not (isinstance(p, str) and p.strip()):
                    continue
                project_files_index.append(
                    {
                        "path": p.strip(),
                        "language": f.get("language"),
                        "exports": list(f.get("exports") or []),
                        "entrypoint": f.get("entrypoint"),
                    }
                )
        except Exception:
            project_files_index = []

        def _sanitize_imports(spec: Dict[str, Any]) -> Dict[str, Any]:
            """Deterministically normalize import entries without language assumptions.

            CRPB must not hardcode stack-specific behaviors (e.g., appending extensions).
            This function only normalizes path separators and strips whitespace.
            """
            out = dict(spec or {})
            imps = out.get("imports")
            if not isinstance(imps, list):
                return out
            fixed: List[str] = []
            for imp in imps:
                if not isinstance(imp, str):
                    continue
                s_norm = imp.strip().replace("\\", "/")
                if s_norm:
                    fixed.append(s_norm)
            out["imports"] = fixed
            return out

        for f in cs.get("files", []) or []:
            if not isinstance(f, dict):
                continue
            if not isinstance(f.get("path"), str) or not f.get("path"):
                continue
            files_to_generate.append({"file_spec": _sanitize_imports(f)})

        self._total_files = len(files_to_generate)

        # Step 4: Generate files
        self._phase = "generate_files"
        generated_files = []
        for file_info in files_to_generate:
            if self._cancel_requested:
                return {"ok": False, "error": "Canceled", "issues": ["canceled"]}
            if workflow is not None:
                await workflow.wait_condition(lambda: (not self._paused))
            file_spec = file_info["file_spec"]
            # Provide compact cross-file context without overfeeding
            per_file_constraints = dict(bi.constraints or {})
            per_file_constraints.setdefault("side_context", {})
            try:
                if isinstance(per_file_constraints.get("side_context"), dict):
                    per_file_constraints["side_context"].update(
                        {
                            "project_files_index": project_files_index,
                        }
                    )
            except Exception:
                pass

            file_result = await _execute_activity_async(
                generate_file_activity,
                file_spec,
                bi.idea,
                per_file_constraints,
                bi.run_id,
                bi.run_dir_path,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )

            if file_result.get("ok"):
                generated_files.append({"path": file_result.get("path"), "ok": True})
                self._generated_files += 1

        # Step 5: Verify exports
        self._phase = "verify_exports"
        export_issues: List[str] = []

        # Build lookup from codespec to avoid re-parsing run outputs inside the workflow.
        meta_by_path: Dict[str, Dict[str, Any]] = {}
        try:
            for f in (cs.get("files", []) or []):
                if not isinstance(f, dict):
                    continue
                p = f.get("path")
                if not (isinstance(p, str) and p.strip()):
                    continue
                meta_by_path[p.strip()] = {
                    "language": f.get("language"),
                    "exports": list(f.get("exports") or []),
                }
        except Exception:
            meta_by_path = {}

        for fr in generated_files:
            if self._cancel_requested:
                return {"ok": False, "error": "Canceled", "issues": ["canceled"]}
            if workflow is not None:
                await workflow.wait_condition(lambda: (not self._paused))

            p = fr.get("path") if isinstance(fr, dict) else None
            if not (isinstance(p, str) and p.strip()):
                continue
            rel_path = p.strip()

            meta = meta_by_path.get(rel_path) or {}
            exports = list(meta.get("exports") or [])
            if not exports:
                continue

            rep = await _execute_activity_async(
                verify_exports_activity,
                bi.run_dir_path,
                rel_path,
                exports,
                str(meta.get("language") or ""),
                bi.run_id,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )
            if not bool(rep.get("ok", False)):
                missing = rep.get("missing") or exports
                export_issues.append(f"missing_exports:{rel_path}:{missing}")

        if export_issues:
            if not is_replaying:
                logger.warning(
                    f"[CRPBWorkflow] Export verification issues (non-fatal): count={len(export_issues)}"
                )

        self._phase = "project_validate"
        try:
            generated_paths = [str(fr.get("path") or "").strip() for fr in (generated_files or [])]
            generated_paths = [p for p in generated_paths if p]
        except Exception:
            generated_paths = []

        proj_inputs = {
            "idea": bi.idea,
            "constraints": bi.constraints,
            "plan": (plan_result.plan or {}),
            "codespec": (plan_result.codespec or {}),
            "file_paths": generated_paths,
            "run_dir_path": bi.run_dir_path,
            "run_id": bi.run_id,
        }
        proj_report = await _execute_activity_async(
            project_validate_activity,
            proj_inputs,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RETRY_POLICY,
        )
        if not bool(proj_report.get("ok", False)):
            # Project validation failures should fail the build by default.
            # Opt-out is available via constraints for exploratory runs.
            nonfatal = False
            try:
                nonfatal = bool((bi.constraints or {}).get("project_validation_nonfatal"))
            except Exception:
                nonfatal = False

            if nonfatal:
                if not is_replaying:
                    logger.warning(
                        "[CRPBWorkflow] Project validation reported issues (non-fatal via config): "
                        f"count={len([*(proj_report.get('issues') or []), *(proj_report.get('warnings') or []), *(proj_report.get('suggestions') or [])])}"
                    )
            else:
                issues = [str(x) for x in (list(proj_report.get("issues") or [])) if str(x)]
                warnings = [str(x) for x in (list(proj_report.get("warnings") or [])) if str(x)]
                suggestions = [str(x) for x in (list(proj_report.get("suggestions") or [])) if str(x)]
                if not is_replaying:
                    logger.error(
                        "[CRPBWorkflow] Project validation failed (fatal): "
                        f"count={len(issues)}"
                    )
                return {
                    "ok": False,
                    "error": "Project validation failed",
                    "issues": issues,
                    "warnings": warnings,
                    "suggestions": suggestions,
                    "project_validation": proj_report,
                }

        # Step 6: Merge outputs
        self._phase = "merge_outputs"
        merge_result = await _execute_activity_async(
            merge_activity,
            bi.run_dir_path,
            generated_paths,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RETRY_POLICY,
        )

        self._phase = "done"
        if not is_replaying:
            logger.info(f"[CRPBWorkflow] Workflow completed for run_id: {bi.run_id}")

        return {
            "ok": True,
            "generated_files": generated_files,
            "merge_result": merge_result,
            "project_validation": proj_report,
        }


# --- Worker bootstrap ---


async def run_worker(
    task_queue: str = "crpb-task-queue",
    namespace: str = "default",
    server_address: str = "localhost:7233",
    connect_timeout_seconds: float = 10.0,
    connect_retries: int = 0,
    connect_retry_backoff_seconds: float = 1.0,
):
    """Run Temporal worker that executes CRPB activities.

    For self-hosted development (user's case), Temporal Server
    can be started with: temporal server start-dev.
    """
    if Worker is None or Client is None:
        raise RuntimeError(
            "temporalio package not installed. Install with: pip install temporalio>=1.0"
        )

    temporal_client = await _connect_client(
        server_address=server_address,
        namespace=namespace,
        connect_timeout_seconds=connect_timeout_seconds,
        connect_retries=connect_retries,
        connect_retry_backoff_seconds=connect_retry_backoff_seconds,
    )

    logger.info(
        f"[run_worker] Connected to Temporal server={server_address} namespace={namespace}. "
        f"Starting worker polling task_queue={task_queue}"
    )

    worker = Worker(
        temporal_client,
        task_queue=task_queue,
        workflows=[CRPBWorkflow],
        activities=[
            plan_activity,
            load_plan_artifacts_activity,
            validate_plan_activity,
            generate_file_activity,
            verify_exports_activity,
            project_validate_activity,
            merge_activity,
        ],
    )

    logger.info(f"[run_worker] Worker started and polling on queue: {task_queue}")

    try:
        await worker.run()
    except Exception as e:
        logger.exception(f"[run_worker] Worker failed: {e}", exc_info=True)
        raise


# --- Client bootstrap for workflow execution ---


async def start_workflow(
    idea: str,
    constraints: Dict[str, Any],
    run_id: str,
    run_dir: str,
    stop_after: str = "build",
    workflow_id: Optional[str] = None,
    server_address: str = "localhost:7233",
    namespace: str = "default",
    task_queue: str = "crpb-task-queue",
    connect_timeout_seconds: float = 10.0,
    connect_retries: int = 0,
    connect_retry_backoff_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Start CRPB Temporal workflow with deterministic run_id.

    Ensures language‑agnostic orchestration: LLM decides framework and language based on
    constraints/context, no hardcoded logic.

    """
    if Client is None:
        raise RuntimeError(
            "temporalio package not installed. Install with: pip install temporalio>=1.0"
        )

    temporal_client = await _connect_client(
        server_address=server_address,
        namespace=namespace,
        connect_timeout_seconds=connect_timeout_seconds,
        connect_retries=connect_retries,
        connect_retry_backoff_seconds=connect_retry_backoff_seconds,
    )

    inputs = BuildInputs(
        idea=idea,
        constraints=constraints,
        run_id=run_id,
        run_dir_path=run_dir,
        stop_after=stop_after,
    )

    stop = str(stop_after or "build").strip().lower()
    if isinstance(workflow_id, str) and workflow_id.strip():
        wid = str(workflow_id).strip()
    else:
        # IMPORTANT: use a workflow id that is stable per run directory.
        # run_id stays deterministic for idempotent artifacts, while workflow id uniqueness
        # prevents collisions across multiple runs of the same idea/constraints.
        run_name = "run"
        try:
            run_name = Path(run_dir).name or "run"
        except Exception:
            run_name = "run"
        wid = f"{run_id}::{run_name}"
        if stop == "plan":
            wid = f"{wid}::plan"

    logger.info(f"[start_workflow] Executing CRPBWorkflow with inputs for run_id: {run_id}")

    try:
        result = await temporal_client.execute_workflow(
            CRPBWorkflow.run,
            inputs.__dict__,
            id=wid,
            task_queue=task_queue,
        )
    except Exception as e:
        en = e.__class__.__name__
        if en in {"WorkflowAlreadyStartedError", "WorkflowExecutionAlreadyStartedError"}:
            handle = temporal_client.get_workflow_handle(wid)
            try:
                result = await handle.result()
            except Exception as e2:
                logger.error(
                    f"[start_workflow] Existing workflow handle result failed: {type(e2).__name__}: {e2}"
                )
                return {"ok": False, "error": "Workflow execution failed", "details": str(e2)}
        else:
            raise

    if not result.get("ok", False):
        error = result.get("error", "unknown")
        # Do not raise here: callers (CLI) expect a structured result to show
        # actionable issues (e.g., validation failures). Raising hides details.
        logger.error(f"[start_workflow] Workflow execution failed: {error}")
        return result

    logger.info(f"[start_workflow] Workflow completed for run_id: {run_id}")

    return result


async def connect_client(
    *,
    server_address: str = "localhost:7233",
    namespace: str = "default",
    connect_timeout_seconds: float = 10.0,
    connect_retries: int = 0,
    connect_retry_backoff_seconds: float = 1.0,
):
    return await _connect_client(
        server_address=server_address,
        namespace=namespace,
        connect_timeout_seconds=connect_timeout_seconds,
        connect_retries=connect_retries,
        connect_retry_backoff_seconds=connect_retry_backoff_seconds,
    )


# --- Public API ---

__all__ = [
    "CRPBWorkflow",
    "run_worker",
    "start_workflow",
    "connect_client",
    "BuildInputs",
    "PlanResult",
    "ValidationResult",
]
