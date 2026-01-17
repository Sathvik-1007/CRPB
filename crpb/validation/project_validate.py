from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            # Ignore repair-lock sidecars.
            if p.name.endswith(".lock"):
                continue
            # skip staging/hidden run internals
            # If validating the main outputs dir, ignore nested outputs/_staging.
            # If validating the staging dir directly, do NOT exclude everything.
            if "_staging" in p.parts and Path(root).name != "_staging":
                continue
            yield p


def validate_project_outputs(
    *,
    outputs_dir: Path,
    constraints: Optional[Dict[str, Any]] = None,
    idea: str = "",
    plan: Optional[Dict[str, Any]] = None,
    file_specs: Optional[Dict[str, Any]] = None,
    validations_dir: Optional[Path] = None,
    label: str = "project",
) -> Dict[str, Any]:
    """Validate outputs (LLM-only) and optionally repair until clean.

    This validator is intentionally language/tool neutral:
    - No deterministic parsing/linting by language.
    - No filename/extension guessing.
    - All validation is done via the LLM (DSPy) using file text + specs + plan.

    Returns a report dict with keys: ok, issues, warnings, suggestions, (optional) details.
    """

    c = constraints or {}
    plan_obj: Dict[str, Any] = dict(plan or {})

    def _trace_enabled() -> bool:
        try:
            if isinstance(c, dict) and bool(c.get("validation_trace")):
                return True
        except Exception:
            pass
        try:
            v = os.environ.get("CRPB_VALIDATION_TRACE", "").strip().lower()
            return v in {"1", "true", "yes", "y", "on"}
        except Exception:
            return False

    # Validation root normalization:
    # If we're pointed at <run>/outputs but all real files are under outputs/_staging,
    # validate the staging subtree. This avoids false "missing_file" findings caused
    # by intentionally ignoring nested _staging during normal outputs validation.
    root_dir = Path(outputs_dir)
    try:
        if root_dir.name != "_staging":
            staging = root_dir / "_staging"
            if staging.exists() and staging.is_dir():
                has_non_staging_files = any(
                    p.is_file() and ("_staging" not in p.parts) for p in root_dir.rglob("*")
                )
                if not has_non_staging_files:
                    root_dir = staging
    except Exception:
        root_dir = Path(outputs_dir)

    def _plan_context_summary() -> Dict[str, Any]:
        """Bounded, schema-tolerant context bundle for validators.

        Goal: make it obvious what we're doing (validate/repair a generated project)
        and what artifacts define expected structure (plan + file_specs).
        """
        summary: Dict[str, Any] = {
            "label": str(label or "project"),
            "idea": str(idea or ""),
            "expected_files": [],
            "task_roots": [],
        }
        try:
            # expected files from codespec-derived file_specs
            if isinstance(file_specs, dict):
                exp = [str(p).replace("\\", "/") for p in file_specs.keys() if str(p).strip()]
                summary["expected_files"] = exp[:80]
        except Exception:
            pass
        try:
            tasks = plan_obj.get("tasks") if isinstance(plan_obj, dict) else None
            if isinstance(tasks, list):
                roots: List[Dict[str, Any]] = []
                for t in tasks[:10]:
                    if not isinstance(t, dict):
                        continue
                    roots.append(
                        {
                            "id": str(t.get("id") or ""),
                            "kind": str(t.get("kind") or ""),
                            "title": str(t.get("title") or ""),
                        }
                    )
                summary["task_roots"] = roots
        except Exception:
            pass
        return summary

    plan_context = _plan_context_summary()

    # Hidden escape hatch for debugging; default is ON.
    repair_enabled = True
    try:
        if isinstance(c, dict) and c.get("repair_enabled") is False:
            repair_enabled = False
    except Exception:
        repair_enabled = True

    try:
        from crpb.agents.dspy_engine import DspyEngine
        from crpb.repairing.provider import DspyRepairProvider

        engine = DspyEngine()
        provider = DspyRepairProvider(engine=engine)

        try:
            max_pv_chars = int(
                (c or {}).get("project_validate_max_chars")
                or os.environ.get("CRPB_PROJECT_VALIDATE_MAX_CHARS", "60000")
            )
        except Exception:
            max_pv_chars = 60000
        try:
            max_pv_files = int(
                (c or {}).get("project_validate_max_files")
                or os.environ.get("CRPB_PROJECT_VALIDATE_MAX_FILES", "20")
            )
        except Exception:
            max_pv_files = 20

        def _read_all_files(root: Path) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for p in _iter_files(root):
                try:
                    rel = str(p.relative_to(root).as_posix())
                except Exception:
                    continue
                try:
                    out[rel] = p.read_text(encoding="utf-8")
                except Exception:
                    out[rel] = ""
            return out

        def _get_run_log_findings(base_constraints: Dict[str, Any]) -> List[str]:
            """Convert run log evidence into machine-coded findings.

            This does not attempt to deterministically interpret failures; it only surfaces
            bounded, structured evidence from constraints.side_context.run_logs.
            """
            try:
                sc = base_constraints.get("side_context") if isinstance(base_constraints, dict) else None
                if not isinstance(sc, dict):
                    return []
                rl = sc.get("run_logs")
                if not isinstance(rl, dict):
                    return []
                ev = rl.get("events")
                if not isinstance(ev, dict):
                    return []
                recent_errors = ev.get("recent_errors")
                if not isinstance(recent_errors, list) or not recent_errors:
                    return []
                out: List[str] = []
                for e in recent_errors[-30:]:
                    if not isinstance(e, dict):
                        continue
                    et = str(e.get("type") or "").strip()
                    et = et.replace(":", "_")
                    file_raw = e.get("file")
                    if isinstance(file_raw, str) and file_raw.strip():
                        p = file_raw.strip().replace("\\", "/")
                        out.append(f"run_error_evidence:{p}:{et}")
                    else:
                        out.append(f"run_error_evidence:logs/events.jsonl:{et}")
                # Deduplicate while preserving order
                seen: set[str] = set()
                deduped: List[str] = []
                for it in out:
                    if it in seen:
                        continue
                    seen.add(it)
                    deduped.append(it)
                return deduped
            except Exception:
                return []

        def _bottom_up_plan_nodes(plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Return a bottom-up ordered list of plan node capsules.

            Each capsule includes: node_id, node, parent, siblings, children, depth, touched_paths.
            Best-effort and intentionally schema-tolerant.
            """

            tasks = plan_data.get("tasks") if isinstance(plan_data, dict) else None
            if not isinstance(tasks, list):
                return []

            node_by_id: Dict[str, Dict[str, Any]] = {}
            parent_of: Dict[str, str] = {}
            children_of: Dict[str, List[str]] = {}

            anon_counter = 0

            def node_key(t: Any) -> str:
                nonlocal anon_counter
                if isinstance(t, dict):
                    tid = t.get("id")
                    if isinstance(tid, str) and tid.strip():
                        return tid.strip()
                    inp = t.get("inputs")
                    if isinstance(inp, dict):
                        p = inp.get("path")
                        if isinstance(p, str) and p.strip():
                            return f"path:{p.strip()}"
                anon_counter += 1
                return f"anon:{anon_counter}"

            def walk(t: Any, parent_id: Optional[str], depth: int) -> set[str]:
                if not isinstance(t, dict):
                    return set()
                nid = node_key(t)
                node_by_id[nid] = t
                if parent_id:
                    parent_of[nid] = parent_id
                    children_of.setdefault(parent_id, []).append(nid)

                kind = str(t.get("kind") or "")
                touched: set[str] = set()
                if kind == "code:function":
                    inp = t.get("inputs")
                    if isinstance(inp, dict):
                        p = inp.get("path")
                        if isinstance(p, str) and p.strip():
                            touched.add(p.strip().replace("\\", "/"))

                kids = t.get("children")
                if isinstance(kids, list):
                    for ch in kids:
                        touched |= walk(ch, nid, depth + 1)
                t.setdefault("_depth", depth)
                t.setdefault("_touched_paths", sorted(touched))
                return touched

            for root in tasks:
                walk(root, None, 0)

            # Build capsules
            capsules: List[Dict[str, Any]] = []
            for nid, n in node_by_id.items():
                pid = parent_of.get(nid)
                parent = node_by_id.get(pid, {}) if pid else {}
                child_ids = children_of.get(nid, [])
                children = [node_by_id.get(cid, {}) for cid in child_ids if cid in node_by_id]
                siblings: List[Dict[str, Any]] = []
                if pid:
                    for sid in children_of.get(pid, []):
                        if sid != nid and sid in node_by_id:
                            siblings.append(node_by_id[sid])
                depth_val = 0
                try:
                    depth_val = int(n.get("_depth") or 0)
                except Exception:
                    depth_val = 0
                touched_paths = n.get("_touched_paths")
                if not isinstance(touched_paths, list):
                    touched_paths = []
                touched_paths2 = [str(x) for x in touched_paths if str(x).strip()]

                capsules.append(
                    {
                        "node_id": nid,
                        "node": n,
                        "parent": parent,
                        "siblings": siblings,
                        "children": children,
                        "depth": depth_val,
                        "touched_paths": touched_paths2,
                    }
                )

            # Bottom-up order: deeper nodes first
            capsules.sort(key=lambda x: int(x.get("depth") or 0), reverse=True)
            return capsules

        def _iter_chunks(files_map: Dict[str, str]) -> List[Dict[str, str]]:
            paths = [p for p in (files_map or {}).keys() if isinstance(p, str) and p.strip()]

            # Optional priority ordering: ensure recently-edited or user-specified focus files
            # are validated first (helps when validation is chunked and agentic).
            priority: List[str] = []
            try:
                sc = (c or {}).get("side_context") if isinstance(c, dict) else None
                if isinstance(sc, dict) and isinstance(sc.get("validation_priority_paths"), list):
                    for it in sc.get("validation_priority_paths") or []:
                        s = str(it or "").strip().replace("\\", "/")
                        if s:
                            priority.append(s)
            except Exception:
                priority = []
            priority = list(dict.fromkeys(priority))

            prio_present = [p for p in priority if p in paths]
            rest = sorted([p for p in paths if p not in set(prio_present)])
            ordered = prio_present + rest
            chunks: List[Dict[str, str]] = []
            cur: Dict[str, str] = {}
            total = 0
            for p in ordered:
                txt = str(files_map.get(p) or "")
                if cur and (
                    len(cur) >= int(max_pv_files) or (total + len(txt)) > int(max_pv_chars)
                ):
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

        def _jury_profiles() -> List[str]:
            # Distinct validators (not rounds). Allow override via constraints.
            v = None
            try:
                v = (c or {}).get("project_jury_profiles")
            except Exception:
                v = None
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                out = [str(x).strip() for x in v if str(x).strip()]
                return out or ["default"]
            return [
                "default",
                "integration",
                "contracts",
                "completeness",
                "neutrality",
                "contrarian",
            ]

        def _apply_profile_rules(base_constraints: Dict[str, Any], profile: str) -> Dict[str, Any]:
            c2: Dict[str, Any] = dict(base_constraints or {})
            adds = list(c2.get("rule_blocks_add") or [])
            rb = {
                "integration": "project_validate_integration",
                "contracts": "project_validate_contracts",
                "completeness": "project_validate_completeness",
                "neutrality": "project_validate_neutrality",
                "contrarian": "project_validate_contrarian",
            }.get(profile)
            if rb:
                adds = list(dict.fromkeys(adds + [rb]))
            if adds:
                c2["rule_blocks_add"] = adds
            return c2

        def _compute_report(root: Path) -> Dict[str, Any]:
            files_local = _read_all_files(root)
            # Run log evidence is treated as first-class context and also emitted as machine-coded findings.
            run_log_findings = _get_run_log_findings(dict(c or {}) if isinstance(c, dict) else {})

            # Bottom-up, node-scoped validation (agentic): validate node-relevant subsets and compress into memory.
            node_details: Dict[str, Any] = {
                "enabled": True,
                "nodes": [],
                "summary": "",
            }
            node_prior_summary = ""
            node_key_issues: List[str] = []

            try:
                max_nodes = int(
                    (c or {}).get("node_validate_max_nodes")
                    or os.environ.get("CRPB_NODE_VALIDATE_MAX_NODES", "40")
                )
            except Exception:
                max_nodes = 40
            try:
                max_node_files = int(
                    (c or {}).get("node_validate_max_files")
                    or os.environ.get("CRPB_NODE_VALIDATE_MAX_FILES", "12")
                )
            except Exception:
                max_node_files = 12

            try:
                capsules = _bottom_up_plan_nodes(plan_obj)
                for cap in capsules[: max(0, int(max_nodes))]:
                    touched = [str(x) for x in (cap.get("touched_paths") or [])]
                    # Bound file subset; fall back to empty if no touched paths.
                    picked: List[str] = []
                    for p in touched:
                        if p in files_local:
                            picked.append(p)
                        if len(picked) >= int(max_node_files):
                            break
                    files_subset = {p: files_local.get(p, "") for p in picked}
                    specs_subset = {p: (file_specs or {}).get(p, {}) for p in picked}

                    scn = (
                        dict((c or {}).get("side_context") or {}) if isinstance(c, dict) else {}
                    )
                    # Make the local intent explicit so node-level validators don't drift.
                    scn.setdefault("plan_context", plan_context)
                    scn.setdefault(
                        "workflow_context",
                        {
                            "phase": "node_validate",
                            "goal": "Validate plan node bottom-up using only provided files/specs/log evidence.",
                        },
                    )
                    if node_prior_summary:
                        scn["prior_findings"] = node_prior_summary
                        scn["prior_key_issues"] = node_key_issues
                    if run_log_findings:
                        scn.setdefault("run_log_findings", run_log_findings)
                    # Provide an explicit capsule as side context for transparency/debug.
                    scn["node_capsule"] = {
                        "node_id": cap.get("node_id"),
                        "depth": cap.get("depth"),
                        "touched_paths": picked,
                    }

                    cnode = dict(c or {})
                    cnode["side_context"] = scn
                    repn = engine.node_validate(
                        idea=str(idea or ""),
                        constraints=cnode,
                        node=dict(cap.get("node") or {}),
                        parent=dict(cap.get("parent") or {}),
                        siblings=list(cap.get("siblings") or []),
                        children=list(cap.get("children") or []),
                        files=files_subset,
                        file_specs=specs_subset,
                    )

                    node_details["nodes"].append(
                        {
                            "node_id": str(cap.get("node_id") or ""),
                            "kind": str((cap.get("node") or {}).get("kind") or ""),
                            "depth": int(cap.get("depth") or 0),
                            "touched_paths": picked,
                            "ok": bool(repn.get("ok", False)),
                            "issues": list(repn.get("issues") or []),
                            "warnings": list(repn.get("warnings") or []),
                            "suggestions": list(repn.get("suggestions") or []),
                        }
                    )

                    # Update node memory capsule
                    try:
                        node_findings_for_summary = (
                            [str(x) for x in (repn.get("issues") or [])]
                            + [str(x) for x in (repn.get("warnings") or [])]
                            + [str(x) for x in (repn.get("suggestions") or [])]
                        )
                        node_findings_for_summary = [x for x in node_findings_for_summary if x]
                        if len(node_findings_for_summary) > 120:
                            node_findings_for_summary = node_findings_for_summary[-120:]
                        srepn = engine.summarize_findings(
                            findings=node_findings_for_summary,
                            constraints={
                                "side_context": {
                                    "prior_summary": node_prior_summary,
                                    "run_logs": scn.get("run_logs") if isinstance(scn, dict) else {},
                                }
                            },
                        )
                        node_prior_summary = str(srepn.get("summary") or "")
                        node_key_issues = list(srepn.get("key_issues") or [])
                        node_details["summary"] = node_prior_summary
                    except Exception:
                        pass
            except Exception:
                node_details["enabled"] = False

            chunks = _iter_chunks(files_local)
            if _trace_enabled():
                try:
                    logger.info(
                        "[project_validate] files=%d chunks=%d profiles=%s max_files=%s max_chars=%s",
                        int(len(files_local)),
                        int(len(chunks)),
                        str(_jury_profiles()),
                        str(max_pv_files),
                        str(max_pv_chars),
                    )
                except Exception:
                    pass
            merged: Dict[str, Any] = {
                "ok": True,
                "issues": [],
                "warnings": [],
                "suggestions": [],
            }
            jury_details: Dict[str, Any] = {"profiles": [], "summary": ""}
            prior_summary = ""
            key_issues: List[str] = []

            # Surface run-log evidence as warnings (never as the only signal).
            if run_log_findings:
                merged["warnings"].extend(run_log_findings)

            for ch_i, ch in enumerate(chunks, start=1):
                ch_specs = {p: (file_specs or {}).get(p, {}) for p in (ch or {}).keys()}
                for profile in _jury_profiles():
                    t_call = time.perf_counter()
                    try:
                        logger.info(
                            "[project_validate] validating chunk=%d/%d profile=%s files=%d",
                            int(ch_i),
                            int(len(chunks)),
                            str(profile),
                            int(len(ch or {})),
                        )
                    except Exception:
                        pass
                    sc = (
                        dict((c or {}).get("side_context") or {})
                        if isinstance(c, dict)
                        else {}
                    )
                    if prior_summary:
                        sc["prior_findings"] = prior_summary
                        sc["prior_key_issues"] = key_issues
                    if node_prior_summary:
                        sc.setdefault("node_findings", node_key_issues)
                        sc.setdefault("node_findings_summary", node_prior_summary)
                    # Make the global intent explicit so jury stays aligned to the plan.
                    sc.setdefault("plan_context", plan_context)
                    sc.setdefault(
                        "workflow_context",
                        {
                            "phase": "project_validate",
                            "goal": "Holistic project validation against plan and codespec-derived file_specs.",
                            "profile": str(profile),
                        },
                    )
                    cbase = dict(c or {})
                    cbase["side_context"] = sc
                    c_profile = _apply_profile_rules(cbase, profile)

                    rep = engine.project_validate(
                        idea=str(idea or ""),
                        constraints=c_profile,
                        plan=plan_obj,
                        files=dict(ch or {}),
                        file_specs=dict(ch_specs or {}),
                    )

                    try:
                        dt = time.perf_counter() - t_call
                        logger.info(
                            "[project_validate] done chunk/profile in %.2fs (issues=%d warnings=%d suggestions=%d)",
                            float(dt),
                            int(len(rep.get("issues") or [])),
                            int(len(rep.get("warnings") or [])),
                            int(len(rep.get("suggestions") or [])),
                        )
                    except Exception:
                        pass

                    for k in ("issues", "warnings", "suggestions"):
                        vals = rep.get(k) or []
                        if isinstance(vals, list):
                            merged[k].extend([str(x) for x in vals if str(x)])

                    jury_details["profiles"].append(
                        {
                            "profile": profile,
                            "issues": list(rep.get("issues") or []),
                            "warnings": list(rep.get("warnings") or []),
                            "suggestions": list(rep.get("suggestions") or []),
                        }
                    )

                    # Update memory capsule using LLM summarizer (bounded).
                    try:
                        findings_for_summary = (
                            [str(x) for x in (merged.get("issues") or [])]
                            + [str(x) for x in (merged.get("warnings") or [])]
                            + [str(x) for x in (merged.get("suggestions") or [])]
                        )
                        findings_for_summary = [x for x in findings_for_summary if x]
                        if len(findings_for_summary) > 250:
                            findings_for_summary = findings_for_summary[-250:]
                        srep = engine.summarize_findings(
                            findings=findings_for_summary,
                            constraints={
                                "side_context": {
                                    "prior_summary": prior_summary,
                                    "run_logs": sc.get("run_logs") if isinstance(sc, dict) else {},
                                }
                            },
                        )
                        prior_summary = str(srep.get("summary") or "")
                        key_issues = list(srep.get("key_issues") or [])
                        jury_details["summary"] = prior_summary
                    except Exception:
                        pass

            # Deduplicate while preserving order.
            for k in ("issues", "warnings", "suggestions"):
                seen: set[str] = set()
                out: List[str] = []
                for it in merged.get(k) or []:
                    s = str(it)
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
                merged[k] = out

            # Chunk-safety: the jury sees files in chunks, but findings are merged globally.
            # Filter out "missing_*" / "empty_*" findings that contradict the actual files.
            # These codes must be evidence-based; when the LM hallucinates them due to chunking
            # or prompt drift, we deterministically reconcile against what is on disk.
            try:
                from crpb.utils.fs import normalize_project_relpath

                def _norm_rel(p: str) -> str:
                    try:
                        return normalize_project_relpath(p)
                    except Exception:
                        # Best-effort normalization for already-relative paths.
                        s2 = str(p or "").strip().replace("\\", "/")
                        while s2.startswith("/"):
                            s2 = s2[1:]
                        while s2.startswith("./"):
                            s2 = s2[2:]
                        return s2

            except Exception:

                def _norm_rel(p: str) -> str:
                    s2 = str(p or "").strip().replace("\\", "/")
                    while s2.startswith("/"):
                        s2 = s2[1:]
                    while s2.startswith("./"):
                        s2 = s2[2:]
                    return s2

            files_index = {_norm_rel(str(p)) for p in (files_local or {}).keys()}
            nonempty_files = {
                _norm_rel(str(p))
                for p, txt in (files_local or {}).items()
                if isinstance(p, str)
                and isinstance(txt, str)
                and txt.strip() != ""
            }

            def _filter_findings(findings: List[str]) -> List[str]:
                out2: List[str] = []
                for s in findings or []:
                    parts = [p.strip() for p in str(s).split(":")]
                    code = parts[0] if parts else ""

                    # Most findings follow code:path:..., but some suggestions are prefixed.
                    # We normalize the first plausible path segment.
                    raw_path = ""
                    if len(parts) >= 2:
                        raw_path = parts[1]
                        if code == "suggestion" and len(parts) >= 3:
                            # suggestion:add_missing_file:<path>
                            raw_path = parts[2]

                    path = _norm_rel(raw_path) if raw_path else ""

                    # Deterministic reconciliation: existing files are not missing.
                    if path and path in files_index:
                        if code in (
                            "missing_file",
                            "missing_init",
                            "missing_init_file",
                            "missing_entrypoint_file",
                            "add_missing_file",
                            "add_init_file",
                            "add_file",
                        ):
                            continue
                        if code == "suggestion" and len(parts) >= 2 and parts[1] in (
                            "add_missing_file",
                            "add_init_file",
                            "add_file",
                            "create_file",
                            "add_entrypoint_file",
                        ):
                            continue

                    # Deterministic reconciliation: non-empty files are not empty.
                    if path and path in nonempty_files:
                        if code in ("empty_file", "empty", "file_empty"):
                            continue

                    out2.append(str(s))
                return out2

            for k in ("issues", "warnings", "suggestions"):
                merged[k] = _filter_findings(list(merged.get(k) or []))

            merged["ok"] = len(list(merged.get("issues") or [])) == 0
            # Deterministic repair focus hints derived from machine-coded findings.
            repair_hints: Dict[str, Any] = {"paths": []}
            try:
                from crpb.repairing.parse import parse_issue_strings
                from crpb.repairing.models import IssueSeverity

                all_findings = (
                    [str(x) for x in (merged.get("issues") or [])]
                    + [str(x) for x in (merged.get("warnings") or [])]
                    + [str(x) for x in (merged.get("suggestions") or [])]
                )
                parsed = parse_issue_strings(all_findings, severity=IssueSeverity.warning)
                paths_seen: set[str] = set()
                paths_out: List[str] = []
                for iss in parsed:
                    p = (iss.path or "").strip()
                    if not p:
                        continue
                    p2 = p.replace("\\", "/")
                    # Prefer existing output file paths.
                    if p2 in files_local and p2 not in paths_seen:
                        paths_seen.add(p2)
                        paths_out.append(p2)
                    if len(paths_out) >= 25:
                        break
                repair_hints["paths"] = paths_out
            except Exception:
                repair_hints = {"paths": []}

            # Probing questions to drive repair/clarification.
            questions: List[Dict[str, Any]] = []
            try:
                scq = dict((c or {}).get("side_context") or {}) if isinstance(c, dict) else {}
                if prior_summary:
                    scq["prior_findings"] = prior_summary
                    scq["prior_key_issues"] = key_issues
                if node_prior_summary:
                    scq.setdefault("node_findings", node_key_issues)
                    scq.setdefault("node_findings_summary", node_prior_summary)
                cq = dict(c or {})
                cq["side_context"] = scq
                qrep = engine.validation_questions(
                    idea=str(idea or ""),
                    constraints=cq,
                    plan=plan_obj,
                    findings=(
                        [str(x) for x in (merged.get("issues") or [])]
                        + [str(x) for x in (merged.get("warnings") or [])]
                        + [str(x) for x in (merged.get("suggestions") or [])]
                    ),
                )
                qlist = qrep.get("questions") if isinstance(qrep, dict) else None
                if isinstance(qlist, list):
                    questions = [q for q in qlist if isinstance(q, dict)]
            except Exception:
                questions = []

            merged["details"] = {
                "jury": jury_details,
                "nodes": node_details,
                "questions": questions,
                "repair_hints": repair_hints,
                "label": str(label or "project"),
            }

            # Provide deterministic numeric importance scores for ordering.
            # This stays language/tool neutral: it only uses machine-coded issue keys.
            try:
                from crpb.repairing.parse import parse_issue_strings
                from crpb.repairing.models import IssueSeverity
                from crpb.repairing.score import importance as _importance

                def _scored(items: List[Any], *, sev: IssueSeverity) -> List[Dict[str, Any]]:
                    parsed = parse_issue_strings(list(items or []), severity=sev)
                    out: List[Dict[str, Any]] = []
                    for it in parsed:
                        out.append(
                            {
                                "item": str(getattr(it, "message", "") or ""),
                                "importance": int(_importance(it)),
                                "code": str(getattr(it, "code", "") or ""),
                                "path": str(getattr(it, "path", "") or ""),
                                "ref": str(getattr(it, "ref", "") or ""),
                                "severity": str(getattr(it, "severity", "") or ""),
                            }
                        )
                    out.sort(key=lambda x: (-int(x.get("importance") or 0), str(x.get("path") or ""), str(x.get("code") or "")))
                    return out

                merged["details"]["importance"] = {
                    "issues": _scored(merged.get("issues") or [], sev=IssueSeverity.fatal),
                    "warnings": _scored(merged.get("warnings") or [], sev=IssueSeverity.warning),
                    "suggestions": _scored(merged.get("suggestions") or [], sev=IssueSeverity.warning),
                }
            except Exception:
                pass
            return merged

        report0 = _compute_report(Path(root_dir))
        try:
            i0 = int(len([x for x in (report0.get("issues") or []) if str(x).strip()]))
            w0 = int(len([x for x in (report0.get("warnings") or []) if str(x).strip()]))
            s0 = int(len([x for x in (report0.get("suggestions") or []) if str(x).strip()]))
            d0 = report0.get("details") if isinstance(report0, dict) else None
            q0 = 0
            try:
                if isinstance(d0, dict) and isinstance(d0.get("questions"), list):
                    q0 = int(len(d0.get("questions") or []))
            except Exception:
                q0 = 0
            logger.info(
                "[project_validate] initial_counts: issues=%d warnings=%d suggestions=%d questions=%d repair_enabled=%s",
                int(i0),
                int(w0),
                int(s0),
                int(q0),
                bool(repair_enabled),
            )
        except Exception:
            pass
        if bool(report0.get("ok", False)) or (not repair_enabled):
            return report0

        # Default lower to reduce token burn; callers can override via constraints or env.
        try:
            max_rounds = int(
                (c or {}).get("repair_max_rounds")
                or os.environ.get("CRPB_REPAIR_MAX_ROUNDS", "3")
            )
        except Exception:
            max_rounds = 3
        try:
            max_files = int(
                (c or {}).get("repair_max_files_per_round")
                or os.environ.get("CRPB_REPAIR_MAX_FILES_PER_ROUND", "8")
            )
        except Exception:
            max_files = 8

        # Import spider lazily to avoid circular imports.
        from crpb.repairing.spider import repair_outputs_until_ok_with_oracle

        # Validator-driven repairs default to child-mode with parallel subcalls.
        repair_constraints: Dict[str, Any] = dict(c) if isinstance(c, dict) else {}
        repair_constraints.setdefault("repair_child_mode", True)
        repair_constraints.setdefault("repair_child_parallel_workers", 10)
        repair_constraints.setdefault("repair_child_batch_size", 10)
        repair_constraints.setdefault("repair_child_max_issues_per_call", 10)

        rep = repair_outputs_until_ok_with_oracle(
            outputs_dir=Path(root_dir),
            oracle=lambda root: _compute_report(root),
            provider=provider,
            idea=str(idea or ""),
            constraints=repair_constraints,
            file_specs=dict(file_specs or {}),
            max_rounds=max_rounds,
            max_files_per_round=max_files,
            validations_dir=validations_dir,
            label=str(label or "project"),
        )

        report1 = _compute_report(Path(root_dir))
        try:
            i1 = int(len([x for x in (report1.get("issues") or []) if str(x).strip()]))
            w1 = int(len([x for x in (report1.get("warnings") or []) if str(x).strip()]))
            s1 = int(len([x for x in (report1.get("suggestions") or []) if str(x).strip()]))
            logger.info(
                "[project_validate] after_repair: ok=%s rounds=%d applied_edits=%d issues=%d warnings=%d suggestions=%d",
                bool(getattr(rep, "ok", False)),
                int(getattr(rep, "rounds", 0)),
                int(getattr(rep, "applied_edits", 0)),
                int(i1),
                int(w1),
                int(s1),
            )
        except Exception:
            pass
        try:
            report1.setdefault("details", {})
            if isinstance(report1.get("details"), dict):
                report1["details"].setdefault(
                    "repair",
                    {
                        "ok": bool(rep.ok),
                        "rounds": int(rep.rounds),
                        "applied_edits": int(rep.applied_edits),
                        "notes": list(rep.notes),
                        "history": (rep.details.get("history") if isinstance(rep.details, dict) else []),
                        "history_summary": (
                            rep.details.get("history_summary") if isinstance(rep.details, dict) else ""
                        ),
                    },
                )
        except Exception:
            pass
        return report1

    except Exception as e:
        return {
            "ok": False,
            "issues": [f"validator_error:{type(e).__name__}"],
            "warnings": [],
            "suggestions": [],
        }
