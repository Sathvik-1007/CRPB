from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from crpb.utils.fs import ensure_parent, write_text_locked

from .models import Issue, IssueSeverity, RepairEdit, RepairReport
from .parse import parse_validation_report
from .provider import RepairProvider
from .score import score

logger = logging.getLogger(__name__)


def _find_run_dir_from_outputs(outputs_dir: Path) -> Optional[Path]:
    """Best-effort inference of run_dir from an outputs directory.

    Supports both:
      - <run>/outputs
      - <run>/outputs/_staging
    """
    try:
        p = Path(outputs_dir)
        if p.name == "_staging":
            return p.parent.parent
        if p.name == "outputs":
            return p.parent
    except Exception:
        return None
    return None


def _patches_to_edits(root: Path, patches: List[Dict[str, Any]]) -> List[RepairEdit]:
    """Convert minimal patch ops into full-file edits deterministically.

    Patch schema (dict):
      - path: project-relative path
      - find: exact substring to match (must be present)
      - replace: replacement substring
      - rationale?: optional
      - allow_multiple?: bool (default False)
      - count?: int (default 1)

    Safety rules:
      - By default, `find` must occur exactly once.
      - If allow_multiple=true, occurrences must equal count (and count <= 3).
      - If file does not exist and find=="", treat replace as full file contents.
    """
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for p in patches or []:
        if not isinstance(p, dict):
            continue
        path = str(p.get("path") or "").strip()
        if not path:
            continue
        by_path.setdefault(path, []).append(p)

    edits: List[RepairEdit] = []
    for path, ops in by_path.items():
        target = root / path
        try:
            original = target.read_text(encoding="utf-8") if target.exists() else ""
        except Exception:
            original = ""

        content = original
        rationale_parts: List[str] = []
        ok = True
        for op in ops:
            find = op.get("find")
            replace = op.get("replace")
            if not isinstance(find, str):
                find = str(find or "")
            if not isinstance(replace, str):
                replace = str(replace or "")
            find = find
            replace = replace

            rationale = op.get("rationale")
            if isinstance(rationale, str) and rationale.strip():
                rationale_parts.append(rationale.strip())

            allow_multiple = bool(op.get("allow_multiple", False))
            try:
                count = int(op.get("count", 1))
            except Exception:
                count = 1
            count = max(1, min(3, count))

            if (not target.exists()) and (find.strip() == ""):
                # Create new file via full contents.
                content = replace
                continue

            if not find:
                ok = False
                break

            occurrences = content.count(find)
            if occurrences == 0:
                ok = False
                break
            if not allow_multiple and occurrences != 1:
                ok = False
                break
            if allow_multiple and occurrences != count:
                ok = False
                break
            content = content.replace(find, replace, count if allow_multiple else 1)

        if not ok:
            # Partial success: skip this path rather than failing the entire proposal.
            continue
        if content != original:
            edits.append(RepairEdit(path=path, new_text=content, rationale="; ".join(rationale_parts)))

    return edits


def _normalize_rel_path(p: str) -> str:
    s = str(p or "").strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def _safe_is_subpath(path: str) -> bool:
    # Reject path traversal or absolute paths.
    s = _normalize_rel_path(path)
    if not s or s.startswith("/") or ":" in s:
        return False
    parts = [x for x in s.split("/") if x]
    if any(x == ".." for x in parts):
        return False
    return True


def _allowed_new_files(
    *,
    issues: List[Issue],
    file_index: List[str],
    file_specs: Dict[str, Any],
) -> set[str]:
    """Compute which new files may be created this round.

    Policy (deterministic, language-neutral):
    - Allowed if validator explicitly reported missing_file for that path
    - Allowed if path exists in file_specs (expected) but is absent from file_index
    """
    idx = {_normalize_rel_path(x) for x in (file_index or []) if str(x).strip()}
    allowed: set[str] = set()
    for it in issues or []:
        code = str(getattr(it, "code", "") or "").strip().lower()
        p = _normalize_rel_path(str(getattr(it, "path", "") or ""))
        if code.startswith("missing_file") and p:
            allowed.add(p)
    try:
        if isinstance(file_specs, dict):
            for p in file_specs.keys():
                rp = _normalize_rel_path(str(p))
                if rp and rp not in idx:
                    allowed.add(rp)
    except Exception:
        pass
    return allowed


def _read_outputs(root: Path, rel_paths: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in rel_paths:
        rp = str(p or "").strip()
        if not rp:
            continue
        f = root / rp
        if not f.exists() or not f.is_file():
            continue
        try:
            out[rp] = f.read_text(encoding="utf-8")
        except Exception:
            out[rp] = ""
    return out


def _write_outputs(root: Path, edits: List[RepairEdit]) -> int:
    applied = 0
    for ed in edits or []:
        rp = str(getattr(ed, "path", "") or "").strip()
        if not rp:
            continue
        target = root / rp
        ensure_parent(target)
        write_text_locked(target, str(getattr(ed, "new_text", "") or ""), encoding="utf-8")
        applied += 1
    return applied


def _edit_sig(edits: List[RepairEdit]) -> Tuple[Tuple[str, str], ...]:
    """Stable signature for an edit set.

    Uses SHA-256 (stable across processes) so repeat detection is deterministic.
    """

    pairs: List[Tuple[str, str]] = []
    for e in edits or []:
        p = _normalize_rel_path(str(getattr(e, "path", "") or ""))
        t = str(getattr(e, "new_text", "") or "")
        if not p:
            continue
        h = hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()
        pairs.append((p, h))
    return tuple(sorted(pairs))


def _issue_focus_paths(issues: List[Issue], *, file_index: List[str]) -> List[str]:
    """Derive a deterministic focus set from issues.

    Rule: if issue has a `path`, include it. If it has a `ref` that looks like a path, include it.
    No regex inference: only conservative path-ish checks.
    """
    focus: List[str] = []
    index = {str(p).replace("\\", "/") for p in (file_index or [])}

    def _maybe_add(s: Optional[str], *, allow_missing: bool) -> None:
        if not isinstance(s, str):
            return
        s2 = s.strip().replace("\\", "/")
        if not s2:
            return
        while s2.startswith("./"):
            s2 = s2[2:]
        # Normally, only consider candidates that exist in the file index.
        # Exception: some issues explicitly refer to missing files we need to create.
        if (not allow_missing) and index and s2 not in index:
            return
        # Conservative: treat as path only if it contains a slash or a dot extension.
        if "/" in s2 or ("." in Path(s2).name):
            if s2 not in focus:
                focus.append(s2)

    # Prefer paths referenced by fatal issues first (deterministic ordering).
    ordered_issues: List[Issue] = list(issues or [])
    try:
        ordered_issues.sort(
            key=lambda x: 0
            if getattr(x, "severity", None) == IssueSeverity.fatal
            else 1
        )
    except Exception:
        ordered_issues = list(issues or [])

    for it in ordered_issues:
        code = str(getattr(it, "code", "") or "")
        # Allow focusing missing targets so providers can create them.
        # Keep this rule language/tool neutral: do not special-case specific stacks.
        code_l = code.strip().lower()
        allow_missing = (
            code_l.startswith("missing_")
            or code_l.endswith("_missing")
            or ("not_found" in code_l)
            or ("unresolved" in code_l)
        )
        _maybe_add(it.path, allow_missing=allow_missing)
        _maybe_add(it.ref, allow_missing=False)

    return focus


def _select_issues_for_provider(
    issues: List[Issue],
    *,
    focus_paths: List[str],
    max_fatal: int = 24,
    max_nonfatal: int = 12,
) -> List[Issue]:
    """Bound and prioritize issues so the provider can actually act.

    We keep this deterministic and language/tool neutral:
    - prioritize fatal issues (blocking)
    - prefer issues tied to files we're loading this round
    - include a small number of non-fatal findings for context
    """

    focus_set = {str(p).replace("\\", "/") for p in (focus_paths or []) if str(p).strip()}

    def _p(iss: Issue) -> str:
        return str(getattr(iss, "path", "") or "").strip().replace("\\", "/")

    fatals = [x for x in (issues or []) if getattr(x, "severity", None) == IssueSeverity.fatal]
    nonfatals = [x for x in (issues or []) if x not in fatals]

    # Prefer focus-localized findings.
    fatals_local = [x for x in fatals if _p(x) in focus_set]
    fatals_other = [x for x in fatals if x not in fatals_local]

    # Keep some warnings/suggestions, preferably localized.
    nonfatals_local = [x for x in nonfatals if _p(x) in focus_set]
    nonfatals_other = [x for x in nonfatals if x not in nonfatals_local]

    out: List[Issue] = []
    out.extend(fatals_local[: max(0, int(max_fatal))])
    if len(out) < int(max_fatal):
        out.extend(fatals_other[: max(0, int(max_fatal)) - len(out)])

    out_nonfatal: List[Issue] = []
    out_nonfatal.extend(nonfatals_local[: max(0, int(max_nonfatal))])
    if len(out_nonfatal) < int(max_nonfatal):
        out_nonfatal.extend(
            nonfatals_other[: max(0, int(max_nonfatal)) - len(out_nonfatal)]
        )
    out.extend(out_nonfatal)
    return out


def repair_outputs_until_ok(
    *,
    outputs_dir: Path,
    provider: RepairProvider,
    idea: str,
    constraints: Dict[str, Any],
    file_specs: Dict[str, Any],
    max_rounds: int = 3,
    max_files_per_round: int = 8,
    validations_dir: Optional[Path] = None,
    label: str = "project",
) -> RepairReport:
    # Mutable focus hint for the oracle to prioritize recently-edited files during chunked validation.
    # This reduces false "no change" rollbacks when the validator is context-bounded.
    oracle_hint: Dict[str, Any] = {"priority_paths": []}

    # Enable the improved child-mode strategy by default for real workflow runs.
    # (Unit tests that call repair_outputs_until_ok_with_oracle directly can control this via constraints.)
    constraints2 = dict(constraints or {})
    constraints2.setdefault("repair_child_mode", True)
    # Default parallel subcalls per batch.
    constraints2.setdefault("repair_child_parallel_workers", 10)
    constraints2.setdefault("repair_child_batch_size", 10)
    # Default to pandas-backed grouping/sorting when available.
    constraints2.setdefault("repair_use_pandas", True)
    # Default: do not rollback an entire child-mode round. Per-edit micro acceptance already gates.
    constraints2.setdefault("repair_child_round_rollback", False)
    # Jury profiles for in-between-round discovery (used by the oracle).
    constraints2.setdefault("repair_round_jury_profiles", ["contracts", "integration", "default"])

    def _oracle_full(root: Path) -> Dict[str, Any]:
        return __llm_validate_only(
            root=root,
            idea=idea,
            constraints=constraints2,
            file_specs=file_specs,
            priority_paths=list(oracle_hint.get("priority_paths") or []),
        )

    def _oracle_micro(root: Path) -> Dict[str, Any]:
        # Micro-oracle: a smaller, localized validation pass used to score candidate edits.
        # This enables per-edit acceptance without paying the cost of a full holistic pass.
        c = dict(constraints2 or {})
        c.setdefault("repair_micro_oracle", True)
        c.setdefault("repair_micro_max_files", 3)
        c.setdefault("repair_micro_max_chars", 12000)
        c.setdefault("repair_micro_jury_profiles", ["contracts"])
        return __llm_validate_only(
            root=root,
            idea=idea,
            constraints=c,
            file_specs=file_specs,
            priority_paths=list(oracle_hint.get("priority_paths") or []),
        )

    return repair_outputs_until_ok_with_oracle(
        outputs_dir=outputs_dir,
        # LLM-only, language-neutral oracle. Do not perform repairs inside the oracle.
        oracle=_oracle_full,
        oracle_micro=_oracle_micro,
        provider=provider,
        idea=idea,
        constraints=constraints2,
        file_specs=file_specs,
        max_rounds=max_rounds,
        max_files_per_round=max_files_per_round,
        validations_dir=validations_dir,
        label=label,
    )


def __llm_validate_only(
    *,
    root: Path,
    idea: str,
    constraints: Dict[str, Any],
    file_specs: Dict[str, Any],
    priority_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a pure LLM project validation pass over a directory tree.

    This is intentionally:
    - language/tool neutral
    - non-repairing (oracle must be stable)
    """
    try:
        from crpb.validation.project_validate import validate_project_outputs

        c2 = dict(constraints or {})
        c2["repair_enabled"] = False
        # This oracle is used inside the repair loop, so it must be fast and stable.
        # Reduce repeated token burn while staying language/tool neutral.
        # The full workflow still runs a holistic validator later.
        micro = False
        try:
            if isinstance(c2, dict) and bool(c2.get("repair_micro_oracle")):
                micro = True
        except Exception:
            micro = False
        if micro:
            c2["project_jury_profiles"] = list(
                c2.get("repair_micro_jury_profiles")
                if isinstance(c2.get("repair_micro_jury_profiles"), list)
                else ["contracts"]
            )
            c2["node_validate_max_nodes"] = 0
            c2["project_validate_max_files"] = int(c2.get("repair_micro_max_files") or 3)
            c2["project_validate_max_chars"] = int(c2.get("repair_micro_max_chars") or 12000)
        else:
            try:
                jr = c2.get("repair_round_jury_profiles")
                if isinstance(jr, list) and jr:
                    c2["project_jury_profiles"] = [str(x) for x in jr if str(x).strip()]
                else:
                    c2.setdefault("project_jury_profiles", ["contracts"])
            except Exception:
                c2.setdefault("project_jury_profiles", ["contracts"])
            c2.setdefault("node_validate_max_nodes", 0)
            c2.setdefault("project_validate_max_files", 8)
            c2.setdefault("project_validate_max_chars", 20000)

        # Hint to validator: when chunking, ensure these files are validated first.
        try:
            sc = c2.get("side_context")
            if not isinstance(sc, dict):
                sc = {}
            sc = dict(sc)
            pp = [str(x).replace("\\", "/") for x in (priority_paths or []) if str(x).strip()]
            if pp:
                sc["validation_priority_paths"] = pp[:50]
            c2["side_context"] = sc
        except Exception:
            pass

        return validate_project_outputs(
            outputs_dir=Path(root),
            constraints=c2,
            idea=str(idea or ""),
            plan={},
            file_specs=dict(file_specs or {}),
            validations_dir=None,
            label="oracle",
        )
    except Exception as e:
        return {
            "ok": False,
            "issues": [f"validation_oracle_error:{type(e).__name__}"],
            "warnings": [],
            "suggestions": [],
        }


def repair_outputs_until_ok_with_oracle(
    *,
    outputs_dir: Path,
    oracle: Callable[[Path], Dict[str, Any]],
    oracle_micro: Optional[Callable[[Path], Dict[str, Any]]] = None,
    provider: RepairProvider,
    idea: str,
    constraints: Dict[str, Any],
    file_specs: Dict[str, Any],
    max_rounds: int = 3,
    max_files_per_round: int = 8,
    validations_dir: Optional[Path] = None,
    label: str = "project",
) -> RepairReport:
    """Iterative validate→repair→revalidate on a directory tree using a validation oracle.

    Deterministic core:
    - scoring is monotone: accept a repair only if it strictly improves the metric
    - rollback on non-improving proposals

    The oracle returns a report dict with keys: ok, issues, warnings, suggestions, (optional) details.
    Provider is responsible for semantics and language decisions.
    """
    logger.info("[repair_spider] Starting repair loop for: %s (label=%s)", outputs_dir, label)

    root = Path(outputs_dir)
    notes: List[str] = []
    details: Dict[str, Any] = {"label": label}

    def _compute_file_index() -> List[str]:
        try:
            return sorted(
                [str(p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file()]
            )
        except Exception:
            return []

    t0 = time.perf_counter()
    try:
        det0 = oracle(root)
    except Exception as e:
        logger.exception("[repair_spider] Initial validation oracle failed: %s", e)
        det0 = {"ok": False, "issues": [f"validation_oracle_error::{type(e).__name__}"], "warnings": []}
    t0_s = time.perf_counter() - t0

    rep0 = {
        "ok": bool(det0.get("ok", False)),
        "issues": list(det0.get("issues") or []),
        "warnings": list(det0.get("warnings") or []),
        "suggestions": list(det0.get("suggestions") or []),
    }

    issues0 = parse_validation_report(rep0)
    initial_count = len(issues0)
    fatal0 = len([x for x in (rep0.get("issues") or []) if str(x).strip()])
    logger.info(
        "[repair_spider] Initial validation: ok=%s fatal=%d findings=%d (%.2fs)",
        det0.get("ok"), fatal0, initial_count, t0_s,
    )

    def _is_clean(det: Dict[str, Any]) -> bool:
        # Align with validate_project_outputs(): ok is determined by the absence of blocking issues.
        return len([x for x in (det.get("issues") or []) if str(x).strip()]) == 0

    if _is_clean(det0):
        logger.info("[repair_spider] Already clean, no repairs needed")
        return RepairReport(
            ok=True,
            rounds=0,
            applied_edits=0,
            initial_issue_count=initial_count,
            final_issue_count=0,
            notes=["already_ok"],
            details={"initial": det0, "final": det0},
        )

    applied_total = 0
    # Current state: must match what's on disk.
    current_det = det0
    current_issues = issues0
    # Best-seen metric (for reporting / monotone progress tracking).
    best_det = det0
    best_metric = None

    # Persisted repair memory across rounds to reduce repeated/no-op edits.
    repair_history: List[Dict[str, Any]] = []
    repair_history_summary: str = ""

    # Captures what happened in the prior round (even if rolled back) so the provider can avoid
    # oscillation and no-op proposals.
    last_round_summary: Dict[str, Any] = {}

    seen_edit_sets: set[Tuple[Tuple[str, str], ...]] = set()
    rejected_edit_sets: set[Tuple[Tuple[str, str], ...]] = set()
    repeat_count = 0

    def _filter_edits(
        raw_edits: List[RepairEdit],
        *,
        issues_for_provider: List[Issue],
        file_index: List[str],
    ) -> List[RepairEdit]:
        """Apply deterministic safety/policy filters to proposed edits."""

        allowed_new = _allowed_new_files(
            issues=issues_for_provider,
            file_index=file_index,
            file_specs=dict(file_specs or {}),
        )

        filtered_edits: List[RepairEdit] = []
        for ed in raw_edits or []:
            rp = _normalize_rel_path(str(getattr(ed, "path", "") or ""))
            if not rp or not _safe_is_subpath(rp):
                continue
            tgt = root / rp
            if (not tgt.exists()) and (rp not in allowed_new):
                continue
            # Prevent converting non-empty files into empty files.
            try:
                old_txt = tgt.read_text(encoding="utf-8") if tgt.exists() else ""
            except Exception:
                old_txt = ""
            new_txt = str(getattr(ed, "new_text", "") or "")
            if old_txt.strip() and (not new_txt.strip()):
                continue
            filtered_edits.append(
                RepairEdit(path=rp, new_text=new_txt, rationale=getattr(ed, "rationale", ""))
            )

        # Drop obviously-bad empty edits deterministically (prevents creating empty files).
        filtered2: List[RepairEdit] = []
        for ed in filtered_edits:
            try:
                rp = str(ed.path or "").strip()
                new_txt = str(ed.new_text or "")
                old_txt = ""
                try:
                    old_txt = (root / rp).read_text(encoding="utf-8") if rp else ""
                except Exception:
                    old_txt = ""
                if (not new_txt.strip()) and old_txt.strip():
                    continue
                if (not rp) or (not new_txt.strip() and not old_txt.strip()):
                    # Allow creating an empty file only if it previously existed empty (rare).
                    continue
                filtered2.append(ed)
            except Exception:
                filtered2.append(ed)
        return filtered2

    def _metric(det: Dict[str, Any]) -> Tuple[int, float, int, int]:
        """Monotone metric that prioritizes eliminating fatal issues.

        Ordering:
          1) fatal issues count (primary objective)
          2) weighted score over all findings (secondary)
          3) total findings count
          4) warnings+suggestions count
        """
        rep = {
            "issues": list(det.get("issues") or []),
            "warnings": list(det.get("warnings") or []),
            "suggestions": list(det.get("suggestions") or []),
        }
        fatal_count = len([x for x in (rep.get("issues") or []) if str(x).strip()])
        parsed = parse_validation_report(rep)
        s = score(parsed)
        warn_count = len(
            [x for x in (rep.get("warnings") or []) + (rep.get("suggestions") or []) if str(x).strip()]
        )
        return (int(fatal_count), float(s), int(len(parsed)), int(warn_count))

    if best_metric is None:
        best_metric = _metric(best_det)

    # Optional micro-oracle used for per-candidate scoring.
    oracle_micro_fn: Callable[[Path], Dict[str, Any]] = oracle_micro or oracle

    def _child_mode_enabled() -> bool:
        try:
            if isinstance(constraints, dict) and isinstance(constraints.get("repair_child_mode"), bool):
                return bool(constraints.get("repair_child_mode"))
        except Exception:
            pass
        return False

    def _max_child_calls() -> int:
        try:
            v = (constraints or {}).get("repair_child_parallel_workers")
            if v is None:
                # Back-compat with earlier key name.
                v = (constraints or {}).get("repair_child_calls_per_round")
            if v is None:
                return 10
            return max(1, min(32, int(v)))
        except Exception:
            return 10

    def _child_batch_size() -> int:
        try:
            v = (constraints or {}).get("repair_child_batch_size")
            if v is None:
                return 10
            return max(1, min(32, int(v)))
        except Exception:
            return 10

    def _child_max_batches_per_round() -> int:
        try:
            v = (constraints or {}).get("repair_child_max_batches_per_round")
            if v is None:
                # Default: attempt all batches for all TODO file-groups.
                return 1_000_000
            return max(1, min(1_000_000, int(v)))
        except Exception:
            return 1_000_000

    def _micro_accepts(det_new: Dict[str, Any], det_old: Dict[str, Any]) -> bool:
        # Use the same monotone metric but evaluated on the (fast) micro oracle.
        return _metric(det_new) < _metric(det_old)

    rounds_executed = 0
    def _count_findings(det: Dict[str, Any]) -> Tuple[int, int, int]:
        try:
            i = int(len([x for x in (det.get("issues") or []) if str(x).strip()]))
        except Exception:
            i = 0
        try:
            w = int(len([x for x in (det.get("warnings") or []) if str(x).strip()]))
        except Exception:
            w = 0
        try:
            s = int(len([x for x in (det.get("suggestions") or []) if str(x).strip()]))
        except Exception:
            s = 0
        return (i, w, s)

    def _log_jury_summary(det: Dict[str, Any], *, prefix: str) -> None:
        try:
            d = det.get("details") if isinstance(det, dict) else None
            jury = d.get("jury") if isinstance(d, dict) else None
            profiles = jury.get("profiles") if isinstance(jury, dict) else None
            if not isinstance(profiles, list):
                return
            parts: List[str] = []
            for p in profiles:
                if not isinstance(p, dict):
                    continue
                nm = str(p.get("profile") or "")
                ci = int(len(p.get("issues") or []))
                cw = int(len(p.get("warnings") or []))
                cs = int(len(p.get("suggestions") or []))
                if nm:
                    parts.append(f"{nm}:i={ci} w={cw} s={cs}")
            if parts:
                logger.info("[repair_spider] %s jury_profiles=%s", prefix, "; ".join(parts)[:500])
        except Exception:
            return

    for rnd in range(1, max(1, int(max_rounds)) + 1):
        rounds_executed += 1
        logger.info("[repair_spider] Round %d/%d starting", rnd, max_rounds)
        try:
            i0, w0, s0 = _count_findings(current_det)
            logger.info(
                "[repair_spider] Round %d start_counts: issues=%d warnings=%d suggestions=%d metric=%s",
                rnd,
                int(i0),
                int(w0),
                int(s0),
                str(_metric(current_det)),
            )
        except Exception:
            pass
        _log_jury_summary(current_det, prefix=f"round_{rnd}_start")
        # Refresh file index each round (repairs may have created/removed files).
        file_index = _compute_file_index()
        # Expand issues with jury findings (if present) so TODOs include cross-profile discoveries.
        issues_all: List[Issue] = list(current_issues or [])
        try:
            from crpb.repairing.parse import parse_issue_strings

            details_in0 = current_det.get("details") if isinstance(current_det, dict) else None
            jury0 = details_in0.get("jury") if isinstance(details_in0, dict) else None
            profiles0 = jury0.get("profiles") if isinstance(jury0, dict) else None
            if isinstance(profiles0, list):
                for prof in profiles0:
                    if not isinstance(prof, dict):
                        continue
                    issues_all.extend(
                        parse_issue_strings(prof.get("issues") or [], severity=IssueSeverity.fatal)
                    )
                    issues_all.extend(
                        parse_issue_strings(prof.get("warnings") or [], severity=IssueSeverity.warning)
                    )
                    issues_all.extend(
                        parse_issue_strings(prof.get("suggestions") or [], severity=IssueSeverity.warning)
                    )
        except Exception:
            pass

        # Dedupe by (severity, message) deterministically.
        try:
            seen: set[Tuple[str, str]] = set()
            deduped: List[Issue] = []
            for it in issues_all:
                k = (str(getattr(it, "severity", "") or ""), str(getattr(it, "message", "") or ""))
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(it)
            issues_all = deduped
        except Exception:
            pass

        focus = _issue_focus_paths(issues_all, file_index=file_index)
        files_to_load = focus[: max(1, int(max_files_per_round))]
        # If we cannot localize issues to existing files, still load a small deterministic slice
        # so the provider has some concrete text to act on.
        if not files_to_load:
            files_to_load = (file_index or [])[: max(1, int(max_files_per_round))]
        files_map = _read_outputs(root, files_to_load)
        # If focus targets are missing files, _read_outputs returns empty; still provide some text.
        if (not files_map) and file_index:
            files_map = _read_outputs(root, (file_index or [])[: max(1, int(max_files_per_round))])
        logger.debug("[repair_spider] Round %d: focus_paths=%s", rnd, focus[:5])

        # Only pass a bounded, relevant set of issues to the provider.
        issues_for_provider = _select_issues_for_provider(
            issues_all,
            focus_paths=files_to_load,
            max_fatal=24,
            max_nonfatal=12,
        )

        # Avoid stuffing huge validator details into the model context; keep a thin capsule.
        try:
            details_in = current_det.get("details") if isinstance(current_det, dict) else None
        except Exception:
            details_in = None
        repair_hints = {}
        if isinstance(details_in, dict):
            repair_hints = details_in.get("repair_hints") if isinstance(details_in.get("repair_hints"), dict) else {}

        # Validator probing questions are internal to validation (to find more issues).
        # They are logged for visibility but intentionally NOT fed into repair contexts.
        questions: List[Dict[str, Any]] = []
        try:
            if isinstance(details_in, dict) and isinstance(details_in.get("questions"), list):
                questions = [q for q in (details_in.get("questions") or []) if isinstance(q, dict)]
        except Exception:
            questions = []
        if questions:
            try:
                logger.info("[repair_spider] Round %d validator_questions=%d", rnd, int(len(questions)))
            except Exception:
                pass

        context = {
            "round": rnd,
            "focus_paths": focus,
            "file_index": file_index,
            "previous_round": dict(last_round_summary or {}),
            "budget": {
                "max_rounds": int(max_rounds),
                "max_files_per_round": int(max_files_per_round),
            },
            "validator": {
                "label": str(label or "project"),
                "fatal_count": int(len([x for x in (current_det.get("issues") or []) if str(x).strip()]))
                if isinstance(current_det, dict)
                else 0,
                "repair_hints": repair_hints,
            },
            "repair_history_summary": repair_history_summary,
            "repair_history": repair_history[-5:],
            "patch_contract": {
                "create_missing_file": "If a target file is missing, you MAY create it via a patch with find=\"\" and replace=<full file contents>.",
                "exact_find": "Each patch.find must be copied verbatim from the provided file text. Prefer a find that matches exactly once.",
            },
        }

        # Add a bounded run log summary when available (helps the LM reason about prior failures).
        try:
            from crpb.utils.run_logs import summarize_run_logs

            run_dir = _find_run_dir_from_outputs(root)
            if run_dir is not None:
                context["run_logs"] = summarize_run_logs(run_dir=run_dir)
        except Exception:
            pass

        # Child-mode: multiple small, localized repair calls per round.
        # This reduces context rot and enables per-candidate (micro) scoring.
        child_mode = _child_mode_enabled()
        step_history: List[Dict[str, Any]] = []
        # Track round-start snapshots for any file we end up touching so we can rollback the round.
        round_snapshot: Dict[str, str] = {}
        round_touched: List[str] = []

        # A TODO list grouped by file, ordered by numeric importance.
        try:
            from crpb.repairing.score import importance as _importance
        except Exception:
            _importance = None  # type: ignore

        # Build TODOs for *all* findings (fatal + warning), grouped by file.
        # Optional pandas fast-path; fallback is pure Python and deterministic.
        issues_by_path: Dict[str, List[Issue]] = {}
        for it in (issues_all or []):
            p = _normalize_rel_path(str(getattr(it, "path", "") or "")) or "<global>"
            issues_by_path.setdefault(p, []).append(it)

        todos: List[Dict[str, Any]] = []

        def _issue_imp(it: Issue) -> int:
            if _importance is None:
                return 0
            try:
                return int(_importance(it))
            except Exception:
                return 0

        use_pandas = False
        try:
            # Default to pandas when available; fall back to pure-Python deterministically.
            v = (constraints or {}).get("repair_use_pandas", True)
            use_pandas = bool(v)
        except Exception:
            use_pandas = False

        pandas_unavailable = False

        if use_pandas:
            try:
                import pandas as pd  # type: ignore

                rows: List[Dict[str, Any]] = []
                for it in (issues_all or []):
                    p = _normalize_rel_path(str(getattr(it, "path", "") or "")) or "<global>"
                    sev = getattr(it, "severity", None)
                    rows.append(
                        {
                            "path": p,
                            "is_fatal": 1 if sev == IssueSeverity.fatal else 0,
                            "imp": int(_issue_imp(it)),
                            "code": str(getattr(it, "code", "") or ""),
                            "msg": str(getattr(it, "message", "") or ""),
                        }
                    )
                df = pd.DataFrame.from_records(rows)
                if not df.empty:
                    g = df.groupby("path", sort=True).agg(
                        fatal_count=("is_fatal", "sum"),
                        finding_count=("path", "size"),
                        importance_sum=("imp", "sum"),
                        importance_max=("imp", "max"),
                    )
                    g = g.reset_index()
                    g = g.sort_values(
                        by=["fatal_count", "importance_sum", "finding_count", "path"],
                        ascending=[False, False, False, True],
                        kind="mergesort",
                    )
                    for _, row in g.iterrows():
                        p = str(row["path"])
                        items = issues_by_path.get(p) or []
                        # Deterministic per-file ordering: fatal first, then importance, then message.
                        items_sorted = list(items)
                        items_sorted.sort(
                            key=lambda x: (
                                0 if getattr(x, "severity", None) == IssueSeverity.fatal else 1,
                                -int(_issue_imp(x)),
                                str(getattr(x, "message", "") or ""),
                            )
                        )
                        todos.append(
                            {
                                "path": p,
                                "fatal_count": int(row["fatal_count"]),
                                "finding_count": int(row["finding_count"]),
                                "importance_sum": int(row["importance_sum"]),
                                "importance_max": int(row["importance_max"]),
                                "codes": [str(getattr(x, "code", "")) for x in items_sorted[:12]],
                                "messages": [str(getattr(x, "message", "")) for x in items_sorted[:10]],
                            }
                        )
            except Exception:
                todos = []
                pandas_unavailable = True

        if not todos:
            for p in sorted(issues_by_path.keys()):
                items = issues_by_path.get(p) or []
                items_sorted = list(items)
                items_sorted.sort(
                    key=lambda x: (
                        0 if getattr(x, "severity", None) == IssueSeverity.fatal else 1,
                        -int(_issue_imp(x)),
                        str(getattr(x, "message", "") or ""),
                    )
                )
                fatal_count = int(len([x for x in items_sorted if getattr(x, "severity", None) == IssueSeverity.fatal]))
                imp_vals = [int(_issue_imp(x)) for x in items_sorted]
                todos.append(
                    {
                        "path": p,
                        "fatal_count": fatal_count,
                        "finding_count": int(len(items_sorted)),
                        "importance_sum": int(sum(imp_vals) if imp_vals else 0),
                        "importance_max": int(max(imp_vals) if imp_vals else 0),
                        "codes": [str(getattr(x, "code", "")) for x in items_sorted[:12]],
                        "messages": [str(getattr(x, "message", "")) for x in items_sorted[:10]],
                    }
                )

            todos.sort(
                key=lambda x: (
                    -int(x.get("fatal_count") or 0),
                    -int(x.get("importance_sum") or 0),
                    -int(x.get("finding_count") or 0),
                    str(x.get("path") or ""),
                )
            )

        # Formalized, language-neutral axioms and objective for the provider.
        axioms = [
            "A1 (Evidence): Any patch.find MUST be copied verbatim from provided file text.",
            "A2 (Safety): Do not propose absolute paths or path traversal; only project-relative subpaths.",
            "A3 (New Files): Create a new file only if explicitly missing per validator or expected by file_specs.",
            "A4 (Minimality): Prefer the smallest change that eliminates a targeted blocking issue.",
            "A5 (Monotone Acceptance): Repairs are accepted only if validation metrics do not worsen.",
        ]

        # Initialize micro baseline at current best state.
        micro_det = current_det

        def _apply_candidate(
            cand_edits: List[RepairEdit],
            *,
            issues_local: List[Issue],
            file_index_local: List[str],
        ) -> Tuple[bool, Dict[str, Any], int]:
            nonlocal applied_total, micro_det

            cand_edits2 = _filter_edits(cand_edits, issues_for_provider=issues_local, file_index=file_index_local)
            if not cand_edits2:
                return (False, micro_det, 0)

            # Record snapshots for rollback at round end.
            for e in cand_edits2:
                if e.path not in round_snapshot:
                    try:
                        f = root / e.path
                        round_snapshot[e.path] = f.read_text(encoding="utf-8") if f.exists() else ""
                    except Exception:
                        round_snapshot[e.path] = ""
                if e.path not in round_touched:
                    round_touched.append(e.path)

            snap = _read_outputs(root, [e.path for e in cand_edits2])
            applied = _write_outputs(root, cand_edits2)
            applied_total += applied

            # Hint the oracle to validate changed files first.
            try:
                if isinstance(oracle, object) and hasattr(oracle, "__closure__") and oracle.__closure__:
                    for cell in oracle.__closure__:
                        val = getattr(cell, "cell_contents", None)
                        if isinstance(val, dict) and "priority_paths" in val:
                            val["priority_paths"] = [str(e.path).replace("\\", "/") for e in cand_edits2][:50]
                            break
            except Exception:
                pass

            t0 = time.perf_counter()
            det_new = oracle_micro_fn(root)
            dt = time.perf_counter() - t0

            accepted = _micro_accepts(det_new, micro_det)
            if accepted:
                micro_det = det_new
                return (True, det_new | {"_micro_seconds": float(dt)}, applied)

            # Rollback this candidate.
            for p, txt in snap.items():
                target = root / p
                ensure_parent(target)
                write_text_locked(target, txt, encoding="utf-8")
            return (False, det_new | {"_micro_seconds": float(dt)}, 0)

        def _propose_once(
            *,
            issues_local: List[Issue],
            files_local: Dict[str, str],
            context_local: Dict[str, Any],
            constraints_local: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], float]:
            t_prop = time.perf_counter()
            prop = provider.propose_repairs(
                idea=idea,
                constraints=constraints_local,
                issues=issues_local,
                files=files_local,
                file_specs=file_specs,
                context=context_local,
            )
            return prop, float(time.perf_counter() - t_prop)

        # Default (legacy) single provider call for the round.
        proposal: Dict[str, Any] = {}
        t_prop_s = 0.0
        if not child_mode:
            logger.debug("[repair_spider] Round %d: requesting repairs from provider", rnd)
            proposal, t_prop_s = _propose_once(
                issues_local=issues_for_provider,
                files_local=files_map,
                context_local=dict(context, axioms=axioms, todos=todos, objective="minimize fatal then total weighted issues"),
                constraints_local=dict(constraints or {}),
            )
        else:
            logger.info(
                "[repair_spider] Round %d: child-mode enabled (parallel_workers=%d batch_size=%d)",
                rnd,
                int(_max_child_calls()),
                int(_child_batch_size()),
            )
            # Process TODOs in parallel batches. Each batch launches up to N propose calls.
            batch_size = int(_child_batch_size())
            max_batches = int(_child_max_batches_per_round())
            workers = int(_max_child_calls())

            def _build_local(todo_item: Dict[str, Any]) -> Dict[str, Any]:
                todo_path = str(todo_item.get("path") or "").strip()
                local_issues = list(issues_by_path.get(todo_path) or [])
                if not local_issues:
                    # Fallback: hand the provider a small slice of the most important findings overall.
                    local_issues = list(issues_all[:12])
                # Bound per-call issue list but keep it dense and relevant.
                local_issues.sort(
                    key=lambda x: (
                        0 if getattr(x, "severity", None) == IssueSeverity.fatal else 1,
                        -int(_issue_imp(x)),
                        str(getattr(x, "message", "") or ""),
                    )
                )
                # Bound per-call issue list (defaults to 10 for tight, file-scoped repair calls).
                max_issues = 10
                try:
                    vmi = (constraints or {}).get("repair_child_max_issues_per_call")
                    if vmi is not None:
                        max_issues = max(1, min(50, int(vmi)))
                except Exception:
                    max_issues = 10
                local_issues = local_issues[: int(max_issues)]

                # Load <=10 files for this call: the target + up to 9 related focus paths.
                local_paths: List[str] = []
                if todo_path and todo_path != "<global>" and todo_path in file_index:
                    local_paths.append(todo_path)
                # Prefer file paths referenced by the local issues.
                for pth in _issue_focus_paths(local_issues, file_index=file_index)[:20]:
                    if pth not in local_paths:
                        local_paths.append(pth)
                    if len(local_paths) >= 10:
                        break
                # Then fill with round focus.
                for pth in focus:
                    if pth not in local_paths:
                        local_paths.append(pth)
                    if len(local_paths) >= 10:
                        break
                if not local_paths:
                    local_paths = (file_index or [])[:10]
                local_files = _read_outputs(root, local_paths)
                if not local_files and files_map:
                    local_files = dict(files_map)

                ctx = dict(context)
                ctx.update(
                    {
                        "axioms": axioms,
                        "todos": todos[:100],
                        "todo": dict(todo_item),
                        "todo_other_top_findings": [
                            str(getattr(x, "message", "") or "")
                            for x in (issues_all or [])
                            if (_normalize_rel_path(str(getattr(x, "path", "") or "")) or "<global>") != todo_path
                        ][:30],
                        "objective": "minimize fatal issues first; do not introduce regressions",
                    }
                )
                return {
                    "todo": dict(todo_item),
                    "todo_path": todo_path,
                    "issues": local_issues,
                    "files": local_files,
                    "context": ctx,
                }

            def _extract_edits(prop: Dict[str, Any]) -> Tuple[List[RepairEdit], List[str]]:
                notes2: List[str] = []
                try:
                    nr = prop.get("notes") if isinstance(prop, dict) else None
                    if isinstance(nr, list):
                        notes2 = [str(x) for x in nr if str(x).strip()]
                except Exception:
                    notes2 = []

                cand: List[RepairEdit] = []
                patches_raw = prop.get("patches") if isinstance(prop, dict) else None
                if isinstance(patches_raw, list) and patches_raw:
                    cand = _patches_to_edits(root, [p for p in patches_raw if isinstance(p, dict)])
                else:
                    edits_raw = prop.get("edits") if isinstance(prop, dict) else None
                    if isinstance(edits_raw, list):
                        for e in edits_raw:
                            if isinstance(e, RepairEdit):
                                cand.append(e)
                            elif isinstance(e, dict):
                                pth = e.get("path")
                                txt = e.get("new_text")
                                rat = e.get("rationale") or e.get("why") or ""
                                if isinstance(pth, str) and isinstance(txt, str):
                                    cand.append(RepairEdit(path=pth, new_text=txt, rationale=str(rat)))
                return cand, notes2

            any_progress = False
            batches_done = 0
            for i0 in range(0, len(todos), batch_size):
                if batches_done >= max_batches:
                    break
                batch = todos[i0 : i0 + batch_size]
                if not batch:
                    break
                batches_done += 1

                built = [_build_local(t) for t in batch]

                # Propose repairs in parallel for this batch.
                props: List[Dict[str, Any]] = []
                times: List[float] = []
                with ThreadPoolExecutor(max_workers=min(workers, len(built))) as ex:
                    futs = []
                    for it in built:
                        futs.append(
                            ex.submit(
                                _propose_once,
                                issues_local=it["issues"],
                                files_local=it["files"],
                                context_local=it["context"],
                                constraints_local=dict(constraints or {}),
                            )
                        )
                    for f in futs:
                        try:
                            pr, dt = f.result()
                            props.append(pr if isinstance(pr, dict) else {})
                            times.append(float(dt))
                        except Exception as e:
                            # Provider may not be thread-safe; fall back to an empty proposal.
                            props.append({"notes": [f"child_parallel_error:{type(e).__name__}"]})
                            times.append(0.0)

                # Apply candidates sequentially in batch order (already importance-sorted by todos).
                batch_accepts = 0
                for idx, it in enumerate(built):
                    todo_item = it["todo"]
                    local_issues = it["issues"]
                    prop = props[idx] if idx < len(props) else {}

                    # RLM-style context expansion (single retry) per subcall.
                    needs = prop.get("needs_paths") if isinstance(prop, dict) else None
                    needs_paths: List[str] = []
                    if isinstance(needs, list):
                        for x in needs:
                            s = _normalize_rel_path(str(x or ""))
                            if s and _safe_is_subpath(s):
                                needs_paths.append(s)
                    needs_paths = list(dict.fromkeys(needs_paths))[:12]
                    files_local = it["files"]
                    if needs_paths:
                        missing = [p for p in needs_paths if p not in files_local]
                        if missing:
                            more = _read_outputs(root, missing)
                            if more:
                                files2 = dict(files_local)
                                files2.update(more)
                                ctx2 = dict(it["context"])
                                ctx2["rlm"] = {"requested_paths": missing, "provided_paths": sorted(list(more.keys()))[:50]}
                                prop2, dt2 = _propose_once(
                                    issues_local=local_issues,
                                    files_local=files2,
                                    context_local=ctx2,
                                    constraints_local=dict(constraints or {}),
                                )
                                prop = prop2 if isinstance(prop2, dict) else prop
                                times[idx] = float(dt2)
                                it["files"] = files2

                    cand_edits, cand_notes = _extract_edits(prop)
                    micro_before = list(_metric(micro_det))
                    # Per-edit scoring/acceptance (default): evaluate each edit independently.
                    # This prevents a single bad edit from discarding good ones.
                    accepted = False
                    applied_step = 0
                    det_step: Dict[str, Any] = {}
                    edit_results: List[Dict[str, Any]] = []
                    if cand_edits:
                        for e_i, one in enumerate(list(cand_edits)[:50], start=1):
                            m_before = list(_metric(micro_det))
                            ok_one, det_one, applied_one = _apply_candidate(
                                [one],
                                issues_local=local_issues,
                                file_index_local=file_index,
                            )
                            m_after = list(_metric(micro_det))
                            accepted = accepted or bool(ok_one)
                            applied_step += int(applied_one)
                            if isinstance(det_one, dict):
                                det_step = det_one
                            rec = {
                                "index": int(e_i),
                                "path": str(getattr(one, "path", "") or ""),
                                "accepted": bool(ok_one),
                                "applied": int(applied_one),
                                "micro_before": m_before,
                                "micro_after": m_after,
                                "micro_seconds": float(det_one.get("_micro_seconds") or 0.0)
                                if isinstance(det_one, dict)
                                else 0.0,
                            }
                            edit_results.append(rec)
                            try:
                                logger.info(
                                    "[repair_spider] Round %d batch=%d todo=%s edit=%d/%d path=%s accepted=%s micro_before=%s micro_after=%s",
                                    rnd,
                                    int(batches_done),
                                    str((todo_item or {}).get("path") or ""),
                                    int(e_i),
                                    int(len(cand_edits)),
                                    str(getattr(one, "path", "") or ""),
                                    bool(ok_one),
                                    str(m_before),
                                    str(m_after),
                                )
                            except Exception:
                                pass

                    micro_candidate_after = (
                        list(_metric(det_step)) if isinstance(det_step, dict) and det_step else None
                    )
                    micro_state_after = list(_metric(micro_det))

                    if accepted:
                        batch_accepts += 1
                        any_progress = True

                    try:
                        todo_p = str((todo_item or {}).get("path") or "")
                        logger.info(
                            "[repair_spider] Round %d batch=%d todo=%s accepted=%s applied=%d micro_before=%s micro_after=%s",
                            rnd,
                            int(batches_done),
                            todo_p,
                            bool(accepted),
                            int(applied_step),
                            str(micro_before),
                            str(micro_state_after),
                        )
                    except Exception:
                        pass

                    step_history.append(
                        {
                            "round": int(rnd),
                            "batch": int(batches_done),
                            "index": int(i0 + idx),
                            "todo": dict(todo_item),
                            "proposal_seconds": float(times[idx] if idx < len(times) else 0.0),
                            "provider_notes": cand_notes[:8],
                            "proposed_paths": [str(getattr(e, "path", "")) for e in (cand_edits or [])][:12],
                            "accepted": bool(accepted),
                            "applied": int(applied_step),
                            "micro_metric_before": micro_before,
                            "micro_metric_after_candidate": micro_candidate_after,
                            "micro_metric_after_state": micro_state_after,
                            "edit_results": edit_results[:50],
                        }
                    )

                if batch_accepts == 0:
                    # No progress in this batch: stop to avoid token burn.
                    break

            proposal = {"notes": ["child_mode"], "todos": todos, "batches": int(batches_done)}
            try:
                proposal["child_proposal_seconds"] = float(sum(times) if isinstance(times, list) else 0.0)
                t_prop_s = float(proposal.get("child_proposal_seconds") or 0.0)
            except Exception:
                pass

            if use_pandas and pandas_unavailable:
                notes.append(f"round_{rnd}:pandas_unavailable")

        # RLM-style context expansion: allow provider to request additional file text.
        # This prevents hallucinations when the needed evidence wasn't included in files_map.
        try:
            needs = proposal.get("needs_paths") if isinstance(proposal, dict) else None
        except Exception:
            needs = None
        needs_paths: List[str] = []
        if isinstance(needs, list):
            for it in needs:
                s = _normalize_rel_path(str(it or ""))
                if s and _safe_is_subpath(s):
                    needs_paths.append(s)
        needs_paths = list(dict.fromkeys(needs_paths))[:12]

        if needs_paths:
            missing = [p for p in needs_paths if p not in files_map]
            if missing:
                more = _read_outputs(root, missing)
                if more:
                    files_map2 = dict(files_map)
                    files_map2.update(more)
                    context2 = dict(context)
                    context2["rlm"] = {
                        "requested_paths": missing,
                        "provided_paths": sorted(list(more.keys()))[:50],
                    }
                    t_prop_r = time.perf_counter()
                    proposal = provider.propose_repairs(
                        idea=idea,
                        constraints=constraints,
                        issues=issues_for_provider,
                        files=files_map2,
                        file_specs=file_specs,
                        context=context2,
                    )
                    t_prop_s = time.perf_counter() - t_prop_r
                    files_map = files_map2

        last_round_summary = {
            "round": int(rnd),
            "provider_seconds": float(t_prop_s),
            "focus_paths": list(files_to_load or [])[:25],
            "issues_given": [str(getattr(i, "message", "") or "") for i in (issues_for_provider or [])][:40],
            "child_mode": bool(child_mode),
            "steps": step_history[-5:],
        }

        edits: List[RepairEdit] = []

        # In child-mode, edits may already have been applied. Skip the legacy apply path.
        if child_mode:
            edits = []
            notes.append(f"round_{rnd}:child_mode_steps={len(step_history)}")
        

        # Patch-mode proposals: convert minimal patches into full-file edits deterministically.
        patch_edits: List[RepairEdit] = []
        patches_raw = proposal.get("patches") if isinstance(proposal, dict) else None
        notes_raw = proposal.get("notes") if isinstance(proposal, dict) else None
        if isinstance(patches_raw, list) and patches_raw:
            patch_edits = _patches_to_edits(root, [p for p in patches_raw if isinstance(p, dict)])

        if patch_edits:
            edits = patch_edits
        else:
            edits_raw = proposal.get("edits") if isinstance(proposal, dict) else None
            if isinstance(edits_raw, list):
                for e in edits_raw:
                    if isinstance(e, RepairEdit):
                        edits.append(e)
                    elif isinstance(e, dict):
                        p = e.get("path")
                        t = e.get("new_text")
                        r = e.get("rationale") or e.get("why") or ""
                        if isinstance(p, str) and isinstance(t, str):
                            edits.append(RepairEdit(path=p, new_text=t, rationale=str(r)))

        if (not child_mode) and (not edits):
            # If patches were provided but could not be applied, compute diagnostics and retry once.
            patch_diag: List[Dict[str, Any]] = []
            if isinstance(patches_raw, list) and patches_raw:
                for p in [x for x in patches_raw if isinstance(x, dict)]:
                    path = str(p.get("path") or "").strip().replace("\\", "/")
                    find = p.get("find")
                    replace = p.get("replace")
                    if not path:
                        continue
                    target = root / path
                    try:
                        text = target.read_text(encoding="utf-8") if target.exists() else ""
                    except Exception:
                        text = ""
                    occ = text.count(find) if isinstance(find, str) and text is not None else 0
                    patch_diag.append(
                        {
                            "path": path,
                            "find_type": type(find).__name__,
                            "replace_type": type(replace).__name__,
                            "file_exists": bool(target.exists()),
                            "find_occurrences": int(occ),
                        }
                    )

            retry_done = False
            if patch_diag:
                retry_done = True
                context2 = dict(context)
                context2["patch_diagnostics"] = patch_diag[:50]
                c2 = dict(constraints or {})
                extra = list(c2.get("format_rules", [])) if isinstance(c2.get("format_rules"), list) else []
                extra += [
                    "RETRY REQUIRED: Your previous patch proposal could not be applied.",
                    "Choose patch.find substrings by COPYING VERBATIM from the provided file text.",
                    "Each patch.find MUST occur exactly once in the target file unless allow_multiple=true and count matches occurrences.",
                    "If a target file is missing, you MAY create it using a patch with find=\"\" and replace=<full file contents>.",
                    "Return at least ONE valid patch if any blocking issue is fixable from the provided files.",
                ]
                c2["format_rules"] = extra
                t_prop2 = time.perf_counter()
                proposal2 = provider.propose_repairs(
                    idea=idea,
                    constraints=c2,
                    issues=issues_for_provider,
                    files=files_map,
                    file_specs=file_specs,
                    context=context2,
                )
                t_prop2_s = time.perf_counter() - t_prop2

                edits2: List[RepairEdit] = []
                patches2 = proposal2.get("patches") if isinstance(proposal2, dict) else None
                patch_edits2: List[RepairEdit] = []
                if isinstance(patches2, list) and patches2:
                    patch_edits2 = _patches_to_edits(root, [p for p in patches2 if isinstance(p, dict)])
                if patch_edits2:
                    edits2 = patch_edits2
                else:
                    edits_raw2 = proposal2.get("edits") if isinstance(proposal2, dict) else None
                    if isinstance(edits_raw2, list):
                        for e in edits_raw2:
                            if isinstance(e, RepairEdit):
                                edits2.append(e)
                            elif isinstance(e, dict):
                                pth = e.get("path")
                                txt = e.get("new_text")
                                rat = e.get("rationale") or e.get("why") or ""
                                if isinstance(pth, str) and isinstance(txt, str):
                                    edits2.append(RepairEdit(path=pth, new_text=txt, rationale=str(rat)))

                if edits2:
                    edits = edits2
                    proposal = proposal2
                    notes.append(f"round_{rnd}:retry_after_invalid_patches")

            if not edits:
                # Last resort: if patch-mode yields no usable edits, try a single full-rewrite proposal.
                # This is still bounded by the same files_map and issues_for_provider.
                try:
                    c3 = dict(constraints or {})
                    c3["repair_patch_mode"] = False
                    c3["repair_allow_full_file_rewrites"] = True
                    extra3 = list(c3.get("format_rules", [])) if isinstance(c3.get("format_rules"), list) else []
                    extra3 += [
                        "FALLBACK: Patch proposal produced no applicable edits.",
                        "Return at least ONE concrete file edit (full file contents) if any blocking issue is fixable from the provided files.",
                        "You MAY create missing files needed to resolve blocking issues.",
                    ]
                    c3["format_rules"] = extra3
                    t_prop3 = time.perf_counter()
                    proposal3 = provider.propose_repairs(
                        idea=idea,
                        constraints=c3,
                        issues=issues_for_provider,
                        files=files_map,
                        file_specs=file_specs,
                        context=dict(context),
                    )
                    t_prop3_s = time.perf_counter() - t_prop3

                    edits3: List[RepairEdit] = []
                    edits_raw3 = proposal3.get("edits") if isinstance(proposal3, dict) else None
                    if isinstance(edits_raw3, list):
                        for e in edits_raw3:
                            if isinstance(e, RepairEdit):
                                edits3.append(e)
                            elif isinstance(e, dict):
                                pth = e.get("path")
                                txt = e.get("new_text")
                                rat = e.get("rationale") or e.get("why") or ""
                                if isinstance(pth, str) and isinstance(txt, str):
                                    edits3.append(RepairEdit(path=pth, new_text=txt, rationale=str(rat)))
                    if edits3:
                        edits = edits3
                        proposal = proposal3
                        notes.append(f"round_{rnd}:fallback_full_rewrites")
                        logger.info(
                            "[repair_spider] Round %d: fallback full rewrites produced %d edits (fallback_time=%.2fs)",
                            rnd,
                            len(edits3),
                            float(t_prop3_s),
                        )
                except Exception:
                    pass

            if not edits:
                logger.info(
                    "[repair_spider] Round %d: provider returned no edits, stopping (proposal_time=%.2fs)",
                    rnd,
                    float(t_prop_s),
                )
                try:
                    if isinstance(notes_raw, list) and notes_raw:
                        logger.info("[repair_spider] Round %d: provider notes=%s", rnd, [str(x) for x in notes_raw[:8]])
                except Exception:
                    pass
                try:
                    if isinstance(patches_raw, list):
                        logger.info("[repair_spider] Round %d: provider patches=%d", rnd, len(patches_raw))
                except Exception:
                    pass
                last_round_summary.update(
                    {
                        "outcome": "no_edits",
                        "proposed_paths": [],
                        "applied_count": 0,
                    }
                )
                if retry_done:
                    try:
                        logger.info(
                            "[repair_spider] Round %d: retry also returned no edits (retry_time=%.2fs)",
                            rnd,
                            float(t_prop2_s),
                        )
                    except Exception:
                        pass
                notes.append(f"round_{rnd}:no_edits")
                if retry_done:
                    notes.append(f"round_{rnd}:retry_no_edits")
                break

        # Enforce strict new-file creation policy + safety filters.
        edits = _filter_edits(edits, issues_for_provider=issues_for_provider, file_index=file_index)

        logger.info(
            "[repair_spider] Round %d: provider proposed %d edits: %s",
            rnd, len(edits), [e.path for e in edits[:5]]
        )
        for ed in edits:
            logger.debug("[repair_spider] Edit: path=%s rationale=%s", ed.path, ed.rationale[:100] if ed.rationale else "")

        if (not child_mode) and (not edits):
            logger.info("[repair_spider] Round %d: all proposed edits were no-ops/empty, stopping", rnd)
            notes.append(f"round_{rnd}:no_usable_edits")
            break

        # Detect repeated proposals deterministically to avoid looping.
        try:
            sig = _edit_sig(edits)
        except Exception:
            sig = tuple(sorted([(str(e.path), str(len(str(e.new_text)))) for e in edits]))

        if (not child_mode) and (sig in rejected_edit_sets or sig in seen_edit_sets):
            repeat_count += 1
            logger.info(
                "[repair_spider] Round %d: detected repeat/rejected edits (repeat_count=%d)",
                rnd,
                int(repeat_count),
            )
            last_round_summary.update(
                {
                    "outcome": "repeat_edits",
                    "proposed_paths": [str(e.path) for e in edits[:25]],
                    "repeat_count": int(repeat_count),
                }
            )
            notes.append(f"round_{rnd}:repeat_edits")
            # Try a single fallback full-rewrite proposal to break out of repetition.
            try:
                c3 = dict(constraints or {})
                c3["repair_patch_mode"] = False
                c3["repair_allow_full_file_rewrites"] = True
                extra3 = list(c3.get("format_rules", [])) if isinstance(c3.get("format_rules"), list) else []
                extra3 += [
                    "REPEAT DETECTED: Your last proposal repeated a rejected edit set.",
                    "Return a DIFFERENT set of concrete edits. Do not repeat prior paths/text.",
                    "If you must create a new file, only do so for files explicitly missing per validator or file_specs.",
                ]
                c3["format_rules"] = extra3
                proposal3 = provider.propose_repairs(
                    idea=idea,
                    constraints=c3,
                    issues=issues_for_provider,
                    files=files_map,
                    file_specs=file_specs,
                    context=dict(context),
                )
                edits3_raw = proposal3.get("edits") if isinstance(proposal3, dict) else None
                edits3: List[RepairEdit] = []
                if isinstance(edits3_raw, list):
                    for e in edits3_raw:
                        if isinstance(e, RepairEdit):
                            edits3.append(e)
                        elif isinstance(e, dict):
                            pth = e.get("path")
                            txt = e.get("new_text")
                            rat = e.get("rationale") or e.get("why") or ""
                            if isinstance(pth, str) and isinstance(txt, str):
                                edits3.append(RepairEdit(path=pth, new_text=txt, rationale=str(rat)))
                if edits3:
                    # Apply the same safety/new-file policy filters to fallback edits.
                    edits = _filter_edits(edits3, issues_for_provider=issues_for_provider, file_index=file_index)
                    if not edits:
                        if repeat_count >= 2:
                            break
                        continue
                    # Reset repeat counter and proceed with apply below.
                    repeat_count = 0
                    try:
                        sig = _edit_sig(edits)
                    except Exception:
                        sig = tuple(sorted([(str(e.path), str(len(str(e.new_text)))) for e in edits]))
                    seen_edit_sets.add(sig)
                else:
                    if repeat_count >= 2:
                        break
                    continue
            except Exception:
                if repeat_count >= 2:
                    break
                continue

        repeat_count = 0
        if not child_mode:
            seen_edit_sets.add(sig)

        if not child_mode:
            snapshot = _read_outputs(root, [e.path for e in edits])
            applied = _write_outputs(root, edits)
            applied_total += applied
            logger.info("[repair_spider] Round %d: applied %d edits", rnd, applied)
        else:
            snapshot = {}
            applied = 0

        # Update oracle hint when available (closure in repair_outputs_until_ok).
        try:
            if isinstance(oracle, object) and hasattr(oracle, "__closure__") and oracle.__closure__:
                for cell in oracle.__closure__:
                    val = getattr(cell, "cell_contents", None)
                    if isinstance(val, dict) and "priority_paths" in val:
                        val["priority_paths"] = [str(e.path).replace("\\", "/") for e in edits][:50]
                        break
        except Exception:
            pass

        last_round_summary.update(
            {
                "proposed_paths": [str(e.path) for e in edits[:25]] if edits else [str(x) for x in round_touched[:25]],
                "applied_count": int(applied) if not child_mode else int(len(round_touched)),
            }
        )

        t_val = time.perf_counter()
        try:
            det1 = oracle(root)
        except Exception as e:
            logger.exception("[repair_spider] Round %d: validation oracle failed after repairs: %s", rnd, e)
            det1 = {"ok": False, "issues": [f"validation_oracle_error::{type(e).__name__}"], "warnings": []}
        t_val_s = time.perf_counter() - t_val

        rep1 = {
            "ok": bool(det1.get("ok", False)),
            "issues": list(det1.get("issues") or []),
            "warnings": list(det1.get("warnings") or []),
            "suggestions": list(det1.get("suggestions") or []),
        }
        issues1 = parse_validation_report(rep1)
        new_metric = _metric(det1)
        old_metric = _metric(current_det)
        logger.info(
            "[repair_spider] Round %d: metric before=%s after=%s",
            rnd, old_metric, new_metric
        )
        logger.info(
            "[repair_spider] Round %d: timing provider=%.2fs validate=%.2fs",
            rnd,
            float(t_prop_s),
            float(t_val_s),
        )

        # Update current state to match what's now on disk.
        prev_det = current_det
        current_det = det1
        current_issues = issues1

        if new_metric < old_metric:
            notes.append(f"round_{rnd}:improved")
            logger.info("[repair_spider] Round %d: repairs IMPROVED the score", rnd)

            if child_mode and step_history:
                try:
                    best_det.setdefault("details", {})
                except Exception:
                    pass
                try:
                    if isinstance(best_det.get("details"), dict):
                        best_det["details"]["repair_steps"] = step_history[-50:]
                except Exception:
                    pass

            # Capture a concise history entry so later rounds avoid repeating the same fixes.
            try:
                before_issues = [str(x) for x in (prev_det.get("issues") or []) if str(x).strip()]
                after_issues = [str(x) for x in (det1.get("issues") or []) if str(x).strip()]
                fixed = sorted(list(set(before_issues) - set(after_issues)))
            except Exception:
                fixed = []
            fixed = [x for x in fixed if x][:40]

            try:
                hist_item = {
                    "round": int(rnd),
                    "edited_paths": [str(e.path) for e in edits[:25]],
                    "fixed_issues": fixed,
                    "fatal_before": int(len([x for x in (prev_det.get("issues") or []) if str(x).strip()])),
                    "fatal_after": int(len([x for x in (det1.get("issues") or []) if str(x).strip()])),
                    "provider_notes": list(proposal.get("notes") or []) if isinstance(proposal, dict) else [],
                }
                repair_history.append(hist_item)
            except Exception:
                pass

            # Update best-seen state if this is the best metric observed.
            try:
                if new_metric < best_metric:
                    best_metric = new_metric
                    best_det = det1
            except Exception:
                pass

            try:
                if fixed:
                    repair_history_summary = "fixed:" + ",".join(fixed[:12])
                else:
                    repair_history_summary = "progress:improved_metric"
            except Exception:
                repair_history_summary = "progress:improved_metric"

            if _is_clean(det1):
                logger.info("[repair_spider] Round %d: project is now clean!", rnd)
                break
            last_round_summary.update(
                {
                    "outcome": "improved",
                    "metric_before": list(old_metric),
                    "metric_after": list(new_metric),
                }
            )
        else:
            # If the global metric doesn't change, still allow accepting a "lateral" repair when
            # it fixes at least one of the *targeted* blocking issues we provided to the model,
            # without increasing fatal count. This prevents token-wasting stalls where the model
            # makes real progress but counts remain constant (issue substitution).
            accept_lateral = False
            try:
                pre_fatals = [str(x) for x in (prev_det.get("issues") or []) if str(x).strip()]
                post_fatals = [str(x) for x in (det1.get("issues") or []) if str(x).strip()]
                pre_set = set(pre_fatals)
                post_set = set(post_fatals)
                targeted = set(
                    [str(getattr(i, "message", "") or "") for i in (issues_for_provider or [])]
                )
                targeted = {x for x in targeted if x}
                fixed_target = (pre_set - post_set) & targeted

                # Allow if at least one targeted fatal issue disappeared and fatal count didn't increase.
                if fixed_target and int(new_metric[0]) <= int(old_metric[0]):
                    # Also require we did not grow total findings.
                    if int(new_metric[2]) <= int(old_metric[2]):
                        accept_lateral = True
            except Exception:
                accept_lateral = False

            if accept_lateral:
                notes.append(f"round_{rnd}:accepted_lateral_target_fix")
                logger.info(
                    "[repair_spider] Round %d: accepted lateral fix (targeted issues fixed, metric unchanged)",
                    rnd,
                )
                try:
                    before_issues = [str(x) for x in (prev_det.get("issues") or []) if str(x).strip()]
                    after_issues = [str(x) for x in (det1.get("issues") or []) if str(x).strip()]
                    fixed = sorted(list(set(before_issues) - set(after_issues)))
                except Exception:
                    fixed = []
                fixed = [x for x in fixed if x][:40]
                try:
                    hist_item = {
                        "round": int(rnd),
                        "edited_paths": [str(e.path) for e in edits[:25]],
                        "fixed_issues": fixed,
                        "fatal_before": int(len([x for x in (prev_det.get("issues") or []) if str(x).strip()])),
                        "fatal_after": int(len([x for x in (det1.get("issues") or []) if str(x).strip()])),
                        "provider_notes": list(proposal.get("notes") or []) if isinstance(proposal, dict) else [],
                    }
                    repair_history.append(hist_item)
                except Exception:
                    pass
                last_round_summary.update(
                    {
                        "outcome": "accepted_lateral_target_fix",
                        "metric_before": list(old_metric),
                        "metric_after": list(new_metric),
                    }
                )
                if _is_clean(det1):
                    logger.info("[repair_spider] Round %d: project is now clean!", rnd)
                    break
                continue

            # In child-mode, do NOT rollback the whole round by default.
            # Per-edit micro acceptance already ensured each accepted edit improved locally.
            # Full-oracle may still regress due to global coupling or oracle noise; keep progress.
            try:
                rb = bool((constraints or {}).get("repair_child_round_rollback", False))
            except Exception:
                rb = False
            if child_mode and round_snapshot and rb:
                logger.info("[repair_spider] Round %d: child-mode round regressed; rolling back round (enabled)", rnd)
                for p, txt in round_snapshot.items():
                    target = root / p
                    ensure_parent(target)
                    write_text_locked(target, txt, encoding="utf-8")
                # Restore current state to previous (since we reverted on disk).
                current_det = prev_det
                current_issues = parse_validation_report(
                    {
                        "issues": prev_det.get("issues") or [],
                        "warnings": prev_det.get("warnings") or [],
                        "suggestions": prev_det.get("suggestions") or [],
                    }
                )
                notes.append(f"round_{rnd}:child_mode_round_rollback")
                last_round_summary.update({"outcome": "child_mode_round_rollback"})
                continue
            if child_mode and round_snapshot and (not rb):
                notes.append(f"round_{rnd}:child_mode_kept_micro_accepts")

            # Mark this edit set as rejected so future rounds can avoid repeating it.
            try:
                rejected_edit_sets.add(sig)
            except Exception:
                pass

            logger.info("[repair_spider] Round %d: repairs DID NOT IMPROVE, rolling back", rnd)
            for p, txt in snapshot.items():
                target = root / p
                ensure_parent(target)
                write_text_locked(target, txt, encoding="utf-8")
            # Restore current state to previous (since we reverted on disk).
            current_det = prev_det
            current_issues = parse_validation_report(
                {
                    "issues": prev_det.get("issues") or [],
                    "warnings": prev_det.get("warnings") or [],
                    "suggestions": prev_det.get("suggestions") or [],
                }
            )
            notes.append(f"round_{rnd}:rollback_non_improving")
            last_round_summary.update(
                {
                    "outcome": "rollback_non_improving",
                    "metric_before": list(old_metric),
                    "metric_after": list(new_metric),
                }
            )
            # Keep trying other rounds; provider may propose a different edit set.
            continue

    ok = _is_clean(current_det)
    logger.info(
        "[repair_spider] Repair loop complete: ok=%s rounds=%d applied=%d initial_issues=%d final_issues=%d",
        ok,
        int(rounds_executed),
        applied_total,
        initial_count,
        len(current_issues),
    )

    if validations_dir is not None:
        try:
            outp = Path(validations_dir) / f"repair_{label}.json"
            ensure_parent(outp)
            outp.write_text(
                json.dumps(
                    {
                        "ok": ok,
                        "rounds": int(rounds_executed),
                        "applied_edits": applied_total,
                        "notes": notes,
                        "history": repair_history,
                        "final": best_det,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            details["report_path"] = str(outp)
            logger.debug("[repair_spider] Repair report saved to: %s", outp)
        except Exception:
            pass

    final_count = len(
        parse_validation_report(
            {
                "issues": current_det.get("issues") or [],
                "warnings": current_det.get("warnings") or [],
                "suggestions": current_det.get("suggestions") or [],
            }
        )
    )

    return RepairReport(
        ok=ok,
        rounds=int(rounds_executed),
        applied_edits=int(applied_total),
        initial_issue_count=int(initial_count),
        final_issue_count=int(final_count),
        notes=notes,
        details={
            "initial": det0,
            "final": current_det,
            "best": best_det,
            "best_metric": list(best_metric),
            "history": repair_history,
            "history_summary": repair_history_summary,
            **details,
        },
    )
