from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from crpb.core.ledger import NodeLedgerStore
from crpb.core.specs import TaskPlan, TaskSpec
from crpb.utils.artifacts import ArtifactRegistry
from crpb.validation.coverage import compute_obligation_coverage
from crpb.validation.validator import validate_taskplan_general


def _safe_list(x: Any) -> List[Any]:
    return list(x) if isinstance(x, list) else []


def _safe_dict(x: Any) -> Dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


def _walk_task_specs(tasks: List[TaskSpec]) -> Iterable[TaskSpec]:
    stack: List[TaskSpec] = list(tasks or [])
    while stack:
        cur = stack.pop()
        yield cur
        for ch in list(getattr(cur, "children", []) or []):
            if isinstance(ch, TaskSpec):
                stack.append(ch)


def _collect_artifact_ids(tp: TaskPlan) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return (consumes_by_task, produces_by_task) as id sets."""

    consumes: Dict[str, Set[str]] = {}
    produces: Dict[str, Set[str]] = {}

    for t in _walk_task_specs(list(getattr(tp, "tasks", []) or [])):
        tid = str(getattr(t, "id", "") or "")
        if not tid:
            continue

        ins = _safe_dict(getattr(t, "inputs", None))
        outs = _safe_dict(getattr(t, "outputs", None))

        for ref in _safe_list(ins.get("consumes")):
            rid = None
            if isinstance(ref, str):
                rid = ref
            elif isinstance(ref, dict):
                rid = ref.get("id")
            if isinstance(rid, str) and rid.strip():
                consumes.setdefault(tid, set()).add(rid.strip())

        for ref in _safe_list(outs.get("produces")):
            rid = None
            if isinstance(ref, str):
                rid = ref
            elif isinstance(ref, dict):
                rid = ref.get("id")
            if isinstance(rid, str) and rid.strip():
                produces.setdefault(tid, set()).add(rid.strip())

    return consumes, produces


@dataclass(frozen=True)
class ClosureConfig:
    require_artifact_registry_presence: bool = True
    require_consumes_exist_in_registry: bool = True
    require_produces_exist_in_registry: bool = True
    require_consumers_depend_on_producers: bool = True
    require_coverage_closed: bool = True


def validate_artifact_registry_closure(
    *,
    tp: TaskPlan,
    registry: Optional[ArtifactRegistry],
    base_dir: Optional[Path],
    config: ClosureConfig,
) -> Dict[str, Any]:
    """Deterministic, language-agnostic closure validator.

    This validator does NOT parse source code.
    It enforces wiring correctness via declared artifacts and deterministic evidence.
    """

    issues: List[str] = []
    warnings: List[str] = []

    # 1) Deterministic plan-level checks (deps, artifact shapes, wiring signals)
    plan_report = validate_taskplan_general(tp)
    if not bool(plan_report.get("ok", True)):
        for it in plan_report.get("issues") or []:
            issues.append(f"taskplan:{it}")
        for n in plan_report.get("nodes") or []:
            nid = str((n or {}).get("id") or "")
            for it in (n or {}).get("issues") or []:
                if nid:
                    issues.append(f"task:{nid}:{it}")
                else:
                    issues.append(f"task:<no-id>:{it}")

    # 2) Artifact registry closure (runtime evidence)
    consumes_by_task, produces_by_task = _collect_artifact_ids(tp)

    if registry is None or base_dir is None:
        if config.require_artifact_registry_presence:
            issues.append("artifact_registry_missing")
        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "details": {
                "plan": plan_report,
                "registry": {"present": False},
                "consumes_by_task": {k: sorted(v) for k, v in consumes_by_task.items()},
                "produces_by_task": {k: sorted(v) for k, v in produces_by_task.items()},
            },
        }

    # Build producer index from the plan (artifact id -> producer task ids)
    producers_by_art: Dict[str, Set[str]] = {}
    for tid, rids in produces_by_task.items():
        for rid in rids:
            producers_by_art.setdefault(rid, set()).add(tid)

    # Enforce: consumes exist and (optionally) depend on at least one producer
    for consumer_id, rids in consumes_by_task.items():
        consumer_task: Optional[TaskSpec] = None
        for t in _walk_task_specs(list(getattr(tp, "tasks", []) or [])):
            if str(getattr(t, "id", "") or "") == consumer_id:
                consumer_task = t
                break
        deps = {
            str(d)
            for d in (getattr(consumer_task, "deps", []) or [])
            if isinstance(d, str) and d
        }

        for rid in sorted(rids):
            if config.require_consumes_exist_in_registry:
                if not registry.exists({"id": rid}, base_dir=base_dir):
                    issues.append(f"artifact_consume_missing_in_registry:{consumer_id}:{rid}")

            if config.require_consumers_depend_on_producers:
                prod_ids = producers_by_art.get(rid, set())
                if prod_ids and not (deps & prod_ids):
                    issues.append(
                        f"artifact_consumer_missing_dep:{consumer_id}:{rid}:expected_dep_on:{','.join(sorted(prod_ids))}"
                    )

    # Enforce: produced artifacts exist in registry
    if config.require_produces_exist_in_registry:
        for producer_id, rids in produces_by_task.items():
            for rid in sorted(rids):
                if not registry.exists({"id": rid}, base_dir=base_dir):
                    issues.append(f"artifact_produce_missing_in_registry:{producer_id}:{rid}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "details": {
            "plan": plan_report,
            "registry": {"present": True, "count": len(registry.list())},
            "consumes_by_task": {k: sorted(v) for k, v in consumes_by_task.items()},
            "produces_by_task": {k: sorted(v) for k, v in produces_by_task.items()},
        },
    }


def validate_run_closure(
    *,
    tp: TaskPlan,
    run_dir_path: str,
    config: Optional[ClosureConfig] = None,
) -> Dict[str, Any]:
    """Validate closure for a run directory.

    Includes:
    - artifact wiring closure via ArtifactRegistry index
    - obligation coverage closure via node ledgers
    """

    cfg = config or ClosureConfig()
    run_dir = Path(str(run_dir_path))

    # Registry + ledger live under run_dir/artifacts
    artifacts_dir = run_dir / "artifacts"
    registry: Optional[ArtifactRegistry]
    ledger_store: Optional[NodeLedgerStore]

    try:
        registry = ArtifactRegistry(base_dir=artifacts_dir)
    except Exception:
        registry = None

    try:
        ledger_store = NodeLedgerStore(base_dir=artifacts_dir)
    except Exception:
        ledger_store = None

    wiring = validate_artifact_registry_closure(
        tp=tp,
        registry=registry,
        base_dir=artifacts_dir,
        config=cfg,
    )

    coverage: Dict[str, Any]
    if cfg.require_coverage_closed:
        try:
            coverage = compute_obligation_coverage(
                plan=tp.model_dump(exclude_none=True),
                run_dir_path=str(run_dir),
                ledger_store=ledger_store,
            )
        except Exception as e:
            coverage = {
                "ok": False,
                "error": f"coverage_error:{e}",
                "counts": {},
                "undischarged": [],
                "discharged": [],
            }
    else:
        coverage = {"ok": True, "skipped": True}

    ok = bool(wiring.get("ok", False)) and bool(coverage.get("ok", False))

    report = {
        "ok": ok,
        "wiring": wiring,
        "coverage": coverage,
    }

    # Best-effort persistence
    try:
        outp = run_dir / "validations" / "closure.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pass

    return report
