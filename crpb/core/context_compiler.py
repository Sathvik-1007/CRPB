from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from .ledger import NodeLedger


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_context_pack(pack: Dict[str, Any]) -> str:
    """Stable digest used for provenance and replay diagnostics."""
    s = _stable_json(pack)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _env_int(name: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        v = int(default)
    else:
        try:
            v = int(str(raw).strip())
        except Exception:
            v = int(default)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


class ContextBudgetExceededError(RuntimeError):
    pass


def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


class ContextCompiler:
    """Deterministic, budgeted context compilation.

    This produces *structured* context (a ContextPack dict) which can be passed via
    `constraints.side_context.context_pack`.

    Budgeting is character-based to avoid provider-specific tokenizers.
    """

    def __init__(self, *, max_chars: Optional[int] = None) -> None:
        # Defaults are env-driven; can be overridden per call via constraints.
        self.max_chars = (
            int(max_chars)
            if max_chars is not None
            else _env_int("CRPB_CONTEXT_MAX_CHARS", 12000, lo=1000)
        )

    def compile(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        node: Dict[str, Any],
        parent: Optional[Dict[str, Any]],
        siblings: List[Dict[str, Any]],
        ledger: Optional[NodeLedger],
        artifacts: List[Dict[str, Any]],
        files: Dict[str, str],
        file_specs: Optional[Dict[str, Any]] = None,
        signals: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        budget = int(constraints.get("context_max_chars") or self.max_chars)
        budget = max(1000, budget)

        if strict is None:
            # Default behavior: strict, no silent truncation.
            # Callers can explicitly pass strict=False for best-effort diagnostic usage.
            strict = True

        def _size(x: Dict[str, Any]) -> int:
            try:
                return len(_stable_json(x))
            except Exception:
                return 10**9

        def _bool(v: Any, default: bool = False) -> bool:
            if v is None:
                return bool(default)
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
            return bool(default)

        # Config: sections and quotas.
        # Back-compat: legacy snippet knobs are treated as whole-file inclusion limits (never truncate).
        legacy_max_snips = constraints.get("context_max_file_snippets")
        legacy_per_snip_chars = constraints.get("context_file_snippet_chars")

        include_file_text = _bool(constraints.get("context_include_file_text"), default=False)
        if legacy_max_snips is not None:
            include_file_text = True

        max_file_texts: Optional[int] = None
        try:
            if legacy_max_snips is not None:
                max_file_texts = max(0, int(legacy_max_snips))
        except Exception:
            max_file_texts = None

        max_single_file_text_chars: Optional[int] = None
        try:
            if legacy_per_snip_chars is not None:
                max_single_file_text_chars = max(0, int(legacy_per_snip_chars))
        except Exception:
            max_single_file_text_chars = None
        require_file_text = constraints.get("context_require_file_text")
        required_file_paths: List[str] = []
        if isinstance(require_file_text, list):
            required_file_paths = [str(p) for p in require_file_text if str(p)]

        section_max_chars: Dict[str, int] = {}
        if isinstance(constraints.get("context_section_max_chars"), dict):
            for k, v in constraints["context_section_max_chars"].items():
                try:
                    section_max_chars[str(k)] = max(0, int(v))
                except Exception:
                    continue

        section_quotas: Dict[str, float] = {}
        if isinstance(constraints.get("context_section_quotas"), dict):
            for k, v in constraints["context_section_quotas"].items():
                try:
                    section_quotas[str(k)] = float(v)
                except Exception:
                    continue

        enabled_sections: List[str] = []
        if isinstance(constraints.get("context_sections"), list):
            enabled_sections = [str(s) for s in constraints.get("context_sections") or [] if str(s)]

        # Small always-on core (no truncation; references used when needed).
        idea_txt = str(idea or "")
        pack: Dict[str, Any] = {
            "schema_version": 2,
            # These are part of the serialized payload, so they must be budgeted too.
            "digest": "0" * 40,
            "budget_chars": int(budget),
            "size_chars": 0,
            "idea": None,
            "idea_ref": {"sha1": _sha1_text(idea_txt), "len": len(idea_txt)},
            "node": {
                "id": node.get("id"),
                "kind": node.get("kind"),
                "title": node.get("title"),
                "description": node.get("description"),
                "node_plan": node.get("node_plan")
                if isinstance(node.get("node_plan"), dict)
                else {},
                "meta": node.get("meta") if isinstance(node.get("meta"), dict) else {},
            },
            "parent": None,
            "siblings": [],
            "ledger": None,
            "artifacts": [],
            "files": {
                "touched": [],
                "refs": [],
                "texts": {},
                # Back-compat alias: previous schema used files.snippets
                "snippets": {},
            },
            "file_specs": None,
            "file_specs_ref": None,
            "signals": None,
            "signals_ref": None,
            "omissions": {
                "siblings_dropped": 0,
                "artifacts_dropped": 0,
                "file_refs_dropped": 0,
                "file_texts_dropped": 0,
                "ledger_todos_dropped": 0,
                "ledger_decisions_dropped": 0,
            },
        }

        # Decide whether to include full idea text.
        include_idea_text = _bool(constraints.get("context_include_idea_text"), default=True)
        if include_idea_text:
            pack["idea"] = idea_txt

        if isinstance(parent, dict):
            pack["parent"] = {
                "id": parent.get("id"),
                "kind": parent.get("kind"),
                "title": parent.get("title"),
                "description": parent.get("description"),
                "node_plan": parent.get("node_plan")
                if isinstance(parent.get("node_plan"), dict)
                else {},
            }

        sib_views: List[Dict[str, Any]] = []
        for s in siblings or []:
            if not isinstance(s, dict):
                continue
            sib_views.append(
                {
                    "id": s.get("id"),
                    "kind": s.get("kind"),
                    "title": s.get("title"),
                    "description": s.get("description"),
                }
            )
        pack["siblings"] = sorted(sib_views, key=lambda x: str(x.get("id") or x.get("title") or ""))

        if ledger is not None:
            # Ledger can be large; include obligations always, then add todos/decisions until budgeted.
            pack["ledger"] = {
                "node_id": ledger.node_id,
                "parent_id": ledger.parent_id,
                "todos": [],
                "decisions": [],
                "obligations": list(ledger.obligations),
                # Structured obligations (compact projection) to preserve durable criteria.
                "obligation_items": [],
                "ref": {
                    "sha1": _sha1_text(_stable_json(ledger.model_dump(exclude_none=True))),
                },
            }

            # Include a bounded, deterministic projection of structured obligations.
            try:
                max_items = _env_int("CRPB_CONTEXT_LEDGER_OBLIGATIONS_MAX", 50, lo=0, hi=500)
                raw_items = list(getattr(ledger, "obligation_items", []) or [])
                items: List[Dict[str, Any]] = []
                for it in raw_items:
                    if not isinstance(it, dict):
                        continue
                    items.append(
                        {
                            "id": it.get("id"),
                            "scope": it.get("scope"),
                            "statement": it.get("statement"),
                            "dod": it.get("dod"),
                            "deps": it.get("deps") if isinstance(it.get("deps"), list) else [],
                            "tags": it.get("tags") if isinstance(it.get("tags"), list) else [],
                        }
                    )
                items = sorted(items, key=lambda x: str(x.get("id") or ""))
                if max_items >= 0:
                    items = items[: int(max_items)]
                pack["ledger"]["obligation_items"] = items
            except Exception:
                pass

        # Artifacts: metadata only.
        arts: List[Dict[str, Any]] = []
        for a in artifacts or []:
            if not isinstance(a, dict):
                continue
            arts.append(
                {
                    "id": a.get("id"),
                    "kind": a.get("kind"),
                    "path": a.get("path"),
                    "validation": a.get("validation"),
                }
            )

        # Files: always include touched list and refs (digest/len). Text inclusion is opt-in and whole-file only.
        touched_all = sorted([str(p) for p in (files or {}).keys()])
        pack["files"]["touched"] = touched_all

        file_refs: List[Dict[str, Any]] = []
        for p in touched_all:
            txt = files.get(p) or ""
            file_refs.append({"path": p, "sha1": _sha1_text(str(txt)), "len": len(str(txt))})
        pack["files"]["refs"] = file_refs

        # Deterministic relevance scoring (no regex, no language assumptions).
        node_plan_txt = ""
        try:
            np = pack.get("node", {}).get("node_plan") or {}
            if isinstance(np, dict):
                node_plan_txt = _stable_json(np)
        except Exception:
            node_plan_txt = ""

        file_spec_txt = ""
        try:
            if isinstance(file_specs, dict) and file_specs:
                file_spec_txt = _stable_json(file_specs)
        except Exception:
            file_spec_txt = ""

        def _rel_file(path: str) -> int:
            p = str(path or "")
            if not p:
                return 0
            score = 1
            # Strong signal: explicit mentions in node_plan / codespec file.
            if p in node_plan_txt:
                score += 10
            if p in file_spec_txt:
                score += 8
            # Medium signal: basename mention.
            base = p.split("/")[-1]
            if base and base in node_plan_txt:
                score += 4
            if base and base in file_spec_txt:
                score += 3
            return score

        # Artifacts: relevance-gated metadata only.
        # Prefer artifacts whose id/path is mentioned in the node_plan/codespec.
        def _rel_art(a: Dict[str, Any]) -> int:
            aid = str(a.get("id") or "")
            ap = str(a.get("path") or "")
            score = 1
            for needle in (aid, ap):
                if not needle:
                    continue
                if needle in node_plan_txt:
                    score += 10
                if needle in file_spec_txt:
                    score += 6
            return score

        # Apply section budgets (chars) and deterministic item selection.
        # Section names are stable and neutral.
        # - core/tree are always included.
        # - ledger_todos/ledger_decisions, siblings, artifacts, file_refs, file_texts, signals, file_specs
        #   are subject to quotas.

        def _section_budget(section: str, remaining_total: int, enabled: List[str]) -> int:
            if section in section_max_chars:
                return max(0, int(section_max_chars[section]))
            if section in section_quotas and section_quotas[section] > 0:
                return max(0, int(remaining_total * float(section_quotas[section])))
            if enabled:
                return max(0, int(remaining_total / max(1, len(enabled))))
            return 0

        def _fit_list_items(*, items: List[Dict[str, Any]], section_budget: int) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            used = 0
            for it in items:
                try:
                    it_size = len(_stable_json(it))
                except Exception:
                    it_size = 10**9
                if section_budget > 0 and (used + it_size) > section_budget:
                    break
                out.append(it)
                used += it_size
            return out

        # Establish which optional sections are enabled.
        optional_sections = [
            "siblings",
            "artifacts",
            "file_refs",
            "file_texts",
            "ledger_todos",
            "ledger_decisions",
            "signals",
            "file_specs",
            "idea",
        ]
        if enabled_sections:
            enabled_opt = [s for s in optional_sections if s in enabled_sections]
        else:
            enabled_opt = list(optional_sections)

        # Start from a minimal baseline and then fill optional sections under budgets.
        # 1) Idea: include full text only if it fits into its section budget (otherwise keep ref only).
        base_size = _size(pack)
        remaining_total = max(0, budget - base_size)
        idea_budget = _section_budget("idea", remaining_total, enabled_opt)
        if pack.get("idea") is not None and idea_budget > 0:
            # If idea text alone exceeds its budget, drop full text and keep ref.
            if len(_stable_json({"idea": pack.get("idea")})) > idea_budget:
                pack["idea"] = None

        # 2) file_specs and signals: include full dict if it fits; else include ref only.
        if isinstance(file_specs, dict) and file_specs:
            file_specs_json = _stable_json(file_specs)
            pack["file_specs_ref"] = {"sha1": _sha1_text(file_specs_json), "len": len(file_specs_json)}
        if isinstance(signals, dict) and signals:
            signals_json = _stable_json(signals)
            pack["signals_ref"] = {"sha1": _sha1_text(signals_json), "len": len(signals_json)}

        base_size = _size(pack)
        remaining_total = max(0, budget - base_size)

        fs_budget = _section_budget("file_specs", remaining_total, enabled_opt)
        if isinstance(file_specs, dict) and file_specs and fs_budget > 0:
            if len(_stable_json(file_specs)) <= fs_budget:
                pack["file_specs"] = file_specs

        sig_budget = _section_budget("signals", remaining_total, enabled_opt)
        if isinstance(signals, dict) and signals and sig_budget > 0:
            if len(_stable_json(signals)) <= sig_budget:
                pack["signals"] = signals

        # 3) siblings
        def _rel_sib(s: Dict[str, Any]) -> int:
            sid = str(s.get("id") or "")
            title = str(s.get("title") or "")
            score = 1
            if sid and sid in node_plan_txt:
                score += 6
            if title and title in node_plan_txt:
                score += 4
            return score

        sib_sorted = sorted(
            pack.get("siblings") or [],
            key=lambda s: (-_rel_sib(s), str(s.get("id") or s.get("title") or "")),
        )
        base_size = _size(pack)
        remaining_total = max(0, budget - base_size)
        sib_budget = _section_budget("siblings", remaining_total, enabled_opt)
        sib_fit = _fit_list_items(items=sib_sorted, section_budget=sib_budget)
        pack["omissions"]["siblings_dropped"] = max(0, len(sib_sorted) - len(sib_fit))
        pack["siblings"] = sib_fit

        # 4) artifacts
        art_sorted = sorted(arts, key=lambda a: (-_rel_art(a), str(a.get("id") or a.get("path") or "")))
        base_size = _size(pack)
        remaining_total = max(0, budget - base_size)
        art_budget = _section_budget("artifacts", remaining_total, enabled_opt)
        art_fit = _fit_list_items(items=art_sorted, section_budget=art_budget)
        pack["omissions"]["artifacts_dropped"] = max(0, len(art_sorted) - len(art_fit))
        pack["artifacts"] = art_fit

        # 5) file refs
        ref_sorted = sorted(file_refs, key=lambda r: (-_rel_file(str(r.get("path") or "")), str(r.get("path") or "")))
        base_size = _size(pack)
        remaining_total = max(0, budget - base_size)
        refs_budget = _section_budget("file_refs", remaining_total, enabled_opt)
        refs_fit = _fit_list_items(items=ref_sorted, section_budget=refs_budget)
        pack["omissions"]["file_refs_dropped"] = max(0, len(ref_sorted) - len(refs_fit))
        pack["files"]["refs"] = refs_fit
        # Keep touched aligned with the included refs to avoid un-budgeted touched explosions.
        pack["files"]["touched"] = [str(r.get("path") or "") for r in refs_fit if str(r.get("path") or "")]

        # 6) optional file texts (whole-file only)
        texts: Dict[str, str] = {}
        dropped_texts = 0
        if include_file_text:
            base_size = _size(pack)
            remaining_total = max(0, budget - base_size)
            text_budget = _section_budget("file_texts", remaining_total, enabled_opt)
            ranked_paths = [str(r.get("path") or "") for r in refs_fit]
            ranked_paths = [p for p in ranked_paths if p]
            used = 0
            for p in ranked_paths:
                if max_file_texts is not None and len(texts) >= max_file_texts:
                    dropped_texts += 1
                    continue
                txt = str(files.get(p) or "")
                if not txt:
                    continue
                if max_single_file_text_chars is not None and max_single_file_text_chars > 0:
                    if len(txt) > max_single_file_text_chars:
                        dropped_texts += 1
                        continue
                entry = {"path": p, "text": txt}
                entry_size = len(_stable_json(entry))
                if text_budget > 0 and (used + entry_size) > text_budget:
                    dropped_texts += 1
                    continue
                texts[p] = txt
                used += entry_size
        pack["files"]["texts"] = texts
        pack["files"]["snippets"] = dict(texts)
        pack["omissions"]["file_texts_dropped"] = int(dropped_texts)

        # Enforce required file text contract in strict mode.
        if strict and required_file_paths:
            missing_required = [p for p in required_file_paths if p not in pack.get("files", {}).get("texts", {})]
            if missing_required:
                raise ContextBudgetExceededError(
                    "context_required_file_text_missing:" + ",".join(sorted(missing_required))
                )

        # 7) ledger todos/decisions under budgets (whole items only)
        if isinstance(pack.get("ledger"), dict) and ledger is not None:
            todos_all = [t.model_dump(exclude_none=True) for t in ledger.todos]
            decisions_all = [d.model_dump(exclude_none=True) for d in ledger.decisions]
            base_size = _size(pack)
            remaining_total = max(0, budget - base_size)
            todo_budget = _section_budget("ledger_todos", remaining_total, enabled_opt)
            todos_fit = _fit_list_items(items=todos_all, section_budget=todo_budget)
            pack["ledger"]["todos"] = todos_fit
            pack["omissions"]["ledger_todos_dropped"] = max(0, len(todos_all) - len(todos_fit))

            base_size = _size(pack)
            remaining_total = max(0, budget - base_size)
            dec_budget = _section_budget("ledger_decisions", remaining_total, enabled_opt)
            dec_fit = _fit_list_items(items=decisions_all, section_budget=dec_budget)
            pack["ledger"]["decisions"] = dec_fit
            pack["omissions"]["ledger_decisions_dropped"] = max(0, len(decisions_all) - len(dec_fit))

        # Final enforce overall budget by dropping optional content (never truncating strings).
        # Drop order: file texts -> signals -> file_specs -> ledger extras -> file refs -> artifacts -> siblings -> idea.
        drop_order = [
            "file_texts",
            "signals",
            "file_specs",
            "ledger_decisions",
            "ledger_todos",
            "file_refs",
            "artifacts",
            "siblings",
            "idea",
        ]

        def _drop_one(section: str) -> bool:
            if section == "file_texts":
                if isinstance(pack.get("files"), dict) and pack["files"].get("texts"):
                    # deterministic: drop lexicographically last
                    keys = sorted([str(k) for k in (pack["files"].get("texts") or {}).keys()])
                    if keys:
                        del pack["files"]["texts"][keys[-1]]
                        try:
                            if isinstance(pack["files"].get("snippets"), dict):
                                pack["files"]["snippets"] = dict(pack["files"].get("texts") or {})
                        except Exception:
                            pass
                        pack["omissions"]["file_texts_dropped"] = int(pack["omissions"].get("file_texts_dropped") or 0) + 1
                        return True
                return False
            if section == "signals":
                if pack.get("signals") is not None:
                    pack["signals"] = None
                    return True
                return False
            if section == "file_specs":
                if pack.get("file_specs") is not None:
                    pack["file_specs"] = None
                    return True
                return False
            if section == "ledger_decisions":
                if isinstance(pack.get("ledger"), dict) and (pack["ledger"].get("decisions") or []):
                    cur = list(pack["ledger"].get("decisions") or [])
                    cur.pop()  # drop last
                    pack["ledger"]["decisions"] = cur
                    pack["omissions"]["ledger_decisions_dropped"] = int(pack["omissions"].get("ledger_decisions_dropped") or 0) + 1
                    return True
                return False
            if section == "ledger_todos":
                if isinstance(pack.get("ledger"), dict) and (pack["ledger"].get("todos") or []):
                    cur = list(pack["ledger"].get("todos") or [])
                    cur.pop()
                    pack["ledger"]["todos"] = cur
                    pack["omissions"]["ledger_todos_dropped"] = int(pack["omissions"].get("ledger_todos_dropped") or 0) + 1
                    return True
                return False
            if section == "file_refs":
                if isinstance(pack.get("files"), dict) and (pack["files"].get("refs") or []):
                    cur = list(pack["files"].get("refs") or [])
                    cur.pop()
                    pack["files"]["refs"] = cur
                    try:
                        pack["files"]["touched"] = [
                            str(r.get("path") or "")
                            for r in (pack["files"].get("refs") or [])
                            if str(r.get("path") or "")
                        ]
                    except Exception:
                        pass
                    pack["omissions"]["file_refs_dropped"] = int(pack["omissions"].get("file_refs_dropped") or 0) + 1
                    return True
                return False
            if section == "artifacts":
                if (pack.get("artifacts") or []):
                    cur = list(pack.get("artifacts") or [])
                    cur.pop()
                    pack["artifacts"] = cur
                    pack["omissions"]["artifacts_dropped"] = int(pack["omissions"].get("artifacts_dropped") or 0) + 1
                    return True
                return False
            if section == "siblings":
                if (pack.get("siblings") or []):
                    cur = list(pack.get("siblings") or [])
                    cur.pop()
                    pack["siblings"] = cur
                    pack["omissions"]["siblings_dropped"] = int(pack["omissions"].get("siblings_dropped") or 0) + 1
                    return True
                return False
            if section == "idea":
                if pack.get("idea") is not None:
                    pack["idea"] = None
                    return True
                return False
            return False

        size_chars = _size(pack)
        if size_chars > budget:
            for _ in range(5000):
                if size_chars <= budget:
                    break
                dropped = False
                for sec in drop_order:
                    if _drop_one(sec):
                        dropped = True
                        break
                if not dropped:
                    break
                size_chars = _size(pack)
            if size_chars > budget:
                # If we cannot fit even after dropping all optional content, fail in strict mode.
                if strict:
                    raise ContextBudgetExceededError(
                        f"context_pack_budget_exceeded:size={size_chars}:budget={budget}"
                    )
                # Non-strict: keep as-is (caller opted out of strict behavior).

        # Finalize digest + stabilize size_chars without changing field sizes materially.
        pack["budget_chars"] = int(budget)
        pack["digest"] = digest_context_pack(pack)
        # size_chars depends on its own encoding length; iterate to a fixed point.
        for _ in range(5):
            sz = _size(pack)
            if int(pack.get("size_chars") or 0) == int(sz):
                break
            pack["size_chars"] = int(sz)

        # Ensure the finalized pack still fits the budget.
        if _size(pack) > budget:
            # Try one more drop pass now that metadata is finalized.
            size_chars = _size(pack)
            for _ in range(5000):
                if size_chars <= budget:
                    break
                dropped = False
                for sec in drop_order:
                    if _drop_one(sec):
                        dropped = True
                        break
                if not dropped:
                    break
                # restabilize size field
                for _j in range(3):
                    sz = _size(pack)
                    if int(pack.get("size_chars") or 0) == int(sz):
                        break
                    pack["size_chars"] = int(sz)
                size_chars = _size(pack)

            if _size(pack) > budget and strict:
                raise ContextBudgetExceededError(
                    f"context_pack_budget_exceeded:size={_size(pack)}:budget={budget}"
                )

        return pack
