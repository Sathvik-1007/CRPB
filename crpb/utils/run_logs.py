from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


def _is_errorish_event(evt_type: str, payload: Dict[str, Any]) -> bool:
    t = str(evt_type or "").upper()
    if "ERROR" in t or "FAILED" in t or "EXCEPTION" in t:
        return True
    # Heuristic: payload includes an error-like field
    for k in ("error", "exception", "traceback", "message"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            if k in ("error", "exception", "traceback"):
                return True
    return False


def summarize_events_jsonl(
    events_path: Path,
    *,
    max_events: int = 200,
    max_lines: int = 5000,
) -> Dict[str, Any]:
    """Summarize an EventBus jsonl file into a bounded, structured object.

    Events are expected to be JSON objects with keys: at, type, payload.

    This is designed to be fed into LLM side_context without overflowing context.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "path": str(events_path),
        "counts_by_type": {},
        "recent": [],
        "recent_errors": [],
        "error_types": [],
    }

    if not events_path.exists() or not events_path.is_file():
        out["reason"] = "missing"
        return out

    lines: Deque[str] = deque(maxlen=max_lines)
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                lines.append(line)
    except Exception as e:
        out["reason"] = f"read_failed:{type(e).__name__}"
        return out

    events: List[Dict[str, Any]] = []
    for raw in list(lines)[-max_events:]:
        s = (raw or "").strip()
        if not s:
            continue
        try:
            evt = json.loads(s)
        except Exception:
            continue
        if isinstance(evt, dict):
            events.append(evt)

    counts = Counter()
    error_types: Counter[str] = Counter()
    recent: List[Dict[str, Any]] = []
    recent_errors: List[Dict[str, Any]] = []

    for evt in events:
        et = str(evt.get("type") or "")
        payload = evt.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        counts[et] += 1

        rec = {
            "type": et,
            "at": evt.get("at"),
            "payload_keys": sorted([str(k) for k in payload.keys()])[:30],
        }
        recent.append(rec)

        if _is_errorish_event(et, payload):
            error_types[et] += 1
            # Keep a slightly richer record for errors
            err = {
                "type": et,
                "at": evt.get("at"),
                "error": payload.get("error") or payload.get("exception") or payload.get("message"),
                "file": payload.get("file"),
                "node_id": payload.get("node_id"),
            }
            recent_errors.append(err)

    out["ok"] = True
    out["counts_by_type"] = dict(counts)
    out["recent"] = recent[-min(len(recent), 50) :]
    out["recent_errors"] = recent_errors[-min(len(recent_errors), 30) :]
    out["error_types"] = [
        {"type": t, "count": int(c)} for t, c in error_types.most_common(20)
    ]
    return out


def summarize_run_logs(
    *,
    run_dir: Path,
    max_events: int = 200,
) -> Dict[str, Any]:
    """Summarize run logs from a run directory.

    Currently supports EventBus logs at: <run_dir>/logs/events.jsonl
    """
    logs_dir = run_dir / "logs"
    events_path = logs_dir / "events.jsonl"
    summary = summarize_events_jsonl(events_path, max_events=max_events)
    return {
        "run_dir": str(run_dir),
        "events": summary,
    }
