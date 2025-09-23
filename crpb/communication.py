from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from .utils.fs import ensure_parent, lock_file

class Comms:
    """
    Lightweight, language-agnostic communication layer for node-to-node messaging.
    Messages are persisted under runs/<run>/graph/comms.jsonl as compact JSON lines.

    Message schema:
    {
      "ts": float epoch seconds,
      "from": str node_id,
      "to": str node_id | "*",
      "kind": str,               # e.g., REQUEST_DETAILS, CONTEXT, MICRO_ADJUSTMENT, NOTE
      "path": Optional[str],     # optional file path this message concerns
      "payload": dict            # arbitrary JSON payload
    }
    """

    def __init__(self, comms_path: Path) -> None:
        self._path = comms_path
        ensure_parent(self._path)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def send(self, from_id: str, to_id: str, kind: str, payload: Dict[str, Any] | None = None, path: Optional[str] = None) -> None:
        msg = {
            "ts": time.time(),
            "from": from_id,
            "to": to_id,
            "kind": kind,
            "path": path,
            "payload": payload or {},
        }
        lock = self._path.with_suffix(self._path.suffix + ".lock")
        with lock_file(lock):
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, separators=(",", ":")) + "\n")

    def fetch(self, to_id: str | None = None, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for ln in lines:
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if to_id is not None and obj.get("to") not in (to_id, "*"):
                continue
            if kinds and obj.get("kind") not in kinds:
                continue
            out.append(obj)
        return out
