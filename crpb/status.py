from __future__ import annotations
from pathlib import Path
from typing import Optional
import time
from .utils.fs import append_jsonl, ensure_parent


class NodeStatus:
    def __init__(self, path: Path):
        self.path = path
        ensure_parent(self.path)

    def write(self, node_id: str, state: str, prev_state: Optional[str] = None, **extra):
        rec = {
            "at": time.time(),
            "node_id": node_id,
            "state": state,
            "prev_state": prev_state,
            **extra,
        }
        append_jsonl(self.path, [rec])
