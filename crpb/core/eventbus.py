from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..utils.fs import append_jsonl, ensure_parent


class EventBus:
    def __init__(self, events_path: Path):
        self.events_path = events_path
        ensure_parent(self.events_path)

    def emit(self, type_: str, **payload: Any) -> None:
        evt = {
            "at": time.time(),
            "type": type_,
            "payload": payload,
        }
        append_jsonl(self.events_path, [evt])
