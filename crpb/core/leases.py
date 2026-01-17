from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from ..utils.fs import atomic_write_json, read_json


class Leases:
    def __init__(self, path: Path):
        self.path = path

    def grant(self, node_id: str, ttl: float = 60.0) -> str:
        data = read_json(self.path, {})
        lease_id = f"lease_{int(time.time() * 1000)}"
        data[node_id] = {"lease_id": lease_id, "expires": time.time() + ttl}
        atomic_write_json(self.path, data)
        return lease_id

    def renew(self, node_id: str, lease_id: str, ttl: float = 60.0) -> None:
        data = read_json(self.path, {})
        rec = data.get(node_id)
        if not rec or rec.get("lease_id") != lease_id:
            return
        rec["expires"] = time.time() + ttl
        atomic_write_json(self.path, data)

    def active(self) -> Dict[str, dict]:
        return read_json(self.path, {})
