from __future__ import annotations
from pathlib import Path
from typing import Any, Tuple
from .utils.fs import read_json, atomic_write_json, lock_file


class Registry:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Tuple[int, dict]:
        data = read_json(self.path, {"version": 0, "files": {}})
        return data.get("version", 0), data

    def update(self, expected_version: int, mutate):
        # optimistic concurrency update with lock
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_file(lock_path):
            cur_ver, data = self.load()
            if cur_ver != expected_version:
                raise ConflictError(f"version conflict: expected {expected_version}, got {cur_ver}")
            new_data = mutate(data)
            new_data["version"] = cur_ver + 1
            atomic_write_json(self.path, new_data)
            return new_data

    def read_only(self) -> dict:
        return self.load()[1]


class ConflictError(Exception):
    pass
