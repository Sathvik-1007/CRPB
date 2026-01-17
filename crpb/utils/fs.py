from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import portalocker


def normalize_project_relpath(path: str) -> str:
    """Normalize a project-relative file path to POSIX form.

    Rules:
    - Uses '/' separators.
    - Strips leading './' and leading '/'.
    - Rejects absolute paths (including Windows drive paths) and paths that escape via '..'.
    """
    s = str(path or "").strip()
    if not s:
        raise ValueError("empty_path")

    # Normalize slashes early.
    s = s.replace("\\", "/")

    # Reject Windows drive paths like C:/... or C:foo.
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        raise ValueError("absolute_drive_path")

    # Reject UNC-ish paths (best-effort).
    if s.startswith("//"):
        raise ValueError("absolute_unc_path")

    # Treat leading '/' as project-root relative.
    while s.startswith("/"):
        s = s[1:]
    while s.startswith("./"):
        s = s[2:]

    # Collapse '.' and '..' segments. Reject escaping above root.
    parts = [p for p in s.split("/") if p not in ("", ".")]
    out_parts: list[str] = []
    for seg in parts:
        if seg == "..":
            if not out_parts:
                raise ValueError("path_escapes_root")
            out_parts.pop()
            continue
        out_parts.append(seg)

    out = "/".join(out_parts).strip()
    if not out:
        raise ValueError("empty_after_normalize")
    return out


def safe_join(root: Path, rel: str) -> Path:
    """Join a potentially messy relative path under root safely."""
    return Path(root) / normalize_project_relpath(rel)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, items: Iterable[dict]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def lock_file(path: Path) -> portalocker.Lock:
    ensure_parent(path)
    return portalocker.Lock(str(path), timeout=5)


def write_text_locked(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text to a file guarded by a sidecar lock file (best-effort cross-process safety)."""
    ensure_parent(path)
    lock_path = path.parent / (path.name + ".lock")
    with lock_file(lock_path):
        path.write_text(text, encoding=encoding)
