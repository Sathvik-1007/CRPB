from __future__ import annotations
from pathlib import Path
import os
from typing import Dict
from .utils.fs import atomic_write_json, ensure_parent


def assemble_python_file(file_spec, impls: Dict[str, str]) -> str:
    """Assemble a Python module:
    - Optional import header from file_spec.imports
    - Deterministic functions: exports order, then remaining
    - Optional __main__ entrypoint calling file_spec.entrypoint
    """
    parts: list[str] = []
    # Configurable annotations behavior to avoid hardcoding per project
    # CRPB_PY_ANNOTATIONS: 'deferred' | 'eager' | 'auto' (default)
    ann_mode = os.getenv("CRPB_PY_ANNOTATIONS", "auto").lower()
    if ann_mode in ("deferred", "auto"):
        parts.append("from __future__ import annotations\n\n")
    # Header imports
    imports = getattr(file_spec, "imports", []) or []
    if imports:
        for mod in sorted(set(imports)):
            parts.append(f"import {mod}\n")
        parts.append("\n")
    # Functions in deterministic order
    seen = set()
    for name in file_spec.exports:
        code = impls.get(name)
        if code:
            parts.append(code.rstrip() + "\n")
            seen.add(name)
    for name, code in impls.items():
        if name not in seen:
            parts.append(code.rstrip() + "\n")
    # Optional entrypoint
    entry = getattr(file_spec, "entrypoint", None)
    if entry:
        parts.append("\n")
        parts.append("if __name__ == \"__main__\":\n")
        parts.append(f"    {entry}()\n")
    return "\n".join(parts).rstrip() + "\n"


def write_code_file(path: Path, code: str) -> None:
    ensure_parent(path)
    path.write_text(code, encoding="utf-8")


def write_file_meta(path: Path, spec: dict) -> None:
    atomic_write_json(path, spec)
