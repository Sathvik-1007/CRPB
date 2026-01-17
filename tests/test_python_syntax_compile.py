from __future__ import annotations

import py_compile
from pathlib import Path


def test_all_crpb_python_files_compile() -> None:
    """Fail fast on SyntaxError anywhere in the package.

    This intentionally compiles files without importing them to avoid side effects.
    """

    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "crpb"
    assert pkg_root.is_dir(), f"Missing package dir: {pkg_root}"

    py_files = sorted(p for p in pkg_root.rglob("*.py") if p.is_file())
    assert py_files, "No Python files found under crpb/"

    # Ensure deterministic paths in error messages
    for p in py_files:
        # Skip any __pycache__ artifacts if present
        if "__pycache__" in p.parts:
            continue
        py_compile.compile(str(p), doraise=True, optimize=0)


def test_cli_entrypoint_compiles() -> None:
    """Extra guard: entrypoint should always at least compile."""

    repo_root = Path(__file__).resolve().parents[1]
    entry = repo_root / "crpb" / "cli.py"
    assert entry.is_file()
    py_compile.compile(str(entry), doraise=True, optimize=0)
