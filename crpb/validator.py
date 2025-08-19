from __future__ import annotations
from typing import Tuple, Set, List, Any
import ast
import os
from .specs import FileSpec


def basic_file_validation(fs: FileSpec) -> Tuple[bool, str]:
    if not fs.functions:
        return False, "FileSpec must declare at least one function"
    for name, f in fs.functions.items():
        if name not in fs.exports:
            return False, f"Function {name} not exported"
        if not f.signature.startswith("def "):
            return False, f"Function {name} must start with 'def ' in signature"
    return True, "ok"


def validate_assembled_python_file(fs: FileSpec, code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    func_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in fs.exports:
        if required not in func_names:
            return False, f"Missing required function in assembled code: {required}"
    return True, "ok"


def simple_style_check(code: str, max_len: int = 120) -> Tuple[bool, str]:
    """Very light style check to avoid bringing heavy deps: only line length."""
    lines = code.splitlines()
    too_long = [i + 1 for i, ln in enumerate(lines) if len(ln) > max_len]
    if too_long:
        return False, f"Lines exceed {max_len} chars: {too_long[:5]}"  # cap list in message
    return True, "ok"


def jsonschema_validate(data: dict, schema: dict) -> Tuple[bool, str]:
    """
    Validate data against a JSON Schema, if jsonschema is available.
    Returns (ok, message). If jsonschema is not installed, returns (True, 'skipped').
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        return True, "skipped: jsonschema not installed"
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, "ok"
    except Exception as e:
        return False, f"schema_error: {e}"


def import_safety_check(code: str, allowed: Set[str] | None = None) -> Tuple[bool, str]:
    """
    Check that imports in the assembled code are limited to an allowed set.
    - Primary source of truth should be the file spec's `imports` list.
    - A minimal stdlib baseline is tolerated (typing, json, re, os, sys, pathlib, time, math, random,
      collections, itertools, functools, dataclasses). You can extend this baseline by setting the
      environment variable CRPB_IMPORT_BASELINE to a comma-separated list of additional module names.
      No third-party packages are permitted unless explicitly declared in the file spec `imports` or
      included in CRPB_IMPORT_BASELINE.
    """
    baseline = {
        "typing",
        "math",
        "random",
        "time",
        "sys",
        "os",
        "dataclasses",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "json",
        "re",
    }
    extra = os.getenv("CRPB_IMPORT_BASELINE", "")
    if extra:
        for token in extra.replace(";", ",").split(","):
            mod = token.strip()
            if mod:
                baseline.add(mod)
    allowed = set(allowed or set()) | baseline
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError before import check: {e}"
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = (alias.name or "").split(".")[0]
                if mod not in allowed:
                    violations.append(mod)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod and mod not in allowed:
                violations.append(mod)
    if violations:
        uniq = sorted(set(violations))
        return False, f"disallowed_imports: {uniq}"
    return True, "ok"


def integrity_check(fs: FileSpec, code: str) -> Tuple[bool, str]:
    """
    Static integrity checks after assembly:
    - If fs.entrypoint is set, ensure function exists in the code.
    - If fs.imports is set, ensure an import header exists for each module.
    Does not execute code; runtime checks are out of scope here.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError before integrity: {e}"
    # Check entrypoint function exists
    if getattr(fs, "entrypoint", None):
        func_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        if fs.entrypoint not in func_names:
            return False, f"entrypoint_missing: {fs.entrypoint}"
    # Check import headers
    imports = getattr(fs, "imports", []) or []
    if imports:
        # Build a set of top-level imported module names from AST
        present: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        present.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    present.add(node.module.split(".")[0])
        missing = [m for m in imports if m.split(".")[0] not in present]
        if missing:
            return False, f"missing_import_headers: {missing}"
    return True, "ok"


def _load_module_from_file(module_name: str, file_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        raise ImportError(f"module_exec_error: {e}")
    return mod


def runtime_validate_examples(fs: FileSpec, assembled_path: str) -> Tuple[bool, List[dict]]:
    """
    Execute FunctionSpec.examples against the assembled file.
    - Each example.inp is treated as kwargs by default. If it contains 'args' (list) and/or 'kwargs' (dict), use them.
    - Compare return value to example.out via ==.
    Returns (ok_all, results[]). Each result has: {function, ok, expected, got, error?}
    """
    results: List[dict] = []
    if not fs.functions:
        return True, results
    mod_name = "crpb_runtime_" + str(abs(hash(assembled_path)))
    try:
        mod = _load_module_from_file(mod_name, assembled_path)
    except Exception as e:
        return False, [{"function": "*", "ok": False, "error": str(e)}]

    ok_all = True
    for fname, fmeta in fs.functions.items():
        if not fmeta.examples:
            continue
        fn = getattr(mod, fname, None)
        if not callable(fn):
            results.append({"function": fname, "ok": False, "error": "function_missing_in_module"})
            ok_all = False
            continue
        for ex in fmeta.examples:
            try:
                if isinstance(ex.inp, dict) and ("args" in ex.inp or "kwargs" in ex.inp):
                    args = list(ex.inp.get("args", []))
                    kwargs = dict(ex.inp.get("kwargs", {}))
                    got: Any = fn(*args, **kwargs)
                elif isinstance(ex.inp, dict):
                    got = fn(**ex.inp)
                elif isinstance(ex.inp, list):
                    got = fn(*ex.inp)
                else:
                    # scalar input treated as single positional arg
                    got = fn(ex.inp)
                ok = got == ex.out
                if not ok:
                    ok_all = False
                results.append({"function": fname, "ok": ok, "expected": ex.out, "got": got})
            except Exception as e:
                ok_all = False
                results.append({"function": fname, "ok": False, "error": f"exec_error: {e}"})
    return ok_all, results
