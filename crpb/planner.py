from __future__ import annotations
from typing import List, Tuple
from pathlib import Path as _Path
from .specs import Plan, ModuleSpec, FileSpec, FunctionSpec, FunctionExample, TaskPlan, TaskSpec


def _fallback_plan(idea: str, constraints: dict) -> Tuple[Plan, List[FileSpec]]:
    """
    Deterministic, domain-agnostic fallback that is constraint-driven.
    - If constraints contains full Plan-like modules/files/functions, honor them.
    - Else if constraints contains a simplified `files` array, wrap in a single module.
    - Else produce a minimal single-file spec using constraint defaults.
    Absolutely no domain-specific assumptions; defaults remain generic.
    """

    def build_function_spec(fd: dict, *, language: str | None = None) -> FunctionSpec:
        name = fd.get("name") or fd.get("function") or "run"
        # Default signature is language-aware; for non-Python code or assets, it can be empty
        if "signature" in fd and isinstance(fd.get("signature"), str):
            signature = fd.get("signature")
        else:
            if language == "python":
                signature = f"def {name}() -> None"
            else:
                # Leave unspecified for non-Python (frontend/assets may have 0 functions)
                signature = ""
        returns = fd.get("returns", "None")
        description = fd.get("description", "")
        deps = fd.get("deps", [])
        examples = [
            FunctionExample(inp=e.get("in", {}), out=e.get("out", {}))
            for e in fd.get("examples", [])
        ]
        tests = fd.get("tests", [])
        return FunctionSpec(
            name=name,
            signature=signature,
            returns=returns,
            description=description,
            examples=examples,
            tests=tests,
            deps=deps,
        )

    def build_file_spec(fd: dict) -> FileSpec:
        # Infer language from path extension if not provided; avoid defaulting to python
        language = fd.get("language")
        raw_path = fd.get("path") or fd.get("default_path")
        if not language and raw_path:
            ext = _Path(raw_path).suffix.lower().lstrip(".")
            language = {
                "py": "python",
                "ts": "typescript",
                "tsx": "typescript",
                "js": "javascript",
                "jsx": "javascript",
                "go": "go",
                "html": "html",
                "css": "css",
                "json": "json",
                "md": "markdown",
                "toml": "toml",
                "yaml": "yaml",
                "yml": "yaml",
            }.get(ext)
        # basic extension mapping without implying domain
        ext_map = {"python": "py", "typescript": "ts", "go": "go", "html": "html", "css": "css", "json": "json"}
        default_name = fd.get("default_filename", "main")
        default_path = fd.get("default_path") or f"src/{default_name}.{ext_map.get(language, 'txt')}"
        path = fd.get("path", default_path)
        entrypoint = fd.get("entrypoint") or fd.get("default_entrypoint") or "run"
        imports = fd.get("imports", [])
        # functions can be dict{name->spec} or list of specs
        funcs_in = fd.get("functions", {})
        functions: dict[str, FunctionSpec] = {}
        if isinstance(funcs_in, dict):
            for fname, meta in funcs_in.items():
                # ensure name present
                meta = dict(meta or {})
                meta.setdefault("name", fname)
                functions[fname] = build_function_spec(meta, language=language)
        elif isinstance(funcs_in, list):
            for item in funcs_in:
                fs = build_function_spec(item or {}, language=language)
                functions[fs.name] = fs
        else:
            # minimal default single function based on entrypoint
            functions[entrypoint] = build_function_spec({"name": entrypoint}, language=language)
        exports = fd.get("exports") or [n for n in functions.keys()]
        if entrypoint and entrypoint not in exports:
            exports = [entrypoint] + [e for e in exports if e != entrypoint]
        return FileSpec(
            path=path,
            language=language,
            functions=functions,
            exports=exports,
            imports=imports,
            entrypoint=entrypoint,
        )

    # 1) Full plan-like constraints with modules
    if isinstance(constraints, dict) and constraints.get("modules"):
        modules: List[ModuleSpec] = []
        files_all: List[FileSpec] = []
        for m in constraints.get("modules", []):
            file_specs: List[FileSpec] = []
            for f in m.get("files", []):
                fs = build_file_spec(f)
                file_specs.append(fs)
                files_all.append(fs)
            modules.append(ModuleSpec(
                name=m.get("name", "app"),
                purpose=m.get("purpose", ""),
                priority=m.get("priority", "medium"),
                deps=m.get("deps", []),
                files=file_specs,
            ))
        return Plan(idea=idea, constraints=constraints, modules=modules), files_all

    # 2) Simplified constraints with a `files` array
    if isinstance(constraints, dict) and constraints.get("files"):
        files: List[FileSpec] = [build_file_spec(f) for f in constraints.get("files", [])]
        modules = [ModuleSpec(name=constraints.get("module_name", "app"), files=files)]
        return Plan(idea=idea, constraints=constraints, modules=modules), files

    # 3) Minimal deterministic default using provided defaults (no domain assumptions)
    language = constraints.get("default_language") if isinstance(constraints, dict) else None
    entrypoint = (constraints.get("entrypoint") or constraints.get("default_entrypoint") or "run") if isinstance(constraints, dict) else "run"
    imports = (constraints.get("imports") or []) if isinstance(constraints, dict) else []
    path = None
    if isinstance(constraints, dict):
        # reuse build_file_spec defaults by constructing a small dict
        path = build_file_spec({
            "language": language,
            "default_filename": constraints.get("default_filename", "main"),
            "default_path": constraints.get("default_path"),
        }).path
    else:
        path = "src/main.py"
    functions = {
        entrypoint: FunctionSpec(
            name=entrypoint,
            signature=f"def {entrypoint}() -> None" if language == "python" else "",
            returns="None",
            description="",
            examples=[],
            tests=[],
            deps=[],
        )
    }
    files = [FileSpec(path=path, language=language, exports=[entrypoint], imports=imports, entrypoint=entrypoint, functions=functions)]
    modules = [ModuleSpec(name=constraints.get("module_name", "app") if isinstance(constraints, dict) else "app", files=files)]
    return Plan(idea=idea, constraints=constraints, modules=modules), files


def generate_plan(idea: str, constraints: dict, use_llm: bool = True) -> Tuple[Plan, List[FileSpec]]:
    """
    Use the LLM to propose a multi-file plan (modules->files->functions).
    LLM is required; if unavailable or fails, this function raises an error.
    """
    if use_llm:
        try:
            from .agents.dspy_engine import DspyEngine
            engine = DspyEngine()
            obj = engine.plan(idea, constraints)
            modules: List[ModuleSpec] = []
            files_all: List[FileSpec] = []
            for m in obj.get("modules", []):
                file_specs: List[FileSpec] = []
                for f in m.get("files", []):
                    funcs = {}
                    ffuncs = f.get("functions", {})
                    # Determine language early to select sane defaults (infer from path if missing)
                    _lang = f.get("language")
                    if not _lang:
                        try:
                            ext = _Path(f.get("path", "")).suffix.lower().lstrip(".")
                            _lang = {
                                "py": "python",
                                "ts": "typescript",
                                "tsx": "typescript",
                                "js": "javascript",
                                "jsx": "javascript",
                                "go": "go",
                                "html": "html",
                                "css": "css",
                                "json": "json",
                                "md": "markdown",
                                "toml": "toml",
                                "yaml": "yaml",
                                "yml": "yaml",
                            }.get(ext)
                        except Exception:
                            _lang = None
                    for fname, fmeta in ffuncs.items():
                        # Default signature: Python gets a concrete def; others may be empty/unspecified
                        _sig = fmeta.get("signature")
                        if _sig is None:
                            _sig = f"def {fname}() -> None" if _lang == "python" else ""
                        funcs[fname] = FunctionSpec(
                            name=fname,
                            signature=_sig,
                            returns=fmeta.get("returns", "None"),
                            description=fmeta.get("description", ""),
                            examples=[FunctionExample(inp=e.get("in", {}), out=e.get("out", {})) for e in fmeta.get("examples", [])],
                            tests=fmeta.get("tests", []),
                            deps=fmeta.get("deps", []),
                        )
                    # Integrity hints: imports/entrypoint (optional)
                    imports = f.get("imports", [])
                    entry = f.get("entrypoint")
                    # Heuristic only for entrypoint if present in functions
                    if not entry and ("run" in funcs):
                        entry = "run"
                    fs = FileSpec(
                        path=f.get("path"),
                        language=_lang,
                        exports=f.get("exports", list(funcs.keys())),
                        functions=funcs,
                        imports=imports,
                        entrypoint=entry,
                    )
                    file_specs.append(fs)
                    files_all.append(fs)
                modules.append(ModuleSpec(
                    name=m.get("name", "app"),
                    purpose=m.get("purpose", ""),
                    priority=m.get("priority", "medium"),
                    deps=m.get("deps", []),
                    files=file_specs,
                ))
            plan = Plan(idea=idea, constraints=constraints, modules=modules)
            # validate minimal
            if not files_all:
                raise ValueError("empty-plan")
            return plan, files_all
        except Exception as e:
            # Always require DSPy; provide a clear, actionable message.
            raise RuntimeError(
                f"DSPy planning failed: {e}. Ensure OPENAI_API_KEY and dspy-ai are installed."
            )
    # If use_llm is False or anything else, we still enforce LLM usage to keep behavior strict.
    raise RuntimeError("LLM planning disabled by configuration, but fallbacks are removed. Enable LLM.")


def generate_task_plan(idea: str, constraints: dict, use_llm: bool = True) -> TaskPlan:
    """
    Produce a generic hierarchical TaskPlan using the LLM. No code, only structure and metadata.
    LLM is required; if unavailable or fails, this function raises an error.
    """
    if use_llm:
        try:
            from .agents.dspy_engine import DspyEngine
            engine = DspyEngine()
            obj = engine.task_plan(idea, constraints)
            def build_task(t: dict) -> TaskSpec:
                children = [build_task(c) for c in t.get("children", [])]
                return TaskSpec(
                    id=t.get("id"),
                    kind=t.get("kind", "composite"),
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    priority=t.get("priority", "medium"),
                    deps=t.get("deps", []),
                    inputs=t.get("inputs", {}),
                    outputs=t.get("outputs", {}),
                    children=children,
                )
            tasks = [build_task(x) for x in obj.get("tasks", [])]
            if not tasks:
                raise ValueError("empty-task-plan")
            return TaskPlan(idea=idea, constraints=constraints, tasks=tasks)
        except Exception as e:
            # Always require DSPy; provide a clear, actionable message.
            raise RuntimeError(
                f"DSPy task planning failed: {e}. Ensure OPENAI_API_KEY and dspy-ai are installed."
            )
    # If use_llm is False or anything else, we still enforce LLM usage to keep behavior strict.
    raise RuntimeError("LLM task planning disabled by configuration, but fallbacks are removed. Enable LLM.")
