from __future__ import annotations
from typing import List, Tuple
from .specs import Plan, ModuleSpec, FileSpec, FunctionSpec, FunctionExample


def _fallback_plan(idea: str, constraints: dict) -> Tuple[Plan, List[FileSpec]]:
    """
    Deterministic, domain-agnostic fallback that is constraint-driven.
    - If constraints contains full Plan-like modules/files/functions, honor them.
    - Else if constraints contains a simplified `files` array, wrap in a single module.
    - Else produce a minimal single-file spec using constraint defaults.
    Absolutely no domain-specific assumptions; defaults remain generic.
    """

    def build_function_spec(fd: dict) -> FunctionSpec:
        name = fd.get("name") or fd.get("function") or "run"
        signature = fd.get("signature") or f"def {name}() -> None"
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
        language = fd.get("language", "python")
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
                functions[fname] = build_function_spec(meta)
        elif isinstance(funcs_in, list):
            for item in funcs_in:
                fs = build_function_spec(item or {})
                functions[fs.name] = fs
        else:
            # minimal default single function based on entrypoint
            functions[entrypoint] = build_function_spec({"name": entrypoint})
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
    language = constraints.get("default_language", "python") if isinstance(constraints, dict) else "python"
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
            signature=f"def {entrypoint}() -> None" if language == "python" else f"function {entrypoint}()",
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


def generate_plan(idea: str, constraints: dict, use_llm: bool = True, require_llm: bool = False) -> Tuple[Plan, List[FileSpec]]:
    """
    Use LLM (if available) to propose a multi-file plan (modules->files->functions).
    Falls back to a deterministic python plan if LLM is unavailable or fails.
    """
    if use_llm:
        try:
            from .agents.llm import LLM
            import json
            llm = LLM()
            system = (
                "You are the Root Planning Agent in a recursive, parent-mediated build system (CRPB).\n"
                "Strictly output ONLY valid minified JSON matching this schema: {\n"
                "  modules: [\n"
                "    { name: string, purpose: string, priority: 'high'|'medium'|'low', deps: string[], files: [\n"
                "      { path: string, language: string, exports: string[], imports?: string[], entrypoint?: string,\n"
                "        functions: { [name: string]: { signature: string, returns: string, description: string, deps: string[], examples?: {in: object, out: object}[], tests?: string[] } }\n"
                "      }\n"
                "    ] }\n"
                "  ]\n"
                "}.\n"
                "Hard rules:\n"
                "- No code in any field. Only signatures and metadata.\n"
                "- Use snake_case for python function names.\n"
                "- Declare explicit deps only by function names within the same module/file unless absolutely necessary; cross-file deps must be minimal and parent will mediate via stubs/futures.\n"
                "- Prefer many small cohesive files over monoliths; keep exports minimal and clear.\n"
                "- Only include imports if concretely required; do not guess domain libraries.\n"
                "- Do not fabricate examples or tests; include them only if constraints provide them or they are trivial.\n"
                "- If a file has a natural entrypoint, set entrypoint to that exported function's name.\n"
                "- Keep the total number of functions reasonable; avoid deep dependency chains in a single step.\n"
                "- Do not include any commentary text outside JSON."
            )
            user = (
                "Plan a project for the following idea and constraints. Ensure valid JSON and include language per file.\n"
                f"idea: {idea}\nconstraints: {json.dumps(constraints)}"
            )
            content = llm.complete(system=system, messages=[{"role": "user", "content": user}], temperature=0.0)
            obj = json.loads(content)
            modules: List[ModuleSpec] = []
            files_all: List[FileSpec] = []
            for m in obj.get("modules", []):
                file_specs: List[FileSpec] = []
                for f in m.get("files", []):
                    funcs = {}
                    ffuncs = f.get("functions", {})
                    for fname, fmeta in ffuncs.items():
                        funcs[fname] = FunctionSpec(
                            name=fname,
                            signature=fmeta.get("signature", f"def {fname}() -> None"),
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
                        language=f.get("language", "python"),
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
            if require_llm:
                raise
            # else: fall back deterministically
            pass
    return _fallback_plan(idea, constraints)
