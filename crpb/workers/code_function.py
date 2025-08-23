from __future__ import annotations
from typing import Tuple, Dict, Any
import os
from pathlib import Path
from ..agents.dspy_engine import DspyEngine


def implement_code_function(
    inputs: Dict[str, Any],
    constraints: Dict[str, Any] | None = None,
    model: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Implement a single function using LLM (python only for now).
    Inputs expected (with safe defaults):
      - language: 'python' (only supported for assembly)
      - path: file path for final assembly (e.g., 'src/main.py')
      - name: function name
      - signature: complete function signature string (e.g., 'def foo(x: int) -> int')
      - description: optional natural-language description
      - deps: optional list of function names required from the same file
      - exports: optional list of exported functions (used for ordering)
      - allowed_imports: optional list of allowed imports
      - file_contract: optional dict { name: signature } of all functions in the file
    Returns (code_text, meta) where meta can include updated exports or hints.
    """
    constraints = constraints or {}
    # Determine language: prefer explicit, else infer from file path; avoid defaulting to python
    language = inputs.get("language")
    if not language:
        p = inputs.get("path")
        if isinstance(p, str):
            ext = (Path(p).suffix or "").lower().lstrip(".")
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
    if language != "python":
        raise NotImplementedError("code:function worker currently supports language=python only")

    name: str = inputs.get("name") or "run"
    signature: str = inputs.get("signature") or f"def {name}() -> None"
    description: str = inputs.get("description", "")
    path: str = inputs.get("path", "src/main.py")
    deps = list(inputs.get("deps", []))
    exports = list(inputs.get("exports", [name]))
    allowed_imports = list(inputs.get("allowed_imports", []))
    file_contract: Dict[str, str] = dict(inputs.get("file_contract", {}))

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("LLM required for code implementation: set OPENAI_API_KEY")

    code_text: str | None = None
    try:
        engine = DspyEngine(model=model)
        dep_sig = {n: file_contract.get(n) for n in deps if n in file_contract}
        all_funcs = {n: s for n, s in file_contract.items()} or {name: signature}
        code_text = engine.generate_function_impl(
            file=path,
            exports=exports,
            all_functions=all_funcs,
            allowed_imports=allowed_imports,
            target_function=name,
            signature=signature,
            description=description,
            dependencies=deps,
            dependency_signatures={k: v for k, v in dep_sig.items() if v is not None},
            constraints=constraints,
            feedback=inputs.get("feedback"),
        )
    except Exception as e:
        raise RuntimeError(f"DSPy code generation failed: {e}")

    if code_text is None:
        raise RuntimeError("LLM returned no valid function code")

    # Basic snippet validation: parse as module
    try:
        import ast
        ast.parse(code_text)
    except SyntaxError as e:
        raise RuntimeError(f"generated snippet invalid syntax: {e}")

    meta = {
        "path": path,
        "language": language,
        "name": name,
        "signature": signature,
        "exports": exports,
        "allowed_imports": allowed_imports,
    }
    return code_text, meta
