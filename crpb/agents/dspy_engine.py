from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List, Set
from ..config import DEFAULT_MODEL


class DspyEngine:
    """
    DSPy-backed orchestration for planning and code/file generation.
    - Configures a single DSPy LM globally.
    - Uses typed Signatures to structure prompts.
    - Provides strict postconditions and parsing with clear errors.
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            import dspy  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "DSPy is not installed. Please add 'dspy-ai' to requirements and pip install."
            ) from e
        # Configure LM once
        self._dspy = dspy
        mdl = model or DEFAULT_MODEL
        lm = None
        # Try multiple provider entry points for compatibility across DSPy versions
        try:
            OpenAICls = getattr(dspy, "OpenAI")
            try:
                lm = OpenAICls(model=mdl, api_key=api_key, temperature=0)  # type: ignore[call-arg]
            except TypeError:
                lm = OpenAICls(model=mdl, api_key=api_key)  # type: ignore[call-arg]
        except Exception:
            lm = None
        if lm is None:
            try:
                OpenAIChatCls = getattr(dspy, "OpenAIChat")
                try:
                    lm = OpenAIChatCls(model=mdl, api_key=api_key, temperature=0)  # type: ignore[call-arg]
                except TypeError:
                    lm = OpenAIChatCls(model=mdl, api_key=api_key)  # type: ignore[call-arg]
            except Exception:
                lm = None
        if lm is None:
            try:
                LMCls = getattr(dspy, "LM")
                try:
                    # Newer API may accept provider kwarg
                    lm = LMCls(provider="openai", model=mdl, api_key=api_key, temperature=0)  # type: ignore[call-arg]
                except TypeError:
                    # Or a single identifier string like "openai/<model>"
                    try:
                        lm = LMCls(f"openai/{mdl}", api_key=api_key, temperature=0)  # type: ignore[call-arg]
                    except TypeError:
                        lm = LMCls(f"openai/{mdl}", api_key=api_key)  # type: ignore[call-arg]
            except Exception:
                lm = None
        if lm is None:
            raise RuntimeError(
                "DSPy OpenAI provider not found. Your installed dspy-ai version may use a different API. "
                "Try upgrading dspy-ai, or ensure the OpenAI provider is available."
            )
        dspy.settings.configure(lm=lm)

        # Reusable rule blocks for composing prompt constraints without hardcoding
        self._RULE_BLOCKS: Dict[str, List[str]] = {
            "format_strict_json": [
                "ABSOLUTE: Return ONLY minified JSON on exactly one line (no raw newlines).",
                "ABSOLUTE: Use double quotes for all keys and strings. Never single quotes.",
                "ABSOLUTE: No trailing commas, no comments, no markdown, no backticks/code fences.",
                "Use only valid JSON escapes: \\, \", \/, \\b, \\f, \\n, \\r, \\t, and \\uXXXX.",
                "NEVER use backslash to escape a single quote (\\'). If a single quote appears, include it raw as '.",
                "Do NOT include standalone code snippets (e.g., if __name__ == '__main__':).",
                "Represent any code only inside designated fields (e.g., signature, file_contract); keep strings concise.",
                "If output would be long, shorten descriptions to brief phrases to keep within length and on one line.",
            ],
            "planning_general": [
                "Honor the user's idea and constraints literally; do not invent unrelated metadata.",
                "Think like a developer: partition by responsibility into modules/files when non-trivial; single-file acceptable only when truly minimal or explicitly requested.",
                "You MUST include at least one file (modules[].files[].path).",
                "Assign a programming language per file appropriate to its role; multi-language stacks are allowed. If language is obvious from the extension, you may omit the language field. If extension is missing/ambiguous, explicitly set the language and do NOT assume Python.",
                "For code files, include: functions{name->spec} and exports (include entrypoint if applicable). For asset files (HTML/CSS/JSON/MD/etc.), functions and exports may be omitted.",
                "Cover relevant cross-cutting concerns via modules/files when in-scope (config, validation, error handling, logging, tests, docs, CI, performance, security, privacy, rate limits, caching, observability, i18n/a11y, API/schema, data/persistence, deployment/packaging).",
                "Minimize prose; focus on implementable file/function specs; avoid placeholder boilerplate.",
            ],
            "task_plan_general": [
                "Honor the user's idea literally; scope tasks to what was asked.",
                "Think like a developer: tasks should map to deliverables/responsibilities (e.g., planning, API, UI, data, tests, docs, deployment).",
                "Prefer fewer, well-scoped tasks; split only when it improves clarity, parallelism, or risk management.",
                "Keep descriptions concise; minimize prose.",
                "Leaf tasks must be minimal, independently actionable units; avoid unnecessary leaves.",
                "Include cross-cutting tasks where relevant (validation, error handling, logging, performance, security, privacy, observability).",
            ],
            "no_language_defaults": [
                "Do not assume any programming language or tool unless specified in constraints.",
                "Explicitly set inputs.language for code tasks when the file extension is absent/ambiguous; NEVER default to Python or any other language.",
                "If the file extension does not make language obvious, explicitly set FileSpec.language; NEVER assume a default (including Python).",
            ],
            "code_function_contract": [
                "For code:function leaves, ensure inputs include: name, path, language, signature, allowed_imports (list), exports (list), and entrypoint when relevant.",
            ],
            "split_general": [
                "Honor the user's idea and parent task literally.",
                "If splitting, return a few meaningful children with clear, non-overlapping deliverables.",
                "Prefer partitioning by deliverable/role/phase; avoid redundant children.",
                "Select 'implement' only if the task is already a minimal actionable leaf (all-in-one acceptable when justified).",
                "When creating children of kind code:function, include minimal inputs contract: name, path, language, signature (and allowed_imports/exports/entrypoint as applicable). Never default the language; derive from path when obvious, otherwise specify explicitly.",
            ],
        }

        # Default rule blocks per mode; callers can override via constraints
        self._MODE_DEFAULT_RULE_BLOCKS: Dict[str, List[str]] = {
            "plan": ["format_strict_json", "planning_general", "no_language_defaults"],
            "task_plan": ["format_strict_json", "task_plan_general", "no_language_defaults", "code_function_contract"],
            "split": ["format_strict_json", "split_general", "no_language_defaults", "code_function_contract"],
        }

        # Define Signatures lazily to avoid top-level import-time failures when dspy missing
        class PlanSignature(dspy.Signature):  # type: ignore
            """Return ONLY minified JSON of a concrete software plan.
            Schema (top-level): {"modules": ModuleSpec[]}

            ModuleSpec fields:
            - name: string
            - purpose: string (short; optional)
            - priority: "high" | "medium" | "low" (default "medium")
            - deps: string[] (module-level deps; may be empty)
            - files: FileSpec[] (REQUIRED; non-empty for at least one module)

            FileSpec fields (developer-grade detail):
            - path: string (posix-like, e.g., "src/app/app.tsx" or "web/index.html"); REQUIRED
            - language?: string (e.g., "python", "typescript", "html", "css", "json"); may be omitted if inferable from path. If extension is missing/ambiguous, EXPLICITLY set "language"; do not default to any language.
            - functions?: { name: FunctionSpec } (for code files; may be empty or omitted for asset files like HTML/CSS/JSON/MD)
            - exports?: string[] (names of top-level functions to expose for code files)
            - imports?: string[] (allowed imports for validators; may be empty)
            - entrypoint?: string (name of function to run; optional; code files only)

            FunctionSpec minimal fields:
            - signature: string (exact signature text)
            - returns: string (type or descriptor)
            - description: string (concise)
            - deps?: string[] (other functions in same file this function depends on)
            - examples?: {in: any, out: any}[] (may be empty)
            - tests?: any[] (may be empty)

            Planning intent (general, rigorous, multi-language):
            - Honor the user's idea and constraints literally; do not invent unrelated metadata.
            - Think like a developer/architect: partition by responsibility into cohesive modules/files when non-trivial; a single-file plan is acceptable only when scope is truly minimal or explicitly requested.
            - Choose appropriate languages per file/module (e.g., HTML/CSS/JS for UI; TS/Go/Python/etc. for services). Multi-language stacks are common and allowed. If the file extension does not make language obvious, explicitly set FileSpec.language; NEVER assume a default (including Python).
            - Cover relevant cross-cutting concerns as applicable (do not bloat): configuration, validation, error handling, logging, testing, docs, CI, performance, security, privacy, rate limiting, caching, observability, i18n/a11y (for UI), API/schema/contracts, data model/persistence (if needed), deployment/packaging/entrypoints).
            - Always include at least one file. For code files, provide functions and exports; for asset files (HTML/CSS/JSON/MD), functions/exports may be omitted.

            Output rules (ABSOLUTE formatting):
            - Exactly one line of minified JSON. No raw newline characters anywhere in the output.
            - Use double quotes only for keys/strings. Never single quotes.
            - No trailing commas. No comments. No markdown/code fences/backticks.
            - If a description needs a line break, encode it as \\n inside the string.
            - Do not include standalone code snippets or shell commands.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            plan_json = dspy.OutputField()  # type: ignore

        class TaskPlanSignature(dspy.Signature):  # type: ignore
            """Return ONLY minified JSON for a hierarchical task plan.
            Output must be a syntactically valid, single-line JSON object:
            {"tasks": Task[]}

            Task fields:
            - id?: string (executor may assign if omitted)
            - kind: string label (e.g., "composite", "design", "research", "planning", "code:function", ...)
            - title: non-empty, specific summary
            - description: short explanation
            - priority: "high" | "medium" | "low"
            - deps: string[] of task ids (may be empty)
            - inputs: object (parameters/metadata for the worker handling this kind)
            - outputs: object (expected result contract or success criteria)
            - children?: Task[] (present only for composites)

            Guidance (developer-oriented, directional):
            - Honor the user's idea literally and keep scope faithful to it.
            - Think like a developer: decompose by deliverable/responsibility (planning, design, data, API, UI, testing, docs).
            - Keep it pragmatic: if the scope is simple, a single actionable task or a tiny set of tasks is acceptable.
            - When splitting, prefer a few specific children with clear contracts; avoid generic/duplicate tasks.
            - Children recursively follow these rules; stop splitting once tasks are smallest independently actionable units.
            - For implementation leaves of kind "code:function", include a minimal executable contract in inputs: name, path, language, signature, allowed_imports (empty list acceptable), exports (list), and entrypoint when relevant. Explicitly set inputs.language (derive from file path if clear; otherwise, specify explicitly). Do NOT assume defaults.

            Rules (ABSOLUTE):
            - Single line, minified JSON. Double quotes only. No trailing commas, comments, markdown, or code fences.
            - No raw newlines; encode as \\n within strings when needed.
            - Keep the number of top-level tasks small and decompose into a few children when warranted.
            - A leaf task is the smallest independently actionable unit for its kind (executable by a worker without further split). Do not add children to a leaf.
            - Do not assume any programming language or tool unless constraints explicitly require it. If language cannot be inferred from file path for code tasks, explicitly set inputs.language; never default to Python or any other language.
            - Do NOT include standalone code snippets; if code or signatures are required by constraints, include them only as concise strings inside inputs/outputs.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            tasks_json = dspy.OutputField()  # type: ignore

        class SplitDecisionSignature(dspy.Signature):  # type: ignore
            """Decide split vs implement for a Task.
            Return ONLY minified JSON of the shape (single line):
            {"action":"split"|"implement","children"?:Task[]}
            Rules (ABSOLUTE):
            - Single-line JSON, double quotes only, no trailing commas, no markdown/comments/backticks.
            - If the task is a composite with no children and is not yet a minimal leaf, return action="split" with a few meaningful children that partition scope by deliverable/role/phase.
            - Choose "implement" only when the task is already a smallest independently actionable unit for its kind.
            - Titles must be non-empty and specific; maintain or infer sensible deps. Children follow the Task schema in TaskPlanSignature.
            - Do not introduce language/tool-specific assumptions unless constraints require them. If creating children of kind "code:function", include in each child's inputs at minimum: name, path, language, signature; avoid defaulting to any language—derive from file path when obvious, otherwise specify explicitly.
            """
            task = dspy.InputField()  # type: ignore
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            decision_json = dspy.OutputField()  # type: ignore

        class FunctionImplSignature(dspy.Signature):  # type: ignore
            """Produce ONLY a single Python def block implementing the target function EXACTLY as specified.
            - No markdown, no fences, no extra commentary.
            - Must start with `def <name>(...):` and compile standalone when placed in the file.
            - Use only allowed imports; prefer pure-Python logic.
            - Do not change parameters or returns; do not create globals.
            """
            file = dspy.InputField()  # type: ignore
            exports = dspy.InputField()  # type: ignore
            all_functions = dspy.InputField()  # type: ignore
            allowed_imports = dspy.InputField()  # type: ignore
            target_function = dspy.InputField()  # type: ignore
            func_signature = dspy.InputField()  # type: ignore
            description = dspy.InputField()  # type: ignore
            dependencies = dspy.InputField()  # type: ignore
            dependency_signatures = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            feedback = dspy.InputField()  # type: ignore
            code = dspy.OutputField(desc="ONLY the Python def block, no markdown")  # type: ignore

        class FileGenSignature(dspy.Signature):  # type: ignore
            """Generate the complete file content appropriate for the language.
            Output MUST be the exact file text with no markdown fences or commentary.
            Keep it professional, robust, and minimal.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            file = dspy.InputField()  # type: ignore
            language = dspy.InputField()  # type: ignore
            exports = dspy.InputField()  # type: ignore
            imports = dspy.InputField()  # type: ignore
            entrypoint = dspy.InputField()  # type: ignore
            functions = dspy.InputField()  # type: ignore
            text = dspy.OutputField(desc="Full file contents only")  # type: ignore

        class ExportVerifySignature(dspy.Signature):  # type: ignore
            """Verify that the provided file text contains the declared exports for the given language.
            Return ONLY one-line minified JSON: {"ok":bool,"missing":string[]}
            Do not include markdown or code fences.
            """
            file = dspy.InputField()  # type: ignore
            language = dspy.InputField()  # type: ignore
            exports = dspy.InputField()  # type: ignore
            text = dspy.InputField()  # type: ignore
            result = dspy.OutputField()  # type: ignore

        self._PlanSignature = PlanSignature
        self._TaskPlanSignature = TaskPlanSignature
        self._SplitDecisionSignature = SplitDecisionSignature
        self._FunctionImplSignature = FunctionImplSignature
        self._FileGenSignature = FileGenSignature
        self._ExportVerifySignature = ExportVerifySignature

        # Modules
        self._plan_predictor = dspy.Predict(PlanSignature)  # type: ignore
        self._task_plan_predictor = dspy.Predict(TaskPlanSignature)  # type: ignore
        self._split_predictor = dspy.Predict(SplitDecisionSignature)  # type: ignore
        self._func_predictor = dspy.Predict(FunctionImplSignature)  # type: ignore
        self._file_predictor = dspy.Predict(FileGenSignature)  # type: ignore
        self._export_verify_predictor = dspy.Predict(ExportVerifySignature)  # type: ignore

        def _unused():
            return None

    def _strip_markdown_fences(self, text: str) -> str:
        """Remove common markdown code fences around JSON/text, if present."""
        s = (text or "").strip()
        if not s.startswith("```"):
            return s
        parts = s.splitlines()
        if len(parts) >= 2 and parts[0].startswith("```") and parts[-1].startswith("```"):
            return "\n".join(parts[1:-1]).strip()
        # Fallback: strip backticks and take content after first newline
        s = s.strip().strip("`")
        return s.split("\n", 1)[-1].strip()

    def _normalize_split_decision(self, obj: Dict[str, Any], parent_task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure split/implement decision has expected shape and child defaults.
        - action: one of {split, implement} (default implement)
        - children: list (default [])
        - if parent has id, synthesize child ids if missing as <parent>:<n>
        - fill child defaults: kind, title, description, priority, deps, inputs, outputs
        """
        if not isinstance(obj, dict):
            return {"action": "implement", "children": []}
        action = obj.get("action")
        if action not in ("split", "implement"):
            action = "implement"
        children = obj.get("children") if isinstance(obj.get("children"), list) else []
        parent_id = parent_task.get("id")
        if action == "split" and children:
            new_children = []
            for idx, ch in enumerate(children, start=1):
                ch = dict(ch) if isinstance(ch, dict) else {"title": str(ch)}
                # synthesize id if parent has id and child lacks id
                if parent_id and not ch.get("id"):
                    ch["id"] = f"{parent_id}:{idx}"
                # fill required/minimal fields
                ch.setdefault("kind", ch.get("type") or "generic")
                title = ch.get("title") or f"Task {idx}"
                ch["title"] = title
                ch.setdefault("description", "")
                ch.setdefault("priority", "medium")
                ch.setdefault("deps", [])
                ch.setdefault("inputs", {})
                ch.setdefault("outputs", {})
                new_children.append(ch)
            obj["children"] = new_children
        else:
            obj["children"] = []
        obj["action"] = action
        return obj

    def _augment_constraints(self, constraints: Dict[str, Any] | None, mode: str) -> Dict[str, Any]:
        """Compose constraint rules from reusable blocks instead of hardcoded strings.
        - Uses default rule blocks per mode via self._MODE_DEFAULT_RULE_BLOCKS.
        - Allows callers to override with:
          - constraints.rule_blocks: exact list of block keys to apply (override defaults)
          - constraints.rule_blocks_add: additional block keys to add
          - constraints.rule_blocks_remove: block keys to remove
          - constraints.format_rules: extra literal rules appended at the end
        """
        c: Dict[str, Any] = dict(constraints or {})
        # Determine block set
        default_blocks: List[str] = list(self._MODE_DEFAULT_RULE_BLOCKS.get(mode, []))
        override_blocks = c.get("rule_blocks")
        if isinstance(override_blocks, list):
            blocks: List[str] = [str(x) for x in override_blocks]
        else:
            blocks = default_blocks
            add = c.get("rule_blocks_add", [])
            if isinstance(add, list):
                for b in add:
                    sb = str(b)
                    if sb not in blocks:
                        blocks.append(sb)
            remove = c.get("rule_blocks_remove", [])
            if isinstance(remove, list):
                to_remove: Set[str] = {str(x) for x in remove}
                blocks = [b for b in blocks if b not in to_remove]

        # Materialize rules from block registry
        rules: List[str] = []
        for b in blocks:
            for r in self._RULE_BLOCKS.get(b, []):
                if r not in rules:
                    rules.append(r)

        # Append any caller-provided literal rules last
        extra_rules = c.get("format_rules", [])
        if isinstance(extra_rules, list):
            for r in extra_rules:
                sr = str(r)
                if sr not in rules:
                    rules.append(sr)

        c["format_rules"] = rules
        # Optional transparency for debugging
        c["_applied_rule_blocks"] = blocks
        return c

    def plan(self, idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured plan JSON object via DSPy and parse it strictly.
        Includes a single retry if the first output lacks files.
        """
        def _predict_once(c: Dict[str, Any]) -> str:
            res = self._plan_predictor(idea=idea, constraints=c)  # type: ignore
            return self._strip_markdown_fences(getattr(res, "plan_json", ""))

        c1 = self._augment_constraints(constraints, "plan")
        content = _predict_once(c1)
        def _parse(s: str) -> Dict[str, Any]:
            try:
                return json.loads(s)
            except Exception:
                return json.loads(s.strip())

        obj = _parse(content)
        # Detect empty files across modules; if so, retry with stricter guidance
        files_count = 0
        try:
            for m in obj.get("modules", []):
                files_count += len(m.get("files", []) or [])
        except Exception:
            files_count = 0
        if files_count == 0:
            c2 = self._augment_constraints(constraints, "plan")
            rules = list(c2.get("format_rules", []))
            rules += [
                "Your previous output omitted files. Now include modules[].files with concrete file paths.",
                "For code files, include concrete functions and exports; for asset-only files (HTML/CSS/JSON/MD), functions and exports can be omitted.",
            ]
            c2["format_rules"] = rules
            retry_content = _predict_once(c2)
            obj = _parse(retry_content)
        return obj

    def task_plan(self, idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a hierarchical task plan JSON object via DSPy and parse it strictly."""
        result = self._task_plan_predictor(idea=idea, constraints=self._augment_constraints(constraints, "task_plan"))  # type: ignore
        content = self._strip_markdown_fences(getattr(result, "tasks_json", ""))
        # First, try strict parse of the raw content
        try:
            return json.loads(content)
        except Exception as e:
            # Already stripped fences; try once more with the same string
            cleaned = content
            try:
                return json.loads(cleaned)
            except Exception:
                # Single retry: ask the LM to reformat strictly to one-line minified JSON
                retry_constraints = self._augment_constraints(constraints, "task_plan")
                retry_rules = retry_constraints.get("format_rules", [])
                retry_rules = list(retry_rules) + [
                    "Your previous output contained formatting that is not valid JSON (e.g., raw newlines or bad escapes).",
                    "Now return the SAME content strictly as a single-line, syntactically valid JSON object with no markdown.",
                ]
                retry_constraints["format_rules"] = retry_rules
                retry = self._task_plan_predictor(idea=idea, constraints=retry_constraints)  # type: ignore
                retry_content = self._strip_markdown_fences(getattr(retry, "tasks_json", ""))
                return json.loads(retry_content)

    def decide_split(self, task: Dict[str, Any], idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to split or implement a task; return structured JSON decision.
        Adds guardrails to ensure empty composite tasks are split into actionable children.
        """
        # Augment the task with hints for the LLM
        t = dict(task)
        is_leaf_composite = (t.get("kind") == "composite") and (not t.get("children"))
        if is_leaf_composite:
            t["must_split"] = True

        def _predict_once(task_payload: Dict[str, Any]) -> Dict[str, Any]:
            result = self._split_predictor(
                task=task_payload, idea=idea, constraints=self._augment_constraints(constraints, "split")
            )  # type: ignore
            content = self._strip_markdown_fences(getattr(result, "decision_json", ""))
            try:
                return json.loads(content)
            except Exception:
                cleaned = content
                return json.loads(cleaned)

        # First attempt
        obj = self._normalize_split_decision(_predict_once(t), t)
        action = obj.get("action")
        children = obj.get("children", []) if isinstance(obj, dict) else []
        # If the model refused to split an empty composite, retry once with a stronger hint
        if is_leaf_composite and (action != "split" or not children):
            t2 = dict(t)
            t2["force_split"] = True
            t2["detail"] = "You must return a few meaningful children. Do not assume any specific programming language or implementation details unless present in constraints. Children should be minimal, independently actionable units for their kind."
            obj = self._normalize_split_decision(_predict_once(t2), t2)
        return obj

    def generate_function_impl(
        self,
        *,
        file: str,
        exports: list[str],
        all_functions: Dict[str, str],
        allowed_imports: list[str],
        target_function: str,
        signature: str,
        description: str,
        dependencies: list[str],
        dependency_signatures: Dict[str, str],
        constraints: Dict[str, Any],
        feedback: Any | None = None,
    ) -> str:
        """Generate a single Python function implementation; enforce strict postconditions."""
        result = self._func_predictor(
            file=file,
            exports=exports,
            all_functions=all_functions,
            allowed_imports=allowed_imports,
            target_function=target_function,
            func_signature=signature,
            description=description,
            dependencies=dependencies,
            dependency_signatures=dependency_signatures,
            constraints=constraints,
            feedback=feedback or "",
        )  # type: ignore
        code = (result.code or "").strip()
        if not code.startswith(f"def {target_function}"):
            raise RuntimeError("DSPy returned invalid function block")
        return code

    def generate_full_file(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        file: str,
        language: str,
        exports: list[str],
        imports: list[str],
        entrypoint: str | None,
        functions: Dict[str, str],
    ) -> str:
        """Generate a complete file content for non-Python files."""
        result = self._file_predictor(
            idea=idea,
            constraints=constraints,
            file=file,
            language=language,
            exports=exports,
            imports=imports,
            entrypoint=entrypoint or "",
            functions=functions,
        )  # type: ignore
        text = (result.text or "").strip()
        # Strip stray fences if any
        if text.startswith("```") and text.endswith("```"):
            # Heuristic: remove first and last fence lines
            parts = text.splitlines()
            if len(parts) >= 2:
                text = "\n".join(parts[1:-1]).strip()
        return text

    def verify_exports_in_text(
        self,
        *,
        file: str,
        language: str,
        exports: list[str],
        text: str,
    ) -> dict:
        """Use DSPy to verify declared exports exist in file text across languages.
        Returns a dict like {"ok": bool, "missing": [..]}
        """
        if not exports:
            return {"ok": True, "missing": []}
        result = self._export_verify_predictor(  # type: ignore
            file=file,
            language=language or "",
            exports=exports,
            text=text,
        )
        content = self._strip_markdown_fences(getattr(result, "result", "").strip())
        try:
            obj = json.loads(content)
            ok = bool(obj.get("ok", False))
            missing = obj.get("missing", [])
            if not isinstance(missing, list):
                missing = []
            missing = [str(x) for x in missing]
            return {"ok": ok, "missing": missing}
        except Exception:
            # If the verifier failed to return JSON, conservatively mark unknown as ok=False
            return {"ok": False, "missing": exports}
