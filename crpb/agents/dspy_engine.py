from __future__ import annotations
import os
import json
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FTimeout
from typing import Any, Dict, List, Set
try:
    # Prefer optional dependency; we will guard usage if missing
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover - optional
    repair_json = None  # type: ignore
from ..config import default_model_for
from ..llm_config import load_selection, LLMSelection, effective_model

logger = logging.getLogger(__name__)


class JSONValidationError(ValueError):
    """Raised when LLM output cannot be parsed into valid JSON even after repair."""


class DspyEngine:
    """
    DSPy-backed orchestration for planning and code/file generation.
    - Configures a single DSPy LM globally.
    - Uses typed Signatures to structure prompts.
    - Provides strict postconditions and parsing with clear errors.
    """

    def __init__(self, model: str | None = None) -> None:
        import dspy  # type: ignore
        # Configure LM once based on saved LLM selection (provider-neutral)
        self._dspy = dspy
        # Load persisted selection; require explicit selection (no implicit guessing to avoid hidden defaults)
        sel = load_selection()
        if not sel:
            raise RuntimeError(
                "No LLM selection found. Run `python -m crpb llm choose` to select a provider and model."
            )
        prov = sel.provider
        # Resolve effective model using provider-specific env defaults only
        mdl = effective_model(sel, override=model, default_model=default_model_for(prov))
        lm = None

        # Enforce model presence neutrally (no provider bias, no built-in defaults)
        if prov in ("server", "local", "huggingface", "openai", "anthropic") and not mdl:
            raise RuntimeError(
                "A model is required for the current configuration. Set an env default (CRPB_OPENAI_MODEL / CRPB_ANTHROPIC_MODEL / CRPB_HF_MODEL) or persist with `python -m crpb llm choose --model <id>`."
            )

        # LM behavior knobs (env-driven, no CLI defaults)
        def _env_int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
            try:
                v = int(os.environ.get(name, str(default)))
            except Exception:
                return default
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
            try:
                v = float(os.environ.get(name, str(default)))
            except Exception:
                return default
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        # Keep max tokens high by default to avoid truncation; users can reduce via env
        max_tokens = _env_int("CRPB_LM_MAX_TOKENS", default=32768, lo=256)
        # Temperature: no override unless env provides a non-empty value
        temperature_env: float | None = None
        _temp_raw = os.environ.get("CRPB_LM_TEMPERATURE")
        if _temp_raw is not None and str(_temp_raw).strip() != "":
            try:
                temperature_env = float(str(_temp_raw).strip())
                if temperature_env < 0.0:
                    temperature_env = 0.0
            except Exception:
                temperature_env = None  # Ignore invalid; use library default

        # Provider-aware kwargs to avoid invalid args (e.g., OpenAI rejects max_new_tokens)
        def _lm_kw(for_provider: str) -> dict[str, object]:
            kw: dict[str, object] = {"max_tokens": max_tokens}
            if for_provider in ("huggingface", "local"):
                kw["max_new_tokens"] = max_tokens
            if temperature_env is not None:
                kw["temperature"] = temperature_env
            return kw

        # Helper to try multiple constructor variants
        def _try_build(attempts):
            for fn in attempts:
                try:
                    obj = fn()
                    if obj is not None:
                        return obj
                except TypeError:
                    continue
                except Exception as e:
                    logger.debug("LM init attempt failed: %s", str(e))
                    continue
            return None

        # Credentials by provider
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        # Optional provider-specific keys (not required by default)
        server_key = os.environ.get("CRPB_SERVER_API_KEY") or os.environ.get("SERVER_API_KEY")
        base_url = getattr(sel, "base_url", None)

        # Additional required fields per provider
        if prov == "server" and not (isinstance(base_url, str) and base_url.strip()):
            raise RuntimeError(
                "Base URL is required for provider='server'. Set it via `python -m crpb llm choose --provider server --base-url http://localhost:11434 --model <name>`."
            )

        # Resolve DSPy entry points
        try:
            OpenAICls = getattr(dspy, "OpenAI")
        except Exception:
            OpenAICls = None  # type: ignore[assignment]
        try:
            OpenAIChatCls = getattr(dspy, "OpenAIChat")
        except Exception:
            OpenAIChatCls = None  # type: ignore[assignment]
        try:
            LMCls = getattr(dspy, "LM")
        except Exception:
            LMCls = None  # type: ignore[assignment]

        # Build for selected provider
        if prov == "openai":
            if not openai_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            attempts = []
            if OpenAICls is not None:
                attempts += [
                    lambda: OpenAICls(model=mdl, api_key=openai_key, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: OpenAICls(model=mdl, api_key=openai_key),  # type: ignore[misc]
                ]
            if OpenAIChatCls is not None:
                attempts += [
                    lambda: OpenAIChatCls(model=mdl, api_key=openai_key, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: OpenAIChatCls(model=mdl, api_key=openai_key),  # type: ignore[misc]
                ]
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(provider="openai", model=mdl, api_key=openai_key, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=openai_key, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=openai_key),  # type: ignore[misc]
                ]
            lm = _try_build(attempts)
        elif prov == "anthropic":
            if not anthropic_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            attempts = []
            # Try via generic LM first for widest compatibility
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(provider="anthropic", model=mdl, api_key=anthropic_key, **_lm_kw("anthropic")),  # type: ignore[misc]
                    lambda: LMCls(f"anthropic/{mdl}", api_key=anthropic_key, **_lm_kw("anthropic")),  # type: ignore[misc]
                    lambda: LMCls(f"anthropic/{mdl}", api_key=anthropic_key),  # type: ignore[misc]
                ]
            # Some DSPy versions may expose Anthropic class
            try:
                AnthropicCls = getattr(dspy, "Anthropic")
                attempts = [
                    lambda: AnthropicCls(model=mdl, api_key=anthropic_key, **_lm_kw("anthropic")),  # type: ignore[misc]
                    lambda: AnthropicCls(model=mdl, api_key=anthropic_key),  # type: ignore[misc]
                ] + attempts
            except Exception:
                pass
            lm = _try_build(attempts)
        elif prov == "huggingface":
            # Token optional for some public models
            attempts = []
            if LMCls is not None:
                if hf_token:
                    attempts += [
                        lambda: LMCls(provider="huggingface", model=mdl, api_key=hf_token, **_lm_kw("huggingface")),  # type: ignore[misc]
                        lambda: LMCls(f"huggingface/{mdl}", api_key=hf_token, **_lm_kw("huggingface")),  # type: ignore[misc]
                        lambda: LMCls(f"huggingface/{mdl}", api_key=hf_token),  # type: ignore[misc]
                    ]
                attempts += [
                    lambda: LMCls(provider="huggingface", model=mdl, **_lm_kw("huggingface")),  # type: ignore[misc]
                    lambda: LMCls(f"huggingface/{mdl}", **_lm_kw("huggingface")),  # type: ignore[misc]
                ]
            lm = _try_build(attempts)
        elif prov == "server":
            # OpenAI-compatible server; often ignores api_key. Use base_url when supported.
            # Prefer server-specific keys if provided; otherwise fall back to OPENAI_API_KEY or 'NA'.
            key = server_key or openai_key or "NA"
            attempts = []
            if OpenAICls is not None:
                attempts += [
                    lambda: OpenAICls(model=mdl, api_key=key, base_url=base_url, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: OpenAICls(model=mdl, api_key=key, api_base=base_url, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: OpenAICls(model=mdl, api_key=key, base_url=base_url),  # type: ignore[misc]
                    lambda: OpenAICls(model=mdl, api_key=key, api_base=base_url),  # type: ignore[misc]
                ]
            if OpenAIChatCls is not None:
                attempts += [
                    lambda: OpenAIChatCls(model=mdl, api_key=key, base_url=base_url, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: OpenAIChatCls(model=mdl, api_key=key, api_base=base_url, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: OpenAIChatCls(model=mdl, api_key=key, base_url=base_url),  # type: ignore[misc]
                    lambda: OpenAIChatCls(model=mdl, api_key=key, api_base=base_url),  # type: ignore[misc]
                ]
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(provider="openai", model=mdl, api_key=key, base_url=base_url, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: LMCls(provider="openai", model=mdl, api_key=key, api_base=base_url, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=key, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=key),  # type: ignore[misc]
                ]
            lm = _try_build(attempts)
        elif prov == "local":
            # Try a generic local provider.
            attempts = []
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(provider="local", model=mdl, **_lm_kw("local")),  # type: ignore[misc]
                    lambda: LMCls(f"local/{mdl}", **_lm_kw("local")),  # type: ignore[misc]
                ]
            lm = _try_build(attempts)
        if lm is None:
            raise RuntimeError(
                "Unable to initialize the language model for the chosen provider. Check provider, model, and environment configuration."
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
            ],
            "format_strict_json_no_shorten": [
                "ABSOLUTE: Return ONLY minified JSON on exactly one line (no raw newlines).",
                "ABSOLUTE: Use double quotes for all keys and strings. Never single quotes.",
                "ABSOLUTE: No trailing commas, no comments, no markdown, no backticks/code fences.",
                "Use only valid JSON escapes: \\, \", \/, \\b, \\f, \\n, \\r, \\t, and \\uXXXX.",
                "NEVER use backslash to escape a single quote (\\'). If a single quote appears, include it raw as '.",
                "Do NOT include standalone code snippets (e.g., if __name__ == '__main__':).",
                "Represent any code only inside designated fields (e.g., signature, file_contract).",
                "Do NOT shorten descriptions or purposes for file entries; multi-sentence paragraphs are acceptable inside JSON strings while keeping the overall output minified on one line.",
            ],
            "interface_first": [
                "Parents own language-neutral interface contracts at boundaries (use OpenAPI, JSON Schema, or equivalent).",
                "If a task designs or defines an interface, represent it as artifacts under outputs.produces with stable ids.",
                "Do NOT embed full code in interface tasks; keep contracts concise and language-agnostic.",
            ],
            "artifact_wiring": [
                "Wire tasks explicitly via artifacts: each task should declare inputs.consumes and outputs.produces when applicable.",
                "Use stable artifact ids in consumes/produces; include path hints when known. Children must consume parent-produced artifacts instead of assuming implicit context.",
                "Do NOT rely on implicit defaults; all cross-task dependencies must be expressed through artifacts or deps.",
            ],
            "boundary_language_neutral": [
                "Keep cross-language boundaries neutral: describe contracts with schemas/specs, not implementation language details.",
                "Never assume Python as a default at boundaries; choose languages explicitly only inside implementation tasks.",
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
            "planning_elaboration": [
                "When scope is non-trivial, prefer a multi-file structure that separates concerns (e.g., interface/UI/layout, behavior/logic, data/persistence/configuration, testing/docs).",
                "Include style/behavior/config/schema/test/docs files when applicable to the domain rather than compressing everything into a single file.",
                "For code files, prefer non-empty function specifications with clear responsibilities and exports; avoid placeholder-only plans.",
            ],
            "no_priority_fields": [
                "Do NOT include any 'priority' fields for modules, files, or tasks; treat all items as equally important unless constraints explicitly require prioritization.",
            ],
            "task_plan_general": [
                "Honor the user's idea literally; scope tasks to what was asked.",
                "Think like a developer: tasks should map to deliverables/responsibilities (e.g., planning, API, UI, data, tests, docs, deployment).",
                "Prefer fewer, well-scoped tasks; split only when it improves clarity, parallelism, or risk management.",
                "Descriptions MUST be detailed, multi-sentence and explicitly cover: WHY (goal/impact, rationale), WHAT (deliverables and responsibilities), and HOW (approach, sequencing, acceptance/verification at a high level). Avoid one-liners for any non-trivial scope.",
                "Leaf tasks must be minimal, independently actionable units; avoid unnecessary leaves.",
                "Include cross-cutting tasks where relevant (validation, error handling, logging, performance, security, privacy, observability).",
            ],
            "no_language_defaults": [
                "Do not assume any programming language or tool unless constraints explicitly require it.",
                "During task planning and splitting, avoid specifying programming languages; defer language/tool selection to implementation-time context or introduce a neutral decision task when selection is a prerequisite.",
                "If constraints pin a language or a path with an obvious extension, you may include it; otherwise omit language fields entirely. If a choice is needed to proceed, add a neutral decision task and describe decision criteria without naming technologies.",
            ],
            "code_function_contract": [
                "For code:function leaves, ensure inputs include: name, path, signature, allowed_imports (list), exports (list), and entrypoint when relevant.",
                "When applicable, also declare inputs.consumes and outputs.produces for artifacts (ids preferred).",
            ],
            "split_general": [
                "Honor the user's idea and parent task literally.",
                "If splitting, return a few meaningful children with clear, non-overlapping deliverables.",
                "Prefer partitioning by deliverable/role/phase; avoid redundant children.",
                "Select 'implement' only if the task is already a minimal actionable leaf (all-in-one acceptable when justified).",
                "Avoid language/tool specifics in children. For code-like leaves, include only general intent and optional path/name; omit language, signatures, and imports/exports unless constraints demand them.",
                "If a technology or framework selection is a prerequisite to progress and not specified by constraints, introduce a neutral decision child that defines decision criteria and acceptance checks without naming any specific technologies.",
                "Composite planning/coordination parents must split and should not directly implement final outputs themselves.",
            ],
            "split_guardrails": [
                "No fixed depth: decide split vs implement locally. Split only if the task is not yet a smallest independently actionable unit.",
                "Prevent overestimation: do not keep broad scope at the parent; introduce a few explicit, non-overlapping children when warranted.",
                "Prevent underestimation: stop splitting once scope is minimal and actionable; avoid deep chains without reducing scope.",
                "Avoid shallow 'one parent -> many leaves' when subareas remain broad: introduce sub-composites as needed. Tree depth is allowed; go to the necessary lowest point.",
                "Anti-infinite recursion: do not re-split a node unless new information changes scope; detect repeated no-op splits and choose implement.",
                "Return only a few meaningful children per split; avoid runaway proliferation.",
            ],
            "clarify_general": [
                "Clarify the given task using only the provided context (idea, constraints, parent, siblings, artifacts, files).",
                "Do NOT add children here; this is not a split step. Return the same task with refined fields.",
                "Preserve the task's id and general intent; fill in missing inputs/outputs/contracts when helpful.",
                "Prefer artifact-centric wiring: specify inputs.consumes and outputs.produces with stable ids/paths if derivable from context.",
                "Avoid language/tool specifics unless constraints demand them or paths/extensions make it obvious.",
                "Strengthen the task description to explicitly articulate WHY (purpose and impact), WHAT (deliverables, boundaries, responsibilities), and HOW (intended approach, sequencing, and high-level verification), using multiple sentences while remaining concise and neutral.",
                "If information is missing or gated on an external choice, record it under node_plan.open_questions with neutral decision criteria; do not invent specifics or name technologies.",
                "Return ONLY single-line minified JSON with the updated task object under key 'task'.",
            ],
            "merge_general": [
                "Merge results from children into coherent artifacts or files as needed by the parent task.",
                "Return a plan of concrete writes as an array 'writes': [{\"path\":string,\"text\":string}] for any files to (re)write in full.",
                "You may also return 'artifacts' (array of ArtifactRef objects) to register as merged outputs.",
                "Do NOT include markdown or code fences; return only one-line minified JSON.",
                "Avoid language defaults; base content and structure on the provided files/artifacts/context.",
            ],
            "plan_quality_check": [
                "Assess whether the plan separates concerns appropriately into modules/files given the idea and constraints.",
                "Flag issues like: trivial single-file when scope is non-trivial, missing style/logic/config/tests where applicable, placeholder functions with no real responsibilities, unclear exports, or missing entrypoints.",
                "Detect plan pathologies: over-splitting (too many tiny or redundant files/modules), under-splitting (monolithic file despite non-trivial scope), duplicate or overlapping modules/files, and repeated no-op elaborations.",
                "Return a compact JSON with ok:boolean, issues:string[], suggestions:string[]; no markdown.",
            ],
            "project_validate_general": [
                "Validate the project holistically using provided plan, file specs, and file texts; remain language-agnostic.",
                "Check for cross-file consistency: declared exports vs. actual contents, unresolved or circular dependencies, duplicate or overlapping responsibilities, and missing entrypoints when required by constraints.",
                "Identify structural smells: over/under-splitting at the file/module level, dead files (never referenced), and missing cross-cutting assets likely in-scope (tests/docs/config) without assuming a particular language.",
                "Integration coverage via artifacts (general, neutral): for each artifact id in plan_artifacts.index, flag (a) consumers with no producers, (b) producers with no consumers when downstream use is implied, and (c) inconsistent or ambiguous multi-producer cases unless a parent merge node is present.",
                "Detect unused exports across files/specs by semantic intent (language-neutral). Call out declared exports not referenced anywhere relevant.",
                "Check plan/spec mismatch: planned capabilities (from plan_view + plan_nodes) not exposed in surfaced files (e.g., UI lacking operations that plan describes), or implemented capabilities not represented in plan/spec (orphan features).",
                "Prefer succinct suggestions that are actionable; avoid verbose prose. Return ONLY minified JSON with keys: ok, issues, warnings, suggestions.",
            ],
            "plan_refine_general": [
                "Refine the current plan to address validation suggestions while remaining language-agnostic.",
                "Promote separation of concerns (e.g., interface/layout vs. logic vs. config/docs/tests) when scope warrants it.",
                "Avoid introducing priority fields; preserve or improve languages inferred by paths. Keep outputs as strict one-line minified JSON.",
            ],
            "task_plan_parent_discipline": [
                "Composite planning/coordination tasks should orchestrate and split work into children;",
                "they should avoid directly implementing final outputs. Leaves are the only implementable units.",
                "Children should be minimal, non-overlapping, and tied to explicit responsibilities; express cross-task wiring via artifacts or deps, not implicit context.",
            ],
            "neutrality_no_examples": [
                "Do not include examples, brand names, frameworks, libraries, tools, or technologies.",
                "Do not assume programming languages, platforms, or deployment environments.",
                "Describe responsibilities, contracts, and constraints in neutral terms only.",
                "Do not mention or imply specific stacks or architectures.",
                "Honor any explicit stack, language, or platform constraints provided by the user exactly as stated; otherwise avoid choosing or implying any specific technology.",
                "If a concrete technology choice is required to proceed, introduce an explicit neutral decision task that defines decision criteria and acceptance checks without naming technologies; defer the concrete selection to that task's outcome or future input.",
            ],
            "task_node_plan": [
                "Every task MUST include a node_plan object with these keys (keys must exist even if values are empty):",
                "intent, in_scope, out_of_scope, constraints, assumptions, deliverables, acceptance_criteria, preconditions, postconditions, interfaces, data_contracts, integration_points, dependencies, sequencing, risks, mitigations, open_questions, test_plan, verification, non_functional, step_outline, completion_definition, success_metrics, ownership, handoffs",
                "Write a detailed yet concise node_plan covering all aspects; ensure completeness rather than verbosity.",
                "Populate node_plan using user-provided details wherever applicable; do not fabricate specifics that were not provided or cannot be neutrally derived.",
                "If information is not provided and cannot be inferred neutrally, leave the field empty (key still present).",
                "Do not add code or technology details inside node_plan.",
            ],
            "socratic": [
                "Return ONLY minified JSON on one line with keys: questions (array of strings), monologue (string).",
                "Ask comprehensive, neutral questions to clarify scope and reduce ambiguity: WHY, WHAT, HOW, WHERE, WHO, WHEN, ASSUMPTIONS, CONSTRAINTS, INTERFACES, DATA/SCHEMAS, DEPENDENCIES, ACCEPTANCE_CRITERIA, TEST_PLAN, RISKS, NON_FUNCTIONAL, INTEGRATIONS, OPEN_QUESTIONS.",
                "Prefer short, sharp questions that drive specificity and independence of subtasks without naming technologies.",
                "Monologue should be a concise neutral reflection tying questions to the current node's role, siblings, and parent context; avoid prescribing technologies.",
                "Avoid language defaults and brand/tool names; stay domain-neutral.",
            ],
            "decomposition_policy": [
                "MANDATORY DECOMPOSITION: Tasks spanning multiple distinct concerns or responsibilities MUST decompose into children.",
                "Split when the task involves multiple separable responsibilities that benefit from specialization or can be approached independently.",
                "Stop splitting when the task is single-focused and represents something an LLM can implement effectively as one coherent unit.",
                "Parent (coordination) tasks organize and delegate to children; they do not directly implement final outputs.",
                "Prefer meaningful hierarchies that isolate concerns over flat structures for complex multi-faceted ideas.",
            ],
            "task_artifact_shape": [
                "inputs.consumes and outputs.produces, when present, are arrays of objects with at least an 'id' field. Additional fields are allowed.",
                "Do not include code or language choices in artifacts.",
            ],
            "task_validation_policy": [
                "After generation, validate and correct the plan before returning:",
                "- All tasks have non-empty title and a node_plan with all required keys (keys present even if values are empty).",
                "- No cycles in deps; all deps reference existing ids.",
                "- HIERARCHICAL STRUCTURE: Complex root tasks MUST have children. Avoid flat sibling structures.",
                "- Parents that coordinate have children; leaves have no children.",
                "- For multi-concern ideas, create parent tasks that decompose into specialized children focused on distinct responsibilities.",
                "- inputs/outputs, when present, follow the artifact shape rules.",
                "- No technology, framework, tool, language, or brand references anywhere.",
                "- Leaves are concretely actionable per their node_plan (step_outline and acceptance_criteria present).",
                "Provide actionable suggestions internally and fix issues before returning output.",
            ],
            "refinement_policy": [
                "When validation finds issues, propose minimal edits to fix them.",
                "Prefer to fill missing node_plan fields; split only if necessary to make a parent non-implementing or to isolate responsibilities.",
                "Keep ids stable; preserve task intent.",
            ],
            "sibling_integration": [
                "Children must communicate explicitly via artifacts and deps: when a child produces outputs needed by siblings, set outputs.produces with stable ids and have consumers declare inputs.consumes of those ids.",
                "Reflect ordering through deps: a consumer child must list the producing child id in deps.",
                "Prefer a parent-level merge/integration child when multiple outputs must be consolidated; avoid implicit integration inside unrelated children.",
                "Boundary-neutral interfaces across languages.",
            ],
        }

        # Default rule blocks per mode; callers can override via constraints
        self._MODE_DEFAULT_RULE_BLOCKS: Dict[str, List[str]] = {
            "plan": ["format_strict_json", "planning_general", "planning_elaboration", "no_priority_fields", "no_language_defaults"],
            "task_plan": [
                "format_strict_json",
                "task_plan_general",
                "no_priority_fields",
                "no_language_defaults",
                "interface_first",
                "artifact_wiring",
                "sibling_integration",
                "boundary_language_neutral",
                "task_plan_parent_discipline",
                "split_guardrails",
                "neutrality_no_examples",
                "task_node_plan",
                "decomposition_policy",
                "task_artifact_shape",
                "task_validation_policy",
            ],
            "split": [
                "format_strict_json",
                "split_general",
                "split_guardrails",
                "no_priority_fields",
                "no_language_defaults",
                "interface_first",
                "artifact_wiring",
                "sibling_integration",
                "boundary_language_neutral",
                "neutrality_no_examples",
                "decomposition_policy",
            ],
            "clarify": [
                "format_strict_json",
                "clarify_general",
                "artifact_wiring",
                "no_priority_fields",
                "no_language_defaults",
                "neutrality_no_examples",
                "task_node_plan",
            ],
            "merge": [
                "format_strict_json",
                "merge_general",
                "artifact_wiring",
                "sibling_integration",
                "boundary_language_neutral",
                "no_priority_fields",
                "no_language_defaults",
            ],
            "plan_validate": [
                "format_strict_json",
                "plan_quality_check",
                "planning_general",
                "planning_elaboration",
                "no_priority_fields",
                "no_language_defaults",
            ],
            "plan_refine": [
                "format_strict_json",
                "plan_refine_general",
                "planning_general",
                "planning_elaboration",
                "no_priority_fields",
                "no_language_defaults",
            ],
            "project_validate": [
                "format_strict_json",
                "project_validate_general",
                "artifact_wiring",
                "boundary_language_neutral",
                "no_language_defaults",
            ],
            "codespec_init": [
                "format_strict_json_no_shorten",
                "planning_general",
                "planning_elaboration",
                "no_language_defaults",
                "boundary_language_neutral",
            ],
            "codespec_enrich": [
                "format_strict_json",
                "boundary_language_neutral",
                "no_language_defaults",
            ],
            "task_plan_validate": [
                "format_strict_json",
                "task_validation_policy",
                "neutrality_no_examples",
            ],
        }

        # Define Signatures lazily to avoid top-level import-time failures when dspy missing
        class PlanSignature(dspy.Signature):  # type: ignore
            """Return ONLY minified JSON of a concrete software plan.
            Schema (top-level): {"modules": ModuleSpec[]}

            ModuleSpec fields:
            - name: string
            - purpose: string (short; optional)
            - deps: string[] (module-level deps; may be empty)
            - files: CodeSpecFile[] (REQUIRED; non-empty for at least one module)

            CodeSpecFile fields (developer-grade detail):
            - path: string (posix-like, e.g., "src/app/app.tsx" or "web/index.html"); REQUIRED
            - language?: string (e.g., "python", "typescript", "html", "css", "json"); may be omitted if inferable from path. If extension is missing/ambiguous, EXPLICITLY set "language"; do not default to any language.
            - purpose: string (brief purpose of this file)
            - description: string (detailed description)
            - functions?: { name: signature } (for code files; name -> signature string mapping)
            - classes?: { name: { methods: string[], description?: string } } (class definitions)
            - constants?: { name: { value: any, description?: string } } (constants/config)
            - exports?: string[] (names of top-level items to expose)
            - imports?: string[] (allowed imports for validators; may be empty)
            - entrypoint?: string (name of function to run; optional; code files only)
            - content?: string (pre-written content for non-code files like README.md)
            - deps?: string[] (other functions in same file this function depends on)
            - tests?: any[] (may be empty)

            Planning intent (general, rigorous, multi-language):
            - Honor the user's idea and constraints literally; do not invent unrelated metadata.
            - Think like a senior architect: partition by responsibility into cohesive modules/files with complete project coverage.
            - Choose appropriate languages per file/module (e.g., HTML/CSS/JS for UI; TS/Go/Python/etc. for services). Multi-language stacks are common and allowed. If the file extension does not make language obvious, explicitly set CodeSpecFile.language; NEVER assume a default (including Python).
            - Cover ALL necessary components for a complete, production-ready solution: frontend, backend, configuration, validation, error handling, logging, testing, documentation, deployment, packaging, entrypoints.
            - Include detailed function signatures with proper parameter types, return types, and comprehensive descriptions.
            - Ensure each file has complete imports, exports, classes, constants, and implementation details.
            - Always include comprehensive file coverage - never generate incomplete project structures.
            - Do NOT include any priority/importance fields; treat all modules/files as equally important unless constraints explicitly require prioritization.
            - Functions must have detailed signatures: name(param1: type, param2: type) -> return_type, not empty parentheses.

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

        class PlanValidateSignature(dspy.Signature):  # type: ignore
            """Return ONLY minified JSON with plan validation.
            Schema: {"ok": boolean, "issues": string[], "suggestions": string[]}

            Use the provided plan to assess separation of concerns, presence of appropriate files,
            clarity of function exports/entrypoints, and avoidance of placeholders.
            Output strictly one-line minified JSON with keys: ok, issues, suggestions.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            plan = dspy.InputField()  # type: ignore
            result_json = dspy.OutputField()  # type: ignore

        class PlanRefineSignature(dspy.Signature):  # type: ignore
            """Return ONLY minified JSON for an improved plan.
            Input: current_plan plus idea/constraints. Address validation suggestions without hardcoding domains.
            Output: same schema as PlanSignature (top-level {"modules": ModuleSpec[]}).
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            current_plan = dspy.InputField()  # type: ignore
            plan_json = dspy.OutputField()  # type: ignore

        class TaskPlanSignature(dspy.Signature):  # type: ignore
            """Return ONLY minified JSON for a hierarchical task plan on exactly one line:
            {"tasks": Task[]}

            Task:
            - id?: string (stable if provided)
            - kind: string (free-form label)
            - title: string (specific and concise)
            - description: string (short)
            - deps: string[] (may be empty)
            - inputs?: object (free-form; may include "consumes": object[])
            - outputs?: object (free-form; may include "produces": object[])
            - node_plan: object (REQUIRED; keys: intent, in_scope, out_of_scope, constraints, assumptions, deliverables, acceptance_criteria, preconditions, postconditions, interfaces, data_contracts, integration_points, dependencies, sequencing, risks, mitigations, open_questions, test_plan, verification, non_functional, step_outline, completion_definition, success_metrics, ownership, handoffs)
            - children?: Task[] (present only for non-leaf tasks)

            Task decomposition policy (CRITICAL - ENFORCE HIERARCHICAL STRUCTURE):
            - HIERARCHICAL DECOMPOSITION: Complex tasks MUST decompose into children. A parent task coordinates and delegates; it does NOT implement directly.
            - PARENT RESPONSIBILITY: Parents with scope spanning multiple distinct concerns or responsibilities MUST split into specialized children.
            - CHILDREN PLACEMENT: Children go in the "children" array of their parent, NOT as separate root tasks.
            - ATOMIC LEAF PRINCIPLE: Only leaves (tasks with no children) are implementable units. Parents orchestrate.
            - COMPLETE COVERAGE: Ensure all aspects are covered by children when decomposing into distinct responsibilities.
            - INTERFACE CLARITY: Parents define what children must produce/consume via inputs/outputs artifacts.
            - DEPENDENCY WIRING: Use deps for sequencing; use inputs.consumes/outputs.produces for data flow between siblings.
            - NO FLAT SIBLINGS: Avoid creating multiple root tasks when they should be children of a coordinating parent.

            Generation rules:
            - Be neutral and example-free. Do NOT mention technologies, frameworks, brands, or programming languages.
            - For complex ideas, create 1-2 root tasks that decompose into children. Avoid many root siblings.
            - Parents coordinate and define interfaces; children implement specific responsibilities.
            - Each task MUST include node_plan with ALL listed keys. Keys may have empty strings/arrays/objects but must be present.
            - Optional artifacts (inputs.consumes / outputs.produces) follow shape: arrays of objects with at least "id".
            - After generating tasks, self-validate: ensure node_plan completeness, valid deps (no cycles), parent-vs-leaf correctness, artifact shapes, and no technology leakage. If issues are found, fix them before returning.
{{ ... }}

            ABSOLUTE formatting:
            - Exactly one line of minified JSON. Double quotes only. No comments, no markdown, no code fences, no trailing commas, no raw newlines.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            tasks_json = dspy.OutputField()  # type: ignore

        class TaskPlanValidateSignature(dspy.Signature):  # type: ignore
            """Validate a hierarchical task plan (tasks only; no files).
            Return ONLY one-line minified JSON: {"ok":bool,"issues":string[],"suggestions":string[]}

            Checklist:
            - All tasks have non-empty title and node_plan keys present (even if values are empty).
            - Parent/leaf correctness: coordination tasks have children; leaves have none.
            - Deps are valid and acyclic; all ids referenced exist.
            - inputs/outputs, when present, follow artifact shape (arrays of objects with at least "id").
            - No technology/framework/language/brand references.
            - Leaves are actionable (node_plan.step_outline and acceptance_criteria exist).

            Suggestions MUST be actionable, e.g., "fill:<task_id>.node_plan.test_plan", "split_needed:<task_id>", "remove_cycle:<a>-><b>-><a>".
            Formatting: single-line minified JSON; no markdown.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            plan = dspy.InputField()  # type: ignore
            result_json = dspy.OutputField()  # type: ignore

        class SplitDecisionSignature(dspy.Signature):  # type: ignore
            """Decide whether the given task should be split further or kept as a leaf.
            Return ONLY one-line minified JSON of shape:
            {"action":"split"|"implement","children"?:Task[]}

            Policy:
            - Use local reasoning. Split only if it reduces ambiguity, clarifies responsibilities, or enables parallel work.
            - Keep as leaf when the task is smallest independently actionable and node_plan is complete.
            - If splitting, return a few meaningful non-overlapping children. Each child MUST include a complete node_plan (all keys present) and must not reference technologies or languages.
            - Maintain or infer sensible deps among siblings and with the parent when evident.

            Formatting: single-line minified JSON; no markdown or examples; no technology references.
            """
            task = dspy.InputField()  # type: ignore
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            decision_json = dspy.OutputField()  # type: ignore

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

        class ArtifactValidateSignature(dspy.Signature):  # type: ignore
            """Validate an artifact (any kind) using content and context.
            Return ONLY one-line minified JSON: {"ok":bool,"issues":string[],"warnings":string[],"normalized"?:object}
            Rules:
            - Do not include markdown or code fences.
            - If json_content is provided, use it as ground truth for JSON parsing instead of inferring from text.
            - Be strict but pragmatic; prefer structural correctness and contract completeness over style.
            """
            artifact_id = dspy.InputField()  # type: ignore
            kind = dspy.InputField()  # type: ignore
            path = dspy.InputField()  # type: ignore
            text = dspy.InputField()  # type: ignore
            json_content = dspy.InputField()  # type: ignore
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            result = dspy.OutputField()  # type: ignore

        class AmendPlanSignature(dspy.Signature):  # type: ignore
            """Amend a hierarchical task plan to resolve issues and improve completion.
            Return ONLY one-line minified JSON: {"edits": Edit[]}

            Allowed edits (non-destructive):
              - {"op":"add_child","parent_id":string,"task":Task}
              - {"op":"update_task","id":string,"set":object}  // e.g., set.node_plan.test_plan, set.description, set.outputs
              - {"op":"add_dep","id":string,"dep_id":string}
              - {"op":"rewire_artifacts","id":string,"consumes"?:object[],"produces"?:object[]}

            Constraints:
              - Keep ids stable. Preserve intent. Prefer filling missing node_plan fields before splitting.
              - No technology/framework/language mentions. No code.
              - Ensure postconditions: node_plan keys present on all tasks; no composite leaves; deps valid; artifact shapes respected.
              - Single-line minified JSON only; no markdown.
            """
            current_plan = dspy.InputField()  # type: ignore
            statuses = dspy.InputField()  # type: ignore
            artifacts = dspy.InputField()  # type: ignore
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            edits_json = dspy.OutputField()  # type: ignore

        class ClarifyTaskSignature(dspy.Signature):  # type: ignore
            """Refine a single Task without adding children.
            Return ONLY one-line minified JSON: {"task": Task}

            Rules:
            - Do not add or remove children. Preserve id and overall intent.
            - Fill or improve node_plan fields; all required node_plan keys must exist after clarification.
            - Keep content neutral and example-free; avoid naming technologies or languages.
            - You may add or normalize inputs/outputs fields and explicit deps if derivable from the provided context, adhering to artifact shape rules.
            - Single line, double quotes, no markdown, no trailing commas.
            """
            task = dspy.InputField()  # type: ignore
            parent = dspy.InputField()  # type: ignore
            siblings = dspy.InputField()  # type: ignore
            artifacts = dspy.InputField()  # type: ignore
            files = dspy.InputField()  # type: ignore
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            result_json = dspy.OutputField()  # type: ignore

        class MergeSubtasksSignature(dspy.Signature):  # type: ignore
            """Merge children's outputs for a parent task.
            Return ONLY one-line minified JSON with optional fields:
            {"writes": [{"path": string, "text": string}] , "artifacts": ArtifactRef[] }
            """
            parent = dspy.InputField()  # type: ignore
            children = dspy.InputField()  # type: ignore
            artifacts = dspy.InputField()  # type: ignore
            files = dspy.InputField()  # type: ignore
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            merge_json = dspy.OutputField()  # type: ignore

        class ProjectValidateSignature(dspy.Signature):  # type: ignore
            """Validate the project holistically.
            Return ONLY one-line minified JSON: {"ok":bool,"issues":string[],"warnings":string[],"suggestions":string[]}
            Inputs are language-neutral.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            plan = dspy.InputField()  # type: ignore
            files = dspy.InputField()  # type: ignore
            file_specs = dspy.InputField()  # type: ignore
            result = dspy.OutputField()  # type: ignore

        class CodeSpecInitSignature(dspy.Signature):  # type: ignore
            """Generate an initial Codespec (files[]) for the entire plan.
            Return ONLY one-line minified JSON: {"files": FileEntry[]}
            FileEntry shape (language-agnostic, may omit optional fields):
            {"path":string, "purpose":string, "description":string,
             "language"?:string, "imports"?:string[], "exports"?:string[],
             "functions"?:{name:string|{signature?:string,description?:string,parameters?:any[],returns?:any}},
             "classes"?:object, "constants"?:object, "entrypoint"?:string, "content"?:string}
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            plan_overview = dspy.InputField()  # type: ignore
            tasks_overview = dspy.InputField()  # type: ignore
            codespec_json = dspy.OutputField()  # type: ignore

        class CodeSpecEnrichSignature(dspy.Signature):  # type: ignore
            """Enrich a single Codespec file entry with missing functions/exports/etc.
            Return ONLY one-line minified JSON: a partial dict to merge into the file entry.
            """
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            file_entry = dspy.InputField()  # type: ignore
            result = dspy.OutputField()  # type: ignore

        class SocraticSignature(dspy.Signature):  # type: ignore
            """Produce clarifying questions and a reflective monologue for a task node.
            Return ONLY one-line minified JSON: {"questions": string[], "monologue": string}
            """
            node = dspy.InputField()  # type: ignore
            parent = dspy.InputField()  # type: ignore
            siblings = dspy.InputField()  # type: ignore
            idea = dspy.InputField()  # type: ignore
            constraints = dspy.InputField()  # type: ignore
            result_json = dspy.OutputField()  # type: ignore

        self._PlanSignature = PlanSignature
        self._TaskPlanSignature = TaskPlanSignature
        self._TaskPlanValidateSignature = TaskPlanValidateSignature
        self._SplitDecisionSignature = SplitDecisionSignature
        self._FileGenSignature = FileGenSignature
        self._ExportVerifySignature = ExportVerifySignature
        self._ArtifactValidateSignature = ArtifactValidateSignature
        self._AmendPlanSignature = AmendPlanSignature
        self._ClarifyTaskSignature = ClarifyTaskSignature
        self._MergeSubtasksSignature = MergeSubtasksSignature
        self._ProjectValidateSignature = ProjectValidateSignature
        self._CodeSpecInitSignature = CodeSpecInitSignature
        self._CodeSpecEnrichSignature = CodeSpecEnrichSignature
        self._SocraticSignature = SocraticSignature

        # Modules
        self._plan_predictor = dspy.Predict(PlanSignature)  # type: ignore
        self._task_plan_predictor = dspy.Predict(TaskPlanSignature)  # type: ignore
        self._task_plan_validate_predictor = dspy.Predict(TaskPlanValidateSignature)  # type: ignore
        self._split_predictor = dspy.Predict(SplitDecisionSignature)  # type: ignore
        self._file_predictor = dspy.Predict(FileGenSignature)  # type: ignore
        self._export_verify_predictor = dspy.Predict(ExportVerifySignature)  # type: ignore
        self._artifact_validate_predictor = dspy.Predict(ArtifactValidateSignature)  # type: ignore
        self._amend_plan_predictor = dspy.Predict(AmendPlanSignature)  # type: ignore
        self._plan_validate_predictor = dspy.Predict(PlanValidateSignature)  # type: ignore
        self._plan_refine_predictor = dspy.Predict(PlanRefineSignature)  # type: ignore
        self._clarify_predictor = dspy.Predict(ClarifyTaskSignature)  # type: ignore
        self._merge_predictor = dspy.Predict(MergeSubtasksSignature)  # type: ignore
        self._project_validate_predictor = dspy.Predict(ProjectValidateSignature)  # type: ignore
        self._codespec_init_predictor = dspy.Predict(CodeSpecInitSignature)  # type: ignore
        self._codespec_enrich_predictor = dspy.Predict(CodeSpecEnrichSignature)  # type: ignore
        self._socratic_predictor = dspy.Predict(SocraticSignature)  # type: ignore


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

    def _parse_json_dict_strict(self, data: str, *, context: str) -> Dict[str, Any]:
        """Parse a JSON object from LLM output strictly with repair-first strategy.

        Order:
        1) Strip markdown fences.
        2) json.loads
        2b) JSONDecoder.raw_decode (accepts leading object with trailing text)
        3) json_repair + json.loads
        On failure, raise JSONValidationError with context.
        """
        text = self._strip_markdown_fences(data)
        text = (text or "").strip()
        # Fast path
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Accept a leading JSON object followed by trailing text (common LM pattern)
        try:
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Repair path
        if repair_json is None:
            raise JSONValidationError(f"{context}: JSON parse failed and json-repair not available. Install 'json-repair'.")
        try:
            repaired = repair_json(text)  # type: ignore[misc]
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            raise JSONValidationError(f"{context}: JSON parse failed after repair: {e}")
        raise JSONValidationError(f"{context}: Expected a JSON object.")

    def _normalize_split_decision(self, obj: Dict[str, Any], parent_task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure split/implement decision has expected shape and child defaults.
        - action: one of {split, implement} (default implement)
        - children: list (default [])
        - if parent has id, synthesize child ids if missing as <parent>:<n>
        - fill child defaults: kind, title, description, deps, inputs, outputs
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
        # First attempt: strict parse with repair
        try:
            obj = self._parse_json_dict_strict(content, context="plan")
        except JSONValidationError:
            # Retry once with stricter JSON-only rules
            c_retry = self._augment_constraints(constraints, "plan")
            rules = list(c_retry.get("format_rules", [])) + [
                "Your previous output was not valid JSON. Return the SAME plan strictly as one-line minified JSON.",
                "No markdown or commentary; just the JSON object.",
            ]
            c_retry["format_rules"] = rules
            retry_content = _predict_once(c_retry)
            obj = self._parse_json_dict_strict(retry_content, context="plan:retry")
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
            obj = self._parse_json_dict_strict(retry_content, context="plan:files_retry")
        return obj

    def validate_plan(self, *, idea: str, constraints: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a plan structure for separation of concerns and minimal quality."""
        result = self._plan_validate_predictor(  # type: ignore
            idea=idea, constraints=self._augment_constraints(constraints, "plan_validate"), plan=plan
        )
        content = getattr(result, "result_json", "")
        try:
            obj = self._parse_json_dict_strict(content, context="validate_plan")
        except JSONValidationError:
            return {"ok": False, "issues": ["invalid_validator_output"], "suggestions": []}
        # Normalize fields
        ok = bool(obj.get("ok", False))
        issues = obj.get("issues", [])
        suggestions = obj.get("suggestions", [])
        if not isinstance(issues, list):
            issues = []
        if not isinstance(suggestions, list):
            suggestions = []
        return {"ok": ok, "issues": [str(x) for x in issues], "suggestions": [str(x) for x in suggestions]}

    def refine_plan(self, *, idea: str, constraints: Dict[str, Any], current_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Refine an existing plan into a more elaborate one."""
        result = self._plan_refine_predictor(  # type: ignore
            idea=idea, constraints=self._augment_constraints(constraints, "plan_refine"), current_plan=current_plan
        )
        content = getattr(result, "plan_json", "")
        try:
            return self._parse_json_dict_strict(content, context="refine_plan")
        except JSONValidationError:
            return current_plan

    def task_plan(self, idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a hierarchical task plan JSON object via DSPy and parse it strictly (repair-first).

        Lenient pre-parse: if the LM returns a top-level array, treat it as tasks => {"tasks": [...]}
        """
        result = self._task_plan_predictor(idea=idea, constraints=self._augment_constraints(constraints, "task_plan"))  # type: ignore
        content = getattr(result, "tasks_json", "")
        raw = self._strip_markdown_fences(content)
        # Lenient pre-parse
        try:
            q = json.loads(raw)
            if isinstance(q, dict):
                return q
            if isinstance(q, list):
                return {"tasks": q}
        except Exception:
            pass
        # Initial strict attempt with repair-first
        try:
            return self._parse_json_dict_strict(content, context="task_plan")
        except JSONValidationError as e:
            # Log raw preview for diagnostics
            try:
                preview = raw[:1000].replace("\n", "\\n")
            except Exception:
                preview = "<unavailable>"
            logger.debug("task_plan parse failed: %s; raw preview: %s", e, preview)
            # Single retry: ask the LM to reformat strictly to one-line minified JSON
            retry_constraints = self._augment_constraints(constraints, "task_plan")
            retry_rules = list(retry_constraints.get("format_rules", [])) + [
                "Your previous output was not valid JSON. Return the SAME content strictly as a syntactically valid one-line JSON object.",
                "No markdown or commentary; just the JSON object.",
            ]
            retry_constraints["format_rules"] = retry_rules
            retry = self._task_plan_predictor(idea=idea, constraints=retry_constraints)  # type: ignore
            retry_content = getattr(retry, "tasks_json", "")
            retry_raw = self._strip_markdown_fences(retry_content)
            # Lenient pre-parse on retry
            try:
                q = json.loads(retry_raw)
                if isinstance(q, dict):
                    return q
                if isinstance(q, list):
                    return {"tasks": q}
            except Exception:
                pass
            # Final strict attempt (repair-first)
            try:
                return self._parse_json_dict_strict(retry_content, context="task_plan:retry")
            except JSONValidationError as e2:
                try:
                    preview2 = retry_raw[:1000].replace("\n", "\\n")
                except Exception:
                    preview2 = "<unavailable>"
                logger.debug("task_plan retry parse failed: %s; raw preview: %s", e2, preview2)
                raise

    def validate_task_plan(self, *, idea: str, constraints: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a hierarchical task plan for node_plan completeness, parent/leaf correctness, deps, and neutrality."""
        result = self._task_plan_validate_predictor(  # type: ignore
            idea=idea,
            constraints=self._augment_constraints(constraints, "task_plan_validate"),
            plan=plan,
        )
        content = getattr(result, "result_json", "")
        try:
            obj = self._parse_json_dict_strict(content, context="validate_task_plan")
        except JSONValidationError:
            return {"ok": False, "issues": ["invalid_validator_output"], "suggestions": []}
        if not isinstance(obj, dict):
            return {"ok": False, "issues": ["invalid_validator_output"], "suggestions": []}
        ok = bool(obj.get("ok", False))
        issues = obj.get("issues", [])
        suggestions = obj.get("suggestions", [])
        if not isinstance(issues, list):
            issues = []
        if not isinstance(suggestions, list):
            suggestions = []
        return {"ok": ok, "issues": [str(x) for x in issues], "suggestions": [str(x) for x in suggestions]}

    def _normalize_split_decision(self, obj: Dict[str, Any], task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a split/implement decision object.
        - Ensures 'action' is a lowercase string; defaults based on task hints when absent.
        - Ensures 'children' is a list of normalized dicts with neutral keys.
        No regex, no hardcoded language assumptions.
        """
        d = dict(obj or {})
        # Normalize action with guardrails
        action_raw = d.get("action")
        action = str(action_raw).strip().lower() if isinstance(action_raw, (str, int)) else ""
        must_split = bool(task_payload.get("must_split") or task_payload.get("force_split"))
        if not action:
            action = "split" if must_split else "implement"
        elif must_split and action != "split":
            action = "split"

        # Normalize children
        children = d.get("children", [])
        if not isinstance(children, list):
            children = []
        norm_children: list[dict] = []
        for c in children:
            if not isinstance(c, dict):
                continue
            cc = dict(c)
            # Neutral structure defaults
            cc.setdefault("kind", cc.get("type") or "leaf")
            cc.setdefault("title", "")
            cc.setdefault("description", "")
            if not isinstance(cc.get("deps"), list):
                cc["deps"] = []
            if not isinstance(cc.get("inputs"), dict):
                cc["inputs"] = {}
            if not isinstance(cc.get("outputs"), dict):
                cc["outputs"] = {}
            if not isinstance(cc.get("children"), list):
                cc["children"] = []
            norm_children.append(cc)

        return {"action": action, "children": norm_children}

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
            content = getattr(result, "decision_json", "")
            try:
                return self._parse_json_dict_strict(content, context="decide_split")
            except JSONValidationError as e:
                # Retry once with stronger instruction for strict JSON
                rc = self._augment_constraints(constraints, "split")
                rrules = list(rc.get("format_rules", [])) + [
                    "Return strictly valid JSON for the decision object with keys {action, children}" ,
                    "No markdown or commentary.",
                ]
                rc["format_rules"] = rrules
                r = self._split_predictor(task=task_payload, idea=idea, constraints=rc)  # type: ignore
                return self._parse_json_dict_strict(getattr(r, "decision_json", ""), context="decide_split:retry")

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
        # Additional rich metadata from CodeSpecFile
        purpose: str = "",
        description: str = "",
        classes: Dict[str, Dict[str, Any]] = None,
        constants: Dict[str, Dict[str, Any]] = None,
        content: str | None = None,
        # Robustness knobs (env-overridable)
        max_retries: int | None = None,
        timeout_s: float | None = None,
        backoff_base: float | None = None,
        backoff_max: float | None = None,
        jitter: bool | None = None,
    ) -> str:
        """Generate a complete file content using rich CodeSpecFile metadata."""
        # If content is provided (e.g., for markdown files), use it directly
        if content and content.strip():
            return content.strip()
        
        # Enrich constraints with additional metadata for better generation
        enriched_constraints = dict(constraints or {})
        enriched_constraints.setdefault("file_metadata", {})
        enriched_constraints["file_metadata"].update({
            "purpose": purpose or "",
            "description": description or "",
            "classes": classes or {},
            "constants": constants or {},
        })
        # Resolve robustness knobs with env defaults if not provided
        def _env_int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
            try:
                v = int(os.environ.get(name, str(default)))
            except Exception:
                return default
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
            try:
                v = float(os.environ.get(name, str(default)))
            except Exception:
                return default
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            s = str(raw).strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
            return default

        _max_retries = int(max_retries if max_retries is not None else _env_int("CRPB_GEN_MAX_RETRIES", 2, lo=0, hi=10))
        _timeout_s = float(timeout_s if timeout_s is not None else _env_float("CRPB_GEN_TIMEOUT_S", 120.0, lo=5.0))
        _backoff_base = float(backoff_base if backoff_base is not None else _env_float("CRPB_GEN_BACKOFF_BASE", 1.0, lo=0.1))
        _backoff_max = float(backoff_max if backoff_max is not None else _env_float("CRPB_GEN_BACKOFF_MAX", 30.0, lo=0.5))
        _jitter = bool(jitter if jitter is not None else _env_bool("CRPB_GEN_BACKOFF_JITTER", True))

        last_err: Exception | None = None

        def _predict_once() -> str:
            # Run the LLM call in a worker thread to enforce timeout
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    self._file_predictor,  # type: ignore
                    idea=idea,
                    constraints=enriched_constraints,
                    file=file,
                    language=language,
                    exports=exports,
                    imports=imports,
                    entrypoint=entrypoint or "",
                    functions=functions,
                )
                try:
                    result = fut.result(timeout=_timeout_s)
                except _FTimeout as te:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                    raise TimeoutError(f"LLM file generation timed out after {_timeout_s:.1f}s for file={file}") from te
                # Normal post-processing
                text = (getattr(result, "text", "") or "").strip()
                if text.startswith("```") and text.endswith("```"):
                    parts = text.splitlines()
                    if len(parts) >= 2:
                        text = "\n".join(parts[1:-1]).strip()
                return text

        for attempt in range(_max_retries + 1):
            try:
                text = _predict_once()
                return text
            except Exception as e:
                last_err = e
                if attempt >= _max_retries:
                    break
                # Exponential backoff with optional jitter
                delay = min(_backoff_base * (2 ** attempt), _backoff_max)
                if _jitter:
                    delay = random.uniform(0, delay)
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                continue

        # If we reached here, all attempts failed
        msg = f"LLM file generation failed after {_max_retries + 1} attempts for file={file}: {last_err}"
        logger.warning(msg)
        raise RuntimeError(msg) from last_err

    def verify_exports_in_text(
        self,
        *,
        file: str,
        language: str,
        exports: list[str],
        text: str,
    ) -> dict:
        """Verify declared exports exist in file text using an LLM-based, language-agnostic verifier.
        Returns a dict like {"ok": bool, "missing": [..]}.
        """
        if not exports:
            return {"ok": True, "missing": []}

        # LLM-based verification (provider-agnostic, applies equally to all languages)
        result = self._export_verify_predictor(  # type: ignore
            file=file,
            language=language or "",
            exports=exports,
            text=text,
        )
        content = getattr(result, "result", "")
        try:
            obj = self._parse_json_dict_strict(content, context="verify_exports")
        except JSONValidationError:
            # If the verifier failed to return JSON, conservatively mark unknown as ok=False
            return {"ok": False, "missing": exports}
        ok = bool(obj.get("ok", False))
        missing = obj.get("missing", [])
        if not isinstance(missing, list):
            missing = []
        missing = [str(x) for x in missing]
        return {"ok": ok, "missing": missing}

    def validate_artifact(
        self,
        *,
        artifact_id: str | None,
        kind: str | None,
        path: str | None,
        text: str,
        json_content: dict | None,
        idea: str,
        constraints: Dict[str, Any],
    ) -> dict:
        """Use DSPy to validate an artifact generically across kinds/languages."""
        result = self._artifact_validate_predictor(  # type: ignore
            artifact_id=artifact_id or "",
            kind=kind or "",
            path=path or "",
            text=text or "",
            json_content=json_content or {},
            idea=idea or "",
            constraints=constraints or {},
        )
        content = getattr(result, "result", "")
        try:
            obj = self._parse_json_dict_strict(content, context="validate_artifact")
        except JSONValidationError:
            return {"ok": False, "issues": ["invalid_validator_output"], "warnings": []}
        if not isinstance(obj, dict):
            return {"ok": False, "issues": ["invalid_validator_output"], "warnings": []}
        ok = bool(obj.get("ok", False))
        issues = obj.get("issues", [])
        warnings = obj.get("warnings", [])
        if not isinstance(issues, list):
            issues = []
        if not isinstance(warnings, list):
            warnings = []
        return {"ok": ok, "issues": issues, "warnings": warnings}

    def clarify_task(
        self,
        *,
        task: Dict[str, Any],
        parent: Dict[str, Any] | None,
        siblings: List[Dict[str, Any]],
        artifacts: List[Dict[str, Any]],
        files: Dict[str, str],
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Clarify a single task via DSPy and return the updated task object (not wrapped)."""
        try:
            result = self._clarify_predictor(  # type: ignore
                task=task or {},
                parent=parent or {},
                siblings=siblings or [],
                artifacts=artifacts or [],
                files=files or {},
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "clarify"),
            )
            content = getattr(result, "result_json", "")
            obj = self._parse_json_dict_strict(content, context="clarify_task")
            if isinstance(obj, dict):
                t = obj.get("task")
                if isinstance(t, dict):
                    return t
                # Fallback: some providers may return the task directly
                return obj
        except JSONValidationError:
            pass
        except Exception:
            pass
        # On failure, return the input to allow no-op behavior
        return dict(task or {})

    def merge_subtasks(
        self,
        *,
        parent: Dict[str, Any],
        children: List[Dict[str, Any]],
        artifacts: List[Dict[str, Any]],
        files: Dict[str, str],
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge children for a parent and return a plan with writes/artifacts."""
        try:
            result = self._merge_predictor(  # type: ignore
                parent=parent or {},
                children=children or [],
                artifacts=artifacts or [],
                files=files or {},
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "merge"),
            )
            content = getattr(result, "merge_json", "")
            obj = self._parse_json_dict_strict(content, context="merge_subtasks")
            # Normalize shapes
            if not isinstance(obj.get("writes"), list):
                obj["writes"] = []
            if not isinstance(obj.get("artifacts"), list):
                obj["artifacts"] = []
            return obj
        except JSONValidationError:
            return {"writes": [], "artifacts": []}
        except Exception:
            return {"writes": [], "artifacts": []}

    def amend_task_plan(
        self,
        *,
        current_plan: Dict[str, Any],
        statuses: Dict[str, Any],
        artifacts: List[Dict[str, Any]],
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Request minimal non-destructive edits to the current task plan."""
        try:
            result = self._amend_plan_predictor(  # type: ignore
                current_plan=current_plan or {},
                statuses=statuses or {},
                artifacts=artifacts or [],
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "task_plan_validate"),
            )
            content = getattr(result, "edits_json", "")
            obj = self._parse_json_dict_strict(content, context="amend_task_plan")
            if not isinstance(obj.get("edits"), list):
                obj["edits"] = []
            return obj
        except JSONValidationError:
            return {"edits": []}
        except Exception:
            return {"edits": []}

    def project_validate(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        plan: Dict[str, Any],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use DSPy to validate the project holistically across files/specs/plan.
        Returns a dict like {"ok": bool, "issues": [], "warnings": [], "suggestions": []}.
        """
        result = self._project_validate_predictor(  # type: ignore
            idea=idea or "",
            constraints=self._augment_constraints(constraints, "project_validate"),
            plan=plan or {},
            files=files or {},
            file_specs=file_specs or {},
        )
        content = getattr(result, "result", "")
        try:
            obj = self._parse_json_dict_strict(content, context="project_validate")
        except JSONValidationError:
            return {"ok": False, "issues": ["invalid_validator_output"], "warnings": [], "suggestions": []}
        if not isinstance(obj, dict):
            return {"ok": False, "issues": ["invalid_validator_output"], "warnings": [], "suggestions": []}
        ok = bool(obj.get("ok", False))
        issues = obj.get("issues", [])
        warnings = obj.get("warnings", [])
        suggestions = obj.get("suggestions", [])
        if not isinstance(issues, list):
            issues = []
        if not isinstance(warnings, list):
            warnings = []
        if not isinstance(suggestions, list):
            suggestions = []
        return {
            "ok": ok,
            "issues": [str(x) for x in issues],
            "warnings": [str(x) for x in warnings],
            "suggestions": [str(x) for x in suggestions],
        }

    def micro_adjust(
        self,
        *,
        file: str,
        language: str,
        text: str,
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LM for micro-adjustments (minor fixes) without changing semantics.
        Returns {"text": str, "notes": [..]} on success, or the original text on failure.
        """
        try:
            result = self._micro_adjust_predictor(  # type: ignore
                file=file,
                language=language or "",
                text=text or "",
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "micro_adjust"),
            )
            content = getattr(result, "result", "")
            obj = self._parse_json_dict_strict(content, context="micro_adjust")
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                if not isinstance(obj.get("notes"), list):
                    obj["notes"] = []
                return obj
        except Exception:
            pass
        return {"text": text, "notes": []}

    def enrich_codespec_entry(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        file_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LM to enrich a single codespec file entry with missing functions/exports/etc.
        Returns a partial dict of fields to merge into the entry.
        """
        try:
            result = self._codespec_enrich_predictor(  # type: ignore
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "codespec_enrich"),
                file_entry=file_entry or {},
            )
            content = getattr(result, "result", "")
            obj = self._parse_json_dict_strict(content, context="codespec_enrich")
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {}

    def socratic_interrogate(
        self,
        *,
        node: Dict[str, Any],
        parent: Dict[str, Any],
        siblings: List[Dict[str, Any]],
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask clarifying questions and produce a neutral monologue for a task node.
        Returns {"questions": string[], "monologue": string} on success or safe defaults on failure.
        """
        try:
            result = self._socratic_predictor(  # type: ignore
                node=node or {},
                parent=parent or {},
                siblings=siblings or [],
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "socratic"),
            )
            content = getattr(result, "result_json", "")
            obj = self._parse_json_dict_strict(content, context="socratic")
            questions = obj.get("questions") if isinstance(obj, dict) else []
            monologue = obj.get("monologue") if isinstance(obj, dict) else ""
            if not isinstance(questions, list):
                questions = []
            q2 = []
            for q in questions:
                if isinstance(q, str) and q.strip():
                    q2.append(q.strip())
            if not isinstance(monologue, str):
                monologue = ""
            return {"questions": q2, "monologue": monologue}
        except Exception:
            return {"questions": [], "monologue": ""}

    

    def _normalize_codespec_files(self, arr: Any) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                p = it.get("path")
                if not (isinstance(p, str) and p.strip()):
                    continue
                entry: Dict[str, Any] = {
                    "path": p.strip(),
                    "purpose": str(it.get("purpose") or ""),
                    "description": str(it.get("description") or ""),
                    "language": str(it.get("language") or "") or None,
                    "imports": [str(x).strip() for x in (it.get("imports") or []) if isinstance(x, str) and x.strip()],
                    "exports": [str(x).strip() for x in (it.get("exports") or []) if isinstance(x, str) and x.strip()],
                    "functions": {},
                    "classes": {},
                    "constants": {},
                    "entrypoint": (str(it.get("entrypoint")).strip() if isinstance(it.get("entrypoint"), str) and str(it.get("entrypoint")).strip() else None),
                    "content": (str(it.get("content")).strip() if isinstance(it.get("content"), str) and str(it.get("content")).strip() else None),
                    "exports_detail": [],
                }
                fn = it.get("functions")
                if isinstance(fn, dict):
                    safe_fn: Dict[str, Any] = {}
                    for k, v in fn.items():
                        if isinstance(k, str) and k.strip():
                            safe_fn[k.strip()] = v
                    entry["functions"] = safe_fn
                cl = it.get("classes")
                if isinstance(cl, dict):
                    entry["classes"] = cl
                co = it.get("constants")
                if isinstance(co, dict):
                    entry["constants"] = co
                files.append(entry)
        return files

    def generate_codespec_root(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        plan_overview: Dict[str, Any],
        tasks_overview: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate an initial Codespec for the whole plan (language-agnostic)."""
        result = self._codespec_init_predictor(  # type: ignore
            idea=idea or "",
            constraints=self._augment_constraints(constraints, "codespec_init"),
            plan_overview=plan_overview or {},
            tasks_overview=tasks_overview or [],
        )
        content = getattr(result, "codespec_json", "").strip()
        try:
            obj = self._parse_json_dict_strict(content, context="codespec_init")
        except JSONValidationError:
            obj = {"files": []}
        files = self._normalize_codespec_files((obj or {}).get("files") or [])
        return {"files": files}
