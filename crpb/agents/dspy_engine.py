from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FTimeout
from typing import Any, Dict, List, Set

try:
    # Prefer optional dependency; we will guard usage if missing
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover - optional
    repair_json = None  # type: ignore

from ..core.config import default_model_for
from ..core.llm_config import effective_model, load_selection
from ..core.strict_json import JSONValidationError

logger = logging.getLogger(__name__)

# Global, process-wide DSPy configuration guard to avoid re-configuring from different async tasks
_DSPY_CONFIGURED: bool = False
_DSPY_CONFIG_LOCK = threading.Lock()


def _configure_dspy_once(dspy_mod, *, lm) -> None:
    global _DSPY_CONFIGURED
    if _DSPY_CONFIGURED:
        return
    # Serialize configuration across concurrent activities
    with _DSPY_CONFIG_LOCK:
        if _DSPY_CONFIGURED:
            return
        # Configure DSPy once. Caching is controlled at LM construction time
        # (e.g., dspy.LM(..., cache=False) in DSPy 3.x).
        if hasattr(dspy_mod, "settings") and hasattr(dspy_mod.settings, "configure"):
            dspy_mod.settings.configure(lm=lm)
        elif hasattr(dspy_mod, "configure"):
            dspy_mod.configure(lm=lm)
        else:
            # Fall back silently if neither exists (unlikely)
            pass
        _DSPY_CONFIGURED = True


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
                "No LLM selection found. Set CRPB_LLM_PROVIDER in .env (recommended) or run `python -m crpb llm choose` to select a provider and model."
            )
        prov = sel.provider
        # Resolve effective model using provider-specific env defaults only
        mdl = effective_model(sel, override=model, default_model=default_model_for(prov))
        lm = None

        # Enforce model presence neutrally (no provider bias, no built-in defaults)
        if (
            prov
            in (
                "server",
                "local",
                "huggingface",
                "openai",
                "anthropic",
                "cerebras",
                "azure_foundry",
            )
            and not mdl
        ):
            raise RuntimeError(
                "A model is required for the current configuration. Set an env default (CRPB_OPENAI_MODEL / CRPB_ANTHROPIC_MODEL / CRPB_CEREBRAS_MODEL / CRPB_HF_MODEL / CRPB_AZURE_FOUNDRY_DEPLOYMENT) or persist with `python -m crpb llm choose --model <id>`."
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

        def _env_float(
            name: str, default: float, lo: float | None = None, hi: float | None = None
        ) -> float:
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
            # IMPORTANT: Disable DSPy LM caching here. For DSPy 3.x this is the
            # documented knob (dspy.LM(..., cache=False)).
            kw: dict[str, object] = {"max_tokens": max_tokens, "cache": False}
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
        cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        azure_foundry_key = os.environ.get("CRPB_AZURE_FOUNDRY_API_KEY")
        azure_foundry_endpoint = os.environ.get("CRPB_AZURE_FOUNDRY_ENDPOINT")
        azure_foundry_api_version = os.environ.get("CRPB_AZURE_FOUNDRY_API_VERSION")
        # Optional provider-specific keys (not required by default)
        server_key = os.environ.get("CRPB_SERVER_API_KEY") or os.environ.get("SERVER_API_KEY")
        base_url = getattr(sel, "base_url", None)

        # Additional required fields per provider
        if prov == "server" and not (isinstance(base_url, str) and base_url.strip()):
            raise RuntimeError(
                "Base URL is required for provider='server'. Set it via `python -m crpb llm choose --provider server --base-url http://localhost:11434 --model <name>`."
            )
        if prov == "azure_foundry":
            if not (isinstance(azure_foundry_endpoint, str) and azure_foundry_endpoint.strip()):
                raise RuntimeError(
                    "CRPB_AZURE_FOUNDRY_ENDPOINT not set (e.g., https://<resource>.cognitiveservices.azure.com)"
                )
            if not (isinstance(azure_foundry_key, str) and azure_foundry_key.strip()):
                raise RuntimeError("CRPB_AZURE_FOUNDRY_API_KEY not set")
            if not (isinstance(azure_foundry_api_version, str) and azure_foundry_api_version.strip()):
                raise RuntimeError(
                    "CRPB_AZURE_FOUNDRY_API_VERSION not set (e.g., 2024-05-01-preview)"
                )

        # Resolve DSPy entry points
        try:
            OpenAICls = dspy.OpenAI
        except Exception:
            OpenAICls = None  # type: ignore[assignment]
        try:
            OpenAIChatCls = dspy.OpenAIChat
        except Exception:
            OpenAIChatCls = None  # type: ignore[assignment]
        try:
            LMCls = dspy.LM
        except Exception:
            LMCls = None  # type: ignore[assignment]

        # Build for selected provider
        if prov == "openai":
            if not openai_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            attempts = []
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(
                        provider="openai",
                        model=mdl,
                        api_key=openai_key,
                        **_lm_kw("openai"),
                    ),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=openai_key, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=openai_key, cache=False),  # type: ignore[misc]
                ]
            # Fallbacks for older/alternate DSPy builds (cannot guarantee cache control)
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
            lm = _try_build(attempts)
        elif prov == "anthropic":
            if not anthropic_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            attempts = []
            # Try via generic LM first for widest compatibility
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(
                        provider="anthropic",
                        model=mdl,
                        api_key=anthropic_key,
                        **_lm_kw("anthropic"),
                    ),  # type: ignore[misc]
                    lambda: LMCls(f"anthropic/{mdl}", api_key=anthropic_key, **_lm_kw("anthropic")),  # type: ignore[misc]
                    lambda: LMCls(f"anthropic/{mdl}", api_key=anthropic_key, cache=False),  # type: ignore[misc]
                ]
            # Some DSPy versions may expose Anthropic class
            try:
                AnthropicCls = dspy.Anthropic
                attempts = [
                    lambda: AnthropicCls(model=mdl, api_key=anthropic_key, **_lm_kw("anthropic")),  # type: ignore[misc]
                    lambda: AnthropicCls(model=mdl, api_key=anthropic_key),  # type: ignore[misc]
                ] + attempts
            except Exception:
                pass
            lm = _try_build(attempts)
        elif prov == "cerebras":
            if not cerebras_key:
                raise RuntimeError("CEREBRAS_API_KEY not set")
            attempts = []
            # Cerebras uses OpenAI-compatible API with custom base URL
            model_str = str(mdl or "").strip()
            if model_str.startswith("openai/"):
                model_no_prefix = model_str.split("/", 1)[1]
            else:
                model_no_prefix = model_str
            model_with_prefix = (
                model_str if model_str.startswith("openai/") else f"openai/{model_no_prefix}"
            )

            # IMPORTANT: Prefer a model string with provider prefix so LiteLLM can infer provider.
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(
                        model_with_prefix,
                        api_key=cerebras_key,
                        api_base="https://api.cerebras.ai/v1",
                        **_lm_kw("openai"),
                    ),
                    lambda: LMCls(
                        model_with_prefix,
                        api_key=cerebras_key,
                        api_base="https://api.cerebras.ai/v1",
                        cache=False,
                    ),
                ]
            lm = _try_build(attempts)
        elif prov == "azure_foundry":
            # Azure AI Foundry OpenAI-compatible endpoint (deployment-based).
            key = str(azure_foundry_key or "").strip()
            api_base = str(azure_foundry_endpoint or "").strip()
            api_version = str(azure_foundry_api_version or "").strip()

            deployment = str(mdl or "").strip()
            model_with_prefix = deployment if deployment.startswith("azure/") else f"azure/{deployment}"

            attempts = []
            if LMCls is not None:
                # Try common LiteLLM Azure styles.
                attempts += [
                    lambda: LMCls(
                        model_with_prefix,
                        api_key=key,
                        api_base=api_base,
                        api_version=api_version,
                        **_lm_kw("openai"),
                    ),
                    lambda: LMCls(
                        provider="azure",
                        model=deployment,
                        api_key=key,
                        api_base=api_base,
                        api_version=api_version,
                        **_lm_kw("openai"),
                    ),
                    lambda: LMCls(
                        model_with_prefix,
                        api_key=key,
                        azure_endpoint=api_base,
                        api_version=api_version,
                        **_lm_kw("openai"),
                    ),
                    lambda: LMCls(
                        model_with_prefix,
                        api_key=key,
                        base_url=api_base,
                        api_version=api_version,
                        **_lm_kw("openai"),
                    ),
                    lambda: LMCls(
                        model_with_prefix,
                        api_key=key,
                        api_base=api_base,
                        api_version=api_version,
                        cache=False,
                    ),
                ]
            lm = _try_build(attempts)
        elif prov == "huggingface":
            # Token optional for some public models
            attempts = []
            if LMCls is not None:
                if hf_token:
                    attempts += [
                        lambda: LMCls(
                            provider="huggingface",
                            model=mdl,
                            api_key=hf_token,
                            **_lm_kw("huggingface"),
                        ),  # type: ignore[misc]
                        lambda: LMCls(
                            f"huggingface/{mdl}",
                            api_key=hf_token,
                            **_lm_kw("huggingface"),
                        ),  # type: ignore[misc]
                        lambda: LMCls(f"huggingface/{mdl}", api_key=hf_token, cache=False),  # type: ignore[misc]
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
            if LMCls is not None:
                attempts += [
                    lambda: LMCls(
                        provider="openai",
                        model=mdl,
                        api_key=key,
                        base_url=base_url,
                        **_lm_kw("openai"),
                    ),  # type: ignore[misc]
                    lambda: LMCls(
                        provider="openai",
                        model=mdl,
                        api_key=key,
                        api_base=base_url,
                        **_lm_kw("openai"),
                    ),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=key, **_lm_kw("openai")),  # type: ignore[misc]
                    lambda: LMCls(f"openai/{mdl}", api_key=key, cache=False),  # type: ignore[misc]
                ]
            # Fallbacks for older/alternate DSPy builds (cannot guarantee cache control)
            if OpenAICls is not None:
                attempts += [
                    lambda: OpenAICls(
                        model=mdl, api_key=key, base_url=base_url, **_lm_kw("openai")
                    ),  # type: ignore[misc]
                    lambda: OpenAICls(
                        model=mdl, api_key=key, api_base=base_url, **_lm_kw("openai")
                    ),  # type: ignore[misc]
                    lambda: OpenAICls(model=mdl, api_key=key, base_url=base_url),  # type: ignore[misc]
                    lambda: OpenAICls(model=mdl, api_key=key, api_base=base_url),  # type: ignore[misc]
                ]
            if OpenAIChatCls is not None:
                attempts += [
                    lambda: OpenAIChatCls(
                        model=mdl, api_key=key, base_url=base_url, **_lm_kw("openai")
                    ),  # type: ignore[misc]
                    lambda: OpenAIChatCls(
                        model=mdl, api_key=key, api_base=base_url, **_lm_kw("openai")
                    ),  # type: ignore[misc]
                    lambda: OpenAIChatCls(model=mdl, api_key=key, base_url=base_url),  # type: ignore[misc]
                    lambda: OpenAIChatCls(model=mdl, api_key=key, api_base=base_url),  # type: ignore[misc]
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

        # Hard requirement: DSPy caching must be disabled.
        # In DSPy 3.x this is controlled via `cache=False` on the LM instance.
        lm_cache = getattr(lm, "cache", None)
        if lm_cache is None:
            raise RuntimeError(
                "Unable to verify DSPy LM cache is disabled (no `cache` attribute found). "
                "CRPB requires a DSPy LM that exposes `cache` so we can ensure cache-off behavior. "
                "If you are using a non-LM DSPy backend, switch to `dspy.LM(..., cache=False)` or upgrade DSPy."
            )
        if lm_cache is not False:
            raise RuntimeError(
                "DSPy LM cache is enabled but must be disabled. Ensure the LM is constructed with `cache=False`."
            )
        _configure_dspy_once(dspy, lm=lm)

        # Reusable rule blocks for composing prompt constraints without hardcoding
        self._RULE_BLOCKS: Dict[str, List[str]] = {
            # NOTE: We intentionally avoid asking the model to return one giant JSON blob.
            # DSPy can return multiple output fields; we assemble JSON/dicts in Python.
            "call_contract": [
                "You are executing ONE specific operation. Use constraints.call.{name,purpose,inputs,outputs} to understand what this call is doing.",
                "Do not narrate or explain. Do not add extra keys. Populate ONLY the signature output fields.",
                "Do not refer to unspecified external sources (e.g., 'in some file', 'elsewhere', 'as mentioned above').",
                "If information is unknown, leave the field empty but still return the field.",
            ],
            "use_side_context": [
                "If constraints.side_context is present, treat it as authoritative context for this operation.",
                "Do not ask for more information. Do not request files, links, or external references.",
                "Prefer using concrete IDs/paths/artifact IDs that already appear in the inputs/side_context; do not invent new ones unless required by the signature output shape.",
            ],
            "format_structured_fields": [
                "Return outputs using the signature's declared output fields only.",
                "FORMAT: Return a single JSON object whose top-level keys are EXACTLY the signature output field names.",
                "Do NOT include markdown or code fences. Do NOT include any extra keys.",
                "For list/object fields, use proper JSON arrays/objects as the field value.",
            ],
            "format_structured_fields_no_shorten": [
                "Return outputs using the signature's declared output fields only.",
                "FORMAT: Return a single JSON object whose top-level keys are EXACTLY the signature output field names.",
                "Do NOT include markdown or code fences. Do NOT include any extra keys.",
                "Do NOT shorten descriptions or purposes; multi-sentence values are acceptable.",
                "For list/object fields, use proper JSON arrays/objects as the field value.",
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
            "no_cross_language_imports": [
                "LANGUAGE CONSISTENCY (CRITICAL): do not create or approve imports/requires/includes that cross programming-language boundaries.",
                "If a file's extension/language implies one language, it must not import a local file whose extension implies a different language (e.g., importing './x.py' from a '.js' file).",
                "If cross-component communication is needed across languages, use an explicit boundary contract (artifact id + interface/schema) and a runtime integration mechanism, not direct source-file imports.",
                "VALIDATION: if you see a local cross-language import in file text or file_specs.imports, emit a BLOCKING issue: cross_language_import:<from_path>:<ref>.",
                "REPAIR: fix cross-language imports by removing the invalid import and replacing it with a boundary-safe call pattern consistent with the plan/spec (e.g., contract-based API).",
            ],
            "jury_evidence_standard": [
                "JURY STANDARD (adversarial): start from 'guilty' and actively search for defects.",
                "Your job is to find the smallest set of HIGH-CONFIDENCE blocking issues that would prevent a correct run or violate the plan/spec/contracts.",
                "The project/node is 'proven ok' ONLY if, after an adversarial search, you cannot find any blocking issues using the provided evidence.",
                "Only emit a BLOCKING issue when there is clear, concrete evidence in: (a) file text, (b) file_specs, (c) plan content, or (d) run_logs evidence.",
                "Do NOT speculate about external dependencies or runtime behavior; reason strictly from provided artifacts.",
                "If something is plausible but unproven, emit a warning/suggestion (machine-coded) instead of a blocking issue.",
                "Prefer fewer, more certain issues over many weak ones.",
            ],
            "repair_must_act": [
                "REPAIR (CRITICAL): If there is at least ONE blocking issue in the provided issues list and the necessary target file text is present in `files` (or the file is explicitly missing and you can create it), you MUST propose at least one concrete change.",
                "If you believe no fix is possible from the provided inputs, return patches=[] and add a note starting with 'no_fix_possible:' explaining exactly what required evidence is missing (e.g., missing target file text, ambiguous contract, conflicting requirements).",
                "Prefer bundling multiple independent patches in ONE round when they touch different files or non-overlapping regions of the same file.",
                "Avoid oscillation: consult constraints.side_context.prior_findings and context.previous_round/repair_history; do NOT undo a prior improvement.",
            ],
            "codespec_quality": [
                "CODESPEC QUALITY: produce a clear, internally consistent file contract set derived from the plan/tasks.",
                "Every file entry MUST have a unique project-relative path using '/' separators (no absolute paths, no drive letters).",
                "Avoid duplicate responsibilities: do not create multiple files with the same purpose unless the plan requires separation.",
                "For each file: purpose/description must be non-empty, concrete, and aligned to specific plan deliverables.",
                "For code-like files: exports should correspond to described functions/classes/constants; avoid listing exports that are not described.",
                "Avoid placeholders (e.g., 'TBD', 'TODO', 'placeholder'). If uncertain, keep it minimal and ask for clarification in a separate process (not here).",
                "Remain language-neutral unless constraints or extensions require specifics.",
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
                "Return the clarified task via the signature output field.",
            ],
            "merge_general": [
                "Merge results from children into coherent artifacts or files as needed by the parent task.",
                "Return a plan of concrete writes via output field 'writes' as a list of {path,text} objects for any files to (re)write in full.",
                "You may also return merged artifacts via output field 'artifacts' as a list of ArtifactRef-like objects.",
                "Do NOT include markdown or code fences.",
                "Avoid language defaults; base content and structure on the provided files/artifacts/context.",
            ],
            "plan_quality_check": [
                "Assess whether the plan separates concerns appropriately into modules/files given the idea and constraints.",
                "Flag issues like: trivial single-file when scope is non-trivial, missing style/logic/config/tests where applicable, placeholder functions with no real responsibilities, unclear exports, or missing entrypoints.",
                "Detect plan pathologies: over-splitting (too many tiny or redundant files/modules), under-splitting (monolithic file despite non-trivial scope), duplicate or overlapping modules/files, and repeated no-op elaborations.",
                "Return ok, issues, suggestions via signature outputs; no markdown.",
            ],
            "project_validate_general": [
                "Validate the project holistically using provided plan, file specs, and file texts; remain language-agnostic.",
                "Check for cross-file consistency: declared exports vs. actual contents, unresolved or circular dependencies, duplicate or overlapping responsibilities, and missing entrypoints when required by constraints.",
                "Identify structural smells: over/under-splitting at the file/module level, dead files (never referenced), and missing cross-cutting assets likely in-scope (tests/docs/config) without assuming a particular language.",
                "Integration coverage via artifacts (general, neutral): for each artifact id in plan_artifacts.index, flag (a) consumers with no producers, (b) producers with no consumers when downstream use is implied, and (c) inconsistent or ambiguous multi-producer cases unless a parent merge node is present.",
                "Detect unused exports across files/specs by semantic intent (language-neutral). Call out declared exports not referenced anywhere relevant.",
                "Check plan/spec mismatch: planned capabilities (from plan_view + plan_nodes) not exposed in surfaced files (e.g., UI lacking operations that plan describes), or implemented capabilities not represented in plan/spec (orphan features).",
                "Connectivity axiom: if a file refers to another local file/module/resource (by path-like token), that target must exist. Prefer machine-coded issues like missing_html_ref, unresolved_local_import, missing_entrypoint_file.",
                "When constraints.side_context includes prior_findings or run_logs, use them as memory to avoid repeating the same items; focus on new contradictions and higher-severity failures.",
                "If constraints.side_context includes node_findings or node_findings_summary, treat them as bottom-up evidence from plan node validation; reconcile with the current file chunk and surface any cross-file contradictions.",
                "Apply the jury evidence standard: do not emit issues without concrete evidence from the provided inputs.",
                "REPAIRABILITY RULE: whenever an issue pertains to a specific file, you MUST include that file path as the second colon-delimited segment (code:<path>:...).",
                "If an issue involves a reference from one file to another, include both (code:<from_path>:<ref>).",
                "Avoid non-localizable global issues; if you truly cannot localize, use a stable placeholder path like logs/events.jsonl or plan/codespec.json if applicable.",
                "Output FORMAT IS STRICT: every item in issues/warnings/suggestions MUST be a single machine-coded string, not a sentence.",
                "Use only this shape: code[:path[:ref[:extra]]].",
                "- code: snake_case, stable, no spaces.",
                "- path: project-relative file path when applicable (use '/' separators).",
                "- ref: referenced path/token/symbol when applicable (avoid ':' inside ref when possible).",
                "- extra: optional tail for additional detail; keep it short.",
                "Do NOT output natural-language issues like 'Missing import ...' or full sentences.",
                "Prefer succinct, actionable suggestions; keep them machine-coded too. Return ok, issues, warnings, suggestions via signature outputs.",
            ],
            "node_validate_general": [
                "Validate a single plan node bottom-up using ONLY the provided node neighborhood and the provided file texts.",
                "Inputs are language-neutral. Do not assume any toolchain; reason from text only.",
                "Use node/parent/siblings/children to check boundary correctness and responsibility split.",
                "Use files/file_specs to check that this node's intended deliverables are actually present and internally coherent.",
                "If constraints.side_context.run_logs is present, treat it as evidence: reconcile recent_errors with the files under review and emit issues when there is a concrete mismatch.",
                "Apply the jury evidence standard: do not emit blocking issues without concrete evidence from the provided inputs.",
                "REPAIRABILITY RULE: whenever an issue pertains to a specific file, you MUST include that file path as the second colon-delimited segment (code:<path>:...).",
                "If an issue involves a reference from one file to another, include both (code:<from_path>:<ref>).",
                "Output FORMAT IS STRICT: every item in issues/warnings/suggestions MUST be a single machine-coded string, not a sentence.",
                "Use only this shape: code[:path[:ref[:extra]]].",
                "- code: snake_case, stable, no spaces.",
                "- path: project-relative file path when applicable (use '/' separators).",
                "- ref: referenced path/token/symbol when applicable (avoid ':' inside ref when possible).",
                "- extra: optional tail for additional detail; keep it short.",
                "Prefer codes that localize to a file when possible (e.g., syntax_error:<path>:indentation_break, unresolved_local_import:<from>:<ref>, missing_file:<path>).",
                "Return ok, issues, warnings, suggestions via signature outputs.",
            ],
            "validation_questions_general": [
                "Generate probing, actionable validation questions based on the current findings.",
                "Use constraints.side_context.prior_findings/prior_key_issues and constraints.side_context.run_logs to avoid redundant questions and to target the highest-risk gaps.",
                "Questions MUST be a flat list of objects with keys: id, question, focus, why_this_matters, expected_answer_shape.",
                "Generate 8-15 questions spanning: goal fit, missing capabilities, integration/entrypoints, contracts/exports, local references/links, error evidence from logs, testability, and ambiguity in plan vs implementation.",
                "Avoid technology names unless explicitly in constraints; stay language-neutral.",
                "Do not include markdown or code fences.",
            ],
            "project_validate_syntax": [
                "SYNTAX VALIDATION (language-agnostic, LLM-only): for EVERY provided file, assess whether its text is syntactically plausible for its file extension.",
                "Compute filename and extension separately by splitting the path at the last '/'+last '.'; do NOT guess extensions or assume a default language.",
                "If extension indicates a structured format (e.g., config/data/markup), check for obvious structural invalidity (unbalanced delimiters, unterminated strings, malformed nesting).",
                "If extension indicates a code-like file, check for obvious syntax errors (unbalanced braces/parens, broken indentation blocks, unterminated strings/comments, mismatched quote styles).",
                "If the file is empty when it should contain code/config per file_specs or plan intent, emit empty_file:<path>.",
                "Report syntax problems with codes like syntax_error:<path>:<kind> where kind is a short token (unbalanced_braces, unterminated_string, invalid_structure, indentation_break, etc.).",
                "Do not run or assume any compiler/interpreter; this is reasoning from text only.",
            ],
            "findings_summarize_general": [
                "Summarize validation findings for memory compression and deduplication.",
                "Input is a flat list of machine-coded findings plus optional prior_summary in constraints.side_context.",
                "Output summary MUST be short (<= 1200 chars) and strictly language-neutral.",
                "Also output key_issues as a de-duplicated list of at most 30 machine-coded issue strings (keep original codes; do not rewrite into prose).",
                "Do not introduce new issue codes that were not present in findings.",
            ],
            "project_validate_contracts": [
                "PRIMARY FOCUS (contracts): verify that each file satisfies its declared contract in file_specs (imports/exports/entrypoint/purpose).",
                "If file_specs[path].entrypoint is present, require that the project has a runnable entry (or a documented run contract) that uses it.",
                "If file_specs declares exports, verify that the file text contains those exported symbols in a manner consistent with its language (do not assume a language; infer from path extension if needed).",
                "Flag missing or mismatched contracts with codes like declared_export_missing:<path>:<symbol> or entrypoint_missing:<path>:<entry>.",
                "Apply the jury evidence standard: do not speculate; only flag missing *local* dependencies when there is concrete evidence in file paths/specs.",
            ],
            "project_validate_contrarian": [
                "PRIMARY FOCUS (contrarian): try to falsify correctness, but ONLY using concrete evidence from the provided inputs.",
                "Prefer reporting the smallest set of HIGH-CONFIDENCE blocking issues that explain why a user run would fail (entrypoints, missing local modules/assets, inconsistent paths, missing referenced files).",
                "No speculation; no explanations; machine-coded only.",
            ],
            "project_validate_integration": [
                "PRIMARY FOCUS (integration): validate the project as an executable system, not just a set of files.",
                "Definition: Build a directed reference graph G=(V,E) where V are files and E are declared local references (imports, script/src, stylesheet/href, markdown links, path-like tokens in configs).",
                "Axiom (closure): for every edge (u->v) in E where v is local, v MUST exist in the project. If not, emit missing_* codes.",
                "Axiom (entry): if there is an entrypoint in file_specs, there MUST be an integration path from a user-facing entry to that entrypoint (or a documented run contract).",
                "Report integration failures as machine-coded issues like missing_entrypoint_file:<path> or unresolved_local_import:<from>:<ref>.",
            ],
            "project_validate_completeness": [
                "PRIMARY FOCUS (completeness): validate that the plan's declared capabilities are realized by artifacts/files.",
                "Definition: Let C be the set of deliverables implied by plan nodes (node_plan.deliverables + acceptance_criteria).",
                "Only emit missing_capability when a deliverable is explicitly present in the plan/task node_plan and there is clear evidence it is not implemented in the provided files.",
                "If the project implements something not represented in plan/specs, emit orphan_feature:<path[:ref]>.",
                "Do not assume a particular language or framework; infer only from extensions and file_specs.",
            ],
            "project_validate_neutrality": [
                "PRIMARY FOCUS (neutrality): ensure language/tool neutrality and avoid hidden bias.",
                "If you detect that the project hardcodes a specific language/stack without an explicit user constraint, emit language_bias:<path[:ref]> .",
                "If code/doc mandates a specific toolchain absent from constraints, emit toolchain_assumption:<path[:ref]> .",
                "If the project claims to be language-agnostic but generates language-specific scaffolding without decision criteria, emit neutrality_contract_breach:<path[:ref]> .",
                "Stay machine-coded; no explanations.",
            ],
            "plan_refine_general": [
                "Refine the current plan to address validation suggestions while remaining language-agnostic.",
                "Promote separation of concerns (e.g., interface/layout vs. logic vs. config/docs/tests) when scope warrants it.",
                "Avoid introducing priority fields; preserve or improve languages inferred by paths.",
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
                "Return questions and monologue via signature outputs.",
                "Questions MUST be a flat list (no grouping) of objects with keys: id, question, focus, why_this_matters, expected_answer_shape.",
                "GENERATE AT LEAST 8-15 COMPREHENSIVE QUESTIONS covering ALL of these aspects:",
                "1. PURPOSE/WHY: What is the core goal? What problem does this solve? Why is this necessary?",
                "2. SCOPE/WHAT: What exactly should be delivered? What is in-scope vs out-of-scope? What are the boundaries?",
                "3. APPROACH/HOW: How should this be implemented? What is the high-level approach? What are the steps?",
                "4. INPUTS: What data/artifacts does this consume? What format/schema? What are the sources?",
                "5. OUTPUTS: What does this produce? What format/schema? Who consumes it?",
                "6. INTERFACES: What contracts exist with other components? What API/protocol is used?",
                "7. DEPENDENCIES: What must be completed first? What does this depend on? What depends on this?",
                "8. DATA/SCHEMAS: What data structures are involved? What validation rules apply?",
                "9. ERROR HANDLING: What can go wrong? How should errors be handled? What are the failure modes?",
                "10. TESTING: How should this be tested? What are acceptance criteria? What edge cases exist?",
                "11. NON-FUNCTIONAL: What are performance, security, accessibility requirements?",
                "12. ASSUMPTIONS: What assumptions are being made? What needs clarification?",
                "13. RISKS: What risks exist? What could delay or block progress?",
                "14. INTEGRATION: How does this integrate with siblings/parent? What coordination is needed?",
                "Questions must be scoped to the given node + its local neighborhood (parent/siblings) only; do not assume global features.",
                "Each question should be specific and actionable - avoid vague questions like 'is this good?'",
                "Prefer short, sharp questions that drive specificity and independence without naming technologies.",
                "Monologue should be a thoughtful reflection analyzing: the task's current completeness, gaps in specification, relationship to parent/siblings, potential risks, and what information is most critically needed.",
                "Avoid language defaults and brand/tool names; stay domain-neutral.",
            ],
            "join_judge_general": [
                "If constraints.side_context.join_judge is present, treat it as the authoritative capsule for the target node.",
                "Validate the target node's decomposition and interfaces in its local neighborhood: coverage, overlap, and explicit contracts (artifacts/deps).",
                "Do not assume any specific technology, language, or UI exists unless stated in constraints.",
                "Derive a rubric from the capsule (do not hardcode criteria). Output rubric as an object with stable schema: dimensions[] (each with name/intent/pass_conditions/fail_signals/priority).",
                "Also output a flat list of high-detail questions (objects) scoped to this node only.",
                "If ok=false, output actions as a list of edit operations with op in: edit_task, add_child, add_dep, rewire_artifacts.",
                "Actions must be minimal and safe: preserve ids; avoid deleting nodes; prefer rescoping and explicit contracts.",
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
            "task_dedup_policy": [
                "If constraints.side_context.context_pack is present, treat it as the authoritative, budgeted context bundle for this node.",
                "If constraints.side_context.task_dedup is present, use it as an external redundancy signal.",
                "If task_dedup.similar_leaves shows two leaves are highly similar, avoid duplicating work: either merge responsibilities, or split by non-overlapping deliverables so they become independent.",
                "Coverage axiom: for any composite task, children should jointly cover the parent's in-scope deliverables (union covers parent) while minimizing overlap.",
                "Independence axiom: siblings should have clear, non-overlapping responsibilities and explicit interfaces (artifacts/deps) rather than implicit shared context.",
                "When removing redundancy, preserve ids where possible; prefer edit_task over delete. You may re-scope titles/descriptions/node_plan to eliminate overlap.",
            ],
            "tree_spider_validate": [
                "If constraints.side_context.tree_spider is present, treat it as the authoritative validation capsule for a specific target node.",
                "The capsule includes: target task, parent summary, siblings summaries, children summaries, artifact producer/consumer slice, similar nodes (embedding neighbors), and deterministic_issues.",
                "Validate the target node in its local neighborhood: division of responsibility, communication contracts, and redundancy/coverage.",
                "Communication axiom: if a sibling/child depends on outputs, require explicit artifacts (outputs.produces / inputs.consumes) and deps wiring; no implicit coupling.",
                "Coverage axiom: if target is composite, children must jointly cover the parent's stated deliverables (in node_plan.deliverables / acceptance_criteria / step_outline) with minimal overlap.",
                "Redundancy axiom: if deterministic_issues or similar nodes indicate overlap, suggest rescoping/merging responsibilities so siblings become independent.",
                "Do NOT request more context; operate only on the provided capsule.",
                "Output suggestions must be actionable edits: fill missing node_plan fields, split_needed:<id>, add_artifact:<producer_id>-><artifact_id>, add_dep:<consumer_id>-><producer_id>, rescope:<id>.",
            ],
            "deterministic_taskplan_fixes": [
                "If constraints.side_context.deterministic_taskplan_validation is present, treat it as authoritative deterministic findings.",
                "Convert deterministic suggestions into concrete edits for AmendPlanSignature when possible.",
                "For add_dep suggestions, add the specified dep edge. For artifact_consumer_missing_dep, add dep(s) to at least one producer.",
                "For parent_artifact_not_delegated suggestions, either move the artifact production/consumption to an appropriate child or introduce an explicit integration child that owns that artifact and wire deps/artifacts.",
                "For sibling_artifact_multi_producer or sibling_similarity_high, rescope or merge responsibilities so there is a single clear producer and other tasks consume it.",
                "Preserve ids; prefer update_task/rewire_artifacts/add_dep/add_child over delete.",
            ],
        }

        # Default rule blocks per mode; callers can override via constraints
        self._MODE_DEFAULT_RULE_BLOCKS: Dict[str, List[str]] = {
            "plan": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "planning_general",
                "planning_elaboration",
                "no_priority_fields",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "task_plan": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "task_plan_general",
                "no_priority_fields",
                "no_language_defaults",
                "interface_first",
                "artifact_wiring",
                "sibling_integration",
                "boundary_language_neutral",
                "no_cross_language_imports",
                "task_plan_parent_discipline",
                "split_guardrails",
                "neutrality_no_examples",
                "task_node_plan",
                "decomposition_policy",
                "task_artifact_shape",
                "task_validation_policy",
            ],
            "split": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "split_general",
                "split_guardrails",
                "no_priority_fields",
                "no_language_defaults",
                "interface_first",
                "artifact_wiring",
                "sibling_integration",
                "boundary_language_neutral",
                "no_cross_language_imports",
                "neutrality_no_examples",
                "decomposition_policy",
            ],
            "clarify": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "clarify_general",
                "artifact_wiring",
                "no_priority_fields",
                "no_language_defaults",
                "no_cross_language_imports",
                "neutrality_no_examples",
                "task_node_plan",
            ],
            "merge": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "merge_general",
                "artifact_wiring",
                "sibling_integration",
                "boundary_language_neutral",
                "no_cross_language_imports",
                "no_priority_fields",
                "no_language_defaults",
            ],
            "plan_validate": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "plan_quality_check",
                "planning_general",
                "planning_elaboration",
                "no_priority_fields",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "plan_refine": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "plan_refine_general",
                "planning_general",
                "planning_elaboration",
                "no_priority_fields",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "project_validate": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "jury_evidence_standard",
                "project_validate_general",
                "project_validate_syntax",
                "artifact_wiring",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "node_validate": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "jury_evidence_standard",
                "node_validate_general",
                "project_validate_syntax",
                "artifact_wiring",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "validation_questions": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "jury_evidence_standard",
                "validation_questions_general",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "repair_project": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "jury_evidence_standard",
                "project_validate_general",
                "project_validate_syntax",
                "artifact_wiring",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
                "repair_must_act",
            ],
            "repair_project_patches": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "jury_evidence_standard",
                "project_validate_general",
                "project_validate_syntax",
                "artifact_wiring",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
                "repair_must_act",
            ],
            "findings_summarize": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "findings_summarize_general",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "codespec_init": [
                "format_structured_fields_no_shorten",
                "call_contract",
                "use_side_context",
                "codespec_quality",
                "planning_general",
                "planning_elaboration",
                "no_language_defaults",
                "boundary_language_neutral",
                "no_cross_language_imports",
            ],
            "codespec_enrich": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "codespec_quality",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
            "task_plan_validate": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "task_validation_policy",
                "neutrality_no_examples",
            ],
            "micro_adjust": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "no_language_defaults",
            ],
            "join_judge": [
                "format_structured_fields",
                "call_contract",
                "use_side_context",
                "join_judge_general",
                "artifact_wiring",
                "sibling_integration",
                "boundary_language_neutral",
                "no_language_defaults",
                "no_cross_language_imports",
            ],
        }

        # Per-mode call metadata used to make each DSPy call self-describing.
        # This is injected into constraints.call by _augment_constraints.
        self._CALL_SPECS: Dict[str, Dict[str, Any]] = {
            "plan": {
                "name": "plan",
                "purpose": "Generate a concrete project plan (modules/files) from idea + constraints.",
                "inputs": ["idea", "constraints"],
                "outputs": ["plan"],
            },
            "plan_validate": {
                "name": "plan_validate",
                "purpose": "Validate plan quality and separation of concerns; return issues/suggestions.",
                "inputs": ["idea", "constraints", "plan"],
                "outputs": ["ok", "issues", "suggestions"],
            },
            "plan_refine": {
                "name": "plan_refine",
                "purpose": "Refine an existing plan to address validation suggestions and improve completeness.",
                "inputs": ["idea", "constraints", "current_plan"],
                "outputs": ["plan"],
            },
            "task_plan": {
                "name": "task_plan",
                "purpose": "Generate a hierarchical, language-neutral TaskPlan (tasks tree) with complete node_plan fields.",
                "inputs": ["idea", "constraints"],
                "outputs": ["tasks"],
            },
            "task_plan_validate": {
                "name": "task_plan_validate",
                "purpose": "Validate the hierarchical task plan structure; return actionable issues/suggestions.",
                "inputs": ["idea", "constraints", "plan"],
                "outputs": ["ok", "issues", "suggestions"],
            },
            "split": {
                "name": "split",
                "purpose": "Decide whether a task should be split; if so return children tasks.",
                "inputs": ["task", "idea", "constraints"],
                "outputs": ["action", "children"],
            },
            "repair_project_patches": {
                "name": "repair_project_patches",
                "purpose": "Propose minimal patch operations (find/replace) to repair project outputs based on validation issues.",
                "inputs": ["idea", "constraints", "issues", "files", "file_specs", "context"],
                "outputs": ["patches", "notes"],
            },
            "clarify": {
                "name": "clarify",
                "purpose": "Clarify a single task in-place (no child changes); fill missing node_plan details.",
                "inputs": [
                    "task",
                    "parent",
                    "siblings",
                    "artifacts",
                    "files",
                    "idea",
                    "constraints",
                ],
                "outputs": ["task_out"],
            },
            "merge": {
                "name": "merge",
                "purpose": "Merge children outputs for a parent into full-file writes and merged artifacts.",
                "inputs": ["parent", "children", "artifacts", "files", "idea", "constraints"],
                "outputs": ["writes", "artifacts_out"],
            },
            "project_validate": {
                "name": "project_validate",
                "purpose": "Validate the project holistically (files/specs/plan integration); return issues/warnings/suggestions.",
                "inputs": ["idea", "constraints", "plan", "files", "file_specs"],
                "outputs": ["ok", "issues", "warnings", "suggestions"],
            },
            "repair_project": {
                "name": "repair_project",
                "purpose": "Propose concrete file edits that repair validation issues while remaining language-agnostic.",
                "inputs": ["idea", "constraints", "issues", "files", "file_specs", "context"],
                "outputs": ["edits", "notes"],
            },
            "codespec_init": {
                "name": "codespec_init",
                "purpose": "Generate initial language-neutral CodeSpec file entries for the plan.",
                "inputs": ["idea", "constraints", "plan_overview", "tasks_overview"],
                "outputs": ["files"],
            },
            "codespec_enrich": {
                "name": "codespec_enrich",
                "purpose": "Enrich a single CodeSpec file entry with missing functions/exports/etc (patch only).",
                "inputs": ["idea", "constraints", "file_entry"],
                "outputs": ["patch"],
            },
            "micro_adjust": {
                "name": "micro_adjust",
                "purpose": "Apply minimal non-semantic fixes to file text.",
                "inputs": ["file", "language", "text_in", "idea", "constraints"],
                "outputs": ["text", "notes"],
            },
            "join_judge": {
                "name": "join_judge",
                "purpose": "Judge a node's decomposition/interfaces using a local capsule; derive rubric, questions, and minimal actions.",
                "inputs": ["node", "idea", "constraints"],
                "outputs": ["ok", "actions", "rubric", "questions", "notes"],
            },
        }

        # Define Signatures lazily to avoid top-level import-time failures when dspy missing
        class PlanSignature(dspy.Signature):  # type: ignore
            """Return a concrete software plan via the signature output fields.

            Output shape (plan): {"modules": ModuleSpec[]}

            ModuleSpec fields:
            - name: string
            - purpose: string (short; optional)
            - deps: string[] (module-level deps; may be empty)
            - files: CodeSpecFile[] (REQUIRED; non-empty for at least one module)

            CodeSpecFile fields (developer-grade detail):
            - path: string (posix-like relative path); REQUIRED
            - language?: string (a language identifier); may be omitted if inferable from path. If extension is missing/ambiguous, EXPLICITLY set "language"; do not default to any language.
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
            - Choose appropriate languages per file/module. Multi-language stacks are allowed. If the file extension does not make language obvious, explicitly set CodeSpecFile.language; NEVER assume a default.
            - Cover ALL necessary components for a complete, production-ready solution: frontend, backend, configuration, validation, error handling, logging, testing, documentation, deployment, packaging, entrypoints.
            - Include detailed function signatures with proper parameter types, return types, and comprehensive descriptions.
            - Ensure each file has complete imports, exports, classes, constants, and implementation details.
            - Always include comprehensive file coverage - never generate incomplete project structures.
            - Do NOT include any priority/importance fields; treat all modules/files as equally important unless constraints explicitly require prioritization.
            - Functions must have detailed signatures: name(param1: type, param2: type) -> return_type, not empty parentheses.

            Formatting:
            - Do NOT include markdown or code fences.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement. Preserve wording and scope.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(
                desc="Structured constraints dict. Treat as binding requirements; remain neutral unless constraints force a choice."
            )  # type: ignore
            plan: Dict[str, Any] = dspy.OutputField(
                desc="Plan object with modules/files. Must be complete enough to build/run and include clear responsibilities and interfaces."
            )  # type: ignore

        class PlanValidateSignature(dspy.Signature):  # type: ignore
            """Validate a plan structure for separation of concerns and minimal quality.

            Use the provided plan to assess separation of concerns, presence of appropriate files,
            clarity of function exports/entrypoints, and avoidance of placeholders.
            Return ok/issues/suggestions via output fields.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict.")  # type: ignore
            plan: Dict[str, Any] = dspy.InputField(desc="Candidate plan to validate.")  # type: ignore
            ok: bool = dspy.OutputField(desc="True if the plan is acceptable without further refinement.")  # type: ignore
            issues: List[str] = dspy.OutputField(desc="Blocking issues; must be fixed.")  # type: ignore
            suggestions: List[str] = dspy.OutputField(desc="Non-blocking improvements; should be actionable.")  # type: ignore

        class PlanRefineSignature(dspy.Signature):  # type: ignore
            """Refine an existing plan.

            Output shape (plan): {"modules": ModuleSpec[]}
            """

            idea: str = dspy.InputField(desc="User idea / problem statement.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict.")  # type: ignore
            current_plan: Dict[str, Any] = dspy.InputField(desc="Existing plan to refine (do not discard; patch/improve).")  # type: ignore
            plan: Dict[str, Any] = dspy.OutputField(desc="Refined plan that resolves issues and improves completeness.")  # type: ignore

        class TaskPlanSignature(dspy.Signature):  # type: ignore
            """Produce a hierarchical task plan.

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

                        Integration & completeness axioms (CRITICAL):
                        - INTEGRATION IS A FIRST-CLASS TASK: For any parent whose children produce multiple files/components/interfaces, include at least one explicit child responsible for wiring/integration across those outputs (not just producing more files). This child defines concrete interfaces and validates cross-component behavior.
                        - VERIFICATION IS A FIRST-CLASS TASK: Include explicit children for verification (tests, run steps, validation evidence) and documentation (how to run/build/use). Do not assume someone else will do it.
                        - NO ISLAND LEAVES: Leaves must not be "write a file" in isolation; each leaf must either (a) implement a concrete exported unit in an existing file, or (b) create/update a file together with its integration surface (imports/exports/entrypoint contract).

                        Leaf contract (REQUIRED for code:function leaves):
                        - kind MUST be exactly "code:function".
                        - inputs MUST include: path (string), name (string), exports (string[]), allowed_imports (string[]), signature (string).
                        - If multiple leaves target the same path, they must coordinate by expanding the same file incrementally: each leaf updates the full file while preserving existing correct code.

                        Generation rules:
                        - Be neutral and example-free. Do NOT mention technologies, frameworks, brands, or programming languages.
                        - For complex ideas, create 1-2 root tasks that decompose into children. Avoid many root siblings.
                        - Parents coordinate and define interfaces; children implement specific responsibilities.
                        - Each task MUST include node_plan with ALL listed keys. Keys may have empty strings/arrays/objects but must be present.
                        - Optional artifacts (inputs.consumes / outputs.produces) follow shape: arrays of objects with at least "id".
                        - After generating tasks, self-validate: ensure node_plan completeness, valid deps (no cycles), parent-vs-leaf correctness, artifact shapes, and no technology leakage. If issues are found, fix them before returning.
            {{ ... }}

                        Formatting:
                        - Do NOT include markdown or code fences.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement. Use as ground truth for scope.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(
                desc="Constraints dict, including any provided context_pack in side_context. Do not invent technologies or languages unless required."
            )  # type: ignore
            tasks: List[Dict[str, Any]] = dspy.OutputField(
                desc="Root tasks list (each may have children). Every task must include a complete node_plan with all required keys present."
            )  # type: ignore

        class TaskPlanValidateSignature(dspy.Signature):  # type: ignore
            """Validate a hierarchical task plan (tasks only; no files).
            Return ok/issues/suggestions via output fields.

            Checklist:
            - All tasks have non-empty title and node_plan keys present (even if values are empty).
            - Parent/leaf correctness: coordination tasks have children; leaves have none.
            - Deps are valid and acyclic; all ids referenced exist.
            - inputs/outputs, when present, follow artifact shape (arrays of objects with at least "id").
            - No technology/framework/language/brand references.
            - Leaves are actionable (node_plan.step_outline and acceptance_criteria exist).

            Suggestions MUST be actionable, e.g., "fill:<task_id>.node_plan.test_plan", "split_needed:<task_id>", "remove_cycle:<a>-><b>-><a>".
            Formatting: no JSON; no markdown.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict.")  # type: ignore
            plan: Dict[str, Any] = dspy.InputField(desc="Candidate task plan object containing tasks.")  # type: ignore
            ok: bool = dspy.OutputField(desc="True if the task plan is acceptable.")  # type: ignore
            issues: List[str] = dspy.OutputField(desc="Blocking issues.")  # type: ignore
            suggestions: List[str] = dspy.OutputField(desc="Actionable suggestions in stable machine-readable strings.")  # type: ignore

        class SplitDecisionSignature(dspy.Signature):  # type: ignore
            """Decide whether the given task should be split further or kept as a leaf.
            Return action and children via output fields.

            Policy:
            - Use local reasoning. Split only if it reduces ambiguity, clarifies responsibilities, or enables parallel work.
            - Keep as leaf when the task is smallest independently actionable and node_plan is complete.
            - If splitting, return a few meaningful non-overlapping children. Each child MUST include a complete node_plan (all keys present) and must not reference technologies or languages.
            - Maintain or infer sensible deps among siblings and with the parent when evident.
            - If the task spans multiple files/components or requires cross-component behavior, prefer splitting and include a dedicated integration/verification child.

            Formatting: no JSON; no markdown or examples; no technology references.
            """

            task: Dict[str, Any] = dspy.InputField(desc="Single task node to evaluate for splitting.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict.")  # type: ignore
            action: str = dspy.OutputField(desc="One of: split | keep. Do not invent other actions.")  # type: ignore
            children: List[Dict[str, Any]] = dspy.OutputField(desc="When action=split: a small list of child tasks with full node_plan keys.")  # type: ignore

        class FileGenSignature(dspy.Signature):  # type: ignore
            """Generate the complete file content appropriate for the language.
            Output MUST be the exact file text with no markdown fences or commentary.
            Keep it professional, robust, and minimal.

            Cross-file contract rules (critical):
            - If this file references other local files (imports/includes/links), those targets MUST exist in the CodeSpec and the referenced symbols/paths must match exactly.
            - If there is an entrypoint/initializer concept (a function to run on startup), ensure it is actually invoked by the real entrypoint mechanism for that ecosystem (do not define-but-never-call).
            - If this file defines an interface consumed elsewhere (functions/classes/events/schema), keep names/signatures consistent with the declared `functions`/`exports` and with other files.
            - Avoid runtime-importing placeholder stubs. If a contract-only stub exists, do not import it from runnable code.
            """

            idea = dspy.InputField(desc="User idea / problem statement.")  # type: ignore
            constraints = dspy.InputField(desc="Constraints dict (may include side_context and context_pack).")  # type: ignore
            file = dspy.InputField(desc="File path to generate (posix-like relative path).")  # type: ignore
            language = dspy.InputField(desc="Language identifier for this file; may be empty if inferable from extension.")  # type: ignore
            exports = dspy.InputField(desc="Declared exports for this file (names that must exist in output text where applicable).")  # type: ignore
            imports = dspy.InputField(desc="Allowed imports/dependencies list; keep within this set when applicable.")  # type: ignore
            entrypoint = dspy.InputField(desc="Entrypoint hint for this file/module (if applicable).")  # type: ignore
            functions = dspy.InputField(desc="Declared functions map for this file; keep signatures and names consistent.")  # type: ignore
            text = dspy.OutputField(desc="Full file contents only")  # type: ignore

        class MicroAdjustSignature(dspy.Signature):  # type: ignore
            """Make minimal non-semantic micro-adjustments to file text.

            Goals:
            - Fix obvious small issues (typos, inconsistent formatting, tiny correctness tweaks) without changing intent.
            - Do not introduce new dependencies or technologies.
            - Output MUST be the exact full file text in the output field `text`.
            - Put brief rationale in `notes` (short strings). No markdown or code fences.
            """

            file: str = dspy.InputField(desc="File path being adjusted (posix-like relative path).")  # type: ignore
            language: str = dspy.InputField(desc="Language identifier for this file; used to apply language-specific formatting conventions where applicable.")  # type: ignore
            text_in: str = dspy.InputField(desc="Original file contents before adjustments; preserve all semantic meaning and contract obligations from this text.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement for context; micro-adjustments must align with this overall goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on project-wide conventions and allowed operations.")  # type: ignore
            text: str = dspy.OutputField(desc="Adjusted file contents as complete text; MUST be full file with all micro-adjustments applied. Never return partial or diff format.")  # type: ignore
            notes: List[str] = dspy.OutputField(desc="Brief rationale strings explaining each micro-adjustment made; keep concise and factual, no markdown.")  # type: ignore

        class ExportVerifySignature(dspy.Signature):  # type: ignore
            """Verify that the provided file text contains the declared exports for the given language.
            Return ok/missing via output fields. Do not include markdown or code fences.
            """

            file: str = dspy.InputField(desc="File path being verified (posix-like relative path); used for context on what kind of exports to expect.")  # type: ignore
            language: str = dspy.InputField(desc="Language identifier (e.g., 'python', 'javascript', 'typescript'); determines what constitutes a valid export for this file.")  # type: ignore
            exports: List[str] = dspy.InputField(desc="Declared exports that MUST exist in the file; these are symbolic names (functions, classes, constants) that other files depend on.")  # type: ignore
            text: str = dspy.InputField(desc="Current file contents to verify; check this text for presence of all declared exports using language-specific rules.")  # type: ignore
            ok: bool = dspy.OutputField(desc="True if all declared exports are present in the text with correct signatures; False if any are missing or have contract mismatches.")  # type: ignore
            missing: List[str] = dspy.OutputField(desc="List of export names that are missing or incorrectly implemented; empty list if ok=True.")  # type: ignore

        class ArtifactValidateSignature(dspy.Signature):  # type: ignore
            """Validate an artifact (any kind) using content and context.
            Return ok/issues/warnings/normalized via output fields.
            Rules:
            - Do not include markdown or code fences.
            - If json_content is provided, use it as ground truth for JSON parsing instead of inferring from text.
            - Be strict but pragmatic; prefer structural correctness and contract completeness over style.
            """

            artifact_id: str = dspy.InputField(desc="Unique identifier for this artifact; used to track dependencies and refer to artifact in error messages.")  # type: ignore
            kind: str = dspy.InputField(desc="Artifact kind (e.g., 'config', 'spec', 'schema'); determines what structural and semantic rules to apply during validation.")  # type: ignore
            path: str = dspy.InputField(desc="File path or logical path for this artifact; used for context on how artifact fits into project structure.")  # type: ignore
            text: str = dspy.InputField(desc="Raw text contents of the artifact; validate this against structural rules and schema expectations for the given kind.")  # type: ignore
            json_content: Dict[str, Any] = dspy.InputField(desc="Pre-parsed JSON content if available; use this as ground truth instead of re-parsing text field.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement for context; artifact must align with this overall goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on project-wide artifact shape rules.")  # type: ignore
            ok: bool = dspy.OutputField(desc="True if artifact passes all structural and semantic validation checks; False if any blocking issues exist.")  # type: ignore
            issues: List[str] = dspy.OutputField(desc="Blocking issues that prevent artifact from being used; must be addressed before artifact can be considered valid.")  # type: ignore
            warnings: List[str] = dspy.OutputField(desc="Non-blocking concerns that should be reviewed; artifact can be used but may have quality or maintainability issues.")  # type: ignore
            normalized: Dict[str, Any] = dspy.OutputField(desc="Canonical representation of artifact after validation; may include inferred fields, normalized formatting, or structural corrections.")  # type: ignore

        class AmendPlanSignature(dspy.Signature):  # type: ignore
            """Amend a hierarchical task plan to resolve issues and improve completion.
            Return edits via the output field 'edits'.

            Allowed edits (non-destructive):
              - {"op":"add_child","parent_id":string,"task":Task}
              - {"op":"update_task","id":string,"set":object}  // e.g., set.node_plan.test_plan, set.description, set.outputs
              - {"op":"add_dep","id":string,"dep_id":string}
              - {"op":"rewire_artifacts","id":string,"consumes"?:object[],"produces"?:object[]}

            Constraints:
              - Keep ids stable. Preserve intent. Prefer filling missing node_plan fields before splitting.
              - No technology/framework/language mentions. No code.
              - Ensure postconditions: node_plan keys present on all tasks; no composite leaves; deps valid; artifact shapes respected.
              - Do not output JSON; no markdown.
            """

            current_plan: Dict[str, Any] = dspy.InputField(desc="Hierarchical task plan as a nested structure; contains tasks with id/description/node_plan/children/deps/inputs/outputs. Preserve all existing task ids and overall decomposition structure.")  # type: ignore
            statuses: Dict[str, Any] = dspy.InputField(desc="Task execution statuses (pending/running/done/failed) keyed by task id; use to understand which tasks need amendments or corrections.")  # type: ignore
            artifacts: List[Dict[str, Any]] = dspy.InputField(desc="Artifact registry showing declared artifacts with id/kind/shape/dependencies; amendments must respect artifact shape rules and inter-artifact contracts.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement for context; amendments must maintain alignment with this overall goal and not introduce scope drift.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on allowed edit operations, node_plan schema requirements, and artifact wiring rules.")  # type: ignore
            edits: List[Dict[str, Any]] = dspy.OutputField(desc="Non-destructive edit operations to apply to the plan; each edit MUST have 'op' field and follow allowed edit schema. Use update_task to fill missing node_plan fields before resorting to add_child splits.")  # type: ignore

        class ClarifyTaskSignature(dspy.Signature):  # type: ignore
            """Refine a single Task without adding children.
            Return the updated Task via output field 'task'.

            Rules:
            - Do not add or remove children. Preserve id and overall intent.
            - Fill or improve node_plan fields; all required node_plan keys must exist after clarification.
            - Keep content neutral and example-free; avoid naming technologies or languages.
            - You may add or normalize inputs/outputs fields and explicit deps if derivable from the provided context, adhering to artifact shape rules.
            - Do not output JSON; no markdown.
            """

            task: Dict[str, Any] = dspy.InputField(desc="Task node to clarify; has id/description/node_plan/inputs/outputs/deps. Fill missing node_plan fields and improve clarity without changing scope or adding children.")  # type: ignore
            parent: Dict[str, Any] = dspy.InputField(desc="Parent task for context; clarification must respect parent's decomposition strategy and not duplicate parent or sibling responsibilities.")  # type: ignore
            siblings: List[Dict[str, Any]] = dspy.InputField(desc="Sibling tasks at same level; clarification must maintain clear boundaries with siblings and not create responsibility overlap.")  # type: ignore
            artifacts: List[Dict[str, Any]] = dspy.InputField(desc="Artifact registry showing declared artifacts; clarification must respect artifact shape rules and ensure inputs/outputs reference existing artifacts correctly.")  # type: ignore
            files: Dict[str, str] = dspy.InputField(desc="Generated files by path; used for context on what has already been implemented and what contracts exist.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement for context; clarified task must maintain alignment with this overall goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on node_plan schema requirements and allowed task transformations.")  # type: ignore
            task_out: Dict[str, Any] = dspy.OutputField(desc="Clarified task with all required node_plan fields filled (decomposition/interfaces/tests); preserve id and children exactly as input, only improve clarity and completeness.")  # type: ignore

        class MergeSubtasksSignature(dspy.Signature):  # type: ignore
            """Merge children's outputs for a parent task.
            Return merged outputs via fields writes and artifacts.
            """

            parent: Dict[str, Any] = dspy.InputField(desc="Parent task whose children are being merged; parent.outputs defines expected merge result shape and artifact contracts.")  # type: ignore
            children: List[Dict[str, Any]] = dspy.InputField(desc="Child tasks that have been executed; each child has outputs/writes that must be composed into parent's output contract.")  # type: ignore
            artifacts: List[Dict[str, Any]] = dspy.InputField(desc="Artifact registry showing declared artifacts; merge must respect artifact shape rules and ensure parent's produced artifacts properly compose children's outputs.")  # type: ignore
            files: Dict[str, str] = dspy.InputField(desc="Generated files by path; used for context on what children produced and what integration points exist.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement for context; merged result must fulfill parent's responsibility toward this overall goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on merge strategies and artifact composition rules.")  # type: ignore
            writes: List[Dict[str, str]] = dspy.OutputField(desc="File writes produced by merging children; each write has path/text. May include orchestration files that wire children together or coordination logic.")  # type: ignore
            artifacts_out: List[Dict[str, Any]] = dspy.OutputField(desc="Artifacts produced by merge; must match parent.outputs artifact contracts and properly reference composed child artifacts.")  # type: ignore

        class ProjectValidateSignature(dspy.Signature):  # type: ignore
            """Validate the project holistically.
            Return ok/issues/warnings/suggestions via output fields.
            Inputs are language-neutral.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement; project must fully implement this goal with all required functionality, completeness, and coherence.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on project-wide requirements, quality standards, and completeness criteria.")  # type: ignore
            plan: Dict[str, Any] = dspy.InputField(desc="Complete hierarchical task plan; validate that decomposition is coherent, all tasks are necessary, and no missing functionality exists.")  # type: ignore
            files: Dict[str, str] = dspy.InputField(desc="Generated files by path; validate cross-file contracts, integration coherence, and that all plan outputs have corresponding implementations.")  # type: ignore
            file_specs: Dict[str, Any] = dspy.InputField(desc="File specifications with exports/imports/entrypoints; validate that declared contracts are fulfilled and dependencies are properly wired.")  # type: ignore
            ok: bool = dspy.OutputField(desc="True if project is complete, coherent, and ready for use; False if any blocking issues prevent project from meeting user's stated goal.")  # type: ignore
            issues: List[str] = dspy.OutputField(desc="Blocking issues that prevent project from being considered complete; must include missing functionality, broken contracts, or integration failures.")  # type: ignore
            warnings: List[str] = dspy.OutputField(desc="Non-blocking concerns about quality, maintainability, or best practices; project can be used but may have technical debt or fragility.")  # type: ignore
            suggestions: List[str] = dspy.OutputField(desc="Actionable improvement suggestions; may include refactoring opportunities, performance optimizations, or feature enhancements beyond minimum requirements.")  # type: ignore

        class NodeValidateSignature(dspy.Signature):  # type: ignore
            """Validate a single plan node bottom-up.

            Return ok/issues/warnings/suggestions via output fields.
            Inputs are language-neutral.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement; node must contribute correctly toward this overall goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context.run_logs and side_context.prior_findings); use for reference on validation standards and memory.")  # type: ignore
            node: Dict[str, Any] = dspy.InputField(desc="Target plan node to validate.")  # type: ignore
            parent: Dict[str, Any] = dspy.InputField(desc="Parent node for boundary/decomposition context; empty dict if root.")  # type: ignore
            siblings: List[Dict[str, Any]] = dspy.InputField(desc="Sibling nodes for boundary validation.")  # type: ignore
            children: List[Dict[str, Any]] = dspy.InputField(desc="Child nodes for decomposition coverage checks.")  # type: ignore
            files: Dict[str, str] = dspy.InputField(desc="Bounded subset of generated files relevant to this node, by project-relative path.")  # type: ignore
            file_specs: Dict[str, Any] = dspy.InputField(desc="Per-file contract specs for the provided files.")  # type: ignore
            ok: bool = dspy.OutputField(desc="True if node is complete/correct within its scope; False if blocking issues exist.")  # type: ignore
            issues: List[str] = dspy.OutputField(desc="Blocking machine-coded issues for this node.")  # type: ignore
            warnings: List[str] = dspy.OutputField(desc="Non-blocking machine-coded warnings for this node.")  # type: ignore
            suggestions: List[str] = dspy.OutputField(desc="Machine-coded improvement suggestions for this node.")  # type: ignore

        class ValidationQuestionsSignature(dspy.Signature):  # type: ignore
            """Generate probing validation questions from findings and context."""

            idea: str = dspy.InputField(desc="User idea / problem statement for context.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context.run_logs and side_context.prior_findings).")  # type: ignore
            plan: Dict[str, Any] = dspy.InputField(desc="Plan for context; may be partial.")  # type: ignore
            findings: List[str] = dspy.InputField(desc="Flat list of machine-coded findings (issues/warnings/suggestions).")  # type: ignore
            questions: List[Dict[str, Any]] = dspy.OutputField(desc="Flat list of question objects: {id, question, focus, why_this_matters, expected_answer_shape}.")  # type: ignore

        class FindingsSummarizeSignature(dspy.Signature):  # type: ignore
            """Summarize findings for context/memory.

            This is used to keep validation agentic without expanding context unboundedly.
            Outputs are typed (no JSON parsing).
            """

            findings: List[str] = dspy.InputField(desc="Flat list of machine-coded issues/warnings/suggestions from prior validator passes. Do not include prose.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context.prior_summary and side_context.run_logs); keep summary strictly language-neutral.")  # type: ignore
            summary: str = dspy.OutputField(desc="Short, language-neutral memory summary of findings (<= 1200 chars).")  # type: ignore
            key_issues: List[str] = dspy.OutputField(desc="De-duplicated subset (<= 30) of the most important machine-coded findings, preserved verbatim.")  # type: ignore

        class RepairProjectSignature(dspy.Signature):  # type: ignore
            """Propose repairs as concrete file edits.

            Inputs:
            - issues: structured issue dicts (code/severity/path/ref/message) derived from validators.
            - files: a bounded subset of whole-file texts for the current repair focus.
            - file_specs: per-file contracts when available (exports/imports/entrypoints).
            - context: deterministic metadata (focus paths, file index, validator details, budgets).

            Output:
            - edits: list of dicts: {"path": string, "new_text": string, "rationale": string}
              `new_text` MUST be the full file contents. No patch formats.
            - notes: brief rationale strings.

            Hard constraints:
            - Do not add remote resource dependencies unless constraints explicitly allow.
            - Prefer minimal surface-area changes that directly eliminate the given issues.
            - You MAY create new files by returning edits for new paths.
            - Do not name technologies/frameworks/brands.
            - Do not output markdown or code fences.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement for context; repairs must maintain alignment with overall goal and not introduce scope changes.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on allowed repair operations, technology restrictions, and cross-file contract rules.")  # type: ignore
            issues: List[Dict[str, Any]] = dspy.InputField(desc="Structured validation issues with code/severity/path/ref/message; each issue must be addressed by corresponding repair edits. Focus on blocking issues first.")  # type: ignore
            files: Dict[str, str] = dspy.InputField(desc="Bounded subset of file texts for repair focus; repairs MUST return complete new_text for each file being fixed, preserving all unaffected content.")  # type: ignore
            file_specs: Dict[str, Any] = dspy.InputField(desc="Per-file contract specifications with exports/imports/entrypoints; repairs must preserve these contracts and not break downstream dependencies.")  # type: ignore
            context: Dict[str, Any] = dspy.InputField(desc="Deterministic repair metadata including focus paths, file index, validator details, and budgets; use to understand repair scope and what files can be modified.")  # type: ignore
            edits: List[Dict[str, Any]] = dspy.OutputField(desc="Repair edits as list of dicts with path/new_text/rationale; new_text MUST be complete file contents (no patches). May create new files if needed to resolve issues.")  # type: ignore
            notes: List[str] = dspy.OutputField(desc="Brief rationale strings explaining repair strategy; highlight which issues each edit addresses and why the repair approach was chosen.")  # type: ignore

        class RepairProjectPatchSignature(dspy.Signature):  # type: ignore
            """Propose minimal repairs as patch operations (preferred).

            Output patches as list items with keys:
                - path: string (project-relative file path)
                - find: exact substring from the current file text (must match exactly)
                - replace: replacement substring
                - rationale: short string
                - allow_multiple?: bool (only if you truly intend multiple replacements)
                - count?: int (when allow_multiple=true, must match actual occurrences)

            Rules:
                - Do NOT rewrite whole files unless unavoidable.
                - Prefer smallest safe edits that directly eliminate the given issues.
                - Preserve formatting/indentation by patching existing code.
                - If you must create a missing file, set find="" and replace=FULL FILE CONTENTS.
                - No markdown or code fences.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement for context; repairs must maintain alignment with overall goal and not introduce scope changes.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on allowed repair operations and restrictions.")  # type: ignore
            issues: List[Dict[str, Any]] = dspy.InputField(desc="Structured validation issues with code/severity/path/ref/message; each issue must be addressed.")  # type: ignore
            files: Dict[str, str] = dspy.InputField(desc="Bounded subset of current file texts for repair focus.")  # type: ignore
            file_specs: Dict[str, Any] = dspy.InputField(desc="Per-file contract specifications with exports/imports/entrypoints.")  # type: ignore
            context: Dict[str, Any] = dspy.InputField(desc="Deterministic repair metadata including focus paths, file index, validator details, and budgets; may include run_logs.")  # type: ignore
            needs_paths: List[str] = dspy.OutputField(desc="Optional list of additional project-relative paths required to make a correct repair. Use only when the necessary evidence is not present in `files`. Do not request more than 12 paths. Prefer paths referenced by issues, file_specs, or interface specs.")  # type: ignore
            patches: List[Dict[str, Any]] = dspy.OutputField(desc="Minimal patch operations list. Each patch dict MUST include path/find/replace/rationale. Use allow_multiple/count only when necessary.")  # type: ignore
            notes: List[str] = dspy.OutputField(desc="Brief rationale strings explaining repair strategy; keep concise and factual.")  # type: ignore

        class CodeSpecInitSignature(dspy.Signature):  # type: ignore
            """Generate an initial Codespec (files[]) for the entire plan.
            Return files via the output field 'files'.
            FileEntry shape (language-agnostic, may omit optional fields):
            {"path":string, "purpose":string, "description":string,
             "language"?:string, "imports"?:string[], "exports"?:string[],
             "functions"?:{name:string|{signature?:string,description?:string,parameters?:any[],returns?:any}},
             "classes"?:object, "constants"?:object, "entrypoint"?:string, "content"?:string}
            """

            idea: str = dspy.InputField(desc="User idea / problem statement; codespec must map to implementation plan that fully realizes this goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on file structure conventions, module organization, and cross-file contract rules.")  # type: ignore
            plan_overview: Dict[str, Any] = dspy.InputField(desc="High-level plan summary showing major decomposition; use to understand overall architecture and how files should be organized.")  # type: ignore
            tasks_overview: List[Dict[str, Any]] = dspy.InputField(desc="Task nodes with outputs/produces; map these to file entries with corresponding exports to ensure all task outputs have concrete file realizations.")  # type: ignore
            files: List[Dict[str, Any]] = dspy.OutputField(desc="File specifications as structured dicts; MUST include path/purpose/description for each file. Include exports/imports/functions to establish cross-file contracts before code generation.")  # type: ignore

        class CodeSpecEnrichSignature(dspy.Signature):  # type: ignore
            """Enrich a single Codespec file entry with missing functions/exports/etc.
            Return a partial dict to merge into the file entry via output field 'patch'.
            """

            idea: str = dspy.InputField(desc="User idea / problem statement for context; enrichment must maintain alignment with overall goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on what additional details are needed for this file.")  # type: ignore
            file_entry: Dict[str, Any] = dspy.InputField(desc="Existing codespec file entry with path/purpose/description and possibly partial exports/imports/functions; enrich missing details to make contract complete.")  # type: ignore
            patch: Dict[str, Any] = dspy.OutputField(desc="Partial dict to merge into file_entry; may include enriched functions map with signatures/parameters/returns, additional exports, or refined imports to complete the contract.")  # type: ignore

        class SocraticSignature(dspy.Signature):  # type: ignore
            """Produce clarifying questions and a reflective monologue for a task node.
            Return questions and monologue via output fields.
            """

            node: Dict[str, Any] = dspy.InputField(desc="Task node to analyze; has id/description/node_plan/inputs/outputs. Generate questions that probe completeness, clarity, and contract correctness for this specific task.")  # type: ignore
            parent: Dict[str, Any] = dspy.InputField(desc="Parent task for context; questions should validate that node fits parent's decomposition strategy and doesn't create gaps or overlaps.")  # type: ignore
            siblings: List[Dict[str, Any]] = dspy.InputField(desc="Sibling tasks at same level; questions should check for responsibility boundaries and ensure no duplicate or conflicting work.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement for context; questions must probe how this node contributes to overall goal fulfillment.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict (may include side_context and context_pack); use for reference on what aspects of the task need clarification.")  # type: ignore
            questions: List[Dict[str, Any]] = dspy.OutputField(desc="Structured clarifying questions as dicts with id/question/focus/why_this_matters/expected_answer_shape; each question should probe a specific aspect of task completeness or clarity.")  # type: ignore
            monologue: str = dspy.OutputField(desc="Reflective monologue analyzing the task's current state, identifying ambiguities, and reasoning through what needs to be clarified; use this to surface implicit assumptions or missing context.")  # type: ignore

        class JoinJudgeSignature(dspy.Signature):  # type: ignore
            """Judge a node's decomposition and interfaces using a node-scoped capsule.

            The authoritative capsule is in constraints.side_context.join_judge and includes:
            - target: the node being judged
            - parent: parent node for decomposition context
            - siblings: sibling nodes for boundary validation
            - children: child nodes to validate decomposition completeness
            - artifacts_slice: relevant artifacts for interface validation
            - deterministic_issues: pre-computed structural issues

            Output contracts:
            - ok: bool
            - actions: list of edit ops (op in edit_task/add_child/add_dep/rewire_artifacts)
            - rubric: object with stable keys (dimensions[])
            - questions: flat list of question objects (id/question/focus/why_this_matters/expected_answer_shape)
            - notes: list of short strings
            """

            node: Dict[str, Any] = dspy.InputField(desc="Task node being judged; authoritative capsule with full context is in constraints.side_context.join_judge. Validate decomposition quality, interface coverage, and artifact wiring.")  # type: ignore
            idea: str = dspy.InputField(desc="User idea / problem statement for context; judge whether node's decomposition and interfaces adequately contribute to this overall goal.")  # type: ignore
            constraints: Dict[str, Any] = dspy.InputField(desc="Constraints dict with side_context.join_judge capsule containing target/parent/siblings/children/artifacts_slice/deterministic_issues. Use capsule as ground truth for all validation dimensions.")  # type: ignore
            ok: bool = dspy.OutputField(desc="True if decomposition is complete, interfaces are well-defined, and no blocking issues exist; False if actions are needed to correct structural or contract problems.")  # type: ignore
            actions: List[Dict[str, Any]] = dspy.OutputField(desc="Edit operations to fix issues; each action has 'op' (edit_task/add_child/add_dep/rewire_artifacts) and operation-specific fields. Use edit_task to fix node_plan, add_child if decomposition is incomplete, rewire_artifacts if interface contracts are broken.")  # type: ignore
            rubric: Dict[str, Any] = dspy.OutputField(desc="Evaluation rubric with stable dimensions[] array; each dimension assesses a specific quality aspect (decomposition completeness, interface clarity, artifact coverage) with score/rationale.")  # type: ignore
            questions: List[Dict[str, Any]] = dspy.OutputField(desc="Clarifying questions as flat list with id/question/focus/why_this_matters/expected_answer_shape; probe ambiguities that prevent definitive judgment on decomposition or interface quality.")  # type: ignore
            notes: List[str] = dspy.OutputField(desc="Brief notes explaining judgment rationale; keep concise and factual, highlight specific structural patterns that informed the ok decision.")  # type: ignore

        self._PlanSignature = PlanSignature
        self._TaskPlanSignature = TaskPlanSignature
        self._TaskPlanValidateSignature = TaskPlanValidateSignature
        self._SplitDecisionSignature = SplitDecisionSignature
        self._FileGenSignature = FileGenSignature
        self._MicroAdjustSignature = MicroAdjustSignature
        self._ExportVerifySignature = ExportVerifySignature
        self._ArtifactValidateSignature = ArtifactValidateSignature
        self._AmendPlanSignature = AmendPlanSignature
        self._ClarifyTaskSignature = ClarifyTaskSignature
        self._MergeSubtasksSignature = MergeSubtasksSignature
        self._ProjectValidateSignature = ProjectValidateSignature
        self._NodeValidateSignature = NodeValidateSignature
        self._ValidationQuestionsSignature = ValidationQuestionsSignature
        self._FindingsSummarizeSignature = FindingsSummarizeSignature
        self._RepairProjectSignature = RepairProjectSignature
        self._RepairProjectPatchSignature = RepairProjectPatchSignature
        self._CodeSpecInitSignature = CodeSpecInitSignature
        self._CodeSpecEnrichSignature = CodeSpecEnrichSignature
        self._SocraticSignature = SocraticSignature
        self._JoinJudgeSignature = JoinJudgeSignature

        # Modules
        self._plan_predictor = dspy.Predict(PlanSignature)  # type: ignore
        self._task_plan_predictor = dspy.Predict(TaskPlanSignature)  # type: ignore
        self._task_plan_validate_predictor = dspy.Predict(TaskPlanValidateSignature)  # type: ignore
        self._split_predictor = dspy.Predict(SplitDecisionSignature)  # type: ignore
        self._file_predictor = dspy.Predict(FileGenSignature)  # type: ignore
        self._micro_adjust_predictor = dspy.Predict(MicroAdjustSignature)  # type: ignore
        self._export_verify_predictor = dspy.Predict(ExportVerifySignature)  # type: ignore
        self._artifact_validate_predictor = dspy.Predict(ArtifactValidateSignature)  # type: ignore
        self._amend_plan_predictor = dspy.Predict(AmendPlanSignature)  # type: ignore
        self._plan_validate_predictor = dspy.Predict(PlanValidateSignature)  # type: ignore
        self._plan_refine_predictor = dspy.Predict(PlanRefineSignature)  # type: ignore
        self._clarify_predictor = dspy.Predict(ClarifyTaskSignature)  # type: ignore
        self._merge_predictor = dspy.Predict(MergeSubtasksSignature)  # type: ignore
        self._project_validate_predictor = dspy.Predict(ProjectValidateSignature)  # type: ignore
        self._node_validate_predictor = dspy.Predict(NodeValidateSignature)  # type: ignore
        self._validation_questions_predictor = dspy.Predict(ValidationQuestionsSignature)  # type: ignore
        self._findings_summarize_predictor = dspy.Predict(FindingsSummarizeSignature)  # type: ignore
        self._repair_project_predictor = dspy.Predict(RepairProjectSignature)  # type: ignore
        self._repair_project_patches_predictor = dspy.Predict(RepairProjectPatchSignature)  # type: ignore
        self._codespec_init_predictor = dspy.Predict(CodeSpecInitSignature)  # type: ignore
        self._codespec_enrich_predictor = dspy.Predict(CodeSpecEnrichSignature)  # type: ignore
        self._socratic_predictor = dspy.Predict(SocraticSignature)  # type: ignore
        self._join_judge_predictor = dspy.Predict(JoinJudgeSignature)  # type: ignore

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

    def _coerce_to_dict(self, value: Any, *, context: str) -> Dict[str, Any]:
        """Best-effort coercion to dict.

        Primary intent: consume typed DSPy output fields without asking for JSON.
        Fallback: if a provider returns strings, attempt JSON parse.
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        # Pydantic v2 models
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()  # type: ignore[attr-defined]
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        if isinstance(value, str):
            s = self._strip_markdown_fences(value).strip()
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                # Best-effort line-based parsing to match format rules:
                #   key: value (one per line)
                out: Dict[str, Any] = {}
                for raw_line in s.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    # Strip common bullet prefixes
                    for prefix in ("- ", "* ", "• "):
                        if line.startswith(prefix):
                            line = line[len(prefix) :].strip()
                            break
                    # Accept "key: value" or "key = value"
                    sep = ":" if ":" in line else ("=" if "=" in line else "")
                    if not sep:
                        continue
                    k, v = line.split(sep, 1)
                    key = k.strip()
                    val_raw = v.strip()
                    if not key:
                        continue
                    # Try to parse structured scalars/containers
                    parsed_val: Any = val_raw
                    try:
                        if val_raw.startswith(("{", "[", '"')) or val_raw in {"true", "false", "null"}:
                            parsed_val = json.loads(val_raw)
                        else:
                            # Try numeric
                            if re.fullmatch(r"-?\d+", val_raw):
                                parsed_val = int(val_raw)
                            elif re.fullmatch(r"-?\d+\.\d+", val_raw):
                                parsed_val = float(val_raw)
                    except Exception:
                        parsed_val = val_raw
                    out[key] = parsed_val
                if out:
                    return out
                logger.debug("%s: could not parse dict from string output", context)
                return {}
        logger.debug("%s: could not coerce %s to dict", context, type(value))
        return {}

    def _coerce_to_list(self, value: Any, *, context: str) -> List[Any]:
        """Best-effort coercion to list.

        Primary intent: consume typed DSPy output fields without asking for JSON.
        Fallback: if a provider returns strings, attempt JSON parse.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        # Pydantic v2 models
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()  # type: ignore[attr-defined]
                if isinstance(dumped, list):
                    return dumped
            except Exception:
                pass
        if isinstance(value, str):
            s = self._strip_markdown_fences(value).strip()
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return obj
            except Exception:
                # Best-effort line-based parsing to match format rules:
                #   one item per line; for list-of-objects, one JSON object per line.
                items: List[Any] = []
                for raw_line in s.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    # Strip common bullet/numbering prefixes
                    for prefix in ("- ", "* ", "• "):
                        if line.startswith(prefix):
                            line = line[len(prefix) :].strip()
                            break
                    line = re.sub(r"^\d+\.[ \t]+", "", line).strip()
                    if not line:
                        continue
                    # Try per-item JSON (objects or arrays)
                    if line.startswith(("{", "[")):
                        try:
                            parsed = json.loads(line)
                            items.append(parsed)
                            continue
                        except Exception:
                            pass
                    items.append(line)
                if items:
                    return items
                logger.debug("%s: could not parse list from string output", context)
                return []
        logger.debug("%s: could not coerce %s to list", context, type(value))
        return []

    def _coerce_optional_bool(
        self,
        value: Any,
        *,
        context: str,
        default: bool | None = None,
    ) -> bool | None:
        """Typed-first coercion for boolean outputs.

        DSPy signatures declare boolean OutputFields, but some backends/providers can still
        return strings. We deliberately do NOT treat arbitrary strings as truthy/falsey
        because bool("no") == True would silently corrupt validation.

        Returns:
          - bool if value is a bool
          - default if provided and value is None
          - None otherwise
        """
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        logger.debug("%s: expected bool output, got %s", context, type(value))
        return None

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
            raise JSONValidationError(
                f"{context}: JSON parse failed and json-repair not available. Install 'json-repair'."
            )
        try:
            repaired = repair_json(text)  # type: ignore[misc]
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            raise JSONValidationError(f"{context}: JSON parse failed after repair: {e}")
        raise JSONValidationError(f"{context}: Expected a JSON object.")

    def _extract_and_validate_json(
        self, result: Any, *, field_name: str = "result_json", context: str = ""
    ) -> Dict[str, Any]:
        """Extract JSON field from DSPy result object and validate using strict_json utility.
        This ensures Axiom A5: single‑line minified JSON contract.
        """
        raw = getattr(result, field_name, "")
        if not isinstance(raw, str):
            raise JSONValidationError(
                f"{context}: Expected string field {field_name}, got {type(raw)}"
            )
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise JSONValidationError(f"{context}: Expected JSON object, got {type(parsed)}")
        except json.JSONDecodeError as e:
            raise JSONValidationError(f"{context}: JSON decode failed: {e}") from e
        # Enforce single‑line deterministic representation (A5)
        minified = json.dumps(parsed, separators=(",", ":"), sort_keys=True)
        logger.debug("strict_json: minified output %s", minified)
        return parsed

    # NOTE: The original duplicate implementation of _normalize_split_decision (parent/task based) has been removed.
    # The canonical implementation is defined later in the file (task_payload based) and is used throughout.

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

        # Inject a stable, self-describing call contract into constraints.
        call_spec = self._CALL_SPECS.get(
            mode,
            {
                "name": str(mode),
                "purpose": "",
                "inputs": [],
                "outputs": [],
            },
        )
        c["call"] = {
            "name": str(call_spec.get("name") or mode),
            "purpose": str(call_spec.get("purpose") or ""),
            "inputs": list(call_spec.get("inputs") or []),
            "outputs": list(call_spec.get("outputs") or []),
        }

        preamble: List[str] = [
            f"Operation: {c['call']['name']}",
            (
                f"Purpose: {c['call']['purpose']}"
                if c["call"].get("purpose")
                else "Purpose: (not provided)"
            ),
            (
                "Inputs: " + ", ".join([str(x) for x in (c["call"].get("inputs") or [])])
                if c["call"].get("inputs")
                else "Inputs: (see signature inputs)"
            ),
            (
                "Outputs: " + ", ".join([str(x) for x in (c["call"].get("outputs") or [])])
                if c["call"].get("outputs")
                else "Outputs: (see signature outputs)"
            ),
            "Context: Use constraints.side_context if present; do not assume any other context.",
            "Return: Populate ONLY the signature output fields; no extra text.",
        ]

        # Ensure preamble is first and stable.
        rules = preamble + [r for r in rules if r not in preamble]

        c["format_rules"] = rules
        # Optional transparency for debugging
        c["_applied_rule_blocks"] = blocks
        return c

    def plan(self, idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured plan object via DSPy.

        Primary path: consume typed signature outputs (no JSON requested).
        Fallback: if a provider returns strings, attempt JSON parse.
        Includes a single retry if the first output lacks files.
        """

        def _predict_once(c: Dict[str, Any]) -> Dict[str, Any]:
            res = self._plan_predictor(idea=idea, constraints=c)  # type: ignore
            obj = self._coerce_to_dict(getattr(res, "plan", None), context="plan")
            if obj:
                return obj
            # Backward-compat: older signature name
            return self._coerce_to_dict(getattr(res, "plan_json", ""), context="plan:fallback")

        c1 = self._augment_constraints(constraints, "plan")
        obj = _predict_once(c1)
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
            obj = _predict_once(c2)
        return obj

    def validate_plan(
        self, *, idea: str, constraints: Dict[str, Any], plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a plan structure for separation of concerns and minimal quality."""
        result = self._plan_validate_predictor(  # type: ignore
            idea=idea,
            constraints=self._augment_constraints(constraints, "plan_validate"),
            plan=plan,
        )
        ok_val = getattr(result, "ok", None)
        issues_val = getattr(result, "issues", None)
        suggestions_val = getattr(result, "suggestions", None)

        if (
            isinstance(ok_val, bool)
            or isinstance(issues_val, list)
            or isinstance(suggestions_val, list)
        ):
            issues = issues_val if isinstance(issues_val, list) else []
            suggestions = suggestions_val if isinstance(suggestions_val, list) else []
            ok2 = self._coerce_optional_bool(ok_val, context="validate_plan:ok")
            ok = ok2 if isinstance(ok2, bool) else (len(issues) == 0)
        else:
            # Backward-compat: JSON blob
            try:
                obj = self._extract_and_validate_json(result, context="validate_plan")
            except JSONValidationError:
                return {
                    "ok": False,
                    "issues": ["invalid_validator_output"],
                    "suggestions": [],
                }
            ok = bool(obj.get("ok", False))
            issues = obj.get("issues", [])
            suggestions = obj.get("suggestions", [])
            if not isinstance(issues, list):
                issues = []
            if not isinstance(suggestions, list):
                suggestions = []
        return {
            "ok": ok,
            "issues": [str(x) for x in issues],
            "suggestions": [str(x) for x in suggestions],
        }

    def refine_plan(
        self, *, idea: str, constraints: Dict[str, Any], current_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Refine an existing plan into a more elaborate one."""
        result = self._plan_refine_predictor(  # type: ignore
            idea=idea,
            constraints=self._augment_constraints(constraints, "plan_refine"),
            current_plan=current_plan,
        )
        obj = self._coerce_to_dict(getattr(result, "plan", None), context="refine_plan")
        if obj:
            return obj
        # Backward-compat
        try:
            return self._extract_and_validate_json(
                result, field_name="plan_json", context="refine_plan"
            )
        except JSONValidationError:
            return current_plan

    def task_plan(self, idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a hierarchical task plan via DSPy.

        Primary path: consume typed signature outputs (no JSON requested).
        Fallback: parse JSON strings if a provider still returns them.
        """
        result = self._task_plan_predictor(
            idea=idea, constraints=self._augment_constraints(constraints, "task_plan")
        )  # type: ignore

        tasks_val = getattr(result, "tasks", None)
        tasks_list = self._coerce_to_list(tasks_val, context="task_plan")
        if tasks_list:
            return {"tasks": tasks_list}

        # Backward-compat: older JSON blob output
        content = getattr(result, "tasks_json", "")
        raw = self._strip_markdown_fences(content)
        try:
            q = json.loads(raw)
            if isinstance(q, dict):
                return q
            if isinstance(q, list):
                return {"tasks": q}
        except Exception:
            pass

        # Retry once with stronger instruction to populate output fields
        retry_constraints = self._augment_constraints(constraints, "task_plan")
        retry_rules = list(retry_constraints.get("format_rules", [])) + [
            "Your previous output did not populate the required output fields.",
            "Populate the 'tasks' output field with the full hierarchical task list.",
        ]
        retry_constraints["format_rules"] = retry_rules
        retry = self._task_plan_predictor(idea=idea, constraints=retry_constraints)  # type: ignore
        tasks_list2 = self._coerce_to_list(getattr(retry, "tasks", None), context="task_plan:retry")
        if tasks_list2:
            return {"tasks": tasks_list2}

        # Last resort: parse any JSON-like string
        retry_raw = self._strip_markdown_fences(getattr(retry, "tasks_json", ""))
        try:
            q2 = json.loads(retry_raw)
            if isinstance(q2, dict):
                return q2
            if isinstance(q2, list):
                return {"tasks": q2}
        except Exception:
            pass
        raise JSONValidationError("task_plan: unable to obtain tasks from DSPy outputs")

    def validate_task_plan(
        self, *, idea: str, constraints: Dict[str, Any], plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a hierarchical task plan for node_plan completeness, parent/leaf correctness, deps, and neutrality."""
        result = self._task_plan_validate_predictor(  # type: ignore
            idea=idea,
            constraints=self._augment_constraints(constraints, "task_plan_validate"),
            plan=plan,
        )

        ok_val = getattr(result, "ok", None)
        issues_val = getattr(result, "issues", None)
        suggestions_val = getattr(result, "suggestions", None)

        if (
            isinstance(ok_val, bool)
            or isinstance(issues_val, list)
            or isinstance(suggestions_val, list)
        ):
            issues = issues_val if isinstance(issues_val, list) else []
            suggestions = suggestions_val if isinstance(suggestions_val, list) else []
            ok2 = self._coerce_optional_bool(ok_val, context="validate_task_plan:ok")
            ok = ok2 if isinstance(ok2, bool) else (len(issues) == 0)
        else:
            # Backward-compat: JSON blob
            try:
                obj = self._extract_and_validate_json(result, context="validate_task_plan")
            except JSONValidationError:
                return {
                    "ok": False,
                    "issues": ["invalid_validator_output"],
                    "suggestions": [],
                }
            ok = bool(obj.get("ok", False))
            issues = obj.get("issues", [])
            suggestions = obj.get("suggestions", [])
            if not isinstance(issues, list):
                issues = []
            if not isinstance(suggestions, list):
                suggestions = []
        return {
            "ok": ok,
            "issues": [str(x) for x in issues],
            "suggestions": [str(x) for x in suggestions],
        }

    def _normalize_split_decision(
        self, obj: Dict[str, Any], task_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            cc.setdefault("kind", cc.get("type") or "composite")
            k = cc.get("kind")
            if not isinstance(k, str):
                cc["kind"] = "composite"
            else:
                k2 = k.strip()
                if k2 == "leaf":
                    k2 = "composite"
                if k2 not in ("composite", "code:function"):
                    k2 = "composite"
                cc["kind"] = k2
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

    def decide_split(
        self, task: Dict[str, Any], idea: str, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide whether to split or implement a task.
        Adds guardrails to ensure empty composite tasks are split into actionable children.
        """
        # Augment the task with hints for the LLM
        t = dict(task)
        is_leaf_composite = (t.get("kind") == "composite") and (not t.get("children"))
        if is_leaf_composite:
            t["must_split"] = True

        def _predict_once(task_payload: Dict[str, Any]) -> Dict[str, Any]:
            result = self._split_predictor(
                task=task_payload,
                idea=idea,
                constraints=self._augment_constraints(constraints, "split"),
            )  # type: ignore
            action_val = getattr(result, "action", None)
            children_val = getattr(result, "children", None)
            if isinstance(action_val, str) or isinstance(children_val, list):
                return {
                    "action": str(action_val or ""),
                    "children": self._coerce_to_list(children_val, context="decide_split"),
                }

            # Backward-compat: older JSON blob
            try:
                return self._extract_and_validate_json(
                    result, field_name="decision_json", context="decide_split"
                )
            except JSONValidationError:
                # Retry once with stronger instruction to populate output fields
                rc = self._augment_constraints(constraints, "split")
                rrules = list(rc.get("format_rules", [])) + [
                    "Your previous output did not populate the required output fields.",
                    "Populate 'action' with either split or implement, and populate 'children' when action=split.",
                ]
                rc["format_rules"] = rrules
                r = self._split_predictor(task=task_payload, idea=idea, constraints=rc)  # type: ignore
                action_val2 = getattr(r, "action", None)
                children_val2 = getattr(r, "children", None)
                if isinstance(action_val2, str) or isinstance(children_val2, list):
                    return {
                        "action": str(action_val2 or ""),
                        "children": self._coerce_to_list(
                            children_val2, context="decide_split:retry"
                        ),
                    }
                return self._extract_and_validate_json(
                    r, field_name="decision_json", context="decide_split:retry"
                )

        # First attempt
        obj = self._normalize_split_decision(_predict_once(t), t)
        action = obj.get("action")
        children = obj.get("children", []) if isinstance(obj, dict) else []
        # If the model refused to split an empty composite, retry once with a stronger hint
        if is_leaf_composite and (action != "split" or not children):
            t2 = dict(t)
            t2["force_split"] = True
            t2["detail"] = (
                "You must return a few meaningful children. Do not assume any specific programming language or implementation details unless present in constraints. Children should be minimal, independently actionable units for their kind."
            )
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
        classes: Dict[str, Dict[str, Any]] | None = None,
        constants: Dict[str, Dict[str, Any]] | None = None,
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
        enriched_constraints["file_metadata"].update(
            {
                "purpose": purpose or "",
                "description": description or "",
                "classes": classes or {},
                "constants": constants or {},
            }
        )

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

        def _env_float(
            name: str, default: float, lo: float | None = None, hi: float | None = None
        ) -> float:
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

        _max_retries = int(
            max_retries
            if max_retries is not None
            else _env_int("CRPB_GEN_MAX_RETRIES", 2, lo=0, hi=10)
        )
        _timeout_s = float(
            timeout_s if timeout_s is not None else _env_float("CRPB_GEN_TIMEOUT_S", 120.0, lo=5.0)
        )
        _backoff_base = float(
            backoff_base
            if backoff_base is not None
            else _env_float("CRPB_GEN_BACKOFF_BASE", 1.0, lo=0.1)
        )
        _backoff_max = float(
            backoff_max
            if backoff_max is not None
            else _env_float("CRPB_GEN_BACKOFF_MAX", 30.0, lo=0.5)
        )
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
                    raise TimeoutError(
                        f"LLM file generation timed out after {_timeout_s:.1f}s for file={file}"
                    ) from te
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
                delay = min(_backoff_base * (2**attempt), _backoff_max)
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

        ok_val = getattr(result, "ok", None)
        missing_val = getattr(result, "missing", None)
        if isinstance(ok_val, bool) or isinstance(missing_val, list):
            missing = missing_val if isinstance(missing_val, list) else []
            ok2 = self._coerce_optional_bool(ok_val, context="verify_exports:ok")
            ok = ok2 if isinstance(ok2, bool) else (len(missing) == 0)
            return {"ok": ok, "missing": [str(x) for x in missing]}

        # Backward-compat: JSON blob
        try:
            obj = self._extract_and_validate_json(
                result, field_name="result", context="verify_exports"
            )
        except JSONValidationError:
            return {"ok": False, "missing": exports}
        ok = bool(obj.get("ok", False))
        missing = obj.get("missing", [])
        if not isinstance(missing, list):
            missing = []
        return {"ok": ok, "missing": [str(x) for x in missing]}

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

        ok_val = getattr(result, "ok", None)
        issues_val = getattr(result, "issues", None)
        warnings_val = getattr(result, "warnings", None)
        if (
            isinstance(ok_val, bool)
            or isinstance(issues_val, list)
            or isinstance(warnings_val, list)
        ):
            issues = issues_val if isinstance(issues_val, list) else []
            warnings = warnings_val if isinstance(warnings_val, list) else []
            ok2 = self._coerce_optional_bool(ok_val, context="validate_artifact:ok")
            ok = ok2 if isinstance(ok2, bool) else (len(issues) == 0)
            return {
                "ok": ok,
                "issues": [str(x) for x in issues],
                "warnings": [str(x) for x in warnings],
                "normalized": self._coerce_to_dict(
                    getattr(result, "normalized", None), context="validate_artifact:normalized"
                ),
            }

        # Backward-compat: JSON blob
        try:
            obj = self._extract_and_validate_json(
                result, field_name="result", context="validate_artifact"
            )
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
        return {
            "ok": ok,
            "issues": issues,
            "warnings": warnings,
            "normalized": obj.get("normalized"),
        }

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
            # Primary path: structured output field
            t = self._coerce_to_dict(getattr(result, "task_out", None), context="clarify_task")
            if t:
                return t
            # Backward-compat: JSON blob
            obj = self._extract_and_validate_json(result, context="clarify_task")
            if isinstance(obj, dict):
                t2 = obj.get("task")
                if isinstance(t2, dict):
                    return t2
                return obj
        except JSONValidationError:
            raise
        except Exception:
            raise

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
            writes = self._coerce_to_list(
                getattr(result, "writes", None), context="merge_subtasks:writes"
            )
            arts = self._coerce_to_list(
                getattr(result, "artifacts_out", None), context="merge_subtasks:artifacts"
            )
            if writes or arts:
                return {"writes": writes, "artifacts": arts}
            # Backward-compat
            obj = self._extract_and_validate_json(
                result, field_name="merge_json", context="merge_subtasks"
            )
            if not isinstance(obj.get("writes"), list):
                obj["writes"] = []
            if not isinstance(obj.get("artifacts"), list):
                obj["artifacts"] = []
            return obj
        except JSONValidationError:
            raise
        except Exception:
            raise

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
            edits = self._coerce_to_list(getattr(result, "edits", None), context="amend_task_plan")
            if edits:
                return {"edits": edits}
            # Backward-compat
            obj = self._extract_and_validate_json(
                result, field_name="edits_json", context="amend_task_plan"
            )
            if not isinstance(obj.get("edits"), list):
                obj["edits"] = []
            return obj
        except JSONValidationError:
            return {"edits": []}
        except Exception:
            raise

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
        ok_val = getattr(result, "ok", None)
        issues_val = getattr(result, "issues", None)
        warnings_val = getattr(result, "warnings", None)
        suggestions_val = getattr(result, "suggestions", None)

        # Typed-only: do not parse JSON fallbacks or treat strings as structured.
        # If a backend returns non-typed outputs, fail loudly with a machine-coded issue.
        if not isinstance(issues_val, list) or not isinstance(warnings_val, list) or not isinstance(
            suggestions_val, list
        ):
            return {
                "ok": False,
                "issues": ["invalid_validator_output"],
                "warnings": [],
                "suggestions": [],
            }

        issues = [str(x) for x in issues_val if str(x).strip()]
        warnings = [str(x) for x in warnings_val if str(x).strip()]
        suggestions = [str(x) for x in suggestions_val if str(x).strip()]

        ok2 = self._coerce_optional_bool(ok_val, context="project_validate:ok")
        ok = ok2 if isinstance(ok2, bool) else (len(issues) == 0)
        return {
            "ok": ok,
            "issues": [str(x) for x in issues],
            "warnings": [str(x) for x in warnings],
            "suggestions": [str(x) for x in suggestions],
        }

    def node_validate(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        node: Dict[str, Any],
        parent: Dict[str, Any],
        siblings: List[Dict[str, Any]],
        children: List[Dict[str, Any]],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a single plan node bottom-up against a bounded file subset."""
        result = self._node_validate_predictor(  # type: ignore
            idea=idea or "",
            constraints=self._augment_constraints(constraints, "node_validate"),
            node=node or {},
            parent=parent or {},
            siblings=siblings or [],
            children=children or [],
            files=files or {},
            file_specs=file_specs or {},
        )

        ok_val = getattr(result, "ok", None)
        issues_val = getattr(result, "issues", None)
        warnings_val = getattr(result, "warnings", None)
        suggestions_val = getattr(result, "suggestions", None)

        if not isinstance(issues_val, list) or not isinstance(warnings_val, list) or not isinstance(
            suggestions_val, list
        ):
            return {
                "ok": False,
                "issues": ["invalid_node_validator_output"],
                "warnings": [],
                "suggestions": [],
            }

        issues = [str(x) for x in issues_val if str(x).strip()]
        warnings = [str(x) for x in warnings_val if str(x).strip()]
        suggestions = [str(x) for x in suggestions_val if str(x).strip()]
        ok2 = self._coerce_optional_bool(ok_val, context="node_validate:ok")
        ok = ok2 if isinstance(ok2, bool) else (len(issues) == 0)

        return {
            "ok": ok,
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions,
        }

    def validation_questions(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        plan: Dict[str, Any],
        findings: List[str],
    ) -> Dict[str, Any]:
        """Generate probing questions to drive clarification/repair."""
        result = self._validation_questions_predictor(  # type: ignore
            idea=idea or "",
            constraints=self._augment_constraints(constraints, "validation_questions"),
            plan=plan or {},
            findings=[str(x) for x in (findings or []) if str(x).strip()],
        )
        questions_val = getattr(result, "questions", None)

        questions: List[Dict[str, Any]] = []
        if isinstance(questions_val, list):
            for q in questions_val:
                if not isinstance(q, dict):
                    continue
                qid = q.get("id")
                question = q.get("question")
                if not (isinstance(qid, str) and qid.strip() and isinstance(question, str) and question.strip()):
                    continue
                questions.append(
                    {
                        "id": qid.strip(),
                        "question": question.strip(),
                        "focus": str(q.get("focus") or ""),
                        "why_this_matters": str(q.get("why_this_matters") or ""),
                        "expected_answer_shape": str(q.get("expected_answer_shape") or ""),
                    }
                )

        return {"questions": questions}

    def summarize_findings(
        self,
        *,
        findings: List[str],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize findings into a bounded memory capsule.

        Returns: {"summary": str, "key_issues": [str, ...]}
        """
        result = self._findings_summarize_predictor(  # type: ignore
            findings=[str(x) for x in (findings or []) if str(x).strip()],
            constraints=self._augment_constraints(constraints, "findings_summarize"),
        )
        summary_val = getattr(result, "summary", None)
        key_val = getattr(result, "key_issues", None)

        summary = str(summary_val or "").strip()
        key_issues = key_val if isinstance(key_val, list) else []
        key_issues = [str(x) for x in key_issues if str(x).strip()]
        # Safety cap regardless of model behavior.
        if len(key_issues) > 30:
            key_issues = key_issues[:30]
        if len(summary) > 1200:
            summary = summary[:1200]

        return {"summary": summary, "key_issues": key_issues}

    def repair_project(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        issues: List[Dict[str, Any]],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LM to propose concrete file edits to fix validation issues.

        Returns:
        - {"edits": [{"path": str, "new_text": str, "rationale": str}], "notes": [str]}

        This method is language-agnostic: the LM decides language/stack based on file paths,
        file specs, and the user idea/constraints.
        """
        result = self._repair_project_predictor(  # type: ignore
            idea=idea or "",
            constraints=self._augment_constraints(constraints, "repair_project"),
            issues=issues or [],
            files=files or {},
            file_specs=file_specs or {},
            context=context or {},
        )

        edits_val = getattr(result, "edits", None)
        notes_val = getattr(result, "notes", None)
        edits: List[Dict[str, Any]] = []
        notes: List[str] = []

        if isinstance(edits_val, list):
            for e in edits_val:
                if isinstance(e, dict):
                    p = e.get("path")
                    t = e.get("new_text")
                    if isinstance(p, str) and isinstance(t, str):
                        edits.append(
                            {
                                "path": p,
                                "new_text": t,
                                "rationale": str(e.get("rationale") or ""),
                            }
                        )
        if isinstance(notes_val, list):
            notes = [str(x) for x in notes_val if str(x)]

        return {"edits": edits, "notes": notes}

    def repair_project_patches(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        issues: List[Dict[str, Any]],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LM to propose minimal patch operations instead of full-file rewrites."""
        result = self._repair_project_patches_predictor(  # type: ignore
            idea=idea or "",
            constraints=self._augment_constraints(constraints, "repair_project_patches"),
            issues=issues or [],
            files=files or {},
            file_specs=file_specs or {},
            context=context or {},
        )

        # Typed-only: require list[dict] outputs.
        patches_val = getattr(result, "patches", None)
        notes_val = getattr(result, "notes", None)
        needs_val = getattr(result, "needs_paths", None)

        patches: List[Dict[str, Any]] = []
        parsed_raw: Any = patches_val
        if not isinstance(parsed_raw, list):
            # Some DSPy backends may return JSON-like strings; parse defensively.
            try:
                if isinstance(parsed_raw, str) and parsed_raw.strip():
                    s = self._strip_markdown_fences(parsed_raw)
                    obj: Any = None
                    try:
                        obj = json.loads(s)
                    except Exception:
                        if repair_json is not None:
                            try:
                                obj = json.loads(repair_json(s))
                            except Exception:
                                obj = None
                    # Accept either a raw list or an object with "patches".
                    if isinstance(obj, dict) and isinstance(obj.get("patches"), list):
                        parsed_raw = obj.get("patches")
                    elif isinstance(obj, list):
                        parsed_raw = obj
            except Exception:
                parsed_raw = patches_val

        if isinstance(parsed_raw, list):
            for p in parsed_raw:
                if not isinstance(p, dict):
                    continue
                path = p.get("path")
                find = p.get("find")
                replace = p.get("replace")
                if not (isinstance(path, str) and path.strip()):
                    continue
                if not isinstance(find, str):
                    continue
                if not isinstance(replace, str):
                    continue

                allow_multiple_raw = p.get("allow_multiple", False)
                allow_multiple = (
                    allow_multiple_raw if isinstance(allow_multiple_raw, bool) else False
                )

                count_raw = p.get("count", 1)
                count = count_raw if isinstance(count_raw, int) and count_raw > 0 else 1

                patches.append(
                    {
                        "path": path.strip(),
                        "find": find,
                        "replace": replace,
                        "rationale": str(p.get("rationale") or ""),
                        "allow_multiple": allow_multiple,
                        "count": count,
                    }
                )

        notes: List[str] = []
        if isinstance(notes_val, list):
            notes = [str(x) for x in notes_val if str(x).strip()]

        needs_paths: List[str] = []
        parsed_needs: Any = needs_val
        if isinstance(parsed_needs, str) and parsed_needs.strip():
            try:
                s = self._strip_markdown_fences(parsed_needs)
                obj: Any = None
                try:
                    obj = json.loads(s)
                except Exception:
                    if repair_json is not None:
                        try:
                            obj = json.loads(repair_json(s))
                        except Exception:
                            obj = None
                if isinstance(obj, list):
                    parsed_needs = obj
                elif isinstance(obj, dict) and isinstance(obj.get("needs_paths"), list):
                    parsed_needs = obj.get("needs_paths")
            except Exception:
                parsed_needs = needs_val

        if isinstance(parsed_needs, list):
            for it in parsed_needs:
                s = str(it or "").strip().replace("\\", "/")
                if s:
                    needs_paths.append(s)
        needs_paths = list(dict.fromkeys(needs_paths))[:12]

        return {"patches": patches, "notes": notes, "needs_paths": needs_paths}

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
                text_in=text or "",
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "micro_adjust"),
            )

            text_out = getattr(result, "text", None)
            notes_out = getattr(result, "notes", None)
            if isinstance(text_out, str):
                notes = notes_out if isinstance(notes_out, list) else []
                return {"text": text_out, "notes": [str(x) for x in notes]}

            # Backward-compat: JSON blob
            obj = self._extract_and_validate_json(
                result, field_name="result", context="micro_adjust"
            )
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                if not isinstance(obj.get("notes"), list):
                    obj["notes"] = []
                return obj
        except JSONValidationError:
            return {"text": text, "notes": []}
        except Exception:
            raise
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
            patch_val = getattr(result, "patch", None)
            if isinstance(patch_val, dict) and patch_val:
                return patch_val
            # Typed-only: do not parse JSON/string fallbacks.
            return {}
        except Exception:
            raise
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
        Returns {"questions": object[], "monologue": string} on success or safe defaults on failure.
        """
        def _min_questions(c: Dict[str, Any]) -> int:
            try:
                v = c.get("socratic_min_questions") if isinstance(c, dict) else None
                if v is None:
                    return 10
                return max(0, min(50, int(v)))
            except Exception:
                return 10

        def _run_once(extra_rules: List[str] | None = None) -> Dict[str, Any]:
            c2: Dict[str, Any] = dict(constraints or {})
            if extra_rules:
                fr = c2.get("format_rules")
                if not isinstance(fr, list):
                    fr = []
                # Add extra rules as late-binding constraints (highest priority).
                c2["format_rules"] = list(fr) + [str(r) for r in extra_rules if str(r)]
            result = self._socratic_predictor(  # type: ignore
                node=node or {},
                parent=parent or {},
                siblings=siblings or [],
                idea=idea or "",
                constraints=self._augment_constraints(c2, "socratic"),
            )
            questions_val = getattr(result, "questions", None)
            monologue_val = getattr(result, "monologue", None)
            if isinstance(questions_val, list) or isinstance(monologue_val, str):
                questions = questions_val if isinstance(questions_val, list) else []
                monologue = monologue_val if isinstance(monologue_val, str) else ""
            else:
                obj = self._extract_and_validate_json(result, context="socratic")
                questions = obj.get("questions") if isinstance(obj, dict) else []
                monologue = obj.get("monologue") if isinstance(obj, dict) else ""

            if not isinstance(monologue, str):
                monologue = ""

            def _qid(qtxt: str) -> str:
                return hashlib.sha1((qtxt or "").encode("utf-8")).hexdigest()[:12]

            q_out: List[Dict[str, Any]] = []
            if isinstance(questions, list):
                for q in questions:
                    if isinstance(q, dict):
                        qtxt = str(q.get("question") or "").strip()
                        if not qtxt:
                            continue
                        q_out.append(
                            {
                                "id": str(q.get("id") or _qid(qtxt)),
                                "question": qtxt,
                                "focus": str(q.get("focus") or "").strip(),
                                "why_this_matters": str(q.get("why_this_matters") or "").strip(),
                                "expected_answer_shape": str(
                                    q.get("expected_answer_shape") or ""
                                ).strip(),
                            }
                        )
                    elif isinstance(q, str) and q.strip():
                        # Back-compat: older providers may return strings.
                        qtxt = q.strip()
                        q_out.append(
                            {
                                "id": _qid(qtxt),
                                "question": qtxt,
                                "focus": "",
                                "why_this_matters": "",
                                "expected_answer_shape": "",
                            }
                        )


            return {"questions": q_out, "monologue": monologue}

        try:
            out = _run_once(extra_rules=None)
            if len(out.get("questions") or []) < _min_questions(constraints):
                out2 = _run_once(
                    extra_rules=[
                        f"You returned too few questions. Return AT LEAST {_min_questions(constraints)} questions.",
                        "Ensure coverage across purpose/scope/inputs/outputs/interfaces/dependencies/testing/risks/integration.",
                        "Do not repeat the same question; each must add new information.",
                    ]
                )
                # If retry is better, take it.
                if len(out2.get("questions") or []) >= len(out.get("questions") or []):
                    out = out2
            return out
        except Exception:
            return {"questions": [], "monologue": ""}

    def join_judge_node(
        self,
        *,
        node: Dict[str, Any],
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Judge a node's local decomposition/interfaces via a capsule.

        The authoritative capsule is expected in constraints.side_context.join_judge.
        Returns: {ok:bool, actions:list[dict], rubric:dict, questions:list[dict], notes:list[str]}
        """
        try:
            res = self._join_judge_predictor(  # type: ignore
                node=node or {},
                idea=idea or "",
                constraints=self._augment_constraints(constraints, "join_judge"),
            )
            ok_val = getattr(res, "ok", None)
            actions_val = getattr(res, "actions", None)
            rubric_val = getattr(res, "rubric", None)
            questions_val = getattr(res, "questions", None)
            notes_val = getattr(res, "notes", None)

            ok2 = self._coerce_optional_bool(ok_val, context="join_judge:ok")
            ok = ok2 if isinstance(ok2, bool) else True
            actions = self._coerce_to_list(actions_val, context="join_judge:actions")
            rubric = self._coerce_to_dict(rubric_val, context="join_judge:rubric")
            questions = self._coerce_to_list(questions_val, context="join_judge:questions")
            notes_raw = self._coerce_to_list(notes_val, context="join_judge:notes")

            def _qid(qtxt: str) -> str:
                return hashlib.sha1((qtxt or "").encode("utf-8")).hexdigest()[:12]

            q_out: List[Dict[str, Any]] = []
            for q in questions:
                if isinstance(q, dict):
                    qtxt = str(q.get("question") or "").strip()
                    if not qtxt:
                        continue
                    q_out.append(
                        {
                            "id": str(q.get("id") or _qid(qtxt)),
                            "question": qtxt,
                            "focus": str(q.get("focus") or "").strip(),
                            "why_this_matters": str(q.get("why_this_matters") or "").strip(),
                            "expected_answer_shape": str(q.get("expected_answer_shape") or "").strip(),
                        }
                    )
                elif isinstance(q, str) and q.strip():
                    qtxt = q.strip()
                    q_out.append(
                        {
                            "id": _qid(qtxt),
                            "question": qtxt,
                            "focus": "",
                            "why_this_matters": "",
                            "expected_answer_shape": "",
                        }
                    )

            actions_out: List[Dict[str, Any]] = []
            for a in actions:
                if not isinstance(a, dict):
                    continue
                op = str(a.get("op") or "").strip()
                if op not in ("edit_task", "add_child", "add_dep", "rewire_artifacts"):
                    continue
                actions_out.append(a)

            notes: List[str] = []
            for n in notes_raw:
                if isinstance(n, str) and n.strip():
                    notes.append(n.strip())

            return {
                "ok": bool(ok),
                "actions": actions_out,
                "rubric": rubric if isinstance(rubric, dict) else {},
                "questions": q_out,
                "notes": notes,
            }
        except Exception:
            return {"ok": True, "actions": [], "rubric": {}, "questions": [], "notes": []}

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
                    "imports": [
                        str(x).strip()
                        for x in (it.get("imports") or [])
                        if isinstance(x, str) and x.strip()
                    ],
                    "exports": [
                        str(x).strip()
                        for x in (it.get("exports") or [])
                        if isinstance(x, str) and x.strip()
                    ],
                    "functions": {},
                    "classes": {},
                    "constants": {},
                    "entrypoint": (
                        str(it.get("entrypoint")).strip()
                        if isinstance(it.get("entrypoint"), str)
                        and str(it.get("entrypoint")).strip()
                        else None
                    ),
                    "content": (
                        str(it.get("content")).strip()
                        if isinstance(it.get("content"), str) and str(it.get("content")).strip()
                        else None
                    ),
                    "exports_detail": [],
                }
                fn = it.get("functions")
                if isinstance(fn, dict):
                    safe_fn: Dict[str, Any] = {}
                    for k, v in fn.items():
                        if isinstance(k, str) and k.strip():
                            kk = k.strip()
                            # CodeSpec schema expects objects for function entries.
                            if isinstance(v, dict):
                                safe_fn[kk] = v
                            elif isinstance(v, str):
                                safe_fn[kk] = {"signature": v}
                            else:
                                safe_fn[kk] = {"signature": str(v)}
                    entry["functions"] = safe_fn
                cl = it.get("classes")
                if isinstance(cl, dict):
                    safe_cl: Dict[str, Any] = {}
                    for k, v in cl.items():
                        if isinstance(k, str) and k.strip():
                            kk = k.strip()
                            # CodeSpec schema expects objects for class entries.
                            safe_cl[kk] = v if isinstance(v, dict) else {"description": str(v)}
                    entry["classes"] = safe_cl
                co = it.get("constants")
                if isinstance(co, dict):
                    safe_co: Dict[str, Any] = {}
                    for k, v in co.items():
                        if isinstance(k, str) and k.strip():
                            kk = k.strip()
                            # CodeSpec schema expects objects for constant entries.
                            safe_co[kk] = v if isinstance(v, dict) else {"value": v}
                    entry["constants"] = safe_co
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

        files_val = getattr(result, "files", None)
        if isinstance(files_val, list) and files_val:
            return {"files": self._normalize_codespec_files(files_val)}

        # Typed-only: do not parse JSON/string fallbacks.
        return {"files": []}
