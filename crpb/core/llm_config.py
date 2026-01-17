from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

# Project-level persisted selection file. Lives at repo root by default.
CONFIG_FILENAME = ".crpb_llm.json"

Provider = Literal[
    "openai",
    "anthropic",
    "huggingface",
    "server",
    "local",
    "cerebras",
    "azure_foundry",
]


class ProviderEnum(str, Enum):
    """Enum version of Provider for CLI parsing (Typer/Click compatible)."""

    openai = "openai"
    anthropic = "anthropic"
    huggingface = "huggingface"
    server = "server"
    local = "local"
    cerebras = "cerebras"
    azure_foundry = "azure_foundry"


# Centralized provider metadata (no secrets here; API keys referenced by name only)
# - required_fields: non-env attributes that must be present on LLMSelection
# - default_model_env: env var name that provides an optional default model
# - token_aliases: alternative env var names for the same token/credential
# - optional_keys: optional env keys users might set, not required by CRPB
PROVIDER_META: Dict[str, Dict[str, Any]] = {
    "openai": {
        "required_fields": [],
        "default_model_env": "CRPB_OPENAI_MODEL",
        "token_aliases": [],
        "optional_keys": ["OPENAI_API_KEY"],  # required by provider, not by CRPB logic here
    },
    "anthropic": {
        "required_fields": [],
        "default_model_env": "CRPB_ANTHROPIC_MODEL",
        "token_aliases": [],
        "optional_keys": ["ANTHROPIC_API_KEY"],
    },
    "huggingface": {
        "required_fields": ["model"],
        "default_model_env": "CRPB_HF_MODEL",
        "token_aliases": ["HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"],
        "optional_keys": ["HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"],
    },
    "server": {
        "required_fields": ["base_url", "model"],
        "default_model_env": None,
        "token_aliases": [],
        # Allow users to set a server-side API key if their server requires one
        "optional_keys": ["CRPB_SERVER_API_KEY", "SERVER_API_KEY"],
    },
    "local": {
        "required_fields": ["model"],
        "default_model_env": None,
        "token_aliases": [],
        # Not required, but allow an opt-in key name for local runtimes that support auth
        "optional_keys": ["CRPB_LOCAL_API_KEY", "LOCAL_API_KEY"],
    },
    "cerebras": {
        "required_fields": [],
        "default_model_env": "CRPB_CEREBRAS_MODEL",
        "token_aliases": [],
        "optional_keys": ["CEREBRAS_API_KEY"],
    },
    "azure_foundry": {
        "required_fields": [],
        # Model here means deployment name for Azure AI Foundry (OpenAI-compatible).
        "default_model_env": "CRPB_AZURE_FOUNDRY_DEPLOYMENT",
        "token_aliases": [],
        # Keep env var names explicit; CRPB never persists secrets.
        "optional_keys": [
            "CRPB_AZURE_FOUNDRY_API_KEY",
            "CRPB_AZURE_FOUNDRY_ENDPOINT",
            "CRPB_AZURE_FOUNDRY_API_VERSION",
        ],
    },
}


@dataclass
class LLMSelection:
    """Represents chosen LLM provider and model/config.

    provider:
      - "openai": uses OPENAI_API_KEY
      - "anthropic": uses ANTHROPIC_API_KEY
      - "cerebras": uses CEREBRAS_API_KEY (fast AI inference)
      - "huggingface": token optional; uses HUGGINGFACEHUB_API_TOKEN/HF_TOKEN if needed
      - "server": a self-hosted HTTP server (e.g., Ollama, LM Studio). Provide base_url.
      - "local": a local/offline model; no API keys; requires a model name string.

    model: optional model identifier; required for provider=="huggingface", provider=="local", and provider=="server".
    base_url: for provider=="server" (e.g., http://localhost:11434 for Ollama or LM Studio proxy URL).
    extra: optional provider-specific fields (kept opaque for forward compatibility).
    """

    provider: Provider
    model: Optional[str] = None
    base_url: Optional[str] = None
    extra: Dict[str, Any] | None = None
    # Feature flags (non-secret). Default is conservative/off.
    embeddings_enabled: bool = False


def _parse_bool(s: object) -> Optional[bool]:
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    v = str(s).strip().lower()
    if not v:
        return None
    if v in ("1", "true", "t", "yes", "y", "on", "enable", "enabled"):
        return True
    if v in ("0", "false", "f", "no", "n", "off", "disable", "disabled"):
        return False
    return None


def _config_path(base: Path | None = None) -> Path:
    base = Path(base) if base else Path.cwd()
    return base / CONFIG_FILENAME


def load_selection(base: Path | None = None) -> Optional[LLMSelection]:
    """Load a persisted selection if present, else None."""
    p = _config_path(base)
    prov = os.environ.get("CRPB_LLM_PROVIDER")
    if isinstance(prov, str) and prov.strip():
        prov2 = prov.strip().lower()
        if prov2 in (
            "openai",
            "anthropic",
            "huggingface",
            "server",
            "local",
            "cerebras",
            "azure_foundry",
        ):
            model = os.environ.get("CRPB_LLM_MODEL")
            base_url = os.environ.get("CRPB_LLM_BASE_URL")
            embed_env = _parse_bool(os.environ.get("CRPB_EMBEDDINGS_ENABLE"))
            return LLMSelection(
                provider=prov2,
                model=(model.strip() if isinstance(model, str) and model.strip() else None),
                base_url=(
                    base_url.strip() if isinstance(base_url, str) and base_url.strip() else None
                ),
                extra=None,
                embeddings_enabled=bool(embed_env) if embed_env is not None else False,
            )

    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        provider = data.get("provider")
        model = data.get("model")
        base_url = data.get("base_url")
        extra = data.get("extra") or None
        emb = data.get("embeddings_enabled")
        if provider not in (
            "openai",
            "anthropic",
            "huggingface",
            "server",
            "local",
            "cerebras",
            "azure_foundry",
        ):
            return None
        emb2 = _parse_bool(emb)
        return LLMSelection(
            provider=provider,
            model=model,
            base_url=base_url,
            extra=extra,
            embeddings_enabled=bool(emb2) if emb2 is not None else False,
        )
    except Exception:
        return None


def load_saved_selection(base: Path | None = None) -> Optional[LLMSelection]:
    """Load a persisted selection from .crpb_llm.json, ignoring env overrides.

    This is useful for diagnostics: CRPB runtime prefers environment-driven selection when
    CRPB_LLM_PROVIDER is set, but users may still want to inspect what is saved.
    """
    p = _config_path(base)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        provider = data.get("provider")
        model = data.get("model")
        base_url = data.get("base_url")
        extra = data.get("extra") or None
        emb = data.get("embeddings_enabled")
        if provider not in (
            "openai",
            "anthropic",
            "huggingface",
            "server",
            "local",
            "cerebras",
            "azure_foundry",
        ):
            return None
        emb2 = _parse_bool(emb)
        return LLMSelection(
            provider=provider,
            model=model,
            base_url=base_url,
            extra=extra,
            embeddings_enabled=bool(emb2) if emb2 is not None else False,
        )
    except Exception:
        return None


def embeddings_enabled(*, base: Path | None = None) -> bool:
    """Return whether embeddings features are enabled.

    Precedence:
    1) CRPB_EMBEDDINGS_ENABLE env var if set to a recognizable boolean.
    2) Saved selection (.crpb_llm.json) field `embeddings_enabled`.
    3) Default: False.
    """
    v = _parse_bool(os.environ.get("CRPB_EMBEDDINGS_ENABLE"))
    if v is not None:
        return bool(v)
    saved = load_saved_selection(base=base)
    if saved is not None:
        return bool(getattr(saved, "embeddings_enabled", False))
    return False


def save_selection(sel: LLMSelection, base: Path | None = None) -> Path:
    """Persist the selection to JSON. Overwrites the previous selection.
    Returns the path written.
    """
    p = _config_path(base)
    p.write_text(json.dumps(asdict(sel), indent=2), encoding="utf-8")
    return p


def require_env_vars(sel: LLMSelection) -> Tuple[bool, list[str]]:
    """Return (ok, missing_env_vars) for the chosen provider.
    Does not modify environment.
    """
    need: list[str] = []
    if sel.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            need.append("OPENAI_API_KEY")
    elif sel.provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            need.append("ANTHROPIC_API_KEY")
    elif sel.provider == "cerebras":
        if not os.environ.get("CEREBRAS_API_KEY"):
            need.append("CEREBRAS_API_KEY")
    elif sel.provider == "huggingface":
        # Accept model from selection or from provider-specific env default to keep models out of JSON
        default_env = PROVIDER_META["huggingface"]["default_model_env"]
        model_from_env = os.environ.get(default_env) if default_env else None
        if not (
            (sel.model and sel.model.strip()) or (model_from_env and str(model_from_env).strip())
        ):
            need.append(f"model (e.g., --model 'google/flan-t5-base' or set {default_env})")
    elif sel.provider == "azure_foundry":
        # CRPB treats Azure AI Foundry as an OpenAI-compatible endpoint with explicit env config.
        # Require explicit endpoint, api key, and api version (no hidden defaults).
        if not os.environ.get("CRPB_AZURE_FOUNDRY_API_KEY"):
            need.append("CRPB_AZURE_FOUNDRY_API_KEY")
        if not os.environ.get("CRPB_AZURE_FOUNDRY_ENDPOINT"):
            need.append("CRPB_AZURE_FOUNDRY_ENDPOINT")
        if not os.environ.get("CRPB_AZURE_FOUNDRY_API_VERSION"):
            need.append("CRPB_AZURE_FOUNDRY_API_VERSION")
        default_env = PROVIDER_META["azure_foundry"]["default_model_env"]
        model_from_env = os.environ.get(default_env) if default_env else None
        if not (
            (sel.model and str(sel.model).strip())
            or (model_from_env and str(model_from_env).strip())
        ):
            need.append(f"deployment (e.g., --model '<deployment>' or set {default_env})")
    elif sel.provider == "server":
        # Typically no API key is strictly required; allow user to secure server as needed.
        if not (sel.base_url and sel.base_url.strip()):
            need.append("base_url (e.g., http://localhost:11434)")
        if not (sel.model and str(sel.model).strip()):
            need.append("model (server-side model name/id)")
    elif sel.provider == "local":
        # No keys, but a local model name is required so downstream knows which to use.
        if not (sel.model and sel.model.strip()):
            need.append("model (local model id/name)")
    return (len(need) == 0, need)


def env_values_for(sel: LLMSelection) -> Dict[str, str]:
    """Read relevant env vars for provider. Returns a dict of key->value for present keys only.
    This function does not validate or raise.
    """
    out: Dict[str, str] = {}
    if sel.provider == "openai":
        v = os.environ.get("OPENAI_API_KEY")
        if v:
            out["OPENAI_API_KEY"] = v
    elif sel.provider == "anthropic":
        v = os.environ.get("ANTHROPIC_API_KEY")
        if v:
            out["ANTHROPIC_API_KEY"] = v
    elif sel.provider == "cerebras":
        v = os.environ.get("CEREBRAS_API_KEY")
        if v:
            out["CEREBRAS_API_KEY"] = v
    elif sel.provider == "huggingface":
        v = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        if v:
            out["HUGGINGFACEHUB_API_TOKEN"] = v
    elif sel.provider == "azure_foundry":
        # Only expose recognized names if present.
        for k in (
            "CRPB_AZURE_FOUNDRY_API_KEY",
            "CRPB_AZURE_FOUNDRY_ENDPOINT",
            "CRPB_AZURE_FOUNDRY_API_VERSION",
            "CRPB_AZURE_FOUNDRY_DEPLOYMENT",
        ):
            v = os.environ.get(k)
            if v:
                out[k] = v
    elif sel.provider in ("server", "local"):
        # Optional keys: expose recognized names if present
        keys = PROVIDER_META[sel.provider]["optional_keys"]
        # Prefer first (canonical) key in output, even if value comes from an alias
        primary = keys[0] if keys else None
        if primary:
            value = os.environ.get(primary)
            if not value and len(keys) > 1:
                for alias in keys[1:]:
                    value = os.environ.get(alias)
                    if value:
                        break
            if value:
                out[primary] = value
    return out


def effective_model(
    sel: LLMSelection,
    override: Optional[str] = None,
    default_model: Optional[str] = None,
) -> Optional[str]:
    """Pick the effective model string to use for this selection.
    - override wins if provided.
    - else sel.model if provided.
    - else default_model (e.g., from provider-specific env via config.default_model_for).
    Some providers (huggingface, local) effectively require a model.
    """
    if override and override.strip():
        return override.strip()
    if sel.model and str(sel.model).strip():
        return str(sel.model).strip()
    return (
        default_model.strip() if isinstance(default_model, str) and default_model.strip() else None
    )


def configuration_warning_if_unset(base: Path | None = None) -> Optional[str]:
    """Return a provider-neutral warning message if no LLM is configured.

    Conditions checked:
    - No saved selection file found.
    - No provider API key or model default env vars present across known providers.
    """
    sel = load_selection(base=base)
    if sel:
        return None
    # Any indicative envs present?
    indicative_envs = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CEREBRAS_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
        "CRPB_OPENAI_MODEL",
        "CRPB_ANTHROPIC_MODEL",
        "CRPB_CEREBRAS_MODEL",
        "CRPB_HF_MODEL",
        "CRPB_AZURE_FOUNDRY_API_KEY",
        "CRPB_AZURE_FOUNDRY_ENDPOINT",
        "CRPB_AZURE_FOUNDRY_API_VERSION",
        "CRPB_AZURE_FOUNDRY_DEPLOYMENT",
        # Optional keys for server/local
        "CRPB_SERVER_API_KEY",
        "SERVER_API_KEY",
        "CRPB_LOCAL_API_KEY",
        "LOCAL_API_KEY",
    ]
    for k in indicative_envs:
        if os.environ.get(k):
            return None
    return (
        "No LLM provider configured. Run `python -m crpb llm choose` to select a provider, or set any provider key or model env var "
        "(e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, CEREBRAS_API_KEY, CRPB_OPENAI_MODEL, CRPB_ANTHROPIC_MODEL, CRPB_CEREBRAS_MODEL, CRPB_HF_MODEL — tokens optional). "
        "Note: server/local providers do not require API keys; CRPB_SERVER_API_KEY/CRPB_LOCAL_API_KEY are optional."
    )


# Optional utility: construct a generic descriptor for downstream engines.
# We DO NOT import or depend on DSPy here to keep this module focused on configuration only.


def as_descriptor(sel: LLMSelection, model: Optional[str] = None) -> Dict[str, Any]:
    """Return a simple provider-agnostic descriptor suitable for an engine factory.
    Example shape:
      {
        "provider": "openai|anthropic|huggingface|server|local",
        "model": "...",           # when applicable
        "base_url": "...",        # for server
        "env": {"KEY": "..."}    # relevant keys present in process env
      }
    """
    desc: Dict[str, Any] = {
        "provider": sel.provider,
        "model": model if model is not None else sel.model,
        "base_url": sel.base_url,
        "env": env_values_for(sel),
    }
    if sel.extra:
        desc["extra"] = sel.extra
    return desc
