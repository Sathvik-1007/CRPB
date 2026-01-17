from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import typer
from rich.console import Console

from ..core.config import default_model_for
from ..core.llm_config import (
    LLMSelection,
    ProviderEnum,
    as_descriptor,
    effective_model,
    embeddings_enabled,
    env_values_for,
    load_saved_selection,
    load_selection,
    require_env_vars,
    save_selection,
)

EPILOG = "\n".join(
    [
        "Use `python -m crpb llm <command> --help` for detailed options.",
        "Commands: `choose`, `show`, `unset`.",
        "Secrets (API keys/tokens) are environment-only. Set them in your shell or .env; the CLI never writes secrets.",
    ]
)

app = typer.Typer(
    help="LLM provider management (saved in .crpb_llm.json).",
    epilog=EPILOG,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
console = Console()


@app.command()
def show() -> None:
    """Show current provider selection.

    Prints JSON fields:
    - provider, model (saved), effective_model (env/default aware)
    - base_url (for server), extra (k=v)
    - env_present, env_missing, ok
    """
    env_provider = os.environ.get("CRPB_LLM_PROVIDER")
    env_embed = os.environ.get("CRPB_EMBEDDINGS_ENABLE")
    sel = load_selection()
    saved = load_saved_selection()
    if not sel:
        console.print(
            "[yellow]No LLM selection saved yet.[/yellow] Use `python -m crpb llm choose --provider <openai|anthropic|cerebras|azure_foundry|huggingface|server|local> [--model <id>]` to save one. Secrets are environment-only; set keys/tokens via your shell or .env."
        )
        return
    ok, missing = require_env_vars(sel)
    _ = as_descriptor(sel)
    effective = effective_model(sel, default_model=default_model_for(sel.provider))
    embed_on = embeddings_enabled()
    console.print("[bold]Current selection[/bold]:")
    if isinstance(env_provider, str) and env_provider.strip():
        console.print(
            "[yellow]Note:[/yellow] CRPB_LLM_PROVIDER is set; runtime uses environment selection and ignores .crpb_llm.json for provider/model/base_url."
        )
    if isinstance(env_embed, str) and env_embed.strip():
        console.print(
            "[yellow]Note:[/yellow] CRPB_EMBEDDINGS_ENABLE is set; it overrides the saved embeddings flag."
        )
    console.print(
        json.dumps(
            {
                "selection_source": "env"
                if (isinstance(env_provider, str) and env_provider.strip())
                else "saved",
                "embeddings_enabled": bool(embed_on),
                "provider": sel.provider,
                "model": sel.model,
                "effective_model": effective,
                "base_url": sel.base_url,
                "extra": sel.extra or {},
                "env_present": list(env_values_for(sel).keys()),
                "env_missing": missing,
                "ok": ok,
            },
            indent=2,
        )
    )
    if saved is not None and (
        not (isinstance(env_provider, str) and env_provider.strip())
        or saved.provider != sel.provider
        or (saved.model or None) != (sel.model or None)
        or (saved.base_url or None) != (sel.base_url or None)
    ):
        console.print("\n[bold]Saved selection (.crpb_llm.json)[/bold]:")
        console.print(
            json.dumps(
                {
                    "provider": saved.provider,
                    "model": saved.model,
                    "base_url": saved.base_url,
                    "extra": saved.extra or {},
                    "embeddings_enabled": bool(getattr(saved, "embeddings_enabled", False)),
                },
                indent=2,
            )
        )


@app.command("embeddings")
def embeddings_toggle(
    enable: bool = typer.Option(
        False,
        "--enable",
        help="Enable embeddings-backed features (planner/workflow/validator).",
    ),
    disable: bool = typer.Option(
        False,
        "--disable",
        help="Disable embeddings-backed features (default).",
    ),
) -> None:
    """Toggle embeddings usage (persisted in .crpb_llm.json).

    This does NOT store secrets. It only controls whether CRPB will make embeddings API calls
    during planning/workflows/validation.

    Env override:
      - CRPB_EMBEDDINGS_ENABLE=true|false overrides the saved value.
    """
    if enable and disable:
        console.print("[red]Choose only one:[/red] --enable or --disable")
        raise typer.Exit(code=2)
    if not enable and not disable:
        console.print(
            json.dumps(
                {
                    "embeddings_enabled": bool(embeddings_enabled()),
                    "saved": bool(getattr(load_saved_selection() or {}, "embeddings_enabled", False)),
                    "env_override": os.environ.get("CRPB_EMBEDDINGS_ENABLE") or None,
                },
                indent=2,
            )
        )
        return

    desired = bool(enable) and not bool(disable)
    existing = load_saved_selection()
    if existing is None:
        # Require a provider selection to exist so we don't create a partially-populated file.
        console.print(
            "[yellow]No saved LLM selection found.[/yellow] Run `python -m crpb llm choose --provider ...` first, then set embeddings."
        )
        raise typer.Exit(code=2)

    updated = LLMSelection(
        provider=existing.provider,
        model=existing.model,
        base_url=existing.base_url,
        extra=existing.extra,
        embeddings_enabled=desired,
    )
    path = save_selection(updated)
    console.print(
        f"[green]Updated embeddings flag[/green] in [bold]{path}[/bold]: embeddings_enabled={desired}"
    )


@app.command()
def choose(
    provider: Optional[ProviderEnum] = typer.Option(
        None,
        "--provider",
        help="Provider: openai | anthropic (Claude) | cerebras | azure_foundry | huggingface (HF) | server | local. Required when no selection is saved; otherwise omitted reuses the saved provider.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model id. Optional for openai/anthropic/cerebras (env defaults available). Required for huggingface/local/server.",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="For provider=server (e.g., http://localhost:11434)"
    ),
    extra: List[str] = typer.Option(
        [], "--extra", help="Optional key=value pairs for provider-specific settings"
    ),
) -> None:
    """Save provider/model/base_url to .crpb_llm.json.

    Notes:
    - Secrets: API keys/tokens are provided via environment only; the CLI does not set or persist secrets. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in your shell or .env. For server/local, keys are optional: CRPB_SERVER_API_KEY/SERVER_API_KEY and CRPB_LOCAL_API_KEY/LOCAL_API_KEY.
    - --provider: must be explicit when no selection is saved. If omitted and a selection exists, that provider is reused. No environment-based inference.
    - Models: prefer env defaults (CRPB_OPENAI_MODEL, CRPB_ANTHROPIC_MODEL, CRPB_HF_MODEL) or pass --model.
      Use --model to persist a model (required for huggingface/local/server).
    - For provider=local: "model" is a free-form identifier passed through to your local runtime. Use whatever your setup expects:
      a simple name (e.g., "my-model") or a filesystem path string if your local runtime resolves models by path.
    - For provider=server: provide both --base-url and --model appropriate for your server backend.

    Examples:
    python -m crpb llm choose --provider openai
    python -m crpb llm choose --provider anthropic
    python -m crpb llm choose --provider cerebras
    python -m crpb llm choose --provider azure_foundry --model <deployment_name>
      python -m crpb llm choose --provider huggingface --model meta-llama/Meta-Llama-3-8B-Instruct
      python -m crpb llm choose --provider server --base-url http://localhost:11434 --model llama3
      python -m crpb llm choose --provider local --model my-local-model-id
      # Use saved provider and just change model
      python -m crpb llm choose --model o4-mini
    """
    # parse extra key=value into dict
    extra_map: Dict[str, str] = {}
    for item in extra:
        if "=" in item:
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                extra_map[k] = v
    # Determine provider: CLI > saved selection (no env inference)
    prov_name: str
    if provider is not None:
        prov_name = provider.value
    else:
        existing = load_selection()
        if existing:
            prov_name = existing.provider
        else:
            # Symmetric warning: model provided but no provider selected
            examples = "openai | anthropic (Claude) | cerebras | azure_foundry | huggingface (HF) | server | local"
            if model and str(model).strip():
                console.print(
                    f"[yellow]You specified --model '{model}' but no --provider.[/yellow] Please pass --provider ({examples})."
                )
                # Optional neutral hints if environment suggests a likely provider (no auto-selection)
                hints = []
                if os.environ.get("OPENAI_API_KEY"):
                    hints.append("openai")
                if os.environ.get("ANTHROPIC_API_KEY"):
                    hints.append("anthropic")
                if os.environ.get("CEREBRAS_API_KEY"):
                    hints.append("cerebras")
                if os.environ.get("CRPB_AZURE_FOUNDRY_API_KEY") and os.environ.get(
                    "CRPB_AZURE_FOUNDRY_ENDPOINT"
                ):
                    hints.append("azure_foundry")
                if (
                    os.environ.get("CRPB_HF_MODEL")
                    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                    or os.environ.get("HF_TOKEN")
                ):
                    hints.append("huggingface")
                if os.environ.get("CRPB_SERVER_API_KEY") or os.environ.get("SERVER_API_KEY"):
                    hints.append("server")
                if os.environ.get("CRPB_LOCAL_API_KEY") or os.environ.get("LOCAL_API_KEY"):
                    hints.append("local")
                if hints:
                    console.print(
                        "[yellow]Hint:[/yellow] Environment suggests: "
                        + ", ".join(sorted(set(hints)))
                        + "."
                    )
                console.print("Examples:")
                console.print("  python -m crpb llm choose --provider openai --model " + str(model))
                console.print(
                    "  python -m crpb llm choose --provider anthropic --model " + str(model)
                )
                console.print(
                    "  python -m crpb llm choose --provider cerebras --model " + str(model)
                )
                console.print(
                    "  python -m crpb llm choose --provider huggingface --model <repo-id>"
                )
                console.print(
                    "  python -m crpb llm choose --provider server --base-url http://host:port --model "
                    + str(model)
                )
                console.print("  python -m crpb llm choose --provider local --model " + str(model))
            else:
                console.print(
                    "[yellow]No provider specified.[/yellow] Please pass --provider (openai | anthropic (Claude) | cerebras | huggingface (HF) | server | local)."
                )
            raise typer.Exit(code=2)

    sel = LLMSelection(provider=prov_name, model=model, base_url=base_url, extra=extra_map or None)
    ok, missing = require_env_vars(sel)
    path = save_selection(sel)
    console.print(f"[green]Saved LLM selection[/green] to [bold]{path}[/bold].")
    env_provider = os.environ.get("CRPB_LLM_PROVIDER")
    if isinstance(env_provider, str) and env_provider.strip():
        console.print(
            "[yellow]Note:[/yellow] CRPB_LLM_PROVIDER is set, so runtime will continue to use the environment selection (and ignore the saved file) until you unset CRPB_LLM_PROVIDER."
        )
    if ok:
        console.print("[green]Environment looks OK for this provider.[/green]")
    else:
        console.print(
            "[yellow]Missing requirements for this provider:[/yellow] " + ", ".join(missing)
        )
        console.print("Set them in your shell or .env (loaded by CLI), then retry.")

    # Warn if no effective model is selected (provider-neutral). This helps users finish setup early.
    eff_model = effective_model(sel, default_model=default_model_for(sel.provider))
    if not eff_model:
        hint_env = {
            "openai": "CRPB_OPENAI_MODEL",
            "anthropic": "CRPB_ANTHROPIC_MODEL",
            "cerebras": "CRPB_CEREBRAS_MODEL",
            "huggingface": "CRPB_HF_MODEL",
            "server": None,
            "local": None,
        }.get(sel.provider)
        if hint_env:
            console.print(
                f"[yellow]No model selected for provider '{sel.provider}'.[/yellow] Set one with `--model <id>` or via env ({hint_env})."
            )
        else:
            # server/local already report missing model via require_env_vars; this is a gentle reiteration if needed
            console.print(
                f"[yellow]No model selected for provider '{sel.provider}'.[/yellow] Set one with `--model <id>` to complete setup."
            )


@app.command()
def unset() -> None:
    """Remove .crpb_llm.json; no provider/model will be saved thereafter.

    Runtime behavior: commands will use environment-based configuration if present (e.g., provider keys or model env vars);
    otherwise they will instruct you to configure an LLM via `python -m crpb llm choose`.
    """
    from ..core.llm_config import _config_path  # local import to avoid exporting utility

    p = _config_path()
    if p.exists():
        p.unlink()
        console.print(f"[green]Removed[/green] {p}")
    else:
        console.print("[yellow]No selection file present.[/yellow]")
