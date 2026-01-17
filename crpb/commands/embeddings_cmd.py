from __future__ import annotations

import json
import os
from typing import List, Optional

import typer
from rich.console import Console

from ..utils.embeddings import make_embedder_from_env, normalize_task_embedding_fields

app = typer.Typer(help="Embeddings utilities (health checks)")
console = Console()


@app.command()
def test(
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="voyage | openai | auto (default: auto)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Embedding model override. If omitted, uses env defaults.",
    ),
    input_type: str = typer.Option(
        "query",
        "--input-type",
        help="Voyage input_type: query | document. (Ignored for OpenAI)",
    ),
    text: List[str] = typer.Option(
        [],
        "--text",
        help="Text to embed (repeatable). Defaults to two short samples.",
    ),
) -> None:
    p = (provider or "auto").strip().lower()
    if p not in ("auto", "voyage", "openai"):
        console.print("[red]Invalid --provider.[/red] Use voyage | openai | auto.")
        raise typer.Exit(code=2)

    texts = [t for t in (text or []) if isinstance(t, str) and t.strip()]
    if not texts:
        texts = [
            "Task A: parse input and normalize",
            "Task B: validate output and persist",
        ]

    bounded: List[str] = []
    for t in texts:
        _, _, tt = normalize_task_embedding_fields(title=t, summary="")
        if tt:
            bounded.append(tt)
    if not bounded:
        console.print("[red]No valid text to embed.[/red]")
        raise typer.Exit(code=2)

    model_override = str(model).strip() if isinstance(model, str) and model.strip() else None

    chosen_provider = p
    use_input_type = (input_type or "query").strip().lower() or "query"
    if chosen_provider == "auto":
        has_v = (
            make_embedder_from_env(
                provider="voyage",
                input_type=use_input_type,
                voyage_model=model_override,
            )
            is not None
        )
        has_o = (
            make_embedder_from_env(
                provider="openai",
                input_type=use_input_type,
                openai_model=model_override,
            )
            is not None
        )
        if has_v and not has_o:
            chosen_provider = "voyage"
        elif has_o and not has_v:
            chosen_provider = "openai"
        elif has_v and has_o:
            pref = os.environ.get("CRPB_TASK_EMBED_PROVIDER_PREFERENCE")
            if isinstance(pref, str) and pref.strip():
                first = pref.split(",", 1)[0].strip().lower()
                chosen_provider = first if first in ("voyage", "openai") else ""
            else:
                chosen_provider = ""
        else:
            chosen_provider = ""

    if chosen_provider == "voyage":
        emb = make_embedder_from_env(
            provider="voyage",
            input_type=use_input_type,
            voyage_model=model_override,
        )
        if emb is None:
            console.print(
                "[red]Voyage embedder is not configured.[/red] Set VOYAGE_API_KEY and CRPB_VOYAGE_EMBED_MODEL (or pass --model)."
            )
            raise typer.Exit(code=2)
        vecs = emb.embed_texts(bounded)
        dim = len(vecs[0]) if vecs and isinstance(vecs[0], list) else 0
        console.print(
            json.dumps(
                {
                    "ok": True,
                    "provider": "voyage",
                    "model": getattr(emb, "model", None),
                    "input_type": use_input_type,
                    "count": len(vecs),
                    "dim": dim,
                },
                indent=2,
            )
        )
        return

    if chosen_provider == "openai":
        emb = make_embedder_from_env(
            provider="openai",
            input_type=use_input_type,
            openai_model=model_override,
        )
        if emb is None:
            console.print(
                "[red]OpenAI embedder is not configured.[/red] Set OPENAI_API_KEY and CRPB_OPENAI_EMBED_MODEL (or pass --model)."
            )
            raise typer.Exit(code=2)
        vecs = emb.embed_texts(bounded)
        dim = len(vecs[0]) if vecs and isinstance(vecs[0], list) else 0
        console.print(
            json.dumps(
                {
                    "ok": True,
                    "provider": "openai",
                    "model": getattr(emb, "model", None),
                    "count": len(vecs),
                    "dim": dim,
                },
                indent=2,
            )
        )
        return

    console.print(
        "[red]No embedding provider selected.[/red] Set --provider, or set CRPB_TASK_EMBED_PROVIDER_PREFERENCE. Ensure the selected provider is fully configured (API key + model env, or pass --model)."
    )
    raise typer.Exit(code=2)
