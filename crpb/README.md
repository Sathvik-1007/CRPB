# Package layout

`crpb/` is the Python package.

## Subpackages

- `agents/` — LLM orchestration engines/providers.
- `commands/` — CLI subcommands.
- `core/` — canonical implementations of shared infrastructure (config, registry, strict JSON, scheduler, etc.).
- `planning/` — planning logic (plan composition, judging, parallel helpers).
- `repairing/` — validate→repair→revalidate loop and supporting models.
- `temporal/` — workflow/activity orchestration.
- `utils/` — small pure utilities (fs, artifacts, embeddings, run log summaries, etc.).
- `validation/` — deterministic validators and project-level validation.

## Compatibility shims

Some top-level modules like `crpb/registry.py` and `crpb/strict_json.py` are intentionally tiny.
They exist to preserve stable import paths while keeping canonical implementations in `crpb/core/`.
