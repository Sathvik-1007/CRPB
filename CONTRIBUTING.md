# Contributing

## Development setup

- Python 3.10+
- Install: `pip install -r requirements.txt`
- Optional dev tools: `pip install -e .[dev]`

## Local-only tests policy

This repo keeps pytest tests local-only by design.

- Put tests under `tests/`
- `tests/` is intentionally git-ignored
- Deterministic validation belongs in `crpb/validation/`

## Style

- Keep changes minimal and spec-driven
- Avoid hardcoding technology stacks or filename assumptions
- Do not truncate/slice context passed to LLM calls; prefer strict failure or multi-pass selection

## Running checks

- `python -m pytest -q`
