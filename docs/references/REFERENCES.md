# References (Upstream Docs)

This repository intentionally avoids vendoring large third-party documentation blobs.
Instead, keep project-specific guidance in README.md / docs/design/plan.md and link to upstream sources.

## Temporal

- Temporal Docs: https://docs.temporal.io/
- Temporal CLI: https://docs.temporal.io/cli
- Temporal Python SDK: https://docs.temporal.io/develop/python

## DSPy

- DSPy (GitHub): https://github.com/stanfordnlp/dspy
- DSPy Docs: https://dspy.ai/

## General

- Keep `.env` and provider keys out of git; use `.env.example` as the tracked template.
- Keep `runs/` (generated artifacts) out of git.
- Keep `tests/` local-only (this repo policy).
