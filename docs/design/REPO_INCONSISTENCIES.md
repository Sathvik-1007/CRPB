# Repo-wide Inconsistencies Report (CRPB)

This report is a snapshot of repo-wide consistency issues and architectural risks found during an audit pass.

## 1) Python version mismatch (runtime vs metadata)

**Problem**
- The codebase uses Python 3.10+ syntax (notably `X | Y` union types), but project metadata/tooling was configured for Python 3.9.

**Impact**
- Running on Python 3.9 will fail at import/parse time.
- Type checking and formatting targets were inconsistent, creating noisy/static-analysis failures.

**Fix applied**
- Updated project metadata and tools to target Python 3.10+:
  - `requires-python = ">=3.10"`
  - Black target-version `py310`
  - Ruff target-version `py310` and moved lint config to `[tool.ruff.lint]`
  - Mypy `python_version = "3.10"`

## 2) “No LLM context truncation” vs existing implementation

**Problem**
- The repo previously used “budgeted context packs” that **truncate** or drop context:
  - `crpb/core/context_compiler.py` truncated `idea` and file snippets and would drop sections when over budget.
  - `crpb/temporal/workflow.py` project validation selected only a bounded subset of files for the LLM call.

**Impact**
- Context was silently reduced before LLM calls, which can cause:
  - missed requirements
  - inconsistent behavior across nodes
  - “it worked on small projects” failure mode

**Fix applied (no silent truncation)**
- `crpb/core/context_compiler.py` is now strict-by-default and does not truncate/slice text.
  - Whole items are included or omitted (with omission metadata).
  - If required items cannot fit, compilation fails loudly (`ContextBudgetExceededError`).
  - There is no environment toggle required for correctness.

**Fix applied (project validation coverage without slicing files)**
- Project-level validation uses **multi-pass** bounded chunks when needed (whole-file only; no slicing) and merges reports.

**Fix applied (regex-based connectivity removed)**
- Regex-driven “connectivity” heuristics were removed in favor of spec-first, evidence-based validation (closure + artifact/ledger evidence).

**Fix applied (no hard-coded filename whitelist in CodeSpec ingestion)**
- `crpb/temporal/workflow.py` no longer drops extensionless file paths based on a hard-coded whitelist (e.g., Dockerfile/README).
- Extensionless file paths are treated as valid file outputs; if a language is ambiguous, the planner/spec should set `language` explicitly.

**Remaining design gap (not fully solved)**
- It is not physically possible to include an unbounded project’s full context in a *single* LLM call.
- The implemented approach is: **no silent truncation**, plus **multi-pass chunking** where needed.
- Extending this same “multi-pass ingestion” approach to *all* generation calls (plan/clarify/generate file/etc.) would be a larger architectural change.

## 3) Ruff configuration noise vs real correctness

**Problem**
- Ruff was configured with upgrade rules (`UP*`) that generate a very large number of stylistic rewrite findings.

**Impact**
- Repo-wide lint runs produced massive noise, obscuring correctness issues.

**Fix applied**
- Ruff is now scoped to correctness + imports + common footguns:
  - `E`, `F`, `W`, `I`, `B`, `C4`
  - avoids forcing repo-wide `typing`-generic rewrites.

## 4) Local-only tests policy

**Requirement**
- Keep certain tests local (not uploaded to GitHub).

**Status**
- Local-only tests live under `tests/` and are git-ignored (via `.gitignore`).
- This repo intentionally keeps unit tests out of version control; only the library code is tracked.

## 8) Lossless event/log payloads

**Problem**
- Display-oriented truncation (e.g., `msg[:N]`, `items[:N]`) can hide critical information.

**Fix applied**
- Oversized messages/lists are now externalized to run artifacts and emitted as small references (path + digest + length).
- CLI commands show bounded tables with explicit totals and pointers to the full data.

## 9) GitHub readiness (secrets, binaries, generated outputs)

- `.env` is git-ignored and contains placeholders only.
- `runs/` and `tests/` are git-ignored by design.
- Do not commit bundled binaries or third-party doc dumps; prefer links and short project-specific docs.

## 5) Implicit “provider preference” defaults (embeddings)

**Problem**
- Several places implicitly preferred embedding providers (e.g., Voyage before OpenAI) when multiple providers were available.

**Impact**
- Provider choice becomes an undocumented behavior change (results differ across providers).
- Violates the “no hard-coded preference decisions” requirement.

**Fix applied**
- Provider ordering/selection is now explicit and environment-driven:
  - `CRPB_TASK_EMBED_PROVIDER_PREFERENCE=voyage,openai` (example)
  - If unset, embeddings indexing/similarity simply does not run rather than choosing.
- Large texts are chunked for embedding rather than being truncated.

## 6) Coverage gaps vs orchestration reality

**Observation**
- Unit tests may be green while orchestration modules remain weakly covered.

**Impact**
- High risk of regressions in:
  - planner refinement loops
  - Temporal workflow glue
  - artifact wiring and ledger integration

**Recommendation**
- Add deterministic tests for:
  - plan validation/rewire behavior (no LLM)
  - artifact producer/consumer graph rules
  - context pack compilation budget errors (strict mode)
  - chunked project validation merging logic (pure function style)

## 7) Next work (high-value, architecture-level)

- Make the strict “no truncation” policy explicit in docs + CLI output.
- Add a first-class multi-pass context ingestion mechanism for *all* LLM operations (not just project validation).
- Document the project’s hard constraints (LLM context limits, Temporal payload limits) and how CRPB resolves them deterministically.

