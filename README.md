# CRPB — Context Recursive Project Builder

A rigorous, deterministic, parent-mediated, event-sourced multi-agent builder that turns an idea into a runnable project.

## Quickstart

1) Create and activate a Python 3.10+ venv
2) Install deps
```bash
pip install -r requirements.txt
```
3) Set your OpenAI key (PowerShell example)
```powershell
$env:OPENAI_API_KEY = "<your-key>"
# optional
$env:CRPB_OPENAI_MODEL = "gpt-4o-mini"
```

4) Run a dry-run scaffold (no LLM calls):
```bash
python -m crpb dry-run --run-dir runs
```

5) Plan (example):
```bash
python -m crpb plan --idea "Build a notes app with tags and sync" --run-dir runs
```
By default, plan requires an LLM. To permit deterministic fallback on LLM errors, add `--allow-fallback`.
```bash
python -m crpb plan --idea "Build a notes app with tags and sync" --run-dir runs --allow-fallback
```

6) Build (end-to-end):
```bash
python -m crpb build --run-dir runs --run latest --model gpt-4o-mini
```
By default, build requires an LLM. To permit a deterministic fallback when the LLM is unavailable or fails, add `--allow-fallback`.
```bash
python -m crpb build --run-dir runs --run latest --allow-fallback
```

7) Watch events
```bash
python -m crpb watch --run "latest"
```

8) Replay a run (summarize events) and write rollup:
```bash
python -m crpb replay --run-dir runs --run latest
```

9) Generate JSON Schemas for core specs:
```bash
python -m crpb schemas --run-dir runs --run latest
```

## What is dry-run?
Dry-run initializes the event log, node status file, leases, and registry structure without invoking any LLM calls. It demonstrates lifecycle transitions (CREATED → LEASED → DONE) and readies the run folder. Use it to verify environment and file structure before spending tokens.

## How does build work without an LLM?
By default, build runs with `--require-llm` and will fail fast if the LLM is unavailable or errors, emitting `NODE_FAILED`.
If you pass `--allow-fallback`, and `OPENAI_API_KEY` is not set (or the LLM call fails), the builder deterministically falls back while emitting `LLM_FALLBACK` events.

## Architecture & Concepts

- __JSON-first artifacts__: all specs, events, leases, and rollups are JSON/JSONL for determinism and replay.
- __Event-sourced flow__: append-only `logs/events.jsonl` and `graph/node_status.jsonl` capture ground truth. `replay` re-derives rollups.
- __Parent-mediated orchestration__: every node write includes `parent_id`, and all events also include `parent_id`. Status changes propagate upward for deterministic coordination and full traceability.
- __Registry with futures__: `registry/registry.json` tracks files, functions, exports, checksums, and statuses (`stub` → `implemented`). Sibling deps emit `FUTURE_WAIT`/`FUTURE_READY` events.
- __Dependency-aware scheduler__: selects the ready-set from the registry (no missing deps) and advances deterministically.
- __Leases & heartbeats__: `graph/leases.json` with TTLs; builders renew leases during long loops to avoid stuck ownership.
- __Validation & repair__: multi-stage checks (schema, structural, AST, style, imports). On failure, writes `repair_plans/repair_<id>.json` and emits `VALIDATION_FAILED` + `REPAIR_PLANNED`. On success, emits `VALIDATION_PASSED` and stores a report under `validations/`.
- __LLM integration with fallback__: uses OpenAI Chat Completions when available; otherwise emits `LLM_FALLBACK` and uses deterministic implementations to guarantee progress.
- __Deterministic assembly__: function chunks assembled in export order, then the rest, to produce stable outputs.

### Repository layout

```text
crpb/
  agents/       # LLM client wrapper
  commands/     # Typer subcommands: plan, dry-run, build, watch, status, replay, schemas
  utils/, specs/, validator/, scheduler/, registry/, leases/, eventbus/
plan.md         # Detailed system spec
README.md       # This file
requirements.txt
.env.example    # Sample environment configuration
```

### Run folder layout (created under your --run-dir)

```text
runs/
  run_YYYYMMDD_HHMMSS/
    artifacts/      # file_spec metadata, chunks/
    graph/          # nodes.jsonl, edges.jsonl, node_status.jsonl, leases.json
    logs/           # events.jsonl (append-only)
    outputs/        # assembled source files and rollups
    plan/           # idea.json, constraints.json
    registry/       # registry.json, stubs/
    repair_plans/   # repair_*.json (written on validation failure)
    specs/          # file_*.json
    validations/    # report_*.json (written on success/failure)
```

## CLI reference

- __Plan__ — save idea and constraints
  - Command: `python -m crpb plan --idea "..." --constraints "{}" --run-dir runs`
  - Output: `plan/idea.json`, `plan/constraints.json`
  - Behavior: requires LLM by default; with `--allow-fallback`, deterministically falls back to a constraint-driven plan on LLM error.

- __Dry-run__ — scaffold structure without LLM calls
  - Command: `python -m crpb dry-run --run-dir runs`
  - Output: initializes `logs/`, `graph/`, `registry/`, and lifecycle demo events

- __Build__ — end-to-end build with registry, scheduler, validations, and LLM
  - Command: `python -m crpb build --run-dir runs --run latest --model gpt-4o-mini`
  - Flags:
    - `--require-llm/--allow-fallback` (default: `--require-llm`)
    - `--lease-ttl <seconds>` (builder lease; default 180)
    - `--child-ttl <seconds>` (child/validator leases; default 120)
    - `--max-children <n>` (parallelism for child tasks; default 2)
    - `--node-id`, `--parent-id` (deterministic IDs for orchestration)
  - Behavior: publishes stubs (`FUNCTION_PUBLISHED`, `FUTURE_WAIT`), implements ready funcs (`FUNCTION_IMPLEMENTED`, `FUTURE_READY`), assembles file, validates, writes reports, emits `VALIDATION_*` and `NODE_DONE`. With `--allow-fallback`, emits `LLM_FALLBACK` on deterministic fallback.

- __Watch__ — tail events
  - Command: `python -m crpb watch --run latest`

- __Status__ — show latest node status entries
  - Command: `python -m crpb status --run latest`

- __Replay__ — summarize events and write rollups
  - Command: `python -m crpb replay --run-dir runs --run latest`

- __Schemas__ — export JSON Schemas for core specs
  - Command: `python -m crpb schemas --run-dir runs --run latest`

## Environment

- Copy `.env.example` to `.env` and set variables (Windows PowerShell):
  ```powershell
  Copy-Item .env.example .env
  # then edit .env
  ```
- Required: `OPENAI_API_KEY` — your OpenAI API key
- Optional: `CRPB_OPENAI_MODEL` — defaults to `gpt-4o-mini`
- Optional: `CRPB_IMPORT_BASELINE` — comma-separated modules added to the import safety whitelist. Example (PowerShell):
  ```powershell
  $env:CRPB_IMPORT_BASELINE = "pathlib,datetime,math"
  ```
- Security: do not commit secrets; prefer environment variables in CI.

## Troubleshooting

- __No events printed by watch__: confirm `runs/<latest>/logs/events.jsonl` exists and grows during build.
- __LLM errors or rate limits__: with `--allow-fallback`, build will fall back deterministically; see `LLM_FALLBACK` events in `events.jsonl`.
- __Validation failed__: inspect `validations/report_*.json` and `repair_plans/repair_*.json`; re-run `build` after addressing issues.
- __Registry conflicts__: the builder retries once on optimistic concurrency errors.

## Design Highlights
- JSON-first artifacts and validation
- Parent-mediated registry and sibling dependency via futures
- Event-sourced lifecycle and replay: `logs/events.jsonl`, `graph/node_status.jsonl`, `replay` command
- Leases and heartbeats; optimistic concurrency on registry
- Deterministic assembly and idempotency keys; scheduler with dep-aware ready-set

See `plan.md` for the full specification.

## Validations and Repair Plans

- After build, a validations report is written under `validations/` per output file, e.g. `validations/report_<id>.json` containing:
  - schema (JSON Schema validation of the FileSpec)
  - basic (structural checks: exports, signatures)
  - ast (assembled Python parses and includes exported functions)
  - style (light line-length check)
  - imports (import safety whitelist)
- On any validation failure, a structured repair plan is written to `repair_plans/repair_<id>.json` with issues and suggestions to retry.

## Replay Rollups (enhanced)

- `python -m crpb replay` also emits a per-function rollup including:
  - last event, first/last timestamps
  - FUTURE_WAIT and FUTURE_READY counts
  - a compact timeline of events per function
  - Rollup JSON written to `outputs/rollups/status.json` for deterministic replay.
