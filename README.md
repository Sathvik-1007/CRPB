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

4) Run a dry-run scaffold (no LLM calls; for environment/structure verification only):
```bash
python -m crpb dry-run --run-dir runs
```

5) Plan (example):
```bash
python -m crpb plan --idea "Build a notes app with tags and sync" --run-dir runs
```
LLM is mandatory for planning. This command fails fast if `OPENAI_API_KEY` is not set or the LLM is unavailable.

5.1) Plan hierarchical tasks (TaskPlan mode):
```bash
python -m crpb plan-tasks --idea "Build a notes app with tags and sync" --run-dir runs
```
Writes `plan/task_plan.json` with a recursive, generic task DAG. LLM is mandatory; the command fails if unavailable.

6) Build (end-to-end):
```bash
python -m crpb build --run-dir runs --run latest --model gpt-4o-mini
```
Build requires an LLM and fails fast if the LLM is unavailable.

6.1) Execute TaskPlan (recursive split-or-implement):
```bash
python -m crpb tasks --run-dir runs --run latest --model gpt-4o-mini
```
This command processes `plan/task_plan.json` recursively. Each task may be split by the LLM into children or implemented directly. Leaf `code:function` tasks generate function chunks that are assembled and validated like the regular build flow.

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

## Why an LLM is required
All planning, task orchestration, and code generation require the LLM. If `OPENAI_API_KEY` is missing or the LLM fails, commands error out early with actionable messages and emit failure events (e.g., `NODE_FAILED`). Deterministic fallbacks have been removed.

## Architecture & Concepts

- __JSON-first artifacts__: all specs, events, leases, and rollups are JSON/JSONL for determinism and replay.
- __Event-sourced flow__: append-only `logs/events.jsonl` and `graph/node_status.jsonl` capture ground truth. `replay` re-derives rollups.
- __Parent-mediated orchestration__: every node write includes `parent_id`, and all events also include `parent_id`. Status changes propagate upward for deterministic coordination and full traceability.
- __Registry with futures__: `registry/registry.json` tracks files, functions, exports, checksums, and statuses (`stub` → `implemented`). Sibling deps emit `FUTURE_WAIT`/`FUTURE_READY` events.
- __Dependency-aware scheduler__: selects the ready-set from the registry (no missing deps) and advances deterministically.
- __Leases & heartbeats__: `graph/leases.json` with TTLs; builders renew leases during long loops to avoid stuck ownership.
- __Validation & repair__: multi-stage checks (schema, structural, AST, style, imports). On failure, writes `repair_plans/repair_<id>.json` and emits `VALIDATION_FAILED` + `REPAIR_PLANNED`. On success, emits `VALIDATION_PASSED` and stores a report under `validations/`.
- __LLM integration (mandatory)__: uses the DSPy-backed OpenAI model. If unavailable, commands fail fast; no fallback modes are provided.
- __Deterministic assembly__: function chunks assembled in export order, then the rest, to produce stable outputs.

### Repository layout

```text
crpb/
  agents/       # LLM client wrapper
  commands/     # Typer subcommands: plan, dry-run, build, watch, status, replay, schemas
                 # New: plan-tasks (TaskPlan planner), tasks (recursive executor)
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
  - Behavior: requires LLM; fails fast if unavailable.
  - Flags:
    - `--idea <str>` (required)
    - `--constraints <json>` (default: `{}`)
    - `--run-dir <path>`
    - `--run <name>` (default: `new`)

- __Plan-Tasks__ — generate a hierarchical `TaskPlan`
  - Command: `python -m crpb plan-tasks --idea "..." --constraints "{}" --run-dir runs`
  - Output: `plan/task_plan.json` (plus `plan/idea.json`, `plan/constraints.json`)
  - Behavior: requires LLM; fails fast if unavailable.
  - Flags:
    - `--idea <str>` (required)
    - `--constraints <json>` (default: `{}`)
    - `--run-dir <path>`
    - `--run <name>` (default: `new`)

- __Dry-run__ — scaffold structure without LLM calls
  - Command: `python -m crpb dry-run --run-dir runs`
  - Output: initializes `logs/`, `graph/`, `registry/`, and lifecycle demo events
  - Flags:
    - `--run-dir <path>`
    - `--run <name>` (default: `new`)
    - `--node-id <str>` (optional deterministic id)
    - `--parent-id <str>` (optional parent id)

- __Build__ — end-to-end build with registry, scheduler, validations, and LLM
  - Command: `python -m crpb build --run-dir runs --run latest --model gpt-4o-mini`
  - Flags:
    - `--run-dir <path>`
    - `--run <name>` (default: `new`)
    - `--idea <str>` (optional; if omitted, reads from `plan/idea.json`)
    - `--model <name>` (override LLM model)
    - `--lease-ttl <seconds>` (builder lease; default 180)
    - `--child-ttl <seconds>` (child/validator leases; default 120)
    - `--max-children <n>` (parallelism for child tasks; default 2)
    - `--node-id`, `--parent-id` (deterministic IDs for orchestration)
  - Behavior: publishes stubs (`FUNCTION_PUBLISHED`, `FUTURE_WAIT`), implements ready funcs (`FUNCTION_IMPLEMENTED`, `FUTURE_READY`), assembles file, validates, writes reports, emits `VALIDATION_*` and `NODE_DONE`. LLM is mandatory; no fallback.

- __Tasks__ — recursive TaskPlan executor (split-or-implement)
  - Command: `python -m crpb tasks --run-dir runs --run latest --model gpt-4o-mini`
  - Flags:
    - `--run-dir <path>`
    - `--run <name>` (default: `new`)
    - `--idea <str>` (optional; if omitted, reads from `plan/idea.json` or `task_plan.json`)
    - `--constraints <json>` (optional; reads `plan/constraints.json` if omitted)
    - `--lease-ttl`, `--child-ttl`, `--max-children` (parallelism)
    - `--node-id`, `--parent-id`, `--model`
    - `--keep-going/--fail-fast` (default: `--keep-going`) — continue executing other ready tasks on failures; final exit code is non-zero if any task/validation failed.
  - Behavior: emits `TASK_ASSIGNED`, `TASK_SPLIT`, `TASK_DONE`, `TASK_FAILED` (and alias `CHILD_*`) and node lifecycle events; selects ready tasks by dependency and priority (high > medium > low); assembles and validates outputs from `code:function` leaves; writes reports under `validations/` and artifacts under `artifacts/chunks/`.

- __Watch__ — tail events
  - Command: `python -m crpb watch --run latest`
  - Flags:
    - `--run-dir <path>`
    - `--run <name>` (default: `latest`)
    - `--raw/--compact` (default: `--compact`) — print raw JSON vs compact summary lines

- __Status__ — show latest node status entries
  - Command: `python -m crpb status --run latest`
  - Flags:
    - `--run-dir <path>`
    - `--run <name>` (default: `latest`)

- __Replay__ — summarize events and write rollups
  - Command: `python -m crpb replay --run-dir runs --run latest`
  - Flags:
    - `--run-dir <path>`
    - `--run <name>` (default: `latest`)
    - `--write/--no-write` (default: `--write`) — write rollup to `outputs/rollups/status.json`

- __Schemas__ — export JSON Schemas for core specs
  - Command: `python -m crpb schemas --run-dir runs --run latest`
  - Flags:
    - `--run-dir <path>`
    - `--run <name>` (default: `latest`)

## Environment

- Copy `.env.example` to `.env` and set variables (Windows PowerShell):
  ```powershell
  Copy-Item .env.example .env
  # then edit .env
  ```
- Required: `OPENAI_API_KEY` — your OpenAI API key
- Optional: `CRPB_OPENAI_MODEL` — defaults to `gpt-4o-mini`
  - Precedence for model selection: CLI `--model` > env `CRPB_OPENAI_MODEL` > default from `crpb/config.py::DEFAULT_MODEL`
- Optional: `CRPB_IMPORT_BASELINE` — comma-separated modules added to the import safety whitelist. Example (PowerShell):
  ```powershell
  $env:CRPB_IMPORT_BASELINE = "pathlib,datetime,math"
  ```
- Security: do not commit secrets; prefer environment variables in CI.

## Troubleshooting

- __No events printed by watch__: confirm `runs/<latest>/logs/events.jsonl` exists and grows during build.
- __LLM errors or rate limits__: commands fail fast. Ensure `OPENAI_API_KEY` is set and your selected model is available.
- __Validation failed__: inspect `validations/report_*.json` and `repair_plans/repair_*.json`; re-run `build` after addressing issues.
- __Registry conflicts__: the builder retries once on optimistic concurrency errors.

## Design Highlights
- JSON-first artifacts and validation
- Parent-mediated registry and sibling dependency via futures
- Event-sourced lifecycle and replay: `logs/events.jsonl`, `graph/node_status.jsonl`, `replay` command
- Leases and heartbeats; optimistic concurrency on registry
- Deterministic assembly and idempotency keys; scheduler with dep-aware ready-set
- __Recursive TaskPlan mode__: generic `TaskPlan`/`TaskSpec` models enable LLM-decided task splitting (DAG with children). `tasks` orchestrates leases, events, and validation for arbitrary depth. Leaf `code:function` tasks are assembled and validated using the same rigorous pipeline as the file/function builder.

See `plan.md` for the full specification.

## Validations and Repair Plans

- After build, a validations report is written under `validations/` per output file, e.g. `validations/report_<id>.json` containing:
  - schema — JSON Schema validation of the `FileSpec` (all languages)
  - basic — structural checks (e.g., declared functions are exported). No language-enforced signature format.
  - ast — Python-only: assembled code parses and includes exported functions. Skipped for non-Python files.
  - style — Python-only: light line-length check. Skipped for non-Python files.
  - imports — Python-only: import safety whitelist. Skipped for non-Python files.
  - integrity — Python-only: entrypoint/import header checks. Skipped for non-Python files.
  - non_python_output — Non-Python: minimal existence/non-empty content check.
  - On any validation failure, a structured repair plan is written to `repair_plans/repair_<id>.json` with issues and suggestions to retry.

## Replay Rollups (enhanced)

- `python -m crpb replay` also emits a per-function rollup including:
  - last event, first/last timestamps
  - FUTURE_WAIT and FUTURE_READY counts
  - a compact timeline of events per function
  - Rollup JSON written to `outputs/rollups/status.json` for deterministic replay.

- Additionally, for __TaskPlan runs__, replay summarizes __per-task__ lifecycle:
  - Task kind, last event, first/last timestamps
  - ASSIGNED/SPLIT/DONE/FAILED counts
  - Compact per-task timeline
  - CHILD_* events are normalized to TASK_* for aggregation

## Status View (enhanced)

- `python -m crpb status` now prints:
  - A summary of node counts by last state
  - Recent node status entries (tail)
  - A filtered table for `task::` nodes showing their last state
