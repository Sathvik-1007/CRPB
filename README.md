# CRPB — Context Recursive Project Builder

CRPB is a general‑purpose, language‑agnostic, spec‑driven project builder. It turns a natural‑language idea plus constraints into a hierarchical plan, a neutral CodeSpec, and a deterministic build with holistic validation and repair loops.

CRPB is not a “code generator for one stack.” It builds anything: a single algorithm, a data pipeline, a CLI tool, a website, a set of configs, or even a long text artifact. The system remains neutral and defers technology choices to the plan/spec or file path inference.

---

## Why CRPB

- **General‑purpose**: Works for any type of project (algorithms, data, content, config, apps).
- **Language‑agnostic**: No default language assumptions; file extensions and `language` fields are optional and never required.
- **Spec‑driven**: Planning produces a hierarchical TaskPlan; execution produces/consumes a neutral CodeSpec.
- **Context‑rich**: Every step is guided by structured context (node plans, artifacts, validations) compiled into budgeted ContextPacks.
- **Persistent node memory**: Each node has a Node Ledger (TODOs, decisions, obligations) stored under run artifacts.
- **Deterministic build**: Files are generated in a stable order, with pre/post validation and repair loops.
- **Holistic validation**: Project‑level checks ensure features are integrated and actually used, not just implemented.

---

## Upstream References

This repo keeps project-specific docs in `README.md` and `docs/design/plan.md` and links out to upstream sources (Temporal/DSPy) in `docs/references/REFERENCES.md`.

## Architecture Overview

CRPB has three major phases:

1. **Plan**
   - Generates a hierarchical `TaskPlan` (`runs/<run>/plan/plan.json`).
   - Embeds rich per‑node context (`node_plan`, `meta.socratic`, `meta.path`).
   - Exports a clean `view` (human‑first parent→children tree, no ids/deps noise).
   - Builds an `artifacts.index` mapping producers/consumers for integration.
   - Validates the plan (`plan_validation.json`).

2. **CodeSpec**
   - Produces a language‑neutral file blueprint: `runs/<run>/plan/codespec.json`.
   - Describes files (`path`, optional `language`), `functions`, `exports`, `content`, `entrypoint`, etc.
   - Can be enriched progressively or refined before build.

3. **Build**
   - Generates files to a staging area, validates per‑file and project‑level, performs repair loops, and then merges to outputs.
   - Writes validation reports under `runs/<run>/validations/` and final files to `runs/<run>/outputs/`.

```mermaid
flowchart TD
  A[Idea + Constraints] --> B[Plan: TaskPlan]
  B --> C[Plan View + Artifacts Index]
  B --> D[Codespec]
  D --> E[Build Staging]
  E --> F[Pre-file Validation]
  F --> G[Project Validation + Repair Loops]
  G --> H[Merge Outputs]
```

## Design Highlights

- **Deterministic Leaf Guardrails** – Only atomic `code:function` nodes without children are generated (`is_deterministic_leaf` in [`crpb/validation/validator.py`](crpb/validation/validator.py)). Enforced in planner and build phases.
- **CAS‑Safe Registry with Deterministic Leases** – Optimistic concurrency control with version checks and checksum tracking; lease API (`acquire_lease`, `renew_lease`, `release_lease`, `lease_info`) ensures exclusive access (`crpb/core/registry.py`).
- **Strict JSON Parsing** – `_parse_json_dict_strict` in [`crpb/utils/artifacts.py`](crpb/utils/artifacts.py) guarantees all LLM‑generated JSON is a single‑line dict, used by the `strict_json` decorator.
- **Export Verification** – Automatic validation that declared exports exist in generated file content (Temporal activities in `crpb/temporal/workflow.py`).
- **Temporal Orchestration** – Plan and build are Temporal‑backed workflows (`crpb/temporal/workflow.py`) with deterministic `run_id` propagation and retry policies.
- **Node Ledger + Context Compiler** – Run-scoped persistent ledgers and deterministic, budgeted context bundles (see `docs/design/plan.md`, `crpb/core/ledger.py`, `crpb/core/context_compiler.py`).
- **Closure Validation (artifact-based)** – Deterministic wiring/closure validation checks declared artifact producers/consumers and run-scoped evidence (ledger + artifact registry) without parsing source code (`crpb/validation/closure.py`).


---

## Key Concepts

- **TaskPlan (planning)**
  - Hierarchical tasks with `children` (parents orchestrate; leaves implementable).
  - Each node has `node_plan` (intent, acceptance_criteria, test_plan, etc.) and `meta.socratic` (questions + monologue).
  - `view.roots` provides a clean parent→children tree for humans/tools.
  - `artifacts.index` captures producers/consumers to enforce integration.

- **CodeSpec (specification)**
  - Per‑file blueprint that remains language‑neutral.
  - Fields: `path`, optional `language`, `purpose`, `description`, `functions`, `exports`, `content`, `entrypoint`, etc.
  - Enables deterministic generation without stack bias.

- **Validation & Repair**
  - `validate_taskplan_general` checks structure, DAG (id/dep), node_plan presence, and artifact shapes/coverage.
  - Project‑level validation flags unused exports, missing producers/consumers, and plan/spec mismatch (capabilities not surfaced or orphan features).
  - Pre‑ and post‑merge repair loops attempt targeted, context‑rich micro‑adjustments to converge on a fully integrated build.

- **Side Context**
  - Every generation/repair step receives structured `side_context`, e.g.:
      - `context_pack` (budgeted, deterministic bundle)
    - `codespec_file`, `plan_nodes`, `plan_artifacts`, `plan_validation`
    - `project_issues`, `project_warnings`, `project_suggestions`
      - Note: LLM calls avoid passing an unbounded raw `files={...}` map; project-level checks are performed via deterministic selection + multi-pass validation over whole files (no slicing).

See `docs/design/plan.md` for the formal axioms and definitions.

---

## Project Layout

- `crpb/` — core library (`agents/`, `commands/`, `core/`, `planning/`, `temporal/`, `utils/`, `validation/`)
- `runs/<run>/` — per‑run artifacts:
  - `plan/plan.json` — TaskPlan with hierarchical `view` and `artifacts.index`
  - `plan/codespec.json` — file blueprint used by build
  - `validations/` — file‑level and project‑level reports
  - `outputs/` — merged final files; `_staging/` is temporary during build
- `.env` — local environment (ignored by Git). Use `.env.example` as template.
- `tests/` — local-only pytest tests. See `docs/design/LOCAL_TESTS.md`.

---

## Quickstart

Prereqs:
- Python 3.10+

1. **Install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment**
   - Copy `.env.example` to `.env` and set provider keys.
   - Do NOT commit real secrets. `.env` is ignored; `.env.example` is tracked.

3. **Start Temporal (required for `plan`/`build`)**
   - Install the Temporal CLI (recommended) and start a local dev server:
   ```bash
   temporal server start-dev
   ```
   - In a second terminal, start the CRPB worker:
   ```bash
   python -m crpb temporal-worker
   ```

4. **Choose LLM provider/model**
    ```bash
    python -m crpb llm --help
    ```

5. **Plan**
   ```bash
   python -m crpb plan --idea "<your idea>" --constraints "{}" --run new
   # Outputs under runs/<run>/plan/:
   # - plan.json (with view + artifacts)
   # - plan_validation.json
   # - codespec.json
   ```

6. **Build**
   ```bash
   python -m crpb build --run latest
   # Validations under runs/<run>/validations/
   # Outputs under runs/<run>/outputs/
   ```

---

## Commands (CLI)

The top‑level entrypoint is `python -m crpb`. Subcommands include:

- `plan` — create a plan and codespec for a new run
- `build` — generate files, validate, repair, and merge outputs
- `doctor` — verify CRPB's local scaffolding (no LLM/provider calls)
- `embeddings` — embeddings utilities (health checks)
- `llm` — manage LLM providers/models
- `schemas` — print schemas (for inspection)
- `tasks` — task orchestration helpers (advanced)
- `watch` — optional file watcher/orchestrator (advanced)
- `status`, `replay` — auxiliary/diagnostics (if available in your build)
- `temporal-worker`, `temporal-ctl` — Temporal workflow infrastructure (optional)

Notes:
- `plan` and `build` are Temporal-backed: they require a reachable Temporal server and a running CRPB worker.
- `tasks` runs locally (non-Temporal) and can be used without a Temporal server.

Use `python -m crpb <command> --help` for up‑to‑date flags.

---

## Lossless Logs (No Truncation)

CRPB avoids silent truncation of messages and lists in runtime events/logs. When a payload would be too large to inline, CRPB writes it to a run artifact and emits a small reference (path + digest + length) instead.

Useful environment variables:

- `CRPB_EVENT_MAX_MESSAGE_CHARS` (default `800`): if an event `message` exceeds this, it is externalized to `runs/<run>/artifacts/messages/` and the event includes `message_ref`.
- `CRPB_EVENT_MAX_LIST_ITEMS` (default `200`): lists larger than this are externalized to `runs/<run>/artifacts/lists/` and the event includes `*_ref`.
- `CRPB_STATUS_TAIL` (default `50`): number of recent node-status entries shown by `python -m crpb status`.
- `CRPB_STATUS_MAX_TASKS` (default `100`): max task nodes shown in the `status` table before externalizing the full list.
- `CRPB_REPLAY_MAX_TASKS` (default `100`): max tasks shown in the `replay` table (the full rollup is written to `runs/<run>/outputs/rollups/status.json`).

---

## Planning Details

- **Hierarchy**: Parents coordinate and split; only leaves are implementable.
- **Node context**: `node_plan` (intent, acceptance_criteria, test_plan, …) + `meta.socratic` to sharpen scope.
- **Plan View**: `view.roots` mirrors the hierarchy for consumption by humans/tools (no ids/deps noise).
- **Artifacts**: `artifacts.index[artifact_id] = { producers: [paths], consumers: [paths] }` enforces integration.

CRPB validates and (optionally) repairs the plan before proceeding. You may configure additional constraints via CLI or `.env`.

---

## Build & Validation

- **Pre‑merge**: Per‑file sanity and export checks.
- **Project‑level**: Checks for unused exports, missing producers/consumers, and plan/spec mismatch.
- **Repair loops**: Targeted, conservative fixes using full project context.
- **Post‑merge**: Re‑validation and optional post‑merge repairs.

Build fails when critical integration issues persist, preventing half‑wired outputs.

---

## Providers & Environment

Set provider env keys in `.env` (or shell/CI):

- OpenAI: `OPENAI_API_KEY`, optional `CRPB_OPENAI_MODEL`
- Anthropic: `ANTHROPIC_API_KEY`, optional `CRPB_ANTHROPIC_MODEL`
- Cerebras: `CEREBRAS_API_KEY`, optional `CRPB_CEREBRAS_MODEL`
- HuggingFace: `HUGGINGFACEHUB_API_TOKEN` or `HF_TOKEN`, optional `CRPB_HF_MODEL`
- Local/server: `CRPB_LOCAL_API_KEY` or `CRPB_SERVER_API_KEY` (if required by your runtime)

Other useful settings (see `.env.example`):
- `CRPB_LLM_RETRIES`, `CRPB_LLM_RETRY_BACKOFF`
- `CRPB_LM_MAX_TOKENS`, `CRPB_LM_TEMPERATURE`
- Planning refinement rounds (e.g., `CRPB_PLAN_REFINE_MAX_ROUNDS`)
- Embeddings (optional, disabled by default):
   - Enable: `python -m crpb llm embeddings --enable` (persists in `.crpb_llm.json`) or `CRPB_EMBEDDINGS_ENABLE=true` (env override)
   - Choose provider(s): `CRPB_TASK_EMBED_PROVIDER_PREFERENCE` (comma-separated; e.g., `voyage,openai`)
   - Configure models explicitly (no built-in defaults): `CRPB_VOYAGE_EMBED_MODEL`, `CRPB_OPENAI_EMBED_MODEL`
   - Tuning knobs: `CRPB_TASK_EMBED_MAX_TEXT_CHARS`, `CRPB_TASK_EMBED_BATCH_SIZE`, `CRPB_*_EMBED_TIMEOUT_S`

> Security: `.env` is ignored by Git. Keep real secrets out of the repo; use `.env.example` as the template.

---

## Troubleshooting

- **Plan invalid**: See `runs/<run>/plan/plan_validation.json`. Fix node titles/kinds, node_plan fields, deps, and artifact shapes.
- **Integration gaps**: Project validation reports in `runs/<run>/validations/` list missing producers/consumers and unused exports.
- **Language missing**: This is allowed. The LLM can decide the language from context and/or the chosen file path; best-effort inference from file extension exists but is never required.
- **LLM/provider errors**: Verify `.env` keys and chosen provider/model via `python -m crpb llm --help`.
- **Embeddings unexpectedly on**: Run `python -m crpb llm embeddings --disable` and ensure `CRPB_EMBEDDINGS_ENABLE` is unset/false.
- **Plan/build cannot connect**: Ensure Temporal is running and reachable (`temporal server start-dev`) and the worker is running (`python -m crpb temporal-worker`).
- **Token/length issues**: Reduce idea/constraints scope or raise `CRPB_LM_MAX_TOKENS` cautiously.

---

## Ready to Push to GitHub

- This repo intentionally ignores `tests/` (local-only), `runs/` (generated outputs), and `.env` (secrets). See `.gitignore`.
- If you ever added these to git before `.gitignore` existed, remove them from the index (keeps local files):
   ```bash
   git rm -r --cached tests runs
   git rm --cached .env
   ```
   Then commit the removals.

---

## Contributing

- Fork and branch from `main`.
- Follow Pydantic and Typer/Rich idioms for style; keep imports at file tops.
- Write tests under `tests/` (pytest). Avoid real provider calls; prefer monkeypatch/mocks.
- Keep the system language‑agnostic. Do not hardcode technology stacks or defaults.
- Prefer small, incremental PRs with clear summaries.

---

## License

See `LICENSE`.
