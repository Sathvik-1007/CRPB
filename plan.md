# CRPB — Context Recursive Project Builder (Perfected Plan)

## Overview
CRPB is a recursive, multi-agent system that converts a natural-language project idea into a complete, runnable software project.
It enforces rigorous context engineering, parent-mediated coordination, JSON-first artifacts, and strict validation at every level. This document contains the full functional spec, problem analysis, and concrete solutions.

---

## 🎯 Goal
Design a robust system that:
- Accepts a high-level idea (e.g., "Build an e-commerce site with payments, search, and admin") and a set of constraints.
- Automatically decomposes work recursively until atomic code artifacts (functions or small cohesive groups of functions) are produced.
- Ensures correctness by validating structured JSON artifacts before any code is written to disk.
- Maintains sound cross-file/type flows, enforces interfaces, and resists failures and hallucinations.
- Produces a reproducible snapshot of the entire decision and generation process.

---

## Core Principles (short)
- JSON-first: all nodes read/write structured JSON; code is produced as an artifact referenced by JSON.
- Parent-mediated communication: no direct sibling-to-sibling messaging; parents curate and publish registries/contracts.
- Atomic leaves: final tasks implement functions or small function-clusters + tests.
- Validator as gatekeeper: validators enforce contracts before file writes.
- Budgeted recursion: limits on depth, width, and retries to avoid runaway expansion.
- Provenance & audit: every decision and artifact is snapshotted and hashed.
- Event-sourced orchestration: node lifecycle is recorded append-only and is the single source of truth for recovery and replay.
- Leases and heartbeats: work is leased with TTLs and renewed via heartbeats; expired leases are safely recovered.
- Idempotency by construction: all updates carry idempotency keys and optimistic concurrency control to allow safe retries.
- Deterministic assembly: file composition order and ownership are explicit and validated.

---

## System Roles (recap)
- Orchestrator (root runner)
- Root Architect
- Tech Stack Synthesizer
- System Planner
- Module Decomposer
- File Decomposer
- Code Generator (Worker)
- Aggregator
- Validator
- Verifier (build/test runner)
- Recovery Agent
- Memory/Registry Manager (parent responsibility)

---

## Parent-Mediated Registry (detailed)
Purpose: enable safe, auditable sharing of function/file metadata so child nodes can depend on sibling-produced artifacts without direct communication.

Registry responsibilities (maintained by parent):
- Maintain registry.json in the parent scope listing published files and function signatures.
- Provide stub artifacts if a function is declared but not yet implemented (signature + docstring + test placeholder).
- Support query API (JSON read): list_functions(filter), get_file(path), claim_stub(name) (claim for implementation).
- Track provenance: which node published a function, checksum, timestamp.
- Enforce uniqueness: parent applies canonical naming and path normalization.

Registry Schema (example)
{
  "files": {
    "backend/utils/math_utils.py": {
      "functions": {
        "add": {
          "signature": "def add(a: int, b: int) -> int",
          "returns": "int",
          "published_by": "node_uuid",
          "status": "stub|implemented",
          "checksum": "sha256:...",
          "examples": [{"in": {"a":1,"b":2},"out": {"result": 3}}],
          "tests": ["issues_add_test"]
        }
      },
      "exports": ["add","multiply"],
      "timestamp": "iso"
    }
  }
}

Parent-mediated flow for dependency
1. Child A publishes math_utils.add to parent registry (status=stub or implemented).
2. Parent records the metadata and optionally creates a stub file artifact in artifacts/.
3. Child B requests math_utils.add via parent query; parent returns signature + import line.
4. Child B uses import; validator later ensures the import exists and matches signature; if mismatch, validator triggers repair.

Stubs & Parallelism
- To support parallel work, parent will create stubs (signatures + tests) for functions that will be implemented later.
- This permits consumers to generate code referencing stubs while workers implement them concurrently.
- Aggregator will replace stubs with implementations when worker returns implemented status.

---

## Problems We Face — Full List (think-through)
Below are the major classes of problems CRPB will face if left unchecked, followed by concrete solutions for each.

1) Sibling Dependency & No Direct Communication
Problem: Sibling nodes need to rely on each other's outputs (functions, classes) but cannot directly talk.
Solution: Parent-Mediated Registry + stubs (see above). Parents mediate imports and stubs, ensuring consistent contracts.

2) Race Conditions & Concurrent Writes
Problem: Parallel children might publish to the registry simultaneously or attempt to modify the same file artifact.
Solution:
- Parent serializes registry writes with a simple file-lock or atomic rename pattern.
- Each registry update is evented in logs/events.jsonl and reflected in registry/registry.json with versioned snapshots; node lifecycle updates are recorded in graph/node_status.jsonl.
- Use optimistic locking with version field; parent compares and applies updates; on conflict it retries merge strategy.
- Limit concurrency: parent controls max_parallel_children.

3) Circular & Tight Coupling Between Modules
Problem: Children might produce designs that cause circular imports or impossible interface cycles.
Solution:
- Dependency graph detection: parent builds an interface graph and runs cycle detection.
- On cycles: parent proposes refactor (abstract interface module, dependency inversion, or co-locate code into the same file).
- Validators block module-level aggregation till cycle resolved.

4) Naming Collisions & Path Ambiguity
Problem: Multiple children might choose the same file/function names.
Solution:
- Parent is canonical namer: all children propose responsibility names; parent maps to file paths and enforces uniqueness via suffixing or namespace prefixes.
- Maintain human-readable canonical names + internal UUIDs.

5) Interface Evolution & Backwards Compatibility
Problem: A function signature changes and many consumers break.
Solution:
- Versioned contracts: when a function signature changes, parent creates v2 entry and keeps v1 until consumers migrate or adaptors are generated.
- Parent can auto-generate adapter shims for backward compatibility.

6) Hallucinations & Incorrect Code (logic bugs)
Problem: LLMs hallucinate or produce logically incorrect implementations.
Solution:
- Strict JSON-first FunctionSpec with examples & tests; generator must return tests.
- Validator checks spec conformance and static checks (AST parsing, type signatures).
- If available, run unit tests in sandbox; otherwise, run static analysis and linter.
- Use lower-temperature deterministic generation for code; store model outputs to allow reruns with identical results.

7) Token & Cost Budgeting
Problem: Large projects cause massive token usage and cost blowouts.
Solution:
- Budget estimation before expansion: each node computes est_tokens and parent refuses expansion if budget exceeded.
- Use smaller models for smaller tasks, reserve high-capacity model calls for architectural decisions only.
- Summarize/ compress context; reuse cached generations.

8) Infinite Recursion / Runaway Decomposition
Problem: Nodes keep splitting forever.
Solution:
- Enforce max_depth and max_children.
- Each node has a decompose_budget and retry_budget.
- At depth cap, node consolidates responsibilities rather than further decomposing.

9) Inconsistent Code Style & Lint Failures
Problem: Generated code uses different styles and linter errors.
Solution:
- Enforce code style via formatters: Black, Prettier, ESLint auto-fix step in aggregator.
- Validator runs style checks and requests autocorrections before commit.

10) Missing Imports & Unresolved Symbols
Problem: Consumer file imports names not actually exported.
Solution:
- Parent registry confirms the import exists; validator compares import lines vs registry entries.
- Aggregator may create import stubs or move implementations to a shared util file.

11) Flaky Tests & Environment Mismatch
Problem: Tests pass locally but fail elsewhere due to environment differences.
Solution:
- Provide deterministic mock environments in tests (mock time, random seeds).
- Use containerized Verifier later for real execution; until then, rely on static checks and mocked tests.

12) Merge Conflicts in Aggregation
Problem: Multiple children change same file sections.
Solution:
- Use componentized file assembly: aggregator composes file from ordered chunks (functions), not free-form concatenation.
- Aggregator resolves conflicts by chunk ownership; if two children claim same chunk, parent arbitrates and issues repair_plan.

13) Secrets & Sensitive Data Leakage
Problem: Generated code may bake secrets or credentials into artifacts.
Solution:
- Validators scan for secret-like patterns and block commits containing secrets.
- Replace secrets with placeholders and document where to set credentials (ENV, vault).
- Require human approval to accept code containing flagged patterns.

14) External API Integrations & Credentials
Problem: End-to-end integration requires keys the system shouldn't have.
Solution:
- Generate adapter stubs and mock clients for tests.
- Expose placeholders in secrets.sample.env and document setup steps.

15) Licensing & Copyright Issues
Problem: Models might generate copyrighted snippets.
Solution:
- Validators check for suspicious long verbatim text (heuristics) and flag for manual review.
- Include a LICENSE header template and recommended license selection.

16) Observability, Auditing & Provenance
Problem: Hard to audit decisions and reproductions.
Solution:
- Snapshot every node: inputs, outputs, decisions, model response, checksum.
- Hash chain snapshots (parent includes child snapshot hashes).
- Keep decisions.jsonl with why and evidence for traceability.

17) Recovery & Fault Tolerance
Problem: If a node fails repeatedly, the project stalls.
Solution:
- Recovery agent applies minimal diffs, attempts alternate implementations, or requests parent refactor.
- Expose manual human-in-the-loop gate with diff and rationale.

18) Cross-Language Interoperability
Problem: Projects may mix languages with different module semantics.
Solution:
- Each language has an adapter that specifies import semantics, module naming rules, and packaging policy.
- Parent enforces consistent interface formats (OpenAPI, Proto, JSON schemas) at module boundaries.

19) Security / Adversarial Prompts
Problem: User prompts or child-generated content may attempt to subvert the system.
Solution:
- Sanitizer agent to pre-process user prompts.
- Validator red-team tests: known jailbreak patterns.
- Quarantine suspicious nodes and require manual review.

20) Non-determinism & Reproducibility
Problem: Different runs produce different code.
Solution:
- Store model outputs and use them as canonical artifacts.
- Allow replay mode that takes stored outputs to reproduce builds deterministically.
- Record model version, temperature, and prompt tokens.

21) Cost of Retries & Exhaustion
Problem: Excess retries are expensive.
Solution:
- Retry budget per node; parent enforces costs and may reassign or consolidate tasks.
- Offer a simulation dry-run mode (no LLM calls) to validate plan structure first.

22) User Interactivity, Edits, and Mid-run Changes
Problem: User changes mind mid-build (switch tech stack).
Solution:
- Versioned plan: changes create new plan version; parent computes diff and only re-generates impacted modules.
- Provide interactive gates: Approve tech stack before decomposition.

23) Type & Data Flow Integrity
Problem: Mismatch of types across files or unclear data models.
Solution:
- Maintain a types.json at module scope; every function spec references types by name.
- Validator performs signature checks; later integrate static typing or schema checks (mypy, TypeScript).

24) File Path Security & Sanitization
Problem: Arbitrary file paths could escape project root.
Solution:
- Parent normalizes paths, disallows .. traversal, enforces project-root sandbox.

25) Estimation, Scheduling & Prioritization
Problem: Long-running projects need prioritization.
Solution:
- Early estimator assigns priority_score to modules based on user-specified importance, value, and complexity.
- Parent schedules high-priority modules first and may produce an MVP subset deliverable early.

---

## Node Lifecycle & Parent Rollups
States: CREATED → PLANNED → READY → LEASED → RUNNING → WAITING_DEP → PRODUCED_STUB | PRODUCED_IMPL → VALIDATING → MERGED → DONE | FAILED_RETRY_i → FAILED_FINAL | CANCELLED.

Parent rollup rules:
- If any child FAILED_FINAL with no repair path: parent = BLOCKED.
- If any child WAITING_DEP or LEASE_EXPIRED: parent = DEGRADED_WAIT.
- If any child RUNNING/VALIDATING: parent = RUNNING.
- All children DONE and gates pass: parent = DONE (post-merge) or READY_TO_MERGE.

Node status record (append-only):
`run_<ts>/graph/node_status.jsonl` entries include `node_id`, `state`, `prev_state`, `at`, `lease_id`, `retries`, `waiting_for`, budgets, and metrics.

## Scheduling, Leases & Backpressure
- Ready set = nodes whose dependencies are satisfied to the minimum required status (stub or implemented) and within budget.
- Priority queue = priority_score with age boost; scheduled under `max_parallel_children` and per-module caps.
- Leases: parent grants `lease_id` with TTL; renew via heartbeats. On expiry → reclaim and re-queue.
- Backpressure: parent throttles fan-out when validations lag or budgets near limits.

## Wait/Notify via Events & Futures
- Futures: registry entries are futures with `status = stub|implemented`. Consumers declare `min_status`.
- Event bus: `run_<ts>/logs/events.jsonl` publishes FUNCTION_PUBLISHED/IMPLEMENTED, VALIDATION_PASSED/FAILED, LEASE_EXPIRED, REPAIR_PLANNED.
- Waiting nodes subscribe (watch) to keys and resume deterministically on matching events.

## Failure Handling & Self-Healing
- Retries: capped exponential backoff with jitter, tracked per node; idempotent operations only.
- Reassignment: parent can reassign to alternate models/strategies or further decompose tasks.
- Repair plans: auto-generated diffs or refactors recorded under `validations/repair_plans/`.
- Partial salvage: keep candidate chunks; validator compares vs FunctionSpec to reuse.

## Locking & Idempotency
- Chunk-level locks for `artifacts/chunks/<function_id>.txt` writes.
- Registry updates use optimistic concurrency with version fields; conflicts are retried deterministically.
- Idempotency keys = hash(spec + lease_id + model_version) attached to all writes.

## Updated Storage Model (include registry & stubs)
  run_<ts>/
    plan/
      idea.json
      constraints.json
      acceptance.json
      tech_stack.json
  graph/
    nodes.jsonl
    edges.jsonl
    node_status.jsonl        # lifecycle state changes (append-only)
    leases.json              # active lease table
  registry/
    registry.json         # parent-curated function/file registry
    stubs/                # generated signature stubs with placeholders
  specs/
    module_<id>.json
    file_<id>.json
    func_<id>.json
  artifacts/
    chunks/               # code chunks per function
    file_<path>.json      # assembled file meta
  validations/
    node_<id>_validation.json
    repair_plans/            # auto-generated repair plans
  logs/
    decisions.jsonl
    events.jsonl
  outputs/
    project_structure.json
    manifest.json
    rollups.json             # parent rollups and critical path

---

## Example Detailed Flow (Sibling coordination)
  1. Planning: Parent planner decides files A and B; both need math utilities.
  2. Registry stub: Parent creates backend/utils/math_utils.py stub with add(a:int,b:int)->int and status: stub.
  3. Child A (implementer) claims add stub and implements it: updates registry status: implemented, and writes chunks/issue_add_<hash>.txt.
  4. Child B (consumer) queries parent, obtains import line and signature, writes code referencing add.
  5. Aggregator composes file chunks; validator checks consumer imports vs registry implementation.
  6. If mismatch: validator offers repair_plan like rename add->sum in both files or generate adapter.

## Deterministic Assembly (order & ownership)
- Order: imports → type defs → public API → private helpers → tests.
- Ownership: each chunk owned by exactly one node; conflicting claims → parent arbitrates and issues a repair plan.
- Normalization: formatters/lint run pre-merge; imports deduped and sorted; exports validated against registry.

## Acceptance & Readiness Gates (recap)
- File gate: All functions declared in FileSpec implemented (or pre-agreed stubs), import graph resolved, at least one test per function, style checks pass.
- Module gate: File gates pass; internal interfaces are consistent; module acceptance tests planned.
- Project gate: Module gates pass; acceptance scenarios mapped to tests; rollups show no BLOCKED nodes.
- Resilience gate: failure injections (lease expiry, signature mismatch, merge conflict) auto-recover to green via retries or repairs.

---

## Developer UX & Debugging Tools
- run_<ts>/decisions.jsonl shows every reason why nodes made choices.
- graph/replay.py replays a run deterministically from snapshots and events.
- CLI commands: `crpb plan`, `crpb dry-run`, `crpb build`, `crpb status`, `crpb resume`, `crpb watch <key>`, `crpb repair <node_id>`, `crpb audit <node_uuid>`, `crpb inject-failure --node <id>`.
- outputs/manifest.json includes all checksums and model traces to allow deterministic reproduction.

---

## Next Steps (what I'll do if you say GO)
1. Repo skeleton: JSON schemas, storage folders, CLI scaffolding.
2. Orchestrator + Registry Manager + Event Bus: implement append-only events and registry with optimistic concurrency.
3. Lifecycle & Leases: node state machine, leases.json, heartbeats, `node_status.jsonl` logging, `crpb status/resume`.
4. Decomposition & Stubs: FileDecomposer + FunctionSpec emitter + parent-created stubs and futures.
5. Validator v1: spec checks, import assertions, style/AST checks; record validations and repair plans.
6. Aggregator v1: deterministic assembly order, chunk ownership enforcement, formatters.
7. Dry-run mode: schedule + validate end-to-end without LLM calls; deterministic replay via `graph/replay.py`.
8. Verifier (optional): sandboxed unit test execution; metrics and rollups.

---

## Manas Principles Alignment
- First-principles context engineering: JSON-first specs, parent-owned contracts, validators as gates.
- Agentic orchestration: event-sourced lifecycle, leases/heartbeats, explicit scheduling and backpressure.
- Determinism and auditability: idempotency keys, append-only logs, reproducible replay.

## Final notes
This plan anticipates the largest real-world pitfalls and provides concrete guardrails and mechanisms that avoid hallucination-driven corruption of the build. We enforce the pattern: plan → specs → stubs → generate → validate → aggregate → write. Parents own contracts; validators gate promotions; registry mediates sibling dependencies; and snapshots enable full reproducibility.
