# CRPB Specification (Axioms → Definitions → Algorithms)

This document is the *formal* behavioral specification for CRPB. It is intended to be the single source of truth for:

- language-agnostic project construction (no hardcoded stacks)
- LLM-driven decision making (paths/extensions/technologies are LLM-selected under constraints)
- deterministic orchestration (especially under Temporal replay)
- budgeted, structured context engineering

The purpose is to close the “holes” you described:

- missing subtask coverage (subtasks implied by acceptance criteria/test plans are not enforced)
- uneven context distribution (some nodes get too much / too little)
- weak persistent memory (TODOs exist but aren’t precise, scoped, and replay-stable)

This spec is written in a research-paper style: explicit notation, total/partial functions, invariants, and algorithms.

---

## 0. Notation

- Let $\Sigma^*$ denote finite strings.
- Let $\mathcal{P}(X)$ denote the power set of a set $X$.
- Let $\text{Dict}(K,V)$ denote finite maps from keys to values.
- A *partial function* $f: X \rightharpoonup Y$ may be undefined for some $x \in X$.
- For a list $L$, $\text{sort}(L)$ denotes a deterministic sort (lexicographic on stable keys).

---

## 1. Core Axioms (Non‑Negotiable)

### A1 (LLM Supremacy)
All *substantive build decisions* are produced by an LLM call subject to constraints: languages, frameworks, file paths, extensions, schemas, and integrations.

**Corollary:** Any inference utility (e.g., language-from-extension) is *best-effort* and must never be required for correctness.

### A2 (Deterministic Orchestration)
Any orchestration layer (especially Temporal workflows) must be deterministic under replay.

Formally: for fixed inputs and fixed external artifact store contents, the workflow must produce the same sequence of decisions and activity invocations.

### A3 (Artifact-by-Reference)
No large payloads are passed through orchestration state.

Temporal activities may read/write large contents, but workflow state and activity args/returns must remain small: references (paths, IDs, digests) only.

### A4 (Budgeted Context)
Every LLM invocation consumes a ContextPack bounded by a budget $B$ (measured in characters, not tokens).

### A5 (No Empty Composites)
Composite nodes have children: if $\text{kind}(n)$ is composite, then $\text{children}(n) \neq []$.

### A6 (Coverage Closure)
All obligations introduced by the plan must be discharged by leaves.

“Discharged” is a formally checkable property defined below.

---

## 2. Data Model Definitions

### 2.1 Run
A **Run** is a tuple $R = (\text{run\_id}, \text{run\_dir})$ where:

- $\text{run\_id} \in \Sigma^*$ is a stable identifier.
- $\text{run\_dir}$ is a filesystem directory containing all persisted artifacts.

### 2.2 Node
A **Node** is a record:

$$
n = (id, kind, title, desc, plan, meta, children)
$$

Where:

- $id \in \Sigma^*$
- $kind \in \Sigma^*$ (e.g., “composite:*” or “code:function”)
- $title, desc \in \Sigma^*$
- $plan$ is a finite dict (“node_plan”) containing (at minimum) acceptance criteria and a test plan.
- $children$ is a list of Nodes.

We define:

- $\text{isLeaf}(n) := (children = [])$.
- $\text{isComposite}(n) := \neg\text{isLeaf}(n)$.

### 2.3 Obligation
An **Obligation** is a structured requirement that must eventually be discharged.

We model an obligation as:

$$
o = (oid, scope, statement, dod, deps, tags)
$$

- $oid \in \Sigma^*$ unique within a run.
- $scope \in \{\text{node}, \text{project}\}$.
- $statement \in \Sigma^*$ (what must become true)
- $dod \in \Sigma^*$ (definition of done; observable)
- $deps \subseteq \Sigma^*$ (artifact IDs or other obligations)
- $tags \subseteq \Sigma^*$

### 2.4 Artifact
An **Artifact** is:

$$
a = (aid, kind, path, meta)
$$

- $aid \in \Sigma^*$ stable ID
- $kind \in \Sigma^*$ (freeform)
- $path \in \Sigma^*$ optional relative path within run artifacts/outputs
- $meta$ is small metadata (validation status, digests, provenance)

### 2.5 Node Ledger
Each node has persistent memory stored in a **NodeLedger**.

Ledger is a record $\ell(n)$ containing:

- TODOs (open/in_progress/done/blocked)
- Decisions (statement + rationale)
- Obligations inherited + created
- Produced/consumed artifacts
- ContextPack digests (provenance)

Concrete storage: `run_dir/artifacts/ledger/<node>.json` (see `crpb/core/ledger.py`).

### 2.6 ContextPack
A **ContextPack** is a dict with stable sections and a bounded size.

We define sections:

- `core`: node identity + idea summary
- `ledger`: open TODOs, decisions, obligations
- `graph`: parent/siblings summaries + plan skeleton
- `artifacts`: artifact metadata only
- `files`: touched paths + file references; optional whole-file texts (no truncation)
- `signals`: deterministic validation findings

Concrete compilation exists in `crpb/core/context_compiler.py`.

**Non-negotiable implementation rule:** Context compilation must not truncate/slice text. If a piece of content cannot fit, it must be omitted as a whole item (with omission metadata recorded) or the compilation must fail in strict mode.

---

## 3. Derived Structures

### 3.1 Obligation Extraction
For each node $n$, define an extraction function:

$$
	ext{Obl}(n): \text{Node} \to \mathcal{P}(\text{Obligation})
$$

`Obl(n)` is computed from:

- `node_plan.acceptance_criteria`
- `node_plan.test_plan`
- any declared artifacts produced/consumed

**Key requirement:** obligations are not vague. Each obligation must include a definition-of-done (DoD) that is externally checkable.

### 3.2 Discharge Relation
Define a discharge predicate for a leaf node $n$:

$$
	ext{Discharges}(n, o) \in \{\text{true},\text{false}\}
$$

Operationally, a leaf discharges an obligation if the run artifacts contain evidence of completion, e.g.:

- produced artifact exists at expected path or registered by ID
- validator report asserts condition
- test plan items are satisfied by deterministic checks

The spec requires evidence be stored as artifacts (JSON reports, digests, etc.), not transient LLM text.

### 3.3 Coverage Closure
Let $root$ be the plan root.

Define the plan’s obligation set:

$$
O = \bigcup_{n \in \text{Nodes}(root)} \text{Obl}(n)
$$

Define leaves $L = \{ n \mid \text{isLeaf}(n) \}$.

Coverage closure requires:

$$
\forall o \in O,\; \exists \ell \in L : \text{Discharges}(\ell, o)
$$

If closure fails, the planner must refine/split until closure holds (or return a proof obligation explaining why it cannot).

---

## 4. Algorithms (Deterministic)

### 4.1 Context Compilation (Budgeted)

**Inputs:** $(idea, constraints, node, parent, siblings, ledger, artifacts, files, signals)$.

**Budget:** $B = \text{constraints.context\_max\_chars} \;\text{or}\; \text{CRPB\_CONTEXT\_MAX\_CHARS}$.

**Algorithm (no truncation):**

1. Build a minimal pack (node identity + refs) that always fits under $B$ unless $B$ is pathologically small.
2. Add optional sections in a deterministic priority order with *explicit section budgets*.
3. For any list-valued section, include whole items in rank order until that section budget is exhausted (never slice items).
4. If size(pack) > $B$, deterministically drop whole optional items/sections in reverse priority order (never truncate text).
5. Record omission metadata (counts) and compute `digest = SHA1(stable_json(pack))` for provenance.

**Design goal:** no node receives “way more context than needed” because each optional section has a quota and is only included when it increases expected utility.

**Concrete knobs (deterministic inputs):**
- `constraints.context_max_chars` total budget in characters.
- `constraints.context_section_max_chars` optional dict: `{section_name: int}`.
- `constraints.context_section_quotas` optional dict: `{section_name: float}` where values are fractions of remaining budget.
- `constraints.context_sections` optional list of enabled optional sections.
- `constraints.context_include_file_text` boolean (default false): allow including whole file texts.
- `constraints.context_require_file_text` optional list of paths that must be included as whole texts (strict mode fails otherwise).

**Files section:**
- Always include file *references* (path + digest + length) when possible.
- Include whole file texts only when explicitly enabled and when they fit (never truncate).

> Current implementation already enforces a deterministic trim order; this spec adds *quota accounting* and *relevance gating* as a next-step improvement.

### 4.2 Relevance Gating (Context Fairness)

Define a relevance scoring function:

$$
	ext{Rel}(node, item) \in \mathbb{N}
$$

Where `item` is a candidate context element (sibling summary, artifact metadata, file snippet).

`Rel` must be deterministic and computable without LLM calls.

Examples of deterministic signals:

- artifact IDs referenced in `node_plan`
- file paths referenced in `codespec_file`
- open TODO tags matching node kind

Context compiler must only include items with $\text{Rel} > 0$ unless required by invariants.

### 4.3 Node Ledger Update (Structured TODOs)

Each node must write TODOs that are not “titles” but structured context.

Define a TODO as:

$$
t = (tid, text, status, created\_at, updated\_at)
$$

Where `text` must contain (as labeled fields):

- `Trigger:` what caused the TODO to exist
- `Work:` the specific action
- `DoD:` measurable completion criteria
- `Deps:` artifact IDs / file paths / obligations
- `Scope:` node-local vs project

**Algorithm:** after every LLM call affecting a node:

1. Extract obligations `Obl(node)`.
2. For each obligation not yet discharged, ensure a corresponding TODO exists (stable ID derived from obligation ID).
3. Partition TODOs into:
  - node-local (can be done within node)
  - project obligations (require cross-file integration)
4. Save ledger deterministically (sorted IDs).

### 4.4 Subtask Coverage Enforcement (Planner)

The planner must not stop at “leaf-looking” nodes if obligations are not fully partitioned.

Define a splitting predicate:

$$
	ext{NeedSplit}(n) := (\exists o \in \text{Obl}(n)\;\text{s.t.}\; o \text{ is not assigned to any child leaf})
$$

Planner refinement loop must continue while any `NeedSplit(n)` holds.

This specifically targets your complaint: “not considering all subtasks from a task”.

---

## 5. DSPy Contract (Typed IO)

DSPy’s central idea is *Signatures* (declarative input/output behavior). CRPB must treat each major LLM operation as a typed signature, not an unstructured prompt.

We define canonical signatures (conceptual):

- `ClarifyNode`: `context_pack, node -> node_plan_delta, obligations`
- `SplitNode`: `context_pack, node, obligations -> children_nodes, obligation_partition`
- `GenerateFile`: `context_pack, codespec_file -> file_text_or_artifact_ref`
- `ValidateProject`: `context_pack, artifact_refs -> report`

Implementation detail: CRPB’s `DspyEngine` should preserve stable field names and keep outputs parseable; optimization/evaluation is a separate stage (per DSPy docs).

---

## 6. Temporal Contract (Replay + Payload Discipline)

Temporal’s event history persists activity args and returns. Therefore:

- workflows only pass small references (paths, IDs, digests)
- large file text is staged in run outputs and read by activities
- workflow logic uses deterministic ordering and no time/random branching

This contract is implemented in `crpb/temporal/workflow.py` via staged outputs and path-only flow.

---

## 7. Implementation Map (Grounded)

Already implemented:

- Node ledgers: `crpb/core/ledger.py`
- Context packs: `crpb/core/context_compiler.py`
- Planner/tasks integration: `crpb/planning/planner.py`, `crpb/commands/tasks_cmd.py`
- Temporal payload hardening: `crpb/temporal/workflow.py`

Still missing (explicit gaps this spec demands):

1. **Coverage validator**: check Coverage Closure over extracted obligations and emitted evidence.
2. **Relevance gating**: context compiler should include items based on deterministic relevance, not only fixed truncation order.
3. **Obligation → TODO ID mapping**: stable derivation so TODOs persist even across minor rephrasings.
4. **Evidence artifacts**: validators should emit small structured artifacts proving discharge.

---

## 8. Acceptance Criteria

- Determinism: same run inputs → same ContextPack digest and stable ledgers.
- Coverage: every acceptance criterion/test-plan item becomes an obligation and is discharged by some leaf.
- Budgeting: no ContextPack exceeds its budget; optional sections obey quotas.
- Temporal: workflows never store large file text in workflow state/history.
- Tooling: `python -m compileall -q crpb` and `python -m pytest -q` pass.
