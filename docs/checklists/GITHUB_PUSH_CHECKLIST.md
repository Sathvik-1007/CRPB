# GitHub Push Checklist

Use this checklist to ensure the repo is safe and clean before pushing.

## 1) Secrets

- Confirm `.env` is not tracked:
  - `git status`
  - If it shows up as tracked: `git rm --cached .env`
- Confirm no API keys are committed:
  - `git grep -n "_API_KEY="`

## 1b) Large local-only reference files

- If you previously committed local binaries or doc dumps, remove them from the index:
  - `git rm --cached tools/temporal/temporal.exe docs/references/dspy-full-docs.md docs/references/temporal-docs.md docs/references/h.md`

## 2) Generated outputs

- Confirm `runs/` is not tracked:
  - If it shows up: `git rm -r --cached runs`

## 3) Local-only tests policy

- Confirm `tests/` is not tracked:
  - If it shows up: `git rm -r --cached tests`

## 4) Repo hygiene

- Ensure `.gitignore` exists and includes:
  - `.env`, `runs/`, `tests/`, caches (`__pycache__`, `.pytest_cache`, etc.)
  - local binaries/doc dumps (`tools/temporal/temporal.exe`, `docs/references/dspy-full-docs.md`, ...)
- Run local verification:
  - `python -m pytest -q`

## 5) Sanity

- `README.md` describes how to run (Temporal + worker + plan/build)
- `docs/design/plan.md` reflects the core axioms (A1–A6)
- `docs/references/REFERENCES.md` links to upstream docs (Temporal/DSPy)
