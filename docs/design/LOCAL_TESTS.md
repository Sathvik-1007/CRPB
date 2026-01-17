# Local-only tests

This repository is designed to keep **all pytest tests local-only**.

- Local tests live in the `tests/` folder.
- `tests/` is git-ignored on purpose (see `.gitignore`).
- Deterministic, project-internal checks live in the library under `crpb/validation/` and can be executed via the normal runtime/doctor flows.

## Running local tests

From the repo root:

```bash
python -m pytest -q tests
```

## Recreating the baseline test suite

If you deleted your local `tests/` folder, recreate it by restoring your local copy (or by re-running whatever local scaffolding you maintain). The upstream repo intentionally does not ship pytest tests.
