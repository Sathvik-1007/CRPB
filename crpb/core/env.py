from __future__ import annotations

import os

from dotenv import dotenv_values, find_dotenv, load_dotenv


def load_env() -> None:
    """Load environment variables from the nearest `.env`.

    Behavior:
    - Does not override non-empty variables already set in the process environment.
    - Treats empty/whitespace-only variables as "unset" and fills them from `.env`.

    This avoids a common pitfall where an env var exists but is empty and python-dotenv
    refuses to override it (override=False), causing confusing "KEY not set" errors.
    """

    try:
        env_path = find_dotenv(usecwd=True)
    except Exception:
        env_path = ""

    # First pass: standard dotenv behavior (no override of existing env vars)
    try:
        if env_path:
            load_dotenv(env_path, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        return

    # Second pass: fill blanks from .env without overriding non-empty values
    if not env_path:
        return

    try:
        vals = dotenv_values(env_path)
        for k, v in (vals or {}).items():
            if not isinstance(k, str) or not k.strip():
                continue
            if v is None:
                continue
            cur = os.environ.get(k)
            if cur is None or str(cur).strip() == "":
                os.environ[k] = str(v)
    except Exception:
        return
