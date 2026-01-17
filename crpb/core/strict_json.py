"""Utility for enforcing strict JSON handling on LLM calls.

This module provides a decorator ``@strict_json`` that wraps any function
returning a JSON string. It parses the output using the project's internal
``_parse_json_dict_strict`` helper (found in ``crpb.utils.artifacts``) and
ensures the result is a *single‑line* minified JSON representation (Axiom A5).

The decorator is deliberately language‑agnostic – it merely enforces a
contract on the LLM output and does not depend on any specific model.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable, cast


# ---------------------------------------------------------------------------
# Lazy import of the project's strict JSON parser to avoid circular imports.
# ---------------------------------------------------------------------------
def _lazy_import_parser() -> Callable[[str], dict]:
    try:
        from crpb.utils.artifacts import _parse_json_dict_strict  # type: ignore

        return _parse_json_dict_strict
    except Exception as exc:  # pragma: no cover – safety net
        raise ImportError(
            "Could not import _parse_json_dict_strict from crpb.utils.artifacts"
        ) from exc


# Resolve the parser once at import time – the function is simple and safe.
_parse_json_dict_strict = _lazy_import_parser()


class JSONValidationError(RuntimeError):
    """Raised when the LLM output cannot be parsed as strict JSON.

    The original payload is attached to the exception for debugging.
    """

    def __init__(self, payload: str, msg: str | None = None):
        self.payload = payload
        super().__init__(msg or "Invalid JSON returned by LLM")


def _minify_json(data: dict) -> str:
    """Return a deterministic, single‑line JSON string.

    ``separators`` removes all unnecessary whitespace. ``sort_keys`` makes the
    output deterministic across runs.
    """
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def strict_json(func: Callable[..., str]) -> Callable[..., dict]:
    """Decorator that enforces strict JSON contracts on LLM callables.

    The wrapped function must return a JSON *string*. This decorator parses it
    using the project's strict parser, validates it, and returns the parsed
    dictionary. If parsing fails, ``JSONValidationError`` is raised with the raw
    payload attached.

    Example
    -------
    >>> @strict_json
    ... def call_llm(prompt: str) -> str:
    ...     return "{\"answer\": 42}"  # LLM output
    >>> call_llm("test")
    {'answer': 42}
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        raw_output = func(*args, **kwargs)
        if not isinstance(raw_output, str):
            raise JSONValidationError(str(raw_output), "LLM function did not return a string")
        try:
            parsed = _parse_json_dict_strict(raw_output)
        except Exception as exc:  # pragma: no cover – parser already strict
            raise JSONValidationError(raw_output) from exc
        # Ensure deterministic single‑line representation (A5)
        minified = _minify_json(parsed)
        logging.debug("strict_json: minified output %s", minified)
        return cast(dict, parsed)

    return wrapper


# Export the public API of this module
__all__ = ["strict_json", "JSONValidationError"]
