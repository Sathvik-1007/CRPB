"""Strict JSON helpers.

The canonical implementation lives in :mod:`crpb.core.strict_json`.

This module exists to provide a stable import path
(``from crpb.strict_json import strict_json``).
"""

from __future__ import annotations

from .core.strict_json import (
    JSONValidationError,
    _lazy_import_parser,
    _minify_json,
    strict_json,
)

__all__ = [
    "strict_json",
    "JSONValidationError",
    "_minify_json",
    "_lazy_import_parser",
]
