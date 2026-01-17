"""Public registry API.

The canonical implementation lives in :mod:`crpb.core.registry`.

This shim preserves the stable import path used by downstream code and this
project's tests (e.g. ``from crpb.registry import Registry``).
"""

from __future__ import annotations

from .core.registry import (
    MAX_REGISTRY_RETRIES,
    ConflictError,
    Registry,
    _compute_checksum,
    _current_timestamp,
    _lease_valid,
)

__all__ = [
    "Registry",
    "ConflictError",
    "MAX_REGISTRY_RETRIES",
    "_compute_checksum",
    "_current_timestamp",
    "_lease_valid",
]
