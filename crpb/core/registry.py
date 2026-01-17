from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Tuple

from ..utils.fs import atomic_write_json, lock_file, read_json


# ----------------------------------------------------------------------
# Lease support utilities (deterministic lease handling for registry entries)
# ----------------------------------------------------------------------
def _current_timestamp() -> float:
    """Return the current time as a Unix timestamp (seconds since epoch)."""
    return time.time()


def _lease_valid(entry: dict) -> bool:
    """
    Determine if a lease entry is still valid.
    A lease is considered valid if both 'lease_owner' and 'lease_expiry'
    fields exist and the expiry timestamp is in the future.
    """
    if not isinstance(entry, dict):
        return False
    owner = entry.get("lease_owner")
    expiry = entry.get("lease_expiry")
    if not isinstance(owner, str) or not isinstance(expiry, (int, float)):
        return False
    return _current_timestamp() < float(expiry)


MAX_REGISTRY_RETRIES = 3


def _compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of content for registry entries."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Registry:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Tuple[int, dict]:
        data = read_json(self.path, {"version": 0, "files": {}})
        return data.get("version", 0), data

    def update(self, expected_version: int, mutate, max_retries: int = MAX_REGISTRY_RETRIES):
        """CAS-safe update with retries and checksum tracking."""
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            try:
                with lock_file(lock_path):
                    cur_ver, data = self.load()
                    if cur_ver != expected_version:
                        raise ConflictError(
                            f"version conflict: expected {expected_version}, got {cur_ver}"
                        )
                    new_data = mutate(data)
                    new_data["version"] = cur_ver + 1

                    # Add checksums to file entries if content is present
                    if isinstance(new_data.get("files"), dict):
                        for _file_path, entry in new_data["files"].items():
                            if isinstance(entry, dict) and "content" in entry:
                                content = entry["content"]
                                if isinstance(content, str) and content:
                                    entry["checksum"] = _compute_checksum(content)

                    # Write to temporary file then atomically replace (T8)
                    temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
                    atomic_write_json(temp_path, new_data)
                    temp_path.replace(self.path)

                    return new_data
            except ConflictError as e:
                last_error = e
                if attempt < max_retries:
                    continue
                raise
            except Exception as e:
                last_error = e
                raise
        if last_error:
            raise last_error

    def get_checksum(self, file_path: str) -> str | None:
        """Get the stored checksum for a file path."""
        _, data = self.load()
        if isinstance(data.get("files"), dict):
            entry = data["files"].get(file_path)
            if isinstance(entry, dict):
                return entry.get("checksum")
        return None

    def read_only(self) -> dict:
        return self.load()[1]

    # ------------------------------------------------------------------
    # Lease management API (deterministic lease handling for registry entries)
    # ------------------------------------------------------------------
    def acquire_lease(self, file_path: str, owner: str, ttl_seconds: int = 30) -> bool:
        """
        Attempt to acquire a lease for a given file path.
        Returns True if the lease was successfully acquired, False otherwise.
        """

        def _mutate(data: dict) -> dict:
            files = data.setdefault("files", {})
            entry = files.setdefault(file_path, {})
            # If no lease or lease expired, grant lease
            if not _lease_valid(entry):
                entry["lease_owner"] = owner
                entry["lease_expiry"] = _current_timestamp() + ttl_seconds
            else:
                # Existing valid lease – do not override
                pass
            return data

        cur_ver, _ = self.load()
        try:
            self.update(cur_ver, _mutate)
            entry = self.read_only()["files"].get(file_path, {})
            return entry.get("lease_owner") == owner
        except ConflictError:
            return False

    def renew_lease(self, file_path: str, owner: str, ttl_seconds: int = 30) -> bool:
        """
        Extend an existing lease if the caller is the current lease owner.
        Returns True on success, False if the lease cannot be renewed.
        """

        def _mutate(data: dict) -> dict:
            files = data.setdefault("files", {})
            entry = files.get(file_path, {})
            if isinstance(entry, dict) and entry.get("lease_owner") == owner:
                entry["lease_expiry"] = _current_timestamp() + ttl_seconds
            return data

        cur_ver, _ = self.load()
        try:
            self.update(cur_ver, _mutate)
            entry = self.read_only()["files"].get(file_path, {})
            return entry.get("lease_owner") == owner
        except ConflictError:
            return False

    def release_lease(self, file_path: str, owner: str) -> bool:
        """
        Release a lease held by the given owner.
        Returns True if the lease was released, False otherwise.
        """

        def _mutate(data: dict) -> dict:
            files = data.setdefault("files", {})
            entry = files.get(file_path, {})
            if isinstance(entry, dict) and entry.get("lease_owner") == owner:
                entry.pop("lease_owner", None)
                entry.pop("lease_expiry", None)
            return data

        cur_ver, _ = self.load()
        try:
            self.update(cur_ver, _mutate)
            entry = self.read_only()["files"].get(file_path, {})
            return not entry.get("lease_owner")
        except ConflictError:
            return False

    def lease_info(self, file_path: str) -> dict | None:
        """
        Retrieve lease information for a given file path.
        Returns a dict with keys 'lease_owner' and 'lease_expiry' if present,
        otherwise None.
        """
        _, data = self.load()
        if isinstance(data.get("files"), dict):
            entry = data["files"].get(file_path)
            if isinstance(entry, dict) and ("lease_owner" in entry or "lease_expiry" in entry):
                return {
                    "lease_owner": entry.get("lease_owner"),
                    "lease_expiry": entry.get("lease_expiry"),
                }
        return None


class ConflictError(Exception):
    """Exception indicating a version conflict during registry update."""

    pass
