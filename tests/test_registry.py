"""Unit tests for registry module with CAS-safe updates."""

import tempfile
from pathlib import Path

import pytest

from crpb.registry import (
    MAX_REGISTRY_RETRIES,
    ConflictError,
    Registry,
    _compute_checksum,
)


class TestComputeChecksum:
    """Tests for checksum computation."""

    def test_checksum_same_content_same_hash(self):
        """Same content should produce same checksum."""
        content = "hello world"
        hash1 = _compute_checksum(content)
        hash2 = _compute_checksum(content)
        assert hash1 == hash2

    def test_checksum_different_content_different_hash(self):
        """Different content should produce different checksum."""
        hash1 = _compute_checksum("hello")
        hash2 = _compute_checksum("world")
        assert hash1 != hash2


class TestRegistryCasSafety:
    """Tests for CAS-safe registry operations."""

    def test_update_with_correct_version(self):
        """Update should succeed with correct expected version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "registry.json"
            registry = Registry(reg_path)

            def mutate(data):
                data["test"] = "value"
                return data

            # Initial update with version 0
            result = registry.update(0, mutate)
            assert result["test"] == "value"
            assert result["version"] == 1

    def test_update_with_wrong_version_raises_conflict(self):
        """Update with wrong version should raise ConflictError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "registry.json"
            registry = Registry(reg_path)

            def mutate(data):
                data["test"] = "value"
                return data

            # Initial update
            registry.update(0, mutate)
            # Update with wrong version (expecting 0, got 1)
            with pytest.raises(ConflictError, match="version conflict"):
                registry.update(0, mutate)

    def test_update_includes_checksum_for_file_content(self):
        """Update should compute and store checksum for file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "registry.json"
            registry = Registry(reg_path)

            def mutate(data):
                data.setdefault("files", {})["test.py"] = {
                    "content": "print('hello')",
                    "status": "generated",
                }
                return data

            result = registry.update(0, mutate)
            entry = result["files"]["test.py"]
            assert "checksum" in entry
            assert entry["checksum"] is not None
            assert len(entry["checksum"]) == 64  # SHA-256 hex length

    def test_get_checksum_retrieves_stored_checksum(self):
        """get_checksum should retrieve previously stored checksum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "registry.json"
            registry = Registry(reg_path)

            content = "hello world"
            expected_checksum = _compute_checksum(content)

            def mutate(data):
                data.setdefault("files", {})["test.py"] = {
                    "content": content,
                }
                return data

            registry.update(0, mutate)

            retrieved = registry.get_checksum("test.py")
            assert retrieved == expected_checksum

    def test_update_retries_on_conflict(self, monkeypatch):
        """Update should retry up to MAX_REGISTRY_RETRIES times on conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "registry.json"
            registry = Registry(reg_path)

            # Track attempts
            attempt_count = [0]

            def mutate_with_conflict(data):
                attempt_count[0] += 1
                # Fail first two attempts, succeed on third
                if attempt_count[0] < MAX_REGISTRY_RETRIES:
                    raise ConflictError("simulated conflict")
                data["success"] = True
                return data

            result = registry.update(0, mutate_with_conflict, max_retries=MAX_REGISTRY_RETRIES)
            assert attempt_count[0] == MAX_REGISTRY_RETRIES
            assert result["success"] is True


class TestRegistryReadonly:
    """Tests for read-only access."""

    def test_read_only_returns_data(self):
        """read_only should return registry data without version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "registry.json"
            registry = Registry(reg_path)

            def mutate(data):
                data["test"] = "value"
                return data

            registry.update(0, mutate)
            data = registry.read_only()
            assert data["test"] == "value"
            assert "version" not in data or data.get("version") is not None
