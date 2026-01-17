"""Unit tests for strict_json module."""

import pytest

from crpb.strict_json import (
    JSONValidationError,
    _lazy_import_parser,
    _minify_json,
    strict_json,
)


class TestStrictJsonDecorator:
    """Tests for @strict_json decorator."""

    def test_decorator_parses_valid_json(self):
        """Decorator should parse and return valid JSON."""

        @strict_json
        def mock_llm_call():
            return '{"answer": 42, "items": ["a", "b"]}'

        result = mock_llm_call()
        assert result == {"answer": 42, "items": ["a", "b"]}

    def test_decorator_enforces_single_line(self):
        """Decorator should enforce single-line minified JSON."""

        @strict_json
        def mock_llm_call():
            return """{"answer": 42,
              "items": ["a"]}"""

        result = mock_llm_call()
        assert result == {"answer": 42, "items": ["a"]}
        # Verify it's single-line by checking no newlines
        minified = _minify_json(result)
        assert "\\n" not in minified

    def test_decorator_raises_error_on_invalid_json(self):
        """Decorator should raise JSONValidationError on invalid JSON."""

        @strict_json
        def mock_llm_call():
            return '{"answer": 42, invalid'

        with pytest.raises(JSONValidationError) as exc_info:
            mock_llm_call()

        assert exc_info.value.payload == '{"answer": 42, invalid'

    def test_decorator_requires_string_output(self):
        """Decorator should raise error if function returns non-string."""

        @strict_json
        def mock_llm_call():
            return {"answer": 42}

        with pytest.raises(JSONValidationError):
            mock_llm_call()

    def test_decorator_sorts_keys(self):
        """Decorator should produce deterministic output with sorted keys."""

        @strict_json
        def mock_llm_call():
            return '{"z": 1, "a": 2, "m": 3}'

        result = mock_llm_call()
        # Keys should be sorted alphabetically
        minified = _minify_json(result)
        # First key should be 'a', last should be 'z'
        assert minified.startswith('{"a":')
        assert '"z":' in minified


class TestMinifyJson:
    """Tests for _minify_json helper."""

    def test_minify_produces_single_line(self):
        """Minified JSON should be single line."""
        data = {"a": 1, "b": {"c": 2}}
        result = _minify_json(data)
        assert "\\n" not in result
        assert result == '{"a":1,"b":{"c":2}}'

    def test_minify_sorts_keys(self):
        """Minified JSON should have sorted keys."""
        data = {"z": 1, "a": 2, "m": 3}
        result = _minify_json(data)
        assert result == '{"a":2,"m":3,"z":1}'


class TestLazyImportParser:
    """Tests for lazy import of _parse_json_dict_strict."""

    def test_imports_parser_from_artifacts(self):
        """Lazy import should successfully import _parse_json_dict_strict."""
        parser = _lazy_import_parser()
        assert parser is not None
        assert callable(parser)

    def test_import_fails_on_missing_module(self):
        """Lazy import should raise ImportError if artifacts module is unavailable."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "crpb.utils.artifacts":
                raise ImportError("No module named 'artifacts'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="Could not import _parse_json_dict_strict"):
                _lazy_import_parser()
        finally:
            builtins.__import__ = original_import
