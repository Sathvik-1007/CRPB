from __future__ import annotations

import pytest

from crpb.utils.fs import normalize_project_relpath


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a/b.txt", "a/b.txt"),
        ("a\\b.txt", "a/b.txt"),
        ("./a/b.txt", "a/b.txt"),
        ("/a/b.txt", "a/b.txt"),
        ("a/./b.txt", "a/b.txt"),
        ("a/x/../b.txt", "a/b.txt"),
    ],
)
def test_normalize_project_relpath_ok(raw: str, expected: str) -> None:
    assert normalize_project_relpath(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        (""),
        ("   "),
        ("../x.txt"),
        ("a/../../x.txt"),
        ("C:/tmp/x.txt"),
        ("C:\\tmp\\x.txt"),
        ("//server/share/x.txt"),
        ("/../x.txt"),
        ("./../x.txt"),
    ],
)
def test_normalize_project_relpath_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_project_relpath(raw)
