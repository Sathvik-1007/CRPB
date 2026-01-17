from __future__ import annotations

import pytest

from crpb.validation.project_crawl import validate_project_outputs_deterministic


def test_validate_project_outputs_deterministic_removed() -> None:
    with pytest.raises(RuntimeError) as e:
        validate_project_outputs_deterministic(outputs_dir=None)  # type: ignore[arg-type]
    assert "Deterministic project crawling has been removed" in str(e.value)
