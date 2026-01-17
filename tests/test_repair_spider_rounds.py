from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from crpb.repairing.models import Issue
from crpb.repairing.provider import RepairProvider
from crpb.repairing.spider import repair_outputs_until_ok_with_oracle


def _oracle_always_bad(_: Path) -> Dict[str, Any]:
    # Deterministic oracle: always returns a single blocking issue.
    return {"ok": False, "issues": ["blocking:root"], "warnings": [], "suggestions": []}


@dataclass
class _NoEditsProvider(RepairProvider):
    def propose_repairs(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        issues: List[Issue],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"edits": [], "notes": ["no_edits"]}


@dataclass
class _SingleEditThenStopProvider(RepairProvider):
    calls: int = 0

    def propose_repairs(
        self,
        *,
        idea: str,
        constraints: Dict[str, Any],
        issues: List[Issue],
        files: Dict[str, str],
        file_specs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.calls == 0:
            self.calls += 1
            # Deterministic full-file edit.
            old = files.get("x.txt", "")
            return {
                "edits": [
                    {
                        "path": "x.txt",
                        "new_text": (old + "\nhello\n"),
                        "rationale": "Create content",
                    }
                ],
                "notes": ["one_edit"],
            }
        return {"edits": [], "notes": ["stop"]}


def test_rounds_reported_on_early_stop(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "x.txt").write_text("", encoding="utf-8")

    # Act: provider returns no edits immediately; loop should stop after the first round.
    report = repair_outputs_until_ok_with_oracle(
        outputs_dir=tmp_path,
        oracle=_oracle_always_bad,
        provider=_NoEditsProvider(),
        idea="fixture",
        constraints={},
        file_specs={},
        max_rounds=3,
        max_files_per_round=8,
        validations_dir=None,
        label="test",
    )

    # Assert
    assert report.ok is False
    assert report.applied_edits == 0
    assert report.rounds == 1


def test_rounds_do_not_exceed_max_rounds(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "x.txt").write_text("", encoding="utf-8")

    provider = _SingleEditThenStopProvider()

    # Act: even if max_rounds is high, the provider stops with no edits after one edit.
    report = repair_outputs_until_ok_with_oracle(
        outputs_dir=tmp_path,
        oracle=_oracle_always_bad,
        provider=provider,
        idea="fixture",
        constraints={},
        file_specs={},
        max_rounds=5,
        max_files_per_round=8,
        validations_dir=None,
        label="test",
    )

    # Assert: should execute round 1 (apply) + round 2 (no edits stop).
    assert report.rounds == 2
    assert report.applied_edits == 1
