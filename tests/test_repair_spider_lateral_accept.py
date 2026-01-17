from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from crpb.repairing.models import Issue, IssueSeverity
from crpb.repairing.provider import RepairProvider
from crpb.repairing.spider import repair_outputs_until_ok_with_oracle


def _oracle_for_fixture(root: Path) -> Dict[str, Any]:
    a = (root / "a.txt").read_text(encoding="utf-8")
    b = (root / "b.txt").read_text(encoding="utf-8")

    issues: List[str] = []
    if "A1" not in a:
        issues.append("need_a:a.txt")
    if "B1" not in b:
        issues.append("need_b:b.txt")

    # After A is fixed, introduce a new issue on B until C1 is added.
    if "A1" in a and "C1" not in b:
        issues.append("new_issue:b.txt")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "warnings": [],
        "suggestions": [],
    }


@dataclass
class _StubProvider(RepairProvider):
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
        # Minimal deterministic stub. We return full-file edits directly.
        # Round 1: fix A (this removes one targeted issue but introduces another elsewhere).
        # Round 2: fix B.
        if self.calls == 0:
            self.calls += 1
            a_old = files.get("a.txt", "")
            a_new = (a_old + "\nA1\n") if "A1" not in a_old else a_old
            return {
                "edits": [
                    {
                        "path": "a.txt",
                        "new_text": a_new,
                        "rationale": "Add A1 marker to satisfy need_a.",
                    }
                ],
                "notes": ["round1_fix_a"],
            }

        if self.calls == 1:
            self.calls += 1
            b_old = files.get("b.txt", "")
            b_new = b_old
            if "B1" not in b_new:
                b_new += "\nB1\n"
            if "C1" not in b_new:
                b_new += "\nC1\n"
            return {
                "edits": [
                    {
                        "path": "b.txt",
                        "new_text": b_new,
                        "rationale": "Add B1/C1 markers to satisfy need_b and resolve new_issue.",
                    }
                ],
                "notes": ["round2_fix_b"],
            }

        return {"edits": [], "notes": ["done"]}


def test_repair_spider_accepts_lateral_target_fix(tmp_path: Path) -> None:
    # Arrange: two files with two blocking issues.
    (tmp_path / "a.txt").write_text("A0\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B0\n", encoding="utf-8")

    provider = _StubProvider()

    # Act
    report = repair_outputs_until_ok_with_oracle(
        outputs_dir=tmp_path,
        oracle=_oracle_for_fixture,
        provider=provider,
        idea="fixture",
        constraints={},
        file_specs={},
        max_rounds=3,
        max_files_per_round=8,
        validations_dir=None,
        label="test",
    )

    # Assert: repairs converge to clean, and we did not roll back the round-1 lateral fix.
    assert report.ok is True
    assert report.applied_edits >= 2

    a = (tmp_path / "a.txt").read_text(encoding="utf-8")
    b = (tmp_path / "b.txt").read_text(encoding="utf-8")
    assert "A1" in a
    assert "B1" in b
    assert "C1" in b

    # Ensure the report rounds is a real count (not derived from notes length).
    assert report.rounds >= 2
