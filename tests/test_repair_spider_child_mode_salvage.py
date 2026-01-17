from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from crpb.repairing.models import Issue
from crpb.repairing.provider import RepairProvider
from crpb.repairing.spider import repair_outputs_until_ok_with_oracle


def _oracle_a_marker(root: Path) -> Dict[str, Any]:
    a = (root / "a.txt").read_text(encoding="utf-8")
    b = (root / "b.txt").read_text(encoding="utf-8")

    issues: List[str] = []
    if "A_OK" not in a:
        issues.append("missing_marker:a.txt:A_OK")
    if "B_OK" not in b:
        issues.append("missing_marker:b.txt:B_OK")

    return {"ok": len(issues) == 0, "issues": issues, "warnings": [], "suggestions": []}


@dataclass
class _GoodPlusBadProvider(RepairProvider):
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
        # Intentionally return a bundle with:
        # - a GOOD edit that fixes the remaining fatal
        # - a BAD edit that breaks an already-correct file
        a_old = files.get("a.txt", "")
        b_old = files.get("b.txt", "")
        return {
            "edits": [
                {"path": "a.txt", "new_text": a_old + "\nA_OK\n", "rationale": "fix a"},
                {"path": "b.txt", "new_text": b_old.replace("B_OK", ""), "rationale": "break b"},
            ],
            "notes": ["good_plus_bad"],
        }


def test_child_mode_salvages_good_edit_and_rejects_bad(tmp_path: Path) -> None:
    # Start with only a.txt missing its marker; b.txt is already correct.
    (tmp_path / "a.txt").write_text("A0\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B0\nB_OK\n", encoding="utf-8")

    provider = _GoodPlusBadProvider()

    report = repair_outputs_until_ok_with_oracle(
        outputs_dir=tmp_path,
        oracle=_oracle_a_marker,
        oracle_micro=_oracle_a_marker,
        provider=provider,
        idea="fixture",
        constraints={"repair_child_mode": True, "repair_child_parallel_workers": 1, "repair_child_batch_size": 1},
        file_specs={"a.txt": {}, "b.txt": {}},
        max_rounds=2,
        max_files_per_round=1,
        validations_dir=None,
        label="test",
    )

    assert report.ok is True
    assert "A_OK" in (tmp_path / "a.txt").read_text(encoding="utf-8")
    assert "B_OK" in (tmp_path / "b.txt").read_text(encoding="utf-8")
