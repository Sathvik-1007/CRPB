from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from crpb.repairing.models import Issue
from crpb.repairing.provider import RepairProvider
from crpb.repairing.spider import repair_outputs_until_ok_with_oracle


def _oracle_needs_two_files(root: Path) -> Dict[str, Any]:
    # Clean only if both files have the marker.
    a = (root / "a.txt").read_text(encoding="utf-8")
    b = (root / "b.txt").read_text(encoding="utf-8")
    issues: List[str] = []
    if "A_OK" not in a:
        issues.append("missing_file:a.txt")
    if "B_OK" not in b:
        issues.append("missing_file:b.txt")
    return {"ok": len(issues) == 0, "issues": issues, "warnings": [], "suggestions": []}


@dataclass
class _NeedsPathsProvider(RepairProvider):
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
        # First call: request b.txt (RLM-style) if not present.
        if self.calls == 0:
            self.calls += 1
            if "b.txt" not in files:
                return {"needs_paths": ["b.txt"], "patches": [], "notes": ["need_b"]}

        # Second call: now we expect b.txt loaded; propose edits for both.
        a_old = files.get("a.txt", "")
        b_old = files.get("b.txt", "")
        return {
            "edits": [
                {"path": "a.txt", "new_text": (a_old + "\nA_OK\n"), "rationale": "mark a"},
                {"path": "b.txt", "new_text": (b_old + "\nB_OK\n"), "rationale": "mark b"},
            ],
            "notes": ["apply_markers"],
        }


def test_repair_spider_expands_context_from_needs_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")

    provider = _NeedsPathsProvider()
    report = repair_outputs_until_ok_with_oracle(
        outputs_dir=tmp_path,
        oracle=_oracle_needs_two_files,
        provider=provider,
        idea="fixture",
        constraints={},
        file_specs={"a.txt": {}, "b.txt": {}},
        max_rounds=2,
        max_files_per_round=1,  # ensures b.txt isn't included initially
        validations_dir=None,
        label="test",
    )

    assert report.ok is True


@dataclass
class _SpuriousNewFileProvider(RepairProvider):
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
        # Propose creating a random new file that is not expected.
        return {
            "edits": [
                {
                    "path": "random/new.txt",
                    "new_text": "SHOULD_NOT_CREATE\n",
                    "rationale": "spurious",
                }
            ],
            "notes": ["spurious_new"],
        }


def test_repair_spider_blocks_spurious_new_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A0\n", encoding="utf-8")

    provider = _SpuriousNewFileProvider()
    report = repair_outputs_until_ok_with_oracle(
        outputs_dir=tmp_path,
        oracle=_oracle_needs_two_files,
        provider=provider,
        idea="fixture",
        constraints={},
        file_specs={"a.txt": {}, "b.txt": {}},
        max_rounds=1,
        max_files_per_round=1,
        validations_dir=None,
        label="test",
    )

    assert report.ok is False
    assert not (tmp_path / "random" / "new.txt").exists()
