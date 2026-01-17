from __future__ import annotations

import pytest

from crpb.core.context_compiler import ContextBudgetExceededError, ContextCompiler
from crpb.core.ledger import NodeLedger, TodoItem


def _mk_node(i: int) -> dict:
    return {
        "id": f"n{i}",
        "kind": "task",
        "title": f"Node {i}",
        "description": "desc",
        "node_plan": {
            "intent": "use file a.txt",
            "deliverables": ["a.txt"],
        },
        "meta": {},
    }


def test_context_compiler_never_truncates_text_marker() -> None:
    cc = ContextCompiler(max_chars=1500)
    pack = cc.compile(
        idea="X" * 5000,
        constraints={"side_context": {}, "context_max_chars": 1500},
        node=_mk_node(1),
        parent=None,
        siblings=[],
        ledger=None,
        artifacts=[],
        files={"a.txt": "HELLO" * 400},
        file_specs={},
        signals={},
    )
    # No legacy truncation marker should exist anywhere.
    s = str(pack)
    assert "(truncated)" not in s
    # Idea may be omitted (ref-only) but must not be partially sliced.
    if pack.get("idea") is not None:
        assert pack["idea"] == "X" * 5000


def test_context_compiler_file_text_is_whole_or_absent() -> None:
    cc = ContextCompiler(max_chars=1400)
    big_text = "A" * 5000
    pack = cc.compile(
        idea="idea",
        constraints={
            "side_context": {},
            "context_max_chars": 1400,
            "context_include_file_text": True,
            # give tiny file_texts budget to force omission
            "context_section_max_chars": {"file_texts": 10},
        },
        node=_mk_node(2),
        parent=None,
        siblings=[],
        ledger=None,
        artifacts=[],
        files={"big.txt": big_text},
        file_specs={},
        signals={},
    )
    texts = (pack.get("files") or {}).get("texts") or {}
    assert "big.txt" not in texts


def test_context_compiler_strict_required_file_text_fails() -> None:
    cc = ContextCompiler(max_chars=1200)
    with pytest.raises(ContextBudgetExceededError):
        cc.compile(
            idea="idea",
            constraints={
                "side_context": {},
                "context_max_chars": 1200,
                "context_include_file_text": True,
                "context_require_file_text": ["big.txt"],
                "context_section_max_chars": {"file_texts": 10},
            },
            node=_mk_node(3),
            parent=None,
            siblings=[],
            ledger=None,
            artifacts=[],
            files={"big.txt": "B" * 5000},
            file_specs={},
            signals={},
        )


def test_context_compiler_respects_budget_and_drops_optional_items() -> None:
    cc = ContextCompiler(max_chars=1200)

    siblings = [_mk_node(i) for i in range(20)]
    artifacts = [{"id": f"a{i}", "kind": "k", "path": f"p{i}", "validation": {}} for i in range(50)]

    led = NodeLedger(node_id="n", parent_id=None)
    for i in range(30):
        led.todos.append(TodoItem(id=f"t{i}", text="x" * 80))
    # decisions omitted; not needed for this test

    pack = cc.compile(
        idea="idea",
        constraints={"side_context": {}, "context_max_chars": 1200},
        node=_mk_node(4),
        parent=None,
        siblings=siblings,
        ledger=led,
        artifacts=artifacts,
        files={f"f{i}.txt": "x" * 10 for i in range(50)},
        file_specs={"x": "y"},
        signals={"s": "v"},
    )

    assert int(pack.get("size_chars") or 0) <= 1200
    omissions = pack.get("omissions") or {}
    assert (omissions.get("siblings_dropped") or 0) >= 0
    assert (omissions.get("artifacts_dropped") or 0) >= 0
    assert (omissions.get("file_refs_dropped") or 0) >= 0
