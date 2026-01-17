from __future__ import annotations

from typing import Any, Dict, List

import crpb.agents.dspy_engine as dspy_engine_mod
from crpb.planning.planner import generate_task_plan


class _StubEngine:
    def __init__(self) -> None:
        self.split_calls: List[str] = []
        self.clarify_calls: List[str] = []
        self.join_judge_calls: List[str] = []

    def task_plan(self, idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        # Single composite root; the planner should decide_split it.
        return {
            "tasks": [
                {
                    "id": "root",
                    "kind": "composite",
                    "title": "Root",
                    "description": "",
                    "deps": [],
                    "inputs": {},
                    "outputs": {},
                    "children": [],
                    "node_plan": {
                        "intent": "",
                        "in_scope": "",
                        "out_of_scope": "",
                        "constraints": "",
                        "assumptions": "",
                        "deliverables": "",
                        "acceptance_criteria": "",
                        "preconditions": "",
                        "postconditions": "",
                        "interfaces": "",
                        "data_contracts": "",
                        "integration_points": "",
                        "dependencies": "",
                        "sequencing": "",
                        "risks": "",
                        "mitigations": "",
                        "open_questions": "",
                        "test_plan": "",
                        "verification": "",
                        "non_functional": "",
                        "step_outline": "",
                        "completion_definition": "",
                        "success_metrics": "",
                        "ownership": "",
                        "handoffs": "",
                    },
                    "meta": {},
                }
            ]
        }

    def decide_split(self, task: Dict[str, Any], idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        tid = str(task.get("id") or "")
        self.split_calls.append(tid)
        if tid == "root":
            return {
                "action": "split",
                "children": [
                    {
                        "id": "c1",
                        "kind": "code:function",
                        "title": "Child 1",
                        "description": "",
                        "deps": [],
                        "inputs": {"path": "a.txt", "name": "f1"},
                        "outputs": {},
                        "children": [],
                        "node_plan": {"intent": "", "acceptance_criteria": "", "test_plan": ""},
                        "meta": {},
                    },
                    {
                        "id": "c2",
                        "kind": "code:function",
                        "title": "Child 2",
                        "description": "",
                        "deps": [],
                        "inputs": {"path": "b.txt", "name": "f2"},
                        "outputs": {},
                        "children": [],
                        "node_plan": {"intent": "", "acceptance_criteria": "", "test_plan": ""},
                        "meta": {},
                    },
                ],
            }
        return {"action": "keep", "children": []}

    def clarify_task(
        self,
        *,
        task: Dict[str, Any],
        parent: Dict[str, Any],
        siblings: List[Dict[str, Any]],
        artifacts: List[Dict[str, Any]],
        files: Dict[str, str],
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        tid = str(task.get("id") or "")
        self.clarify_calls.append(tid)
        return {"description": f"clarified:{tid}"}

    def amend_task_plan(
        self,
        *,
        current_plan: Dict[str, Any],
        statuses: Dict[str, Any],
        artifacts: List[Dict[str, Any]],
        idea: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        # No-op amendments for tests.
        return {"edits": []}

    def join_judge_node(self, *, node: Dict[str, Any], idea: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        tid = str(node.get("id") or "")
        self.join_judge_calls.append(tid)
        # Always ok; return a rubric + flat questions.
        return {
            "ok": True,
            "actions": [],
            "rubric": {"dimensions": [{"name": "coverage", "pass_conditions": ["children cover parent"]}]},
            "questions": [
                {
                    "id": "q1",
                    "question": "What is the acceptance criteria for this node?",
                    "focus": "acceptance",
                    "why_this_matters": "Needed to verify completion",
                    "expected_answer_shape": "bullet list",
                }
            ],
            "notes": [],
        }


def test_generate_task_plan_parallel_refine_smoke(monkeypatch) -> None:
    stub = _StubEngine()

    monkeypatch.setattr(dspy_engine_mod, "DspyEngine", lambda: stub)

    tp = generate_task_plan(
        "idea",
        {
            "planner_parallel_enable": True,
            "planner_parallel_workers": 4,
            "join_judge_enable": True,
            "join_judge_max_nodes": 10,
            "taskplan_refine_max_rounds": 2,
        },
        use_llm=True,
        enforce_deterministic_leaves=True,
    )

    assert tp.tasks
    assert tp.tasks[0].id == "root"
    assert len(tp.tasks[0].children) == 2
    assert {c.id for c in tp.tasks[0].children} == {"c1", "c2"}

    # Ensure join-judge path is exercised at least once.
    assert "root" in stub.join_judge_calls
