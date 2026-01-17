import json

from crpb.core.ledger import NodeLedgerStore, TodoItem
from crpb.core.obligations import stable_todo_id_from_obligation
from crpb.validation.coverage import compute_obligation_coverage


def test_compute_obligation_coverage_accepts_taskplan_shape(tmp_path):
    # TaskPlan-shaped dict (no view.roots)
    plan = {
        "idea": "x",
        "constraints": {},
        "tasks": [
            {
                "id": "n1",
                "kind": "code:function",
                "title": "t",
                "description": "",
                "node_plan": {
                    "acceptance_criteria": "AC1",
                    "test_plan": "TP1",
                },
                "children": [],
            }
        ],
    }

    ledger_store = NodeLedgerStore(base_dir=tmp_path)

    # Pre-create ledger with TODOs marked done for extracted obligations.
    # We don't need to know the exact obligation ids; compute once and store.
    cov0 = compute_obligation_coverage(plan=plan, run_dir_path=str(tmp_path), ledger_store=None)
    assert cov0["counts"]["obligations"] >= 1

    led = ledger_store.load("n1")
    for o in cov0["undischarged"]:
        tid = stable_todo_id_from_obligation(str(o.get("id") or ""))
        led.todos.append(TodoItem(id=tid, text="x", status="done"))
    ledger_store.save(led)

    cov = compute_obligation_coverage(plan=plan, run_dir_path=str(tmp_path), ledger_store=ledger_store)
    assert cov["ok"] is True
    assert cov["counts"]["undischarged"] == 0


def test_coverage_report_is_not_truncated(tmp_path):
    # Create many obligations by making many lines.
    ac_lines = "\n".join([f"AC{i}" for i in range(50)])
    plan = {
        "idea": "x",
        "constraints": {},
        "tasks": [
            {
                "id": "n1",
                "kind": "code:function",
                "title": "t",
                "description": "",
                "node_plan": {
                    "acceptance_criteria": ac_lines,
                    "test_plan": "",
                },
                "children": [],
            }
        ],
    }

    cov = compute_obligation_coverage(plan=plan, run_dir_path=str(tmp_path), ledger_store=None)
    # Without ledger evidence, all are undischarged; discharged list is empty.
    assert isinstance(cov.get("discharged"), list)
    assert len(cov.get("discharged")) == 0

    # Persisted report should be valid JSON.
    p = tmp_path / "validations" / "coverage.json"
    assert p.exists()
    json.loads(p.read_text(encoding="utf-8"))
