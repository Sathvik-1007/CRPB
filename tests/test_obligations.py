from crpb.core.obligations import (
    extract_obligations_from_node,
    format_structured_todo_text,
    stable_obligation_id,
    stable_todo_id_from_obligation,
)


def test_extract_obligations_stable_ids():
    node = {
        "id": "n1",
        "node_plan": {
            "acceptance_criteria": "- produces outputs\n- integrates components\n",
            "test_plan": "- run unit tests\n- validate schema\n",
        },
    }

    obs1 = extract_obligations_from_node(node=node, inherited_deps=["artifact:a"])
    obs2 = extract_obligations_from_node(node=node, inherited_deps=["artifact:a"])

    assert [o.id for o in obs1] == [o.id for o in obs2]
    assert all(o.id.startswith("obl:") for o in obs1)
    assert all(o.statement for o in obs1)
    assert all(o.dod for o in obs1)


def test_todo_id_is_deterministic_and_structured():
    node = {
        "id": "n2",
        "node_plan": {
            "acceptance_criteria": "- thing A\n",
            "test_plan": "- thing B\n",
        },
    }
    obs = extract_obligations_from_node(node=node, inherited_deps=[])
    assert obs

    tid = stable_todo_id_from_obligation(obs[0].id)
    assert tid.startswith("todo:")

    text = format_structured_todo_text(obligation=obs[0])
    # Must include key fields so it can be used as persistent context.
    for needle in ["Trigger:", "Work:", "DoD:", "Deps:", "Scope:", "ObligationId:"]:
        assert needle in text


def test_stable_obligation_id_changes_on_semantics():
    a = stable_obligation_id(node_id="n", scope="node", statement="A", dod="D", deps=[])
    b = stable_obligation_id(node_id="n", scope="node", statement="B", dod="D", deps=[])
    assert a != b
