from crpb.core.context_compiler import ContextCompiler


def test_context_compiler_prefers_referenced_file_snippets():
    cc = ContextCompiler(max_chars=8000)

    files = {
        "a.txt": "AAA" * 500,
        "b.txt": "BBB" * 500,
        "c.txt": "CCC" * 500,
    }

    node = {
        "id": "n",
        "kind": "code:function",
        "title": "t",
        "description": "d",
        "node_plan": {
            "acceptance_criteria": "Must update b.txt and integrate with c.txt",
            "test_plan": "Check b.txt is correct",
        },
        "meta": {},
    }

    pack = cc.compile(
        idea="x",
        constraints={"context_max_file_snippets": 1, "context_file_snippet_chars": 50},
        node=node,
        parent=None,
        siblings=[],
        ledger=None,
        artifacts=[],
        files=files,
        file_specs={},
        signals={},
    )

    # No truncation is allowed: oversized file texts are not included as snippets.
    # Relevance should still determine ordering of file refs.
    refs = (pack.get("files") or {}).get("refs") or []
    assert refs and refs[0].get("path") == "b.txt"
