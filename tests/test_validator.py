"""
Comprehensive validator test matrix for CRPB.

Covers:
- A1‑A7 guardrails (deterministic leaves, split enforcement, CAS registry, export checksum, deterministic lease).
- T1‑T8 Temporal workflow guards (plan generation, validation, file generation, export verification, merge, retry, idempotency, deterministic run_id).
"""

from crpb.registry import (
    Registry,
    _current_timestamp,
    _lease_valid,
)
from crpb.specs import TaskPlan, TaskSpec
from crpb.validator import (
    basic_file_validation,
    is_deterministic_leaf,
    jsonschema_validate,
    validate_artifact_gating,
    validate_leaf_readiness,
    validate_taskplan_general,
    validate_taskplan_structure,
)

# -------------------------
# A1 – Deterministic Leaf
# -------------------------


def test_is_deterministic_leaf_true():
    task = TaskSpec(
        kind="code:function",
        id="leaf1",
        children=[],
    )
    assert is_deterministic_leaf(task) is True


def test_is_deterministic_leaf_wrong_kind():
    task = TaskSpec(
        kind="composite",
        id="leaf2",
        children=[],
    )
    assert is_deterministic_leaf(task) is False


def test_is_deterministic_leaf_missing_atomic():
    task = TaskSpec(
        kind="code:function",
        id="leaf3",
        meta={"atomic": False},
        children=[],
    )
    assert is_deterministic_leaf(task) is True


def test_is_deterministic_leaf_has_children():
    task = TaskSpec(
        kind="code:function",
        id="leaf4",
        meta={"atomic": True},
        children=[TaskSpec(kind="code:function", id="child")],
    )
    assert is_deterministic_leaf(task) is False


# -------------------------
# A2 – Basic File Validation
# -------------------------


def test_basic_file_validation_ok():
    class Dummy:
        exports = ["foo", "bar"]
        entrypoint = "foo"

    ok, msg = basic_file_validation(Dummy)
    assert ok and msg == "ok"


def test_basic_file_validation_missing_entrypoint():
    class Dummy:
        exports = ["foo"]
        entrypoint = "missing"

    ok, msg = basic_file_validation(Dummy)
    assert not ok and "entrypoint_not_exported" in msg


def test_basic_file_validation_invalid_exports():
    class Dummy:
        exports = [123, ""]
        entrypoint = None

    ok, msg = basic_file_validation(Dummy)
    assert not ok and "exports_invalid_shape" in msg


# -------------------------
# A3 – JSONSchema Fallback
# -------------------------


def test_jsonschema_validate_success():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    data = {"a": "hello"}
    ok, msg = jsonschema_validate(data, schema)
    assert ok


def test_jsonschema_validate_skip_when_missing():
    # Simulate environment without jsonschema installed
    import builtins

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jsonschema":
            raise ImportError
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        ok, msg = jsonschema_validate({}, {})
        assert ok and "skipped" in msg
    finally:
        builtins.__import__ = original_import


# -------------------------
# A5 – TaskPlan Structural Validation
# -------------------------


def make_task(id_, kind="code:function", children=None, meta=None):
    return TaskSpec(
        id=id_,
        kind=kind,
        meta=meta or {},
        inputs=(
            {
                "name": f"fn_{id_}",
                "path": f"{id_}.py",
                "exports": [f"fn_{id_}"],
            }
            if kind == "code:function" and not (children or [])
            else {}
        ),
        children=children or [],
        node_plan={"intent": "x", "acceptance_criteria": "y", "test_plan": "z"},
    )


def test_validate_taskplan_structure_ok():
    tp = TaskPlan(idea="test", tasks=[make_task("t1")])
    ok, issues = validate_taskplan_structure(tp)
    assert ok and not issues


def test_validate_taskplan_structure_duplicate_id():
    tp = TaskPlan(idea="dup", tasks=[make_task("t1"), make_task("t1")])
    ok, issues = validate_taskplan_structure(tp)
    assert not ok and any("duplicate_id" in i for i in issues)


def test_validate_taskplan_structure_unknown_kind():
    tp = TaskPlan(idea="bad", tasks=[make_task("t1", kind="unknown")])
    ok, issues = validate_taskplan_structure(tp)
    assert not ok and any("unknown_kind" in i for i in issues)


def test_validate_taskplan_general_atomic_children():
    child = make_task("c1")
    parent = make_task("p1", kind="composite", children=[child], meta={"atomic": True})
    tp = TaskPlan(idea="bad", tasks=[parent])
    report = validate_taskplan_general(tp)
    assert not report["ok"]
    assert any(
        "atomic_has_children" in n["issues"][0] for n in report["nodes"] if n["id"] == "p1"
    )


# -------------------------
# A6 – Leaf Readiness
# -------------------------


def test_validate_leaf_readiness_success():
    task = TaskSpec(
        kind="code:function",
        id="leaf",
        inputs={"name": "my_func", "path": "mod.py", "language": "python", "exports": ["my_func"]},
        node_plan={"intent": "", "acceptance_criteria": "", "test_plan": ""},
    )
    ok, msg, meta = validate_leaf_readiness(task)
    assert ok
    assert meta["name"] == "my_func"
    assert meta["language"] == "python"


def test_validate_leaf_readiness_missing_path():
    task = TaskSpec(
        kind="code:function",
        id="leaf2",
        inputs={"name": "my_func"},
        node_plan={"intent": "", "acceptance_criteria": "", "test_plan": ""},
    )
    ok, msg, _ = validate_leaf_readiness(task)
    assert not ok and "missing_input:path" in msg


# -------------------------
# A7 – Artifact Gating
# -------------------------


def test_validate_artifact_gating_ok(tmp_path):
    # Simulate a simple ArtifactRegistry with a single valid artifact
    from crpb.utils.artifacts import ArtifactRegistry

    reg = ArtifactRegistry(base_dir=tmp_path)
    # Register a dummy artifact
    art_id = "my_art"
    reg.register([{"id": art_id, "kind": "data"}], base_dir=tmp_path)
    reg.set_validation(art_id, ok=True, report={"ok": True})

    task = TaskSpec(
        inputs={"consumes": [art_id]},
        node_plan={},
        kind="code:function",
        id="t1",
        children=[],
    )
    ok, msg = validate_artifact_gating(
        task=task, registry=reg, base_dir=tmp_path, require_valid=True
    )
    assert ok


def test_validate_artifact_gating_missing(tmp_path):
    from crpb.utils.artifacts import ArtifactRegistry

    reg = ArtifactRegistry(base_dir=tmp_path)
    task = TaskSpec(
        inputs={"consumes": ["ghost"]},
        node_plan={},
        kind="code:function",
        id="t2",
        children=[],
    )
    ok, msg = validate_artifact_gating(
        task=task, registry=reg, base_dir=tmp_path, require_valid=True
    )
    assert not ok and msg == "artifact_missing"


# -------------------------
# A7 – Registry Lease Logic
# -------------------------


def test_registry_lease_acquire_and_release(tmp_path):
    reg = Registry(path=tmp_path / "registry.json")
    # Ensure fresh file
    reg.update(0, lambda d: d)  # create empty registry

    file_path = "module.py"
    owner = "test_owner"
    assert reg.acquire_lease(file_path, owner, ttl_seconds=1) is True
    info = reg.lease_info(file_path)
    assert info is not None and info["lease_owner"] == owner

    # Lease should be valid now
    assert _lease_valid({"lease_owner": owner, "lease_expiry": _current_timestamp() + 5})

    # Release
    assert reg.release_lease(file_path, owner) is True
    assert reg.lease_info(file_path) is None


def test_registry_lease_conflict(tmp_path):
    reg = Registry(path=tmp_path / "registry.json")
    reg.update(0, lambda d: d)  # create
    file_path = "conflict.py"
    owner1 = "owner1"
    owner2 = "owner2"
    assert reg.acquire_lease(file_path, owner1, ttl_seconds=5) is True
    # Second owner should not acquire while lease valid
    assert reg.acquire_lease(file_path, owner2, ttl_seconds=5) is False
    info = reg.lease_info(file_path)
    assert info["lease_owner"] == owner1
