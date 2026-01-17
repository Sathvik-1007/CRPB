from pathlib import Path

from crpb.core.specs import TaskPlan, TaskSpec
from crpb.utils.artifacts import ArtifactRegistry
from crpb.validation.closure import ClosureConfig, validate_artifact_registry_closure


def test_validate_artifact_registry_closure_flags_missing_registry(tmp_path):
    tp = TaskPlan(
        idea="x",
        constraints={},
        tasks=[
            TaskSpec(
                id="t1",
                kind="code:function",
                title="",
                description="",
                inputs={"name": "a", "path": "a.txt", "consumes": [{"id": "X"}]},
                node_plan={"intent": "", "acceptance_criteria": "", "test_plan": ""},
                children=[],
            )
        ],
    )

    rep = validate_artifact_registry_closure(
        tp=tp,
        registry=None,
        base_dir=None,
        config=ClosureConfig(require_artifact_registry_presence=True),
    )
    assert rep["ok"] is False
    assert "artifact_registry_missing" in rep["issues"]


def test_validate_artifact_registry_closure_detects_missing_consumed_artifacts(tmp_path):
    tp = TaskPlan(
        idea="x",
        constraints={},
        tasks=[
            TaskSpec(
                id="p1",
                kind="code:function",
                title="",
                description="",
                inputs={"name": "a", "path": "a.txt", "consumes": [{"id": "X"}]},
                outputs={},
                deps=[],
                node_plan={"intent": "", "acceptance_criteria": "", "test_plan": ""},
                children=[],
            )
        ],
    )

    reg = ArtifactRegistry(base_dir=tmp_path)
    rep = validate_artifact_registry_closure(
        tp=tp,
        registry=reg,
        base_dir=Path(tmp_path),
        config=ClosureConfig(
            require_artifact_registry_presence=True,
            require_consumes_exist_in_registry=True,
            require_produces_exist_in_registry=False,
            require_consumers_depend_on_producers=False,
            require_coverage_closed=False,
        ),
    )
    assert rep["ok"] is False
    assert any(i.startswith("artifact_consume_missing_in_registry:p1:X") for i in rep["issues"])


def test_validate_artifact_registry_closure_requires_dep_edge(tmp_path):
    # Producer produces artifact A, consumer consumes A but forgets deps.
    tp = TaskPlan(
        idea="x",
        constraints={},
        tasks=[
            TaskSpec(
                id="prod",
                kind="code:function",
                title="",
                description="",
                inputs={"name": "p", "path": "p.txt"},
                outputs={"produces": [{"id": "A"}]},
                node_plan={"intent": "", "acceptance_criteria": "", "test_plan": ""},
                children=[],
            ),
            TaskSpec(
                id="cons",
                kind="code:function",
                title="",
                description="",
                inputs={"name": "c", "path": "c.txt", "consumes": [{"id": "A"}]},
                outputs={},
                deps=[],
                node_plan={"intent": "", "acceptance_criteria": "", "test_plan": ""},
                children=[],
            ),
        ],
    )

    reg = ArtifactRegistry(base_dir=tmp_path)
    # Register artifact A so registry existence doesn't fail.
    reg.register([{"id": "A", "kind": "x"}], base_dir=Path(tmp_path))

    rep = validate_artifact_registry_closure(
        tp=tp,
        registry=reg,
        base_dir=Path(tmp_path),
        config=ClosureConfig(
            require_artifact_registry_presence=True,
            require_consumes_exist_in_registry=True,
            require_produces_exist_in_registry=False,
            require_consumers_depend_on_producers=True,
            require_coverage_closed=False,
        ),
    )
    assert rep["ok"] is False
    assert any(i.startswith("artifact_consumer_missing_dep:cons:A") for i in rep["issues"])
