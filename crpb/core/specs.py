from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Generic, hierarchical task planning (LLM-decided recursive subtasking)
class TaskSpec(BaseModel):
    """
    A generic task node for recursive orchestration.
    - kind: arbitrary domain label, e.g., 'composite' | 'code:function' | 'design:api' | ...
    - id: stable identifier to reference in deps; if omitted, an engine may synthesize one.
    - deps: list of other task ids this task depends on (DAG semantics)
    - inputs/outputs: unstructured JSON payloads understood by the corresponding worker for the kind.
    - children: nested tasks if the node is split into subtasks
    """

    kind: str = Field(default="composite")
    title: str = Field(default="")
    description: str = Field(default="")
    id: Optional[str] = None
    deps: List[str] = Field(default_factory=list)
    inputs: Dict[str, object] = Field(default_factory=dict)
    outputs: Dict[str, object] = Field(default_factory=dict)
    children: List[TaskSpec] = Field(default_factory=list)
    # Extended planning context (kept language-agnostic)
    node_plan: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class TaskPlan(BaseModel):
    idea: str
    constraints: dict = Field(default_factory=dict)
    tasks: List[TaskSpec] = Field(default_factory=list)


# Ensure forward refs are resolved for self-referential models
TaskSpec.model_rebuild()
TaskPlan.model_rebuild()


# ---------------------- Progressive CodeSpec (top-down architecture) ----------------------
class CodeSpecFile(BaseModel):
    """
    Language-neutral, progressive file blueprint generated during task execution.
    This is the authoritative spec the builder must follow for generation and verification.

    Fields keep rich narrative context (purpose/description) while avoiding language bias.
    The optional top-level 'content' allows non-code or large content-first files to be
    specified without requiring functions/exports. For non-code assets (e.g., README, JSON,
    YAML, Markdown, data files), builders may write this content verbatim when provided.
    """

    path: str
    language: Optional[str] = (
        None  # Optional; may be omitted (best-effort inference is allowed but never required)
    )
    purpose: str = ""
    description: str = ""
    # Optional content payload for content-first files (e.g., README, JSON, YAML, Markdown, data)
    content: Optional[str] = None

    # Structural guidance (language-agnostic shapes)
    imports: List[str] = Field(default_factory=list)
    imports_description: str = ""
    exports: List[str] = Field(default_factory=list)
    exports_description: str = ""
    # New: richer shapes, optional and backward-compatible
    # functions: name -> metadata (signature, description, parameters, returns, etc.)
    functions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Human-readable descriptors (optional)
    functions_description: str = ""
    # Optional detailed exports items (does not replace 'exports')
    # Each item: {name, type: function|class|constant|unknown, signature?, description?, value?}
    exports_detail: List[Dict[str, Any]] = Field(default_factory=list)
    # Free-form class/constant descriptors kept flexible to remain language-neutral
    classes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    classes_description: str = ""
    constants: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    constants_description: str = ""
    # Execution and misc metadata
    entrypoint: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeSpec(BaseModel):
    """
    Consolidated, deterministic blueprint for the entire codebase produced in a
    Spec-first, language-agnostic flow. Builders and orchestrators rely on this
    artifact (together with plan context) to generate, validate, and assemble
    outputs deterministically.
    """

    description: str = ""
    files: List[CodeSpecFile] = Field(default_factory=list)
    # Optional catalog of non-code assets
    assets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # name -> {type, path}
    version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
