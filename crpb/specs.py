from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class FunctionExample(BaseModel):
    inp: dict
    out: dict


class FunctionSpec(BaseModel):
    name: str
    signature: str
    returns: str
    description: str = ""
    examples: List[FunctionExample] = Field(default_factory=list)
    tests: List[str] = Field(default_factory=list)
    status: str = Field(default="stub")  # stub|implemented
    deps: List[str] = Field(default_factory=list)


class FileSpec(BaseModel):
    path: str
    language: Optional[str] = None
    functions: Dict[str, FunctionSpec] = Field(default_factory=dict)
    exports: List[str] = Field(default_factory=list)
    # Optional integrity hints for the assembler/validator
    imports: List[str] = Field(default_factory=list)
    entrypoint: Optional[str] = None


class ModuleSpec(BaseModel):
    name: str
    purpose: str = ""
    priority: str = Field(default="medium")  # high|medium|low
    deps: List[str] = Field(default_factory=list)
    files: List[FileSpec] = Field(default_factory=list)


class Plan(BaseModel):
    idea: str
    constraints: dict = Field(default_factory=dict)
    modules: List[ModuleSpec] = Field(default_factory=list)


# Generic, hierarchical task planning (LLM-decided recursive subtasking)
class TaskSpec(BaseModel):
    """
    A generic task node for recursive orchestration.
    - kind: arbitrary domain label, e.g., 'composite' | 'code:function' | 'design:api' | ...
    - id: stable identifier to reference in deps; if omitted, an engine may synthesize one.
    - deps: list of other task ids this task depends on (DAG semantics)
    - inputs/outputs: unstructured JSON payloads understood by the corresponding worker for the kind
    - children: nested tasks if the node is split into subtasks
    """
    kind: str = Field(default="composite")
    title: str = Field(default="")
    description: str = Field(default="")
    id: Optional[str] = None
    priority: str = Field(default="medium")  # high|medium|low
    deps: List[str] = Field(default_factory=list)
    inputs: Dict[str, object] = Field(default_factory=dict)
    outputs: Dict[str, object] = Field(default_factory=dict)
    children: List["TaskSpec"] = Field(default_factory=list)


class TaskPlan(BaseModel):
    idea: str
    constraints: dict = Field(default_factory=dict)
    tasks: List[TaskSpec] = Field(default_factory=list)

# Ensure forward refs are resolved for self-referential models
TaskSpec.model_rebuild()
TaskPlan.model_rebuild()
