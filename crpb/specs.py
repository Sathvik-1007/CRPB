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
    language: str = Field(default="python")
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
