from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_file_stem(node_id: str) -> str:
    raw = (node_id or "").strip()
    if not raw:
        return "node"
    # Keep readable stems for simple ids; otherwise hash.
    if all(ch.isalnum() or ch in ("-", "_", ".") for ch in raw) and len(raw) <= 80:
        return raw
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TodoItem(BaseModel):
    id: str
    text: str
    status: str = Field(default="open", description="open|in_progress|done|blocked")
    created_at: str = Field(default_factory=_now_ts)
    updated_at: str = Field(default_factory=_now_ts)


class DecisionItem(BaseModel):
    id: str
    statement: str
    rationale: str = ""
    created_at: str = Field(default_factory=_now_ts)


class NodeLedger(BaseModel):
    schema_version: int = 1
    node_id: str
    parent_id: Optional[str] = None
    created_at: str = Field(default_factory=_now_ts)
    updated_at: str = Field(default_factory=_now_ts)

    todos: List[TodoItem] = Field(default_factory=list)
    decisions: List[DecisionItem] = Field(default_factory=list)
    obligations: List[str] = Field(default_factory=list)
    # Optional structured obligations (backward compatible with obligations[str]).
    obligation_items: List[Dict[str, Any]] = Field(default_factory=list)

    consumed_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    produced_artifacts: List[Dict[str, Any]] = Field(default_factory=list)

    context_pack_digests: List[str] = Field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now_ts()

    def stable_dump(self) -> Dict[str, Any]:
        # Ensure deterministic ordering for replay and diffs.
        data = self.model_dump(exclude_none=True)
        data["todos"] = sorted(data.get("todos") or [], key=lambda x: str(x.get("id") or ""))
        data["decisions"] = sorted(
            data.get("decisions") or [], key=lambda x: str(x.get("id") or "")
        )
        data["obligations"] = sorted(
            data.get("obligations") or [], key=lambda x: str(x or "")
        )
        data["obligation_items"] = sorted(
            data.get("obligation_items") or [],
            key=lambda x: str((x or {}).get("id") or (x or {}).get("statement") or ""),
        )
        data["consumed_artifacts"] = sorted(
            data.get("consumed_artifacts") or [],
            key=lambda x: str(x.get("id") or x.get("path") or ""),
        )
        data["produced_artifacts"] = sorted(
            data.get("produced_artifacts") or [],
            key=lambda x: str(x.get("id") or x.get("path") or ""),
        )
        data["context_pack_digests"] = list(
            dict.fromkeys([str(x) for x in (data.get("context_pack_digests") or []) if str(x)])
        )
        return data


class NodeLedgerStore:
    """Run-scoped persistence for NodeLedgers.

    This is deliberately file-based and deterministic. It stores ledgers under:
    `run_dir/artifacts/ledger/<node_id>.json`

    The ledger is also registerable as an artifact by id (`ledger::<node_id>`) via ArtifactRegistry.
    """

    def __init__(self, *, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.ledger_dir = self.base_dir / "ledger"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, node_id: str) -> Path:
        return self.ledger_dir / f"{_safe_file_stem(node_id)}.json"

    def load(self, node_id: str, *, parent_id: Optional[str] = None) -> NodeLedger:
        p = self.path_for(node_id)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                led = NodeLedger.model_validate(data)
                # Ensure node_id is correct even if file stem collided.
                if led.node_id != node_id:
                    led.node_id = node_id
                    if parent_id is not None:
                        led.parent_id = parent_id
                return led
            except Exception:
                # Fall through to new ledger
                pass
        return NodeLedger(node_id=node_id, parent_id=parent_id)

    def save(self, ledger: NodeLedger) -> Path:
        ledger.touch()
        p = self.path_for(ledger.node_id)
        p.write_text(json.dumps(ledger.stable_dump(), indent=2), encoding="utf-8")
        return p
