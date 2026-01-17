from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .fs import ensure_parent, write_text_locked


def _sha1_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def externalize_text(*, base_dir: Path, category: str, text: str, ext: str = ".txt") -> Dict[str, Any]:
    """Persist large text payloads as artifacts and return a small reference dict.

    Deterministic naming: SHA1(text).
    """
    base_dir = Path(base_dir)
    category = str(category or "payloads")
    digest = _sha1_text(text)
    out_path = base_dir / category / f"{digest}{ext}"
    ensure_parent(out_path)
    write_text_locked(out_path, str(text or ""), encoding="utf-8")
    return {
        "path": str(out_path),
        "sha1": digest,
        "len": len(str(text or "")),
    }


def externalize_json(*, base_dir: Path, category: str, obj: Any, ext: str = ".json") -> Dict[str, Any]:
    """Persist JSON payloads as artifacts and return a small reference dict.

    Deterministic naming: SHA1(stable_json(obj)).
    """
    s = _stable_json(obj)
    digest = _sha1_text(s)
    out_path = Path(base_dir) / str(category or "payloads") / f"{digest}{ext}"
    ensure_parent(out_path)
    write_text_locked(out_path, s, encoding="utf-8")
    return {
        "path": str(out_path),
        "sha1": digest,
        "len": len(s),
    }
