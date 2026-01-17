from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .fs import ensure_parent


def _strip_markdown_fences(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    parts = s.splitlines()
    if (
        len(parts) >= 2
        and parts[0].lstrip().startswith("```")
        and parts[-1].lstrip().startswith("```")
    ):
        return "\n".join(parts[1:-1]).strip()
    return s


def _parse_json_dict_strict(json_str: str) -> dict:
    """Parse a JSON string strictly, ensuring the result is a dict."""
    try:
        text = _strip_markdown_fences(json_str)
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Accept a leading JSON object followed by trailing text (common LLM pattern)
    try:
        text = _strip_markdown_fences(json_str)
        decoder = json.JSONDecoder()
        obj, _idx = decoder.raw_decode((text or "").lstrip())
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        raise ValueError(f"Strict JSON parsing failed: {e}")


class ArtifactRegistry:
    """
    Minimal run-scoped registry for produced artifacts to enable task gating and wiring.
    - Persists a JSON index at run_dir/artifacts/index.json
    - Index keyed by artifact id; value holds latest version metadata and path.
    - Also supports lookup by path existence if id is absent.
    """

    def __init__(self, index_path: Path | None = None, base_dir: Path | None = None) -> None:
        """
        Initialise the registry.

        * If ``index_path`` is provided, it is used directly.
        * Otherwise, if ``base_dir`` is given, the registry stores its
          index at ``base_dir / "artifacts" / "index.json"``.
        * If neither is supplied, a temporary directory is used (unlikely in
          production code but convenient for tests).
        """
        if index_path is not None:
            self.index_path = Path(index_path)
        else:
            if base_dir is None:
                raise ValueError("Either index_path or base_dir must be provided")
            self.index_path = base_dir / "artifacts" / "index.json"
        ensure_parent(self.index_path)
        self._index: Dict[str, Dict[str, Any]] = {}
        # Special bucket for validations keyed by file path (absolute or relative as provided)
        self._by_path_key = "__validation_by_path__"
        self._load()

    def _load(self) -> None:
        try:
            if self.index_path.exists():
                self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
            else:
                self._index = {}
        except Exception:
            self._index = {}
        # Ensure special path map exists
        if not isinstance(self._index.get(self._by_path_key), dict):
            self._index[self._by_path_key] = {}

    def _save(self) -> None:
        try:
            self.index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        except Exception:
            # Non-fatal; best-effort persistence
            pass

    def exists(self, ref: Dict[str, Any], base_dir: Optional[Path] = None) -> bool:
        """
        Return True if an artifact reference is satisfied.
        A ref may include:
          - id: stable identifier
          - path: file path (relative to outputs directory or absolute)
          - kind/version/metadata: ignored for existence, but preserved on register
        """
        if not isinstance(ref, dict):
            return False
        art_id = ref.get("id")
        if isinstance(art_id, str) and art_id:
            if art_id in self._index:
                # Existence is based on registration regardless of path presence
                return True
        pth = ref.get("path")
        if isinstance(pth, str) and pth:
            try:
                p = Path(pth)
                if not p.is_absolute() and base_dir is not None:
                    p = base_dir / p
                return p.exists()
            except Exception:
                return False
        return False

    def resolve_path(self, ref: Dict[str, Any], base_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Resolve an artifact reference to a concrete file path when possible.
        Preference order: registry path by id -> provided path (relative to base_dir if needed)
        """
        if not isinstance(ref, dict):
            return None
        art_id = ref.get("id")
        if isinstance(art_id, str) and art_id and art_id in self._index:
            p = self._index[art_id].get("path")
            if isinstance(p, str) and p:
                return Path(p)
        pth = ref.get("path")
        if isinstance(pth, str) and pth:
            p = Path(pth)
            if not p.is_absolute() and base_dir is not None:
                p = base_dir / p
            return p
        return None

    def register(self, items: List[Dict[str, Any]], base_dir: Optional[Path] = None) -> None:
        """
        Register produced artifacts.
        For each item, persist id, kind, version, metadata, and resolved path if present.
        """
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            art_id = it.get("id")
            if not isinstance(art_id, str) or not art_id:
                # Skip items without id; optional: derive from path
                continue
            entry: Dict[str, Any] = {
                "id": art_id,
                "kind": it.get("kind"),
                "version": it.get("version"),
                "metadata": it.get("metadata") or {},
            }
            # resolve path
            p = self.resolve_path(it, base_dir=base_dir)
            if p is not None:
                entry["path"] = str(p)
            # Preserve existing validation if present
            if art_id in self._index and isinstance(self._index.get(art_id), dict):
                prev = self._index[art_id]
                if isinstance(prev.get("validation"), dict):
                    entry["validation"] = prev.get("validation")
            # If we have a path-level validation already, mirror it into the entry
            by_path = self._index.get(self._by_path_key, {})
            if (
                isinstance(entry.get("path"), str)
                and entry.get("path") in by_path
                and "validation" not in entry
            ):
                v = by_path.get(entry["path"])  # type: ignore[index]
                if isinstance(v, dict):
                    entry["validation"] = v
            self._index[art_id] = entry
        self._save()

    def list(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for k, v in self._index.items():
            if k == self._by_path_key:
                continue
            if isinstance(v, dict):
                out.append(v)
        return out

    # Validation tracking
    def set_validation(
        self, art_id: str, ok: bool, report: Optional[Dict[str, Any]] = None
    ) -> None:
        if not art_id:
            return
        entry = self._index.get(art_id) or {"id": art_id}
        entry["validation"] = {"ok": bool(ok), "report": report or {}}
        self._index[art_id] = entry
        # If entry carries a path, also mirror into path-level index
        p = entry.get("path")
        if isinstance(p, str) and p:
            by_path = self._index.get(self._by_path_key, {})
            if not isinstance(by_path, dict):
                by_path = {}
            by_path[p] = entry["validation"]
            self._index[self._by_path_key] = by_path
        self._save()

    def get_validation(self, ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(ref, dict):
            return None
        art_id = ref.get("id")
        if isinstance(art_id, str) and art_id and art_id in self._index:
            v = self._index[art_id].get("validation")
            if isinstance(v, dict):
                return v
        # fallback: lookup by path
        pth = ref.get("path")
        if isinstance(pth, str) and pth:
            # 1) direct path-level map
            by_path = self._index.get(self._by_path_key, {})
            if isinstance(by_path, dict):
                v = by_path.get(pth)
                if isinstance(v, dict):
                    return v
            # 2) scan entries for matching path
            for k, entry in self._index.items():
                if k == self._by_path_key or not isinstance(entry, dict):
                    continue
                if entry.get("path") == pth and isinstance(entry.get("validation"), dict):
                    return entry.get("validation")  # type: ignore[return-value]
        return None

    def set_validation_for_path(
        self, path: str | Path, ok: bool, report: Optional[Dict[str, Any]] = None
    ) -> None:
        """Persist validation for an artifact identified only by its file path.
        Also mirrors into any id-mapped entry that points to the same path.
        """
        try:
            pstr = str(path)
            by_path = self._index.get(self._by_path_key, {})
            if not isinstance(by_path, dict):
                by_path = {}
            v = {"ok": bool(ok), "report": report or {}}
            by_path[pstr] = v
            self._index[self._by_path_key] = by_path
            # Mirror into any entries with this path
            for k, entry in list(self._index.items()):
                if k == self._by_path_key or not isinstance(entry, dict):
                    continue
                if entry.get("path") == pstr:
                    entry["validation"] = v
                    self._index[k] = entry
            self._save()
        except Exception:
            # best-effort only
            pass

    # Compatibility alias for tests expecting `add_validation`
    def add_validation(self, art_id: str, report: Dict[str, Any]) -> None:
        """
        Alias for ``set_validation`` used in test suite.
        The ``report`` dict should contain at least an ``ok`` key.
        """
        ok_flag = bool(report.get("ok", False))
        self.set_validation(art_id, ok=ok_flag, report=report)
