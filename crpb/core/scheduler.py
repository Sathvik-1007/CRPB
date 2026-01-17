from __future__ import annotations

from typing import Dict, List, Tuple


class Scheduler:
    def __init__(self, max_parallel_children: int = 4):
        self.max_parallel_children = max_parallel_children

    def ready_set(self, nodes: List[dict]) -> List[dict]:
        # very basic: nodes with deps satisfied (missing deps default to satisfied in dry-run)
        return nodes[: self.max_parallel_children]

    def ready_from_registry(self, registry: Dict) -> List[Tuple[str, str]]:
        """
        Compute a ready set of (file_path, function_name) where function status == 'stub'
        and all deps are implemented. Limits to max_parallel_children.
        Dep resolution rule:
        - If a dep name exists in the same file, it must be implemented in the same file.
        - Otherwise (dep not present in the same file), consider it satisfied if implemented in any file.
        """
        ready: List[Tuple[str, str]] = []
        files = registry.get("files", {})
        for fpath, fdata in files.items():
            funcs = fdata.get("functions", {})
            # precompute implemented map (same-file)
            implemented_in_file = {
                name for name, meta in funcs.items() if meta.get("status") == "implemented"
            }
            for name, meta in funcs.items():
                if meta.get("status") != "stub":
                    continue
                deps = meta.get("deps", [])

                def _dep_satisfied(
                    dep: str,
                    *,
                    funcs=funcs,
                    implemented_in_file=implemented_in_file,
                    files=files,
                ) -> bool:
                    # same-file dep must be implemented in the same file
                    if dep in funcs:
                        return dep in implemented_in_file
                    # otherwise allow satisfaction by any file
                    return self._is_dep_implemented_anywhere(dep, files)

                if all(_dep_satisfied(dep) for dep in deps):
                    ready.append((fpath, name))
                    if len(ready) >= self.max_parallel_children:
                        return ready
        return ready

    @staticmethod
    def _is_dep_implemented_anywhere(dep: str, files: Dict) -> bool:
        for _fpath, fdata in files.items():
            meta = fdata.get("functions", {}).get(dep)
            if meta and meta.get("status") == "implemented":
                return True
        return False
