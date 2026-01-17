from __future__ import annotations

import time

from crpb.planning.parallel import stable_parallel_map


def test_stable_parallel_map_preserves_order() -> None:
    def work(x: int) -> str:
        # Reverse completion order deterministically.
        time.sleep(0.03 * (3 - x))
        return f"v{x}"

    out = stable_parallel_map(work, [1, 2, 3], max_workers=3)
    assert out == ["v1", "v2", "v3"]


def test_stable_parallel_map_sequential_fastpath() -> None:
    out = stable_parallel_map(lambda x: x + 1, [1, 2, 3], max_workers=1)
    assert out == [2, 3, 4]
