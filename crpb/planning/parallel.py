from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def stable_parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int,
) -> List[R]:
    """Run func over items in parallel while preserving input order.

    Determinism rule: the returned list order always matches the input order,
    regardless of completion timing.

    This utility is intentionally minimal: it does not batch, retry, or swallow
    exceptions.
    """

    if max_workers <= 1 or len(items) <= 1:
        return [func(x) for x in items]

    results: List[Tuple[int, R]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(func, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results.append((idx, fut.result()))

    results.sort(key=lambda t: t[0])
    return [r for _, r in results]
