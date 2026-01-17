from __future__ import annotations

from .models import Issue, IssueSeverity


def weight_for(issue: Issue) -> float:
    """Return a non-negative weight for an issue.

    Design goals (language/tool neutral):
    - Monotone: resolving issues should never increase the score.
    - Deterministic: depends only on structured issue fields.
    - Functional bias: prefer eliminating blockers that tend to prevent execution or contract conformance.

    We keep a simple, explicit mapping from machine-coded issue keys to weights.
    This is not tied to any specific language; it uses generic contract/runtime categories
    expressed in `Issue.code`.
    """

    base = 2.5 if issue.severity == IssueSeverity.warning else 10.0

    code = str(getattr(issue, "code", "") or "").strip().lower()
    if not code:
        return float(base)

    # Higher weights for common "hard blockers".
    hard_blockers = (
        "syntax",
        "parse_error",
        "unresolved_import",
        "missing_export",
        "missing_import",
        "signature_mismatch",
        "unexpected_signature",
        "missing_entry",
        "missing_file",
    )
    if any(k in code for k in hard_blockers):
        return float(base * 2.0)

    # Medium weight for interface/contract mismatches that often break functionality.
    medium = (
        "binding_mismatch",
        "attribute_mismatch",
        "evaluate_mismatch",
        "call_mismatch",
        "missing_button_label",
    )
    if any(k in code for k in medium):
        return float(base * 1.25)

    return float(base)


def score(issues: list[Issue]) -> float:
    return float(sum(weight_for(i) for i in (issues or [])))


def importance(issue: Issue) -> int:
    """Deterministic integer importance for ordering.

    Scale `weight_for()` into a stable 0..100 integer bucket.
    """

    try:
        w = float(weight_for(issue))
    except Exception:
        w = 0.0
    # Base weights are small; scale to a readable range.
    val = int(round(w * 5.0))
    return max(0, min(100, val))
