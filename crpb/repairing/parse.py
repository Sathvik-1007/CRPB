from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import Issue, IssueSeverity


def _split_n(s: str, n: int) -> List[str]:
    parts = str(s).split(":")
    if len(parts) <= n:
        return parts
    # keep tail intact
    head = parts[:n]
    tail = ":".join(parts[n:])
    return head + [tail]


def parse_issue_strings(
    issues: List[Any],
    *,
    severity: IssueSeverity,
) -> List[Issue]:
    """Parse string issues of form `code[:path[:ref[:extra...]]]` into Issue objects.

    This is intentionally conservative and uses only delimiter splitting (no regex heuristics).
    Unknown shapes fall back to `code="unknown"`.
    """

    out: List[Issue] = []
    for it in issues or []:
        s = str(it) if it is not None else ""
        s = s.strip()
        if not s:
            continue
        parts = _split_n(s, 3)
        code = parts[0].strip() if parts and parts[0].strip() else "unknown"
        path: Optional[str] = None
        ref: Optional[str] = None
        msg = s
        data: Optional[Dict[str, Any]] = None

        if len(parts) >= 2 and parts[1].strip():
            path = parts[1].strip()
        if len(parts) >= 3 and parts[2].strip():
            ref = parts[2].strip()
        if len(parts) >= 4 and parts[3].strip():
            # Keep tail as structured data; do not mutate ref into a non-path token.
            data = {"extra": parts[3].strip()}

        out.append(Issue(code=code, severity=severity, message=msg, path=path, ref=ref, data=data))
    return out


def parse_validation_report(report: Dict[str, Any]) -> List[Issue]:
    """Extract structured issues from a CRPB validation report dict."""
    if not isinstance(report, dict):
        return []
    issues = parse_issue_strings(report.get("issues") or [], severity=IssueSeverity.fatal)
    warnings = parse_issue_strings(report.get("warnings") or [], severity=IssueSeverity.warning)
    suggestions = parse_issue_strings(report.get("suggestions") or [], severity=IssueSeverity.warning)
    return issues + warnings + suggestions
