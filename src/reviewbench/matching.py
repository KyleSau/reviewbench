"""Deterministic matcher: decides which findings hit which ground-truth issues.

A finding matches an issue when all three hold:

1. Same file (after path normalisation).
2. The finding's line range overlaps the issue's range widened by ``line_tolerance``
   on each side. Tolerance absorbs "commented on the call site instead of the
   assignment one line up" without letting a finding in a different function count.
3. Compatible category: identical, or one of the explicitly listed near-synonym pairs.

Matching is one-to-one. Candidate pairs are sorted by line distance and assigned
greedily, so the closest finding wins an issue and a second finding about the same
issue is a duplicate, which counts as a false positive (duplicate review comments
cost reviewer attention too).

This is intentionally strict and explainable. Semantic matching ("does this comment
describe the same root cause?") is the LLM judge's job; see judge.py.
"""

from __future__ import annotations

from reviewbench.diff import normalize_path
from reviewbench.models import Category, ExpectedIssue, Finding, Match

DEFAULT_LINE_TOLERANCE = 3

# Pairs a reasonable reviewer could use interchangeably for the same defect.
# Kept short on purpose: every pair added here makes the matcher more lenient.
COMPATIBLE_CATEGORIES: frozenset[frozenset[Category]] = frozenset(
    {
        frozenset({Category.SWALLOWED_EXCEPTION, Category.ERROR_HANDLING}),
        frozenset({Category.OFF_BY_ONE, Category.LOGIC_ERROR}),
    }
)


def categories_compatible(a: Category, b: Category) -> bool:
    return a == b or frozenset({a, b}) in COMPATIBLE_CATEGORIES


def line_distance(finding: Finding, issue: ExpectedIssue) -> int:
    """0 if the ranges overlap, otherwise the gap in lines between them."""
    if finding.line_end < issue.line_start:
        return issue.line_start - finding.line_end
    if finding.line_start > issue.line_end:
        return finding.line_start - issue.line_end
    return 0


def is_candidate(finding: Finding, issue: ExpectedIssue, line_tolerance: int) -> bool:
    return (
        normalize_path(finding.file) == normalize_path(issue.file)
        and line_distance(finding, issue) <= line_tolerance
        and categories_compatible(finding.category, issue.category)
    )


def match_findings(
    findings: list[Finding],
    issues: list[ExpectedIssue],
    line_tolerance: int = DEFAULT_LINE_TOLERANCE,
) -> list[Match]:
    """Return one-to-one matches between findings and expected issues."""
    candidates = [
        (line_distance(f, issue), fi, ii)
        for fi, f in enumerate(findings)
        for ii, issue in enumerate(issues)
        if is_candidate(f, issue, line_tolerance)
    ]
    # Sort by distance, then by index so ties resolve the same way on every run.
    candidates.sort()
    used_findings: set[int] = set()
    used_issues: set[int] = set()
    matches: list[Match] = []
    for _, fi, ii in candidates:
        if fi in used_findings or ii in used_issues:
            continue
        used_findings.add(fi)
        used_issues.add(ii)
        matches.append(Match(finding_index=fi, issue_index=ii))
    return sorted(matches, key=lambda m: m.finding_index)
