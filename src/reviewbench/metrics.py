"""Turn per-case results into the numbers that go in reports and gates.

Counting rules (also in the README):

* Only cases with status ``ok`` are scored. Refusals, truncations and API errors are
  counted separately and never treated as "the reviewer found nothing".
* TP: a finding matched to an issue. FN: an issue with no matched finding. FP: a
  finding with no matched issue (including duplicates of an already-matched issue).
* Per-category TP and FN are attributed to the ground-truth issue's category; FP to
  the finding's own category.
* A ratio with a zero denominator is ``None`` ("n/a"), not 0 and not 1.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable

from pydantic import BaseModel

from reviewbench.models import Case, CaseResult, Match, Usage


class PRF(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class LatencyStats(BaseModel):
    count: int = 0
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None


class JudgeMetrics(BaseModel):
    judged_cases: int
    failed_cases: int
    overall: PRF
    compared_findings: int
    agreement: float | None
    kappa: float | None
    usage: Usage


class Metrics(BaseModel):
    total_cases: int
    scored_cases: int
    status_counts: dict[str, int]
    overall: PRF
    per_category: dict[str, PRF]
    clean_cases: int
    clean_cases_flagged: int
    clean_case_fp_rate: float | None
    findings_on_clean_cases: int
    latency: LatencyStats
    usage: Usage
    fallback_cases: int
    judge: JudgeMetrics | None = None


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def prf(tp: int, fp: int, fn: int) -> PRF:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PRF(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile: always returns an observed value."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion. Better than the normal approximation
    at the small counts a 25-case benchmark produces."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Agreement between two binary raters, corrected for chance. None if undefined
    (no items, or both raters gave every item the same label)."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def _count(
    pairs: Iterable[tuple[Case, CaseResult]], matches_of: Callable[[CaseResult], list[Match]]
) -> tuple[PRF, dict[str, PRF]]:
    tp = fp = fn = 0
    by_cat: dict[str, Counter[str]] = {}
    for case, result in pairs:
        matches = matches_of(result)
        matched_findings = {m.finding_index for m in matches}
        matched_issues = {m.issue_index for m in matches}
        for idx, issue in enumerate(case.expected_issues):
            key = "tp" if idx in matched_issues else "fn"
            by_cat.setdefault(issue.category.value, Counter())[key] += 1
        for idx, finding in enumerate(result.findings):
            if idx not in matched_findings:
                by_cat.setdefault(finding.category.value, Counter())["fp"] += 1
        tp += len(matched_issues)
        fn += len(case.expected_issues) - len(matched_issues)
        fp += len(result.findings) - len(matched_findings)
    per_category = {cat: prf(c["tp"], c["fp"], c["fn"]) for cat, c in sorted(by_cat.items())}
    return prf(tp, fp, fn), per_category


def _sum_usage(usages: Iterable[Usage | None]) -> Usage:
    total = Usage(cost_usd=0.0)
    for usage in usages:
        if usage is not None:
            total = total + usage
    return total


def _judge_metrics(pairs: list[tuple[Case, CaseResult]]) -> JudgeMetrics | None:
    judged = [(c, r) for c, r in pairs if r.judge is not None]
    if not judged:
        return None
    judged_ok = [(c, r) for c, r in judged if r.judge and r.judge.status == "ok"]
    # Cases the judge never needed to see (no findings, or no issues) are decided the
    # same way by any judge, so they count toward judge P/R as-is.
    trivial = [(c, r) for c, r in pairs if r.judge is None]
    overall, _ = _count(judged_ok + trivial, lambda r: r.judge.matches if r.judge else [])
    matcher_labels: list[int | None] = []
    judge_labels: list[int | None] = []
    for _, result in judged_ok:
        assert result.judge is not None
        by_matcher = {m.finding_index: m.issue_index for m in result.matches}
        by_judge = {m.finding_index: m.issue_index for m in result.judge.matches}
        for idx in range(len(result.findings)):
            matcher_labels.append(by_matcher.get(idx))
            judge_labels.append(by_judge.get(idx))
    agreement = safe_div(
        sum(m == j for m, j in zip(matcher_labels, judge_labels, strict=True)), len(matcher_labels)
    )
    kappa = cohens_kappa([m is not None for m in matcher_labels], [j is not None for j in judge_labels])
    return JudgeMetrics(
        judged_cases=len(judged_ok),
        failed_cases=len(judged) - len(judged_ok),
        overall=overall,
        compared_findings=len(matcher_labels),
        agreement=agreement,
        kappa=kappa,
        usage=_sum_usage(r.judge.usage for _, r in judged if r.judge),
    )


def compute_metrics(cases: list[Case], results: list[CaseResult]) -> Metrics:
    by_id = {c.id: c for c in cases}
    pairs = [(by_id[r.case_id], r) for r in results]
    scored = [(c, r) for c, r in pairs if r.status == "ok"]
    overall, per_category = _count(scored, lambda r: r.matches)
    clean = [(c, r) for c, r in scored if c.is_clean]
    flagged = [r for _, r in clean if r.findings]
    latencies = [r.latency_ms for _, r in scored if r.latency_ms is not None]
    return Metrics(
        total_cases=len(pairs),
        scored_cases=len(scored),
        status_counts=dict(sorted(Counter(r.status for _, r in pairs).items())),
        overall=overall,
        per_category=per_category,
        clean_cases=len(clean),
        clean_cases_flagged=len(flagged),
        clean_case_fp_rate=safe_div(len(flagged), len(clean)),
        findings_on_clean_cases=sum(len(r.findings) for r in flagged),
        latency=LatencyStats(
            count=len(latencies),
            mean_ms=safe_div(sum(latencies), len(latencies)),
            p50_ms=percentile(latencies, 50),
            p95_ms=percentile(latencies, 95),
            max_ms=max(latencies) if latencies else None,
        ),
        # Cost counts every call, including refused/failed ones: that money was spent.
        usage=_sum_usage(r.usage for _, r in pairs),
        fallback_cases=sum(r.fallback_used for _, r in pairs),
        judge=_judge_metrics(scored),
    )
