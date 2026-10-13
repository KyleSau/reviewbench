from reviewbench.gate import evaluate_gate
from reviewbench.metrics import PRF, LatencyStats, Metrics, prf
from reviewbench.models import Usage
from reviewbench.reviewers.base import ReviewerInfo
from reviewbench.runner import DatasetInfo, RunResult


def make_run(tp, fp, fn, *, fingerprint="abc", tolerance=3, unscored=0) -> RunResult:
    overall = prf(tp, fp, fn)
    metrics = Metrics(
        total_cases=25, scored_cases=25 - unscored, status_counts={}, overall=overall,
        per_category={}, clean_cases=6, clean_cases_flagged=0, clean_case_fp_rate=0.0,
        findings_on_clean_cases=0, latency=LatencyStats(), usage=Usage(), fallback_cases=0,
    )  # fmt: skip
    return RunResult(
        name="r", created_at="now", reviewbench_version="0", reviewer=ReviewerInfo(name="x"),
        dataset=DatasetInfo(name="v1", path="p", fingerprint=fingerprint, case_count=25, issue_count=20),
        line_tolerance=tolerance, cases=[], metrics=metrics,
    )  # fmt: skip


def gate(baseline, candidate, **kw):
    kw.setdefault("max_recall_drop", 0.05)
    kw.setdefault("max_precision_drop", 0.05)
    return evaluate_gate(baseline, candidate, **kw)


def failed(result):
    return [c.name for c in result.checks if not c.passed]


def test_identical_runs_pass():
    assert gate(make_run(10, 2, 10), make_run(10, 2, 10)).passed


def test_improvement_passes():
    assert gate(make_run(10, 2, 10), make_run(15, 1, 5)).passed


def test_recall_drop_beyond_limit_fails():
    # recall 0.50 -> 0.40
    result = gate(make_run(10, 0, 10), make_run(8, 0, 12))
    assert failed(result) == ["recall drop"]


def test_precision_drop_beyond_limit_fails():
    # precision 1.0 -> 0.8
    result = gate(make_run(8, 0, 2), make_run(8, 2, 2))
    assert failed(result) == ["precision drop"]


def test_drop_exactly_at_the_limit_passes_despite_float_error():
    # recall 0.80 -> 0.75: 0.80 - 0.75 == 0.05000000000000004 in floating point
    baseline, candidate = make_run(16, 0, 4), make_run(15, 0, 5)
    assert baseline.metrics.overall.recall - candidate.metrics.overall.recall > 0.05
    assert gate(baseline, candidate).passed


def test_candidate_with_no_findings_fails_precision():
    result = gate(make_run(10, 2, 10), make_run(0, 0, 20))
    assert set(failed(result)) == {"recall drop", "precision drop"}


def test_different_dataset_is_not_comparable():
    result = gate(make_run(10, 2, 10), make_run(10, 2, 10, fingerprint="other"))
    assert failed(result) == ["same dataset"]
    assert not any(c.name.endswith("drop") for c in result.checks)  # no meaningless numbers


def test_different_tolerance_is_not_comparable():
    assert failed(gate(make_run(10, 2, 10), make_run(10, 2, 10, tolerance=5))) == ["same line tolerance"]


def test_unscored_cases_fail_by_default_and_can_be_allowed():
    baseline, candidate = make_run(10, 2, 10), make_run(10, 2, 10, unscored=2)
    assert failed(gate(baseline, candidate)) == ["unscored cases"]
    assert gate(baseline, candidate, max_unscored=2).passed


def test_prf_none_treated_as_zero():
    run = make_run(0, 0, 0)
    assert run.metrics.overall == PRF(tp=0, fp=0, fn=0)
    assert gate(run, run).passed
