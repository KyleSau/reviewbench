import pytest

from helpers import case, finding, issue
from reviewbench.metrics import cohens_kappa, compute_metrics, percentile, prf, wilson_interval
from reviewbench.models import CaseResult, Match, Usage


def test_prf_math():
    m = prf(tp=3, fp=1, fn=2)
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.6)
    assert m.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_prf_zero_denominators_are_none_not_zero():
    m = prf(tp=0, fp=0, fn=4)
    assert m.precision is None
    assert m.recall == 0.0
    assert m.f1 is None
    assert prf(0, 0, 0).recall is None


def test_f1_is_zero_when_nothing_is_right():
    assert prf(tp=0, fp=3, fn=2).f1 == 0.0


def _result(case_id, n_issues, findings, matches, status="ok", usage=None, latency=None):
    return CaseResult(
        case_id=case_id, expected_issue_count=n_issues, status=status, findings=findings,
        matches=[Match(finding_index=f, issue_index=i) for f, i in matches],
        usage=usage, latency_ms=latency,
    )  # fmt: skip


@pytest.fixture
def scenario():
    cases = [
        case(
            "buggy-a", [issue(start=5, category="sql_injection"), issue(start=20, category="race_condition")]
        ),
        case("buggy-b", [issue(start=8, category="sql_injection")]),
        case("clean-a", []),
        case("clean-b", []),
        case("errored", [issue(start=3, category="off_by_one")]),
    ]
    results = [
        # 1 TP (sql), 1 FN (race), 1 FP labelled resource_leak
        _result("buggy-a", 2, [finding(start=5), finding(start=25, category="resource_leak")], [(0, 0)],
                usage=Usage(input_tokens=100, output_tokens=10, cost_usd=0.01), latency=100),
        # 1 TP
        _result("buggy-b", 1, [finding(start=8)], [(0, 0)], latency=300),
        # clean case flagged with 2 findings (2 FP)
        _result("clean-a", 0, [finding(start=1, category="other"), finding(start=9, category="other")], [],
                latency=200),
        _result("clean-b", 0, [], [], latency=400),
        # API error: excluded from scoring, but its cost still counts
        _result("errored", 1, [], [], status="error", usage=Usage(input_tokens=50, cost_usd=0.002)),
    ]  # fmt: skip
    return cases, results


def test_compute_metrics_overall(scenario):
    m = compute_metrics(*scenario)
    assert (m.overall.tp, m.overall.fp, m.overall.fn) == (2, 3, 1)
    assert m.overall.precision == pytest.approx(2 / 5)
    assert m.overall.recall == pytest.approx(2 / 3)
    assert m.scored_cases == 4
    assert m.status_counts == {"error": 1, "ok": 4}


def test_errored_case_is_not_scored_as_a_miss(scenario):
    m = compute_metrics(*scenario)
    assert "off_by_one" not in m.per_category  # the errored case's issue is not an FN


def test_per_category_attribution(scenario):
    per = compute_metrics(*scenario).per_category
    assert (per["sql_injection"].tp, per["sql_injection"].fn) == (2, 0)
    assert (per["race_condition"].tp, per["race_condition"].fn) == (0, 1)
    assert per["resource_leak"].fp == 1  # FP attributed to the finding's category
    assert per["other"].fp == 2


def test_clean_case_metrics(scenario):
    m = compute_metrics(*scenario)
    assert (m.clean_cases, m.clean_cases_flagged, m.findings_on_clean_cases) == (2, 1, 2)
    assert m.clean_case_fp_rate == pytest.approx(0.5)


def test_usage_includes_unscored_calls_and_latency_excludes_them(scenario):
    m = compute_metrics(*scenario)
    assert m.usage.input_tokens == 150
    assert m.usage.cost_usd == pytest.approx(0.012)
    assert m.latency.count == 4
    assert m.latency.p50_ms == 200
    assert m.latency.max_ms == 400


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile([], 50) is None


def test_wilson_interval():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(12, 21)
    assert 0 < low < 12 / 21 < high < 1
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_cohens_kappa():
    assert cohens_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)
    assert cohens_kappa([True, False], [True, False]) == pytest.approx(1.0)
    assert cohens_kappa([True, True], [True, True]) is None  # undefined: no variation
    assert cohens_kappa([], []) is None
