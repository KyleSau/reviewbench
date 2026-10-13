import pytest

from helpers import finding, issue
from reviewbench.matching import categories_compatible, match_findings
from reviewbench.models import Category


def pairs(matches):
    return [(m.finding_index, m.issue_index) for m in matches]


def test_exact_match():
    assert pairs(match_findings([finding(start=10)], [issue(start=10)])) == [(0, 0)]


@pytest.mark.parametrize(("line", "matched"), [(7, True), (6, False), (15, True), (16, False)])
def test_line_tolerance_edges(line, matched):
    # Issue spans 10-12; tolerance 3 accepts 7..15.
    result = match_findings([finding(start=line)], [issue(start=10, end=12)], line_tolerance=3)
    assert bool(result) is matched


def test_finding_range_overlapping_issue_matches():
    assert match_findings([finding(start=1, end=20)], [issue(start=10)], line_tolerance=0)


def test_wrong_file_never_matches():
    assert not match_findings([finding(file="b.py")], [issue(file="a.py")])


def test_paths_are_normalised():
    assert match_findings([finding(file="./src\\a.py")], [issue(file="src/a.py")])


def test_incompatible_category_does_not_match():
    assert not match_findings([finding(category="resource_leak")], [issue(category="sql_injection")])


def test_compatible_categories_match_both_ways():
    assert categories_compatible(Category.ERROR_HANDLING, Category.SWALLOWED_EXCEPTION)
    assert categories_compatible(Category.SWALLOWED_EXCEPTION, Category.ERROR_HANDLING)
    assert not categories_compatible(Category.SECURITY_OTHER, Category.SQL_INJECTION)
    assert match_findings([finding(category="error_handling")], [issue(category="swallowed_exception")])


def test_two_issues_on_the_same_line_are_separated_by_category():
    issues = [issue(start=11, category="n_plus_one_query"), issue(start=11, category="null_handling")]
    findings = [finding(start=11, category="null_handling"), finding(start=11, category="n_plus_one_query")]
    assert pairs(match_findings(findings, issues)) == [(0, 1), (1, 0)]


def test_duplicate_findings_match_once_and_closest_wins():
    findings = [finding(start=13), finding(start=10)]
    result = match_findings(findings, [issue(start=10)])
    assert pairs(result) == [(1, 0)]  # the exact-line finding wins; the other is a duplicate


def test_no_findings_or_no_issues():
    assert match_findings([], [issue()]) == []
    assert match_findings([finding()], []) == []
