import json

import pytest

from helpers import FakeClient, case, finding, issue, message
from reviewbench.judge import ClaudeJudge, parse_verdicts
from reviewbench.llm import LLMSettings
from reviewbench.models import PullRequest
from reviewbench.reviewers.base import ReviewerInfo, ReviewOutput
from reviewbench.runner import run_benchmark

RUBRIC = "# Rubric\nMatch only the same root cause."


def verdicts(*items):
    return {"verdicts": [{"finding_id": f, "rationale": "because", "matched_issue_id": i} for f, i in items]}


def test_parse_verdicts_maps_ids_to_indexes():
    vs, matches = parse_verdicts(verdicts(("F1", "I2"), ("F2", "none")), finding_count=2, issue_count=2)
    assert [(v.finding_index, v.issue_index) for v in vs] == [(0, 1), (1, None)]
    assert [(m.finding_index, m.issue_index) for m in matches] == [(0, 1)]


def test_parse_verdicts_distrusts_bad_ids_and_fills_gaps():
    data = verdicts(("F1", "I9"), ("F7", "I1"), ("garbage", "I1"))
    vs, matches = parse_verdicts(data, finding_count=2, issue_count=1)
    assert [v.issue_index for v in vs] == [None, None]  # I9 out of range; F2 had no verdict
    assert vs[1].rationale == "(no verdict returned)"
    assert matches == []


def test_parse_verdicts_one_to_one():
    _, matches = parse_verdicts(verdicts(("F1", "I1"), ("F2", "I1")), finding_count=2, issue_count=1)
    assert [(m.finding_index, m.issue_index) for m in matches] == [(0, 0)]  # F2 is a duplicate


def test_judge_call_uses_rubric_and_structured_output():
    client = FakeClient(message(text=json.dumps(verdicts(("F1", "I1")))))
    judge = ClaudeJudge(LLMSettings(), RUBRIC, client=client)
    c = case("c", [issue(start=5)])
    result = judge.judge(c, [finding(start=5)])
    assert result.status == "ok"
    assert [(m.finding_index, m.issue_index) for m in result.matches] == [(0, 0)]
    request = client.messages.calls[0]
    assert request["system"][0]["text"] == RUBRIC
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "I1: file=a.py lines=5-5" in request["messages"][0]["content"]
    assert judge.info().rubric_sha256


def test_judge_refusal_is_recorded():
    refusal = message(
        text=None, stop_reason="refusal", extra={"stop_details": {"type": "refusal", "category": None}}
    )
    judge = ClaudeJudge(LLMSettings(), RUBRIC, client=FakeClient(refusal))
    assert judge.judge(case("c", [issue()]), [finding()]).status == "refused"


class ScriptedReviewer:
    """Returns pre-baked findings per case id."""

    name = "scripted"

    def __init__(self, findings_by_case):
        self.findings_by_case = findings_by_case

    def info(self):
        return ReviewerInfo(name=self.name)

    def review(self, pr: PullRequest):
        return ReviewOutput(findings=self.findings_by_case.get(pr.id, []))


def test_run_with_judge_reports_agreement(dataset):
    small = dataset.select(["py-orders-pagination", "py-ui-prefs-cookie", "py-clean-env-config"])
    by_id = {c.id: c.expected_issues for c in small.cases}
    pag, pickle_issue = by_id["py-orders-pagination"][0], by_id["py-ui-prefs-cookie"][0]
    findings = {
        # Right place, but a category the matcher won't accept; the judge will call it a match.
        "py-orders-pagination": [finding(file=pag.file, start=pag.line_start, category="security_other")],
        # Matcher and judge both accept this one.
        "py-ui-prefs-cookie": [
            finding(
                file=pickle_issue.file, start=pickle_issue.line_start, category="insecure_deserialization"
            )
        ],
        "py-clean-env-config": [finding(file="app/config/env.py", start=20, category="other")],
    }
    client = FakeClient(
        message(text=json.dumps(verdicts(("F1", "I1")))),  # pagination: judge says match
        message(text=json.dumps(verdicts(("F1", "I1")))),  # pickle: judge agrees with matcher
    )
    judge = ClaudeJudge(LLMSettings(), RUBRIC, client=client)
    run = run_benchmark(small, ScriptedReviewer(findings), name="j", judge=judge, workers=1)

    # The clean case has findings but no issues: decided without calling the judge.
    assert len(client.messages.calls) == 2
    jm = run.metrics.judge
    assert jm.judged_cases == 2
    assert jm.compared_findings == 2
    assert jm.agreement == pytest.approx(0.5)
    assert (run.metrics.overall.tp, jm.overall.tp) == (1, 2)
    assert jm.overall.fp == 1  # the clean-case finding is an FP under either scorer
    assert run.judge.model == "claude-opus-5-5"
