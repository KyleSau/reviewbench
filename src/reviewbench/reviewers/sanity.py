"""Two reviewers that exist only to test the harness itself.

* ``oracle`` returns the ground truth as findings. Every metric must be perfect; if it
  isn't, the matcher, metrics or dataset loader has a bug.
* ``null`` returns no findings. Recall must be zero and clean-case false positives must
  be zero; if not, something is scoring "no answer" as an answer.

Running both through the full pipeline catches most wiring bugs in seconds, before any
money is spent on model calls.
"""

from __future__ import annotations

from reviewbench.dataset import Dataset
from reviewbench.models import Finding, PullRequest
from reviewbench.reviewers.base import ReviewerInfo, ReviewOutput


class OracleReviewer:
    name = "oracle"

    def __init__(self, dataset: Dataset) -> None:
        # The oracle is the only reviewer allowed to see labels, and it has to be
        # handed them explicitly.
        self._answers = {case.id: case.expected_issues for case in dataset.cases}

    def info(self) -> ReviewerInfo:
        return ReviewerInfo(name=self.name)

    def review(self, pr: PullRequest) -> ReviewOutput:
        findings = [
            Finding(
                file=issue.file,
                line_start=issue.line_start,
                line_end=issue.line_end,
                category=issue.category,
                severity=issue.severity,
                explanation=issue.description,
            )
            for issue in self._answers[pr.id]
        ]
        return ReviewOutput(status="ok", findings=findings)


class NullReviewer:
    name = "null"

    def info(self) -> ReviewerInfo:
        return ReviewerInfo(name=self.name)

    def review(self, pr: PullRequest) -> ReviewOutput:
        return ReviewOutput(status="ok", findings=[])
