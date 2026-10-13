"""Reviewer implementations. ``REVIEWER_NAMES`` is what the CLI accepts."""

from reviewbench.reviewers.base import Reviewer, ReviewerInfo, ReviewOutput
from reviewbench.reviewers.heuristic import HeuristicReviewer
from reviewbench.reviewers.sanity import NullReviewer, OracleReviewer

REVIEWER_NAMES = ("heuristic", "claude", "oracle", "null")

__all__ = [
    "REVIEWER_NAMES",
    "HeuristicReviewer",
    "NullReviewer",
    "OracleReviewer",
    "ReviewOutput",
    "Reviewer",
    "ReviewerInfo",
]
