"""Data models shared by every part of the harness.

Everything that crosses a boundary (dataset files on disk, reviewer output, run files)
is a pydantic model, so malformed data fails loudly at the edge instead of producing
quietly wrong metrics later.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(StrEnum):
    """Defect taxonomy. The first ten appear in ground truth; the last three are
    escape hatches for reviewers when nothing else fits."""

    OFF_BY_ONE = "off_by_one"
    NULL_HANDLING = "null_handling"
    SQL_INJECTION = "sql_injection"
    MISSING_AUTHORIZATION = "missing_authorization"
    RESOURCE_LEAK = "resource_leak"
    RACE_CONDITION = "race_condition"
    SWALLOWED_EXCEPTION = "swallowed_exception"
    ERROR_HANDLING = "error_handling"
    N_PLUS_ONE_QUERY = "n_plus_one_query"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    LOGIC_ERROR = "logic_error"
    SECURITY_OTHER = "security_other"
    OTHER = "other"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


Language = Literal["python", "typescript", "csharp"]


class ExpectedIssue(BaseModel):
    """One ground-truth defect. Line numbers refer to the *new* version of the file."""

    model_config = ConfigDict(extra="forbid")

    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    category: Category
    severity: Severity
    description: str = Field(min_length=20)

    @model_validator(mode="after")
    def _check_range(self) -> ExpectedIssue:
        if self.line_end < self.line_start:
            raise ValueError(f"line_end {self.line_end} < line_start {self.line_start}")
        return self


class PullRequest(BaseModel):
    """What a reviewer is allowed to see. Deliberately has no ground-truth fields, so a
    reviewer cannot peek at labels even by accident."""

    id: str
    title: str
    description: str
    diff: str


class Case(BaseModel):
    """A benchmark case: a pull request plus its ground truth."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    language: Language
    title: str
    description: str
    expected_issues: list[ExpectedIssue]
    diff: str

    @property
    def is_clean(self) -> bool:
        return not self.expected_issues

    def to_pull_request(self) -> PullRequest:
        return PullRequest(id=self.id, title=self.title, description=self.description, diff=self.diff)


class Finding(BaseModel):
    """One review comment produced by a reviewer."""

    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    category: Category
    severity: Severity
    explanation: str

    @model_validator(mode="after")
    def _order_range(self) -> Finding:
        # Models occasionally emit a reversed range; normalise rather than reject.
        if self.line_end < self.line_start:
            self.line_start, self.line_end = self.line_end, self.line_start
        return self


class Usage(BaseModel):
    """Token usage and estimated cost of the model calls made for one case."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float | None = None  # None when a model is missing from the price table

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None and other.cost_usd is not None:
            cost = self.cost_usd + other.cost_usd
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cost_usd=cost,
        )


# "ok" is the only scorable status. Everything else is infrastructure or a model
# declining, and must never be scored as "found nothing".
CaseStatus = Literal["ok", "refused", "truncated", "error"]


class Match(BaseModel):
    finding_index: int
    issue_index: int


class JudgeVerdict(BaseModel):
    finding_index: int
    issue_index: int | None  # None = the judge says this finding matches no issue
    rationale: str


class JudgeResult(BaseModel):
    status: CaseStatus
    error: str | None = None
    verdicts: list[JudgeVerdict] = Field(default_factory=list)
    matches: list[Match] = Field(default_factory=list)  # after 1:1 de-duplication
    usage: Usage | None = None
    latency_ms: float | None = None


class CaseResult(BaseModel):
    case_id: str
    expected_issue_count: int  # stored so a run file can be reported on without the dataset
    status: CaseStatus
    error: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    matches: list[Match] = Field(default_factory=list)
    latency_ms: float | None = None
    usage: Usage | None = None
    served_model: str | None = None
    fallback_used: bool = False
    judge: JudgeResult | None = None
