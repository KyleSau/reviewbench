"""The reviewer interface. A reviewer turns a pull request into findings; that's all."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from reviewbench.models import CaseStatus, Finding, PullRequest, Usage


class ReviewerInfo(BaseModel):
    """Recorded in every run file so results can be traced to an exact configuration."""

    name: str
    model: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ReviewOutput:
    status: CaseStatus = "ok"
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None
    latency_ms: float | None = None
    usage: Usage | None = None
    served_model: str | None = None
    fallback_used: bool = False


class Reviewer(Protocol):
    name: str

    def info(self) -> ReviewerInfo: ...

    def review(self, pr: PullRequest) -> ReviewOutput: ...
