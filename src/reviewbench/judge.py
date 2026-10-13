"""LLM-as-judge: semantic matching of findings to ground-truth issues.

The deterministic matcher checks *where* and *what kind*. The judge checks *whether it
is the same defect*, using the written rubric in ``rubrics/match_rubric.md``. Running
both and reporting their agreement tells you how much to trust the cheap matcher, and
the disagreements are exactly the cases worth reading by hand.

Design notes:

* The judge sees the diff, the issues and the findings, and never the matcher's verdict
  (so agreement is measured, not induced) or which reviewer produced the findings.
* It is only called for cases with at least one finding *and* at least one issue. With
  no findings every issue is a miss; on a clean case every finding is a false positive.
  Neither needs a model to decide.
* Verdicts are reduced to one-to-one matches the same way the matcher does: the first
  finding (by index) to claim an issue keeps it; later claims are duplicates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from reviewbench.diff import render_numbered
from reviewbench.llm import CallFailed, LLMSettings, call_structured, make_client
from reviewbench.models import Case, Finding, JudgeResult, JudgeVerdict, Match

DEFAULT_RUBRIC_PATH = Path("rubrics/match_rubric.md")

VERDICTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_id": {"type": "string"},
                    "rationale": {"type": "string"},
                    "matched_issue_id": {"type": "string"},
                },
                "required": ["finding_id", "rationale", "matched_issue_id"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


class JudgeInfo(BaseModel):
    name: str
    model: str
    settings: dict[str, Any]
    rubric_sha256: str


def build_judge_prompt(case: Case, findings: list[Finding]) -> str:
    issues = "\n".join(
        f"I{n}: file={i.file} lines={i.line_start}-{i.line_end} category={i.category.value}\n"
        f"    {i.description}"
        for n, i in enumerate(case.expected_issues, start=1)
    )
    found = "\n".join(
        f"F{n}: file={f.file} lines={f.line_start}-{f.line_end} category={f.category.value} "
        f"severity={f.severity.value}\n    <explanation>{f.explanation}</explanation>"
        for n, f in enumerate(findings, start=1)
    )
    return (
        f"<diff>\n{render_numbered(case.diff)}</diff>\n\n"
        f"<known_issues>\n{issues}\n</known_issues>\n\n"
        f"<review_findings>\n{found}\n</review_findings>\n\n"
        f"Return exactly one verdict for each of F1..F{len(findings)}."
    )


def parse_verdicts(
    data: dict[str, Any], finding_count: int, issue_count: int
) -> tuple[list[JudgeVerdict], list[Match]]:
    """Turn the judge's raw JSON into verdicts and one-to-one matches.

    Unknown or malformed ids are treated as "no match" rather than trusted.
    """
    by_finding: dict[int, JudgeVerdict] = {}
    for raw in data.get("verdicts", []):
        finding_index = _parse_id(raw.get("finding_id", ""), "F", finding_count)
        if finding_index is None or finding_index in by_finding:
            continue
        by_finding[finding_index] = JudgeVerdict(
            finding_index=finding_index,
            issue_index=_parse_id(raw.get("matched_issue_id", ""), "I", issue_count),
            rationale=str(raw.get("rationale", "")),
        )
    verdicts = [
        by_finding.get(n, JudgeVerdict(finding_index=n, issue_index=None, rationale="(no verdict returned)"))
        for n in range(finding_count)
    ]
    claimed: set[int] = set()
    matches: list[Match] = []
    for verdict in verdicts:
        if verdict.issue_index is not None and verdict.issue_index not in claimed:
            claimed.add(verdict.issue_index)
            matches.append(Match(finding_index=verdict.finding_index, issue_index=verdict.issue_index))
    return verdicts, matches


def _parse_id(value: str, prefix: str, count: int) -> int | None:
    value = value.strip().upper()
    if not value.startswith(prefix) or not value[1:].isdigit():
        return None
    index = int(value[1:]) - 1
    return index if 0 <= index < count else None


class ClaudeJudge:
    name = "claude"

    def __init__(self, settings: LLMSettings, rubric: str, client: Any | None = None) -> None:
        self.settings = settings
        self.rubric = rubric
        self._client = client if client is not None else make_client()

    @classmethod
    def from_rubric_file(
        cls, settings: LLMSettings, path: Path = DEFAULT_RUBRIC_PATH, client: Any | None = None
    ) -> ClaudeJudge:
        return cls(settings, path.read_text(encoding="utf-8"), client)

    def info(self) -> JudgeInfo:
        return JudgeInfo(
            name=self.name,
            model=self.settings.model,
            settings=self.settings.describe(),
            rubric_sha256=hashlib.sha256(self.rubric.encode("utf-8")).hexdigest()[:16],
        )

    def judge(self, case: Case, findings: list[Finding]) -> JudgeResult:
        try:
            result = call_structured(
                self._client,
                self.settings,
                system=self.rubric,
                user=build_judge_prompt(case, findings),
                schema=VERDICTS_SCHEMA,
            )
        except CallFailed as failure:
            return JudgeResult(
                status=failure.status,
                error=failure.message,
                usage=failure.usage,
                latency_ms=failure.latency_ms,
            )
        verdicts, matches = parse_verdicts(result.data, len(findings), len(case.expected_issues))
        return JudgeResult(
            status="ok",
            verdicts=verdicts,
            matches=matches,
            usage=result.usage,
            latency_ms=result.latency_ms,
        )
