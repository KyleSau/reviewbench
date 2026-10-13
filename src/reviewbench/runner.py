"""Run a reviewer over a dataset and produce a self-describing run file.

A run file records *what* was measured (dataset fingerprint, reviewer and judge config,
matcher tolerance) next to the raw findings and the computed metrics, so it can be
reported on or gated against later without re-running anything.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from reviewbench import __version__
from reviewbench.dataset import Dataset
from reviewbench.judge import ClaudeJudge, JudgeInfo
from reviewbench.llm import FatalLLMError
from reviewbench.matching import DEFAULT_LINE_TOLERANCE, match_findings
from reviewbench.metrics import Metrics, compute_metrics
from reviewbench.models import Case, CaseResult
from reviewbench.reviewers.base import Reviewer, ReviewerInfo

RUN_SCHEMA_VERSION = 1


class DatasetInfo(BaseModel):
    name: str
    path: str
    fingerprint: str
    case_count: int
    issue_count: int


class RunResult(BaseModel):
    schema_version: int = RUN_SCHEMA_VERSION
    name: str
    created_at: str
    reviewbench_version: str
    reviewer: ReviewerInfo
    judge: JudgeInfo | None = None
    dataset: DatasetInfo
    line_tolerance: int
    cases: list[CaseResult]
    metrics: Metrics


def review_case(case: Case, reviewer: Reviewer, judge: ClaudeJudge | None, line_tolerance: int) -> CaseResult:
    started = time.perf_counter()
    output = reviewer.review(case.to_pull_request())
    if output.latency_ms is None and output.status == "ok":
        # Model-backed reviewers time the API call themselves; time the rest here.
        output.latency_ms = (time.perf_counter() - started) * 1000
    result = CaseResult(
        case_id=case.id,
        expected_issue_count=len(case.expected_issues),
        status=output.status,
        error=output.error,
        findings=output.findings,
        latency_ms=output.latency_ms,
        usage=output.usage,
        served_model=output.served_model,
        fallback_used=output.fallback_used,
    )
    if result.status != "ok":
        return result
    result.matches = match_findings(result.findings, case.expected_issues, line_tolerance)
    if judge is not None and result.findings and case.expected_issues:
        result.judge = judge.judge(case, result.findings)
    return result


def run_benchmark(
    dataset: Dataset,
    reviewer: Reviewer,
    *,
    name: str,
    judge: ClaudeJudge | None = None,
    line_tolerance: int = DEFAULT_LINE_TOLERANCE,
    workers: int = 1,
    progress: Callable[[CaseResult], None] | None = None,
) -> RunResult:
    lock = threading.Lock()

    def work(case: Case) -> CaseResult:
        result = review_case(case, reviewer, judge, line_tolerance)
        if progress:
            with lock:  # progress callbacks need not be thread-safe
                progress(result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, case) for case in dataset.cases]
        try:
            results = [f.result() for f in futures]  # keeps dataset order
        except FatalLLMError:
            for f in futures:
                f.cancel()
            raise

    return RunResult(
        name=name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        reviewbench_version=__version__,
        reviewer=reviewer.info(),
        judge=judge.info() if judge else None,
        dataset=DatasetInfo(
            name=dataset.name,
            path=dataset.path,
            fingerprint=dataset.fingerprint,
            case_count=len(dataset.cases),
            issue_count=dataset.issue_count,
        ),
        line_tolerance=line_tolerance,
        cases=results,
        metrics=compute_metrics(dataset.cases, results),
    )


def save_run(run: RunResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # LF on every OS, so a baseline regenerated on Windows diffs cleanly against Linux CI.
    target.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")


def load_run(path: str | Path) -> RunResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != RUN_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported run schema_version {version!r}")
    return RunResult.model_validate(data)
