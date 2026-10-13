"""Render one or two run files as a Markdown report (the second is compared to the first)."""

from __future__ import annotations

from reviewbench.metrics import PRF, wilson_interval
from reviewbench.models import CaseResult
from reviewbench.pricing import PRICES_RETRIEVED
from reviewbench.runner import RunResult

MAX_DISAGREEMENTS = 10


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _pp(a: float | None, b: float | None) -> str:
    return "" if a is None or b is None else f"{(b - a) * 100:+.1f} pp"


def _ci(successes: int, n: int) -> str:
    interval = wilson_interval(successes, n)
    return "" if interval is None else f" ({interval[0] * 100:.0f}-{interval[1] * 100:.0f}%)"


def _ms(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f} ms" if x < 10 else f"{x:,.0f} ms"


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def _issue_label(index: int | None) -> str:
    return "none" if index is None else f"I{index + 1}"


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _reviewer_label(run: RunResult) -> str:
    reviewer = run.reviewer
    if not reviewer.model:
        return reviewer.name
    effort = reviewer.settings.get("effort")
    return f"{reviewer.name} ({reviewer.model}" + (f", effort={effort})" if effort else ")")


def _summary_table(runs: list[RunResult]) -> list[str]:
    two = len(runs) == 2
    metrics = [r.metrics for r in runs]
    overall = [m.overall for m in metrics]
    rows: list[list[str]] = []

    def add(label: str, values: list[str], change: str = "") -> None:
        rows.append([label, *values, *([change] if two else [])])

    same_data = runs[0].dataset.fingerprint == runs[-1].dataset.fingerprint
    add("Reviewer", [_reviewer_label(r) for r in runs])
    add(
        "Dataset",
        [f"{r.dataset.name} (`{r.dataset.fingerprint}`)" for r in runs],
        "" if same_data else "**different datasets: not comparable**",
    )
    add("Scored cases", [f"{m.scored_cases}/{m.total_cases}" for m in metrics])
    add(
        "Precision (95% CI)",
        [_pct(p.precision) + _ci(p.tp, p.tp + p.fp) for p in overall],
        _pp(overall[0].precision, overall[-1].precision),
    )
    add(
        "Recall (95% CI)",
        [_pct(p.recall) + _ci(p.tp, p.tp + p.fn) for p in overall],
        _pp(overall[0].recall, overall[-1].recall),
    )
    add("F1", [_pct(p.f1) for p in overall], _pp(overall[0].f1, overall[-1].f1))
    add("TP / FP / FN", [f"{p.tp} / {p.fp} / {p.fn}" for p in overall])
    add(
        "Clean PRs with any finding",
        [f"{m.clean_cases_flagged}/{m.clean_cases} ({_pct(m.clean_case_fp_rate)})" for m in metrics],
        _pp(metrics[0].clean_case_fp_rate, metrics[-1].clean_case_fp_rate),
    )
    add("Latency p50 / p95", [f"{_ms(m.latency.p50_ms)} / {_ms(m.latency.p95_ms)}" for m in metrics])
    add("Tokens in / out", [f"{m.usage.input_tokens:,} / {m.usage.output_tokens:,}" for m in metrics])
    add("Estimated reviewer cost", [_usd(m.usage.cost_usd) for m in metrics])
    add("Cases served by fallback model", [str(m.fallback_cases) for m in metrics])
    header = ["Metric", *[f"`{r.name}`" for r in runs], *(["Change"] if two else [])]
    return _table(header, rows)


def _category_table(runs: list[RunResult]) -> list[str]:
    categories = sorted({cat for r in runs for cat in r.metrics.per_category})
    rows = []
    for cat in categories:
        stats = [r.metrics.per_category.get(cat, PRF()) for r in runs]
        issues = max(s.tp + s.fn for s in stats)
        found = [f"{s.tp}/{s.tp + s.fn}" if s.tp + s.fn else "-" for s in stats]
        rows.append([cat, str(issues), *found, *[str(s.fp) for s in stats]])
    header = [
        "Category",
        "Issues",
        *[f"`{r.name}` found" for r in runs],
        *[f"`{r.name}` FP" for r in runs],
    ]
    return _table(header, rows)


def _case_cell(result: CaseResult | None) -> str:
    if result is None:
        return "-"
    if result.status != "ok":
        return f"**{result.status}**"
    tp = len(result.matches)
    return f"{tp} / {len(result.findings) - tp} / {result.expected_issue_count - tp}"


def _case_table(runs: list[RunResult]) -> list[str]:
    by_run = [{c.case_id: c for c in r.cases} for r in runs]
    rows = []
    for case in runs[0].cases:
        cells = [_case_cell(lookup.get(case.case_id)) for lookup in by_run]
        rows.append([f"`{case.case_id}`", str(case.expected_issue_count), *cells])
    return _table(["Case", "Issues", *[f"`{r.name}` TP / FP / FN" for r in runs]], rows)


def _judge_section(run: RunResult) -> list[str]:
    j = run.metrics.judge
    if j is None or run.judge is None:
        return []
    kappa = "n/a" if j.kappa is None else f"{j.kappa:.2f}"
    m = run.metrics.overall
    lines = [
        f"### `{run.name}` judged by {run.judge.model} (rubric `{run.judge.rubric_sha256}`)",
        "",
        f"- Cases sent to the judge: {j.judged_cases} ok, {j.failed_cases} failed",
        f"- Precision / recall / F1 by judge: {_pct(j.overall.precision)} / "
        f"{_pct(j.overall.recall)} / {_pct(j.overall.f1)}; by matcher: {_pct(m.precision)} / "
        f"{_pct(m.recall)} / {_pct(m.f1)}",
        f"- Judge-matcher agreement over {j.compared_findings} judged findings: "
        f"{_pct(j.agreement)} (Cohen's kappa {kappa})",
        f"- Judge cost: {_usd(j.usage.cost_usd)}",
        "",
    ]
    rows = []
    for case in run.cases:
        if case.judge is None or case.judge.status != "ok":
            continue
        by_matcher = {m.finding_index: m.issue_index for m in case.matches}
        by_judge = {m.finding_index: m.issue_index for m in case.judge.matches}
        for verdict in case.judge.verdicts:
            idx = verdict.finding_index
            if by_matcher.get(idx) != by_judge.get(idx):
                rows.append(
                    [
                        f"`{case.case_id}`",
                        f"F{idx + 1}",
                        _issue_label(by_matcher.get(idx)),
                        _issue_label(by_judge.get(idx)),
                        _cell(verdict.rationale),
                    ]
                )
    if rows:
        lines += ["Disagreements (worth reading by hand):", ""]
        lines += _table(["Case", "Finding", "Matcher", "Judge", "Judge rationale"], rows[:MAX_DISAGREEMENTS])
        lines.append("")
    return lines


def _unscored_section(runs: list[RunResult]) -> list[str]:
    rows = [
        [f"`{r.name}`", f"`{c.case_id}`", c.status, _cell(c.error or "")]
        for r in runs
        for c in r.cases
        if c.status != "ok"
    ]
    if not rows:
        return []
    return ["## Unscored cases", "", *_table(["Run", "Case", "Status", "Detail"], rows), ""]


def render_report(runs: list[RunResult]) -> str:
    if not 1 <= len(runs) <= 2:
        raise ValueError("report takes one or two runs")
    names = " vs ".join(f"`{r.name}`" for r in runs)
    out = [f"# reviewbench report: {names}", "", *_summary_table(runs), ""]
    out += ["## By category", "", *_category_table(runs), ""]
    out += ["## By case", "", *_case_table(runs), ""]
    judge_lines = [line for r in runs for line in _judge_section(r)]
    if judge_lines:
        out += ["## LLM judge", "", *judge_lines]
    out += _unscored_section(runs)
    out += [
        "Precision = TP / (TP + FP), recall = TP / (TP + FN); intervals are 95% Wilson score "
        "intervals. Only cases with status `ok` are scored. Costs are estimates from recorded "
        f"token usage and list prices retrieved {PRICES_RETRIEVED}.",
        "",
    ]
    return "\n".join(out)
