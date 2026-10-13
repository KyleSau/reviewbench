"""Command-line entry point: ``reviewbench run | report | gate``."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from reviewbench import __version__
from reviewbench.dataset import DatasetError, load_dataset
from reviewbench.gate import evaluate_gate
from reviewbench.judge import DEFAULT_RUBRIC_PATH, ClaudeJudge
from reviewbench.llm import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EFFORT_LEVELS,
    MISSING_KEY_HELP,
    FatalLLMError,
    LLMSettings,
    has_credentials,
)
from reviewbench.matching import DEFAULT_LINE_TOLERANCE
from reviewbench.models import CaseResult
from reviewbench.report import render_report
from reviewbench.reviewers import REVIEWER_NAMES, HeuristicReviewer, NullReviewer, OracleReviewer
from reviewbench.reviewers.base import Reviewer
from reviewbench.runner import load_run, run_benchmark, save_run

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def cmd_run(args: argparse.Namespace) -> int:
    needs_claude = args.reviewer == "claude" or args.judge == "claude"
    if needs_claude and not has_credentials():
        _err(MISSING_KEY_HELP)
        return EXIT_USAGE
    try:
        dataset = load_dataset(args.dataset)
        if args.case:
            dataset = dataset.select(args.case)
    except DatasetError as exc:
        _err(f"error: invalid dataset: {exc}")
        return EXIT_USAGE

    reviewer: Reviewer
    if args.reviewer == "heuristic":
        reviewer = HeuristicReviewer()
    elif args.reviewer == "oracle":
        reviewer = OracleReviewer(dataset)
    elif args.reviewer == "null":
        reviewer = NullReviewer()
    else:
        from reviewbench.reviewers.claude import ClaudeReviewer

        reviewer = ClaudeReviewer(LLMSettings(args.model, args.effort, args.max_tokens, not args.no_fallback))

    judge = None
    if args.judge == "claude":
        if args.judge_model == args.model and args.reviewer == "claude":
            _err(
                "note: judge and reviewer are the same model; consider --judge-model to "
                "reduce self-preference bias"
            )
        judge = ClaudeJudge.from_rubric_file(
            LLMSettings(args.judge_model, args.judge_effort, args.max_tokens, not args.no_fallback),
            Path(args.rubric),
        )

    out = Path(args.out or f"runs/{args.reviewer}-{datetime.now():%Y%m%d-%H%M%S}.json")
    total = len(dataset.cases)
    done = 0

    def progress(result: CaseResult) -> None:
        nonlocal done
        done += 1
        detail = f"{len(result.findings)} finding(s)" if result.status == "ok" else result.error
        _err(f"[{done:>2}/{total}] {result.case_id}: {result.status} - {detail}")

    try:
        run = run_benchmark(
            dataset,
            reviewer,
            name=args.name or out.stem,
            judge=judge,
            line_tolerance=args.line_tolerance,
            workers=args.workers,
            progress=progress,
        )
    except FatalLLMError as exc:
        _err(f"error: run aborted: {exc}")
        return EXIT_FAILED

    save_run(run, out)
    m = run.metrics
    print(f"wrote {out}")
    print(
        f"precision={_fmt(m.overall.precision)} recall={_fmt(m.overall.recall)} "
        f"f1={_fmt(m.overall.f1)} (TP={m.overall.tp} FP={m.overall.fp} FN={m.overall.fn}) "
        f"clean-PR FP rate={_fmt(m.clean_case_fp_rate)} scored={m.scored_cases}/{m.total_cases}"
    )
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    try:
        runs = [load_run(p) for p in args.runs]
    except (OSError, ValueError) as exc:
        _err(f"error: cannot read run file: {exc}")
        return EXIT_USAGE
    text = render_report(runs)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return EXIT_OK


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        baseline, candidate = load_run(args.baseline), load_run(args.candidate)
    except (OSError, ValueError) as exc:
        _err(f"error: cannot read run file: {exc}")
        return EXIT_USAGE
    result = evaluate_gate(
        baseline,
        candidate,
        max_recall_drop=args.max_recall_drop,
        max_precision_drop=args.max_precision_drop,
        max_unscored=args.max_unscored,
    )
    print(f"gate: baseline={args.baseline} candidate={args.candidate}")
    for check in result.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print("gate PASSED" if result.passed else "gate FAILED")
    return EXIT_OK if result.passed else EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reviewbench", description="Evaluate AI code reviewers on seeded-bug pull requests."
    )
    parser.add_argument("--version", action="version", version=f"reviewbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a reviewer over the dataset and write a run file")
    run.add_argument("--reviewer", choices=REVIEWER_NAMES, required=True)
    run.add_argument("--out", help="run file to write (default: runs/<reviewer>-<timestamp>.json)")
    run.add_argument("--name", help="label used in reports (default: output file name)")
    run.add_argument("--dataset", default="datasets/v1")
    run.add_argument("--case", action="append", help="only run this case id (repeatable)")
    run.add_argument("--line-tolerance", type=int, default=DEFAULT_LINE_TOLERANCE)
    run.add_argument("--workers", type=int, default=4, help="cases reviewed in parallel")
    claude = run.add_argument_group("Claude options")
    claude.add_argument("--model", default=os.environ.get("REVIEWBENCH_MODEL", DEFAULT_MODEL))
    claude.add_argument("--effort", choices=EFFORT_LEVELS, default=DEFAULT_EFFORT)
    claude.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    claude.add_argument(
        "--no-fallback",
        action="store_true",
        help="disable the server-side refusal fallback (e.g. for models without it)",
    )
    claude.add_argument("--judge", choices=("none", "claude"), default="none")
    claude.add_argument("--judge-model", default=os.environ.get("REVIEWBENCH_JUDGE_MODEL", DEFAULT_MODEL))
    claude.add_argument("--judge-effort", choices=EFFORT_LEVELS, default=DEFAULT_EFFORT)
    claude.add_argument("--rubric", default=str(DEFAULT_RUBRIC_PATH))
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="render one run, or compare two, as Markdown")
    report.add_argument("runs", nargs="+", metavar="RUN_JSON")
    report.add_argument("--out", help="write the Markdown here instead of stdout")
    report.set_defaults(func=cmd_report)

    gate = sub.add_parser("gate", help="exit non-zero if the candidate regressed vs the baseline")
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--max-recall-drop", type=float, default=0.05)
    gate.add_argument("--max-precision-drop", type=float, default=0.05)
    gate.add_argument("--max-unscored", type=int, default=0)
    gate.set_defaults(func=cmd_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report" and len(args.runs) > 2:
        _err("error: report takes one run, or two to compare")
        return EXIT_USAGE
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
