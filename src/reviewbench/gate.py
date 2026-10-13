"""Regression gate: fail CI when a candidate run is meaningfully worse than a baseline.

Rules, in order:

1. The runs must be comparable: same dataset fingerprint and same matcher tolerance.
   Comparing scores across different label sets is meaningless, so a mismatch fails
   with an explanation instead of producing a number.
2. Unscored cases (API errors, refusals, truncations) in the candidate must not exceed
   ``max_unscored``. Otherwise a reviewer that crashes on its hardest cases could pass.
3. Recall and precision may not drop by more than the allowed absolute amount.
   A metric that is n/a (no findings at all) is treated as 0: a reviewer that says
   nothing has no precision to defend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reviewbench.runner import RunResult

# Metrics are ratios of small integers, so 0.80 - 0.75 is 0.05000000000000004 in floating
# point. Compare with a tolerance so "exactly at the limit" passes as documented.
EPSILON = 1e-9


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _value(x: float | None) -> float:
    return 0.0 if x is None else x


def evaluate_gate(
    baseline: RunResult,
    candidate: RunResult,
    *,
    max_recall_drop: float,
    max_precision_drop: float,
    max_unscored: int = 0,
) -> GateResult:
    result = GateResult()

    same_data = baseline.dataset.fingerprint == candidate.dataset.fingerprint
    result.checks.append(
        GateCheck(
            "same dataset",
            same_data,
            f"baseline {baseline.dataset.fingerprint} vs candidate {candidate.dataset.fingerprint}"
            + ("" if same_data else " (labels or cases differ; regenerate the baseline)"),
        )
    )
    same_tolerance = baseline.line_tolerance == candidate.line_tolerance
    result.checks.append(
        GateCheck(
            "same line tolerance",
            same_tolerance,
            f"baseline {baseline.line_tolerance} vs candidate {candidate.line_tolerance}",
        )
    )
    if not (same_data and same_tolerance):
        return result  # nothing below would be meaningful

    unscored = candidate.metrics.total_cases - candidate.metrics.scored_cases
    result.checks.append(
        GateCheck(
            "unscored cases",
            unscored <= max_unscored,
            f"{unscored} unscored (errors/refusals/truncations), allowed {max_unscored}",
        )
    )

    for label, base, cand, limit in (
        ("recall", baseline.metrics.overall.recall, candidate.metrics.overall.recall, max_recall_drop),
        (
            "precision",
            baseline.metrics.overall.precision,
            candidate.metrics.overall.precision,
            max_precision_drop,
        ),
    ):
        drop = _value(base) - _value(cand)
        result.checks.append(
            GateCheck(
                f"{label} drop",
                drop <= limit + EPSILON,
                f"{_value(base):.3f} -> {_value(cand):.3f} (drop {drop:+.3f}, allowed {limit:.3f})",
            )
        )
    return result
