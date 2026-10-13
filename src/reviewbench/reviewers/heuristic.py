"""A deterministic, offline baseline reviewer built from regex rules.

What it is for: a floor. It runs in CI with no API key, it never changes between runs,
and it gives the LLM reviewer something concrete to beat.

What it is not: a static analyzer. It only looks at lines the PR adds, one file at a
time, with no parsing, types or data flow. Honest limits:

* It does not attempt missing-authorization or race-condition detection at all;
  both need to understand intent and concurrency, which regexes cannot.
* N+1 detection only sees an explicit DB call inside a loop body. An ORM lazy load
  (``invoice.customer.email`` in a loop) is invisible to it, and deliberate batching
  inside a loop is flagged anyway.
* Null handling is limited to two syntactic smells (``x!.y`` in TypeScript and
  dereferencing ``FirstOrDefault(...)`` on the same line in C#).
* The rules and the dataset were written by the same person, so its score on this
  dataset is an optimistic estimate of regex-level tooling, not a neutral one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reviewbench.diff import FileDiff, parse_diff
from reviewbench.models import Category, Finding, PullRequest, Severity
from reviewbench.reviewers.base import ReviewerInfo, ReviewOutput

MERGE_WINDOW = 5  # same-category hits this close together are reported once

_PY = (".py",)
_TS = (".ts", ".tsx", ".js", ".jsx")
_CS = (".cs",)


@dataclass(frozen=True)
class LineRule:
    name: str
    category: Category
    severity: Severity
    pattern: re.Pattern[str]
    message: str
    unless: re.Pattern[str] | None = None
    extensions: tuple[str, ...] = ()  # empty = any file

    def applies_to(self, path: str) -> bool:
        return not self.extensions or path.endswith(self.extensions)

    def hits(self, text: str) -> bool:
        return bool(self.pattern.search(text)) and not (self.unless and self.unless.search(text))


_SQL_KEYWORD = r"\b(?:SELECT|INSERT INTO|UPDATE|DELETE FROM|WHERE|ORDER BY|VALUES)\b"
_INTERPOLATION = r"(?:\{[^}]*\}|[\"']\s*\+\s*\w)"

LINE_RULES: tuple[LineRule, ...] = (
    LineRule(
        "sql-string-building",
        Category.SQL_INJECTION,
        Severity.CRITICAL,
        re.compile(rf"{_SQL_KEYWORD}.*{_INTERPOLATION}|{_INTERPOLATION}.*{_SQL_KEYWORD}"),
        "A value is interpolated or concatenated into SQL text; use bound parameters.",
    ),
    LineRule(
        "unsafe-deserializer",
        Category.INSECURE_DESERIALIZATION,
        Severity.CRITICAL,
        re.compile(r"\bpickle\.loads?\(|\bmarshal\.loads?\(|\byaml\.load\(|\bBinaryFormatter\b"),
        "Deserializing untrusted data with this API can execute arbitrary code.",
        unless=re.compile(r"SafeLoader"),
    ),
    LineRule(
        "json-type-name-handling",
        Category.INSECURE_DESERIALIZATION,
        Severity.CRITICAL,
        re.compile(r"TypeNameHandling\.(?:All|Auto|Objects|Arrays)\b"),
        "TypeNameHandling other than None lets the payload choose which types are created.",
        extensions=_CS,
    ),
    LineRule(
        "open-without-with",
        Category.RESOURCE_LEAK,
        Severity.MEDIUM,
        re.compile(r"^\s*\w+\s*=\s*open\("),
        "File opened outside a `with` block is not closed if an exception is raised.",
        extensions=_PY,
    ),
    LineRule(
        "disposable-without-using",
        Category.RESOURCE_LEAK,
        Severity.MEDIUM,
        re.compile(r"\bnew\s+(?:SqlConnection|SqlCommand|FileStream|StreamReader|StreamWriter)\("),
        "Disposable created without `using`; it leaks if not disposed on every path.",
        unless=re.compile(r"\busing\b"),
        extensions=_CS,
    ),
    LineRule(
        "async-void",
        Category.ERROR_HANDLING,
        Severity.HIGH,
        re.compile(r"\basync\s+void\b"),
        "Exceptions from an `async void` method cannot be observed and crash the process.",
        extensions=_CS,
    ),
    LineRule(
        "rethrow-loses-stack",
        Category.ERROR_HANDLING,
        Severity.LOW,
        re.compile(r"^\s*throw\s+[a-z]\w*\s*;"),
        "`throw ex;` resets the stack trace; use `throw;`.",
        extensions=_CS,
    ),
    LineRule(
        "inclusive-length-bound",
        Category.OFF_BY_ONE,
        Severity.MEDIUM,
        re.compile(r"<=\s*[\w.]+\.(?:length|Length|Count)\b(?!\s*-)|<=\s*len\("),
        "Loop bound `<= length` runs one past the last index.",
    ),
    LineRule(
        "non-null-assertion",
        Category.NULL_HANDLING,
        Severity.MEDIUM,
        re.compile(r"[\w\])]!\.[A-Za-z_]"),
        "Non-null assertion hides a possibly undefined value.",
        extensions=_TS,
    ),
    LineRule(
        "first-or-default-deref",
        Category.NULL_HANDLING,
        Severity.MEDIUM,
        re.compile(r"\.(?:First|Single)OrDefault(?:Async)?\([^;]*\)\s*\.\w"),
        "Result of *OrDefault() is dereferenced without a null check.",
        extensions=_CS,
    ),
)

_LOOP_START = re.compile(r"^\s*(?:for\s+.+\s+in\s+.+:|for\s*\(|foreach\s*\(|while\s*\()|\.forEach\(")
_DB_CALL = re.compile(
    r"\.objects\.|\b_?db\.\w+|\bsession\.(?:query|execute)\(|\bcursor\.execute\(|"
    r"\b(?:knex|prisma)\.\w+"
)
_PY_EXCEPT = re.compile(r"^\s*except\b[^:]*:\s*(?:pass|\.\.\.)?\s*(?:#.*)?$")
_CATCH_OPEN = re.compile(r"\bcatch\b[^{]*\{\s*(?://.*)?$")
_CATCH_NO_BRACE = re.compile(r"\bcatch\b[^{]*$")
_CATCH_EMPTY_INLINE = re.compile(r"\bcatch\b[^{]*\{\s*\}")
_COMMENT_ONLY = re.compile(r"^\s*(?://.*|/\*.*|\*.*|#.*)?$")
_RELEASE_PAIRS = (("addEventListener(", "removeEventListener("), ("setInterval(", "clearInterval("))


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


class HeuristicReviewer:
    name = "heuristic"

    def info(self) -> ReviewerInfo:
        block_rules = ["swallowed-exception", "listener-without-cleanup", "query-in-loop"]
        return ReviewerInfo(name=self.name, settings={"rules": [r.name for r in LINE_RULES] + block_rules})

    def review(self, pr: PullRequest) -> ReviewOutput:
        findings: list[Finding] = []
        for file_diff in parse_diff(pr.diff):
            if file_diff.is_deleted:
                continue
            raw = (
                self._line_rules(file_diff)
                + self._swallowed_exceptions(file_diff)
                + self._listener_leaks(file_diff)
                + self._queries_in_loops(file_diff)
            )
            findings.extend(_merge_nearby(raw))
        return ReviewOutput(status="ok", findings=findings)

    # -- rules -----------------------------------------------------------------------

    def _line_rules(self, f: FileDiff) -> list[Finding]:
        out = []
        for lineno, text in sorted(f.added_lines().items()):
            for rule in LINE_RULES:
                if rule.applies_to(f.path) and rule.hits(text):
                    out.append(
                        _finding(f.path, lineno, rule.category, rule.severity, rule.name, rule.message)
                    )
        return out

    def _swallowed_exceptions(self, f: FileDiff) -> list[Finding]:
        out = []
        lines = sorted(f.new_side_lines().items())
        added = f.added_lines()
        for idx, (lineno, text) in enumerate(lines):
            if lineno not in added:
                continue
            following = [t for _, t in lines[idx + 1 : idx + 6]]
            body = [t.strip() for t in following if not _COMMENT_ONLY.match(t)]
            if _PY_EXCEPT.match(text):
                inline = text.split(":", 1)[1].split("#")[0].strip()
                swallowed = inline in ("pass", "...") or (
                    not inline and bool(body) and body[0] in ("pass", "...")
                )
            elif _CATCH_EMPTY_INLINE.search(text):
                swallowed = True
            elif _CATCH_OPEN.search(text):
                swallowed = bool(body) and body[0].startswith("}")
            elif _CATCH_NO_BRACE.search(text):  # C# style: brace on the next line
                swallowed = len(body) >= 2 and body[0] == "{" and body[1].startswith("}")
            else:
                swallowed = False
            if swallowed:
                out.append(
                    _finding(
                        f.path,
                        lineno,
                        Category.SWALLOWED_EXCEPTION,
                        Severity.HIGH,
                        "swallowed-exception",
                        "Exception is caught and discarded without logging or rethrowing.",
                    )
                )
        return out

    def _listener_leaks(self, f: FileDiff) -> list[Finding]:
        out = []
        added = f.added_lines()
        all_added = "\n".join(added.values())
        for lineno, text in sorted(added.items()):
            for acquire, release in _RELEASE_PAIRS:
                if acquire in text and release not in all_added:
                    out.append(
                        _finding(
                            f.path,
                            lineno,
                            Category.RESOURCE_LEAK,
                            Severity.MEDIUM,
                            "listener-without-cleanup",
                            f"`{acquire[:-1]}` is added without a matching `{release[:-1]}`.",
                        )
                    )
        return out

    def _queries_in_loops(self, f: FileDiff) -> list[Finding]:
        out = []
        lines = sorted(f.new_side_lines().items())
        added = f.added_lines()
        for idx, (_, text) in enumerate(lines):
            if not _LOOP_START.search(text):
                continue
            loop_indent = _indent(text)
            for lineno, body in lines[idx + 1 :]:
                stripped = body.strip()
                if not stripped or stripped == "{":
                    continue
                if _indent(body) <= loop_indent:
                    break  # left the loop body
                if lineno in added and _DB_CALL.search(body):
                    out.append(
                        _finding(
                            f.path,
                            lineno,
                            Category.N_PLUS_ONE_QUERY,
                            Severity.MEDIUM,
                            "query-in-loop",
                            "Database call inside a loop runs once per item (N+1 queries).",
                        )
                    )
        return out


def _finding(
    path: str, lineno: int, category: Category, severity: Severity, rule: str, message: str
) -> Finding:
    return Finding(
        file=path,
        line_start=lineno,
        line_end=lineno,
        category=category,
        severity=severity,
        explanation=f"[{rule}] {message}",
    )


def _merge_nearby(findings: list[Finding]) -> list[Finding]:
    """Collapse same-category hits within MERGE_WINDOW lines into one finding, the way a
    human reviewer leaves one comment on a multi-line problem."""
    merged: list[Finding] = []
    for f in sorted(findings, key=lambda x: (x.category, x.line_start)):
        last = merged[-1] if merged else None
        if last and last.category == f.category and f.line_start - last.line_end <= MERGE_WINDOW:
            last.line_end = max(last.line_end, f.line_end)
            continue
        merged.append(f.model_copy())
    return sorted(merged, key=lambda x: (x.line_start, x.category))
