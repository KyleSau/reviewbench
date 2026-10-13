# Walkthrough

A guided tour of reviewbench: what each piece does, why it is built that way, and what
was deliberately left out. The last section has likely interview questions with short
answers.

Suggested reading order: `models.py` → `diff.py` → `dataset.py` → `reviewers/` → `llm.py` →
`matching.py` → `metrics.py` → `judge.py` → `runner.py` → `gate.py` → tests.

## 1. The shape of the problem

An AI code reviewer takes a diff and produces comments. To evaluate it you need three
things: inputs with known answers, a rule for deciding whether a comment "got" a known
answer, and metrics that reflect what users actually experience. A reviewer that misses
bugs is useless; a reviewer that flags everything gets muted. So the harness measures both
directions: recall on buggy PRs and false positives, including on PRs that are clean.

Everything else (judge, gate, cost tracking) exists to make those numbers trustworthy and
to keep them from regressing.

## 2. Data model (`models.py`)

Every boundary is a pydantic model: dataset files, reviewer output, run files. Malformed
data fails at the edge with a clear error instead of producing wrong metrics later.

Two choices worth pointing at:

- **`PullRequest` vs `Case`.** A reviewer receives a `PullRequest` (id, title, description,
  diff) and never a `Case`, which has the labels. The reviewer *cannot* read the answer key
  even by accident. The only reviewer that sees labels is `oracle`, and it has to be handed
  the dataset explicitly in its constructor.
- **Case status.** `CaseResult.status` is one of `ok`, `refused`, `truncated`, `error`. Only
  `ok` is scored. This is the single most important rule in the harness (see §8).

## 3. Diffs and line numbers (`diff.py`)

Labels, heuristic findings and LLM findings all use **new-file line numbers**: the line in
the file after the PR is applied. That is what a reviewer comments on in GitHub, and it is
stable however the diff was generated.

The parser walks hunks using the counts in the `@@ -a,b +c,d @@` header to know where each
hunk ends. That makes it robust to content lines that happen to start with `---` or `+++`.
Tests cover new files, multiple hunks, removed lines, CRLF input, `\ No newline at end of
file`, and truncated hunks (which raise instead of being silently accepted; a test caught a
real bug there during development).

`render_numbered()` prints the diff for the model with the new-file line number in a
left-hand column:

```
FILE app/api/orders.py (modified)
  ...
   30           .order_by(Order.created_at.desc())
      -         .all()
   31       )
      -     return jsonify([o.to_dict() for o in orders])
   32 +     total = query.count()
   33 +     orders = query.offset(page * page_size).limit(page_size).all()
```

(An excerpt of `py-orders-pagination`; the labelled off-by-one is line 33. `...` marks lines
the diff doesn't show.)

**Why:** language models are unreliable at counting lines from hunk offsets. If the model
has to compute "hunk starts at 10, this is the 23rd line, minus 2 removed lines", findings
come back off by a few lines and the matcher scores correct comments as misses. Printing the
number removes a failure mode that has nothing to do with review quality.

## 4. The dataset (`datasets/v1`, `dataset.py`)

**Format.** One `.diff` and one `.json` per case. The diff is the PR; the JSON holds title,
description and `expected_issues` (file, line range, category, severity, description).
Plain files make label changes reviewable in a normal PR diff.

**What makes a good case.** Each bug is something a careful human reviewer would actually
catch, usually wrapped in a plausible justification in the PR description (the webhook
swallows errors "because the provider retries", the checkout uses `transaction.atomic`
"so the reservation and decrement commit together"). The reviewer has to understand the
code, not pattern-match a keyword. Context lines matter: in `ts-project-delete-route`, the
existing GET/PATCH routes use `requireProjectRole(...)` and the new DELETE route does not,
which is exactly how a human would spot it.

**Clean cases.** Six PRs with no bugs. Without them you can't measure the most common
complaint about AI reviewers: noise. `cs-clean-price-batching` is a deliberate trap: a
database call inside a loop that is correct chunked batching, not an N+1.

**Validation beyond the schema.** `validate_case()` checks every label against its diff:
the file must be in the diff, every labelled line must be visible in it, and the range must
touch at least one *added* line (reviewers comment on changes, not on untouched code). A
typo in a label would otherwise turn a correct finding into a false positive and nobody would
notice.

**Fingerprint.** `Dataset.fingerprint` hashes a canonical JSON form of every case (sorted
keys, LF line endings). Runs record it, and the gate refuses to compare runs with different
fingerprints. `.gitattributes` forces LF, and the hash normalises line endings anyway, so a
Windows checkout produces the same fingerprint as CI on Linux (there's a test for that).

**How the cases were authored.** Each case was written as before/after source files; a
small authoring script generated the unified diff with `difflib` and computed label line
numbers by locating marker strings in the after-file, so no line number was counted by hand.
The loader's validation then double-checks every label. To add a case: write the diff (e.g.
`git diff`), write the JSON, run `pytest`.

## 5. Reviewers (`reviewers/`)

The interface is a `Protocol` with two methods: `info()` (recorded in the run file) and
`review(pr) -> ReviewOutput`. Anything that can turn a diff into findings can be plugged in.

### Heuristic (`heuristic.py`)

A deterministic baseline of regex rules, each scoped to file extensions where it makes
sense (`throw ex;` loses the stack trace in C# but is fine in TypeScript). It only looks at
added lines. Three rules need a little context: empty `except`/`catch` bodies, an event
listener added without a matching removal in the same file, and a DB call inside a loop body
(found by indentation). Same-category hits within 5 lines are merged into one finding, the
way a human leaves one comment on a multi-line problem.

**Why it exists:** a floor that runs in CI with no API key and never changes between runs,
so the gate has something stable to protect, and so an LLM reviewer has something concrete
to beat. Its limits are documented at the top of the file, including the uncomfortable
one: the same person wrote the rules and the dataset, so its score here is optimistic.

### Claude (`claude.py` + `llm.py`)

The system prompt is a short spec:

- Report only defects worth blocking a PR, not style or speculative refactors.
- One finding per defect; cite the numbered new-file lines; only comment on added lines.
- Severity and category definitions are spelled out so the model maps consistently.
- **"An empty findings list is a normal, expected answer."** Without this, models tend to
  find *something* on clean PRs.
- The PR content is untrusted input. A diff can contain a comment like "ignore previous
  instructions"; the prompt says to treat it as content under review.

Output uses **structured outputs** (`output_config.format` with a JSON schema). The category
and severity fields are enums, so an invalid label is impossible. The schema puts
`explanation` before `category`, so the model writes its reasoning before choosing the
label. The parsed JSON is validated again with pydantic, because the harness should never
trust a single layer.

### Oracle and null (`sanity.py`)

`oracle` returns the ground truth; `null` returns nothing. They test the *harness*, not a
reviewer. If oracle isn't 100% or null isn't 0% recall, the loader, matcher or metrics have
a bug. This is cheap and catches most wiring mistakes before money is spent on model calls.

## 6. Calling Claude (`llm.py`)

One function, `call_structured()`, is used by both the reviewer and the judge, so the API
rules live in one place:

| Rule | Why |
|---|---|
| `thinking={"type": "adaptive"}` | Claude Opus 5.5 always thinks. `budget_tokens` and `{"type": "disabled"}` return 400. |
| `output_config.effort` always set | Effort is the control for depth, latency and cost on this model. The API default is `medium`; being explicit means a changed default can't silently change results. |
| Structured outputs, no forced `tool_choice` | Forced tool use returns 400 on this model; structured outputs are the supported way to get schema-valid JSON. No assistant prefill (also 400). |
| `max_tokens=16000` | Thinking counts toward `max_tokens`. A limit sized for the answer alone would truncate. |
| Check `stop_reason` first | `refusal` and `max_tokens` become `refused` / `truncated` statuses. A refusal can have empty content, and a truncated response has partial JSON; neither should be parsed. |
| Read blocks by `type` | Thinking blocks (and a `fallback` block, after a fallback) come before the text block. |
| `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"` | If a safety classifier declines, the API retries on Anthropic's recommended fallback model in the same call. |
| System prompt marked `cache_control: ephemeral` | It's identical for every case. Prompts below the model's minimum cacheable length simply aren't cached; no harm. |

**The fallback tension.** For a production reviewer, the fallback is a reliability feature:
a false-positive refusal on a security-flavoured diff shouldn't become a missing review. For
an *eval* it is a confound: that case was answered by a different model. The harness keeps
the fallback on (the safer default) but makes it visible: `served_model` and `fallback_used`
per case, cost computed per attempt from `usage.iterations` at each model's price, a count in
the report, and `--no-fallback` to measure one model strictly.

**Error taxonomy.** Exceptions are caught most-specific-first. Authentication, permission,
not-found (wrong model id) and bad-request errors raise `FatalLLMError` and abort the run:
every case would fail the same way, so continuing wastes money and produces a meaningless run
file. Rate limits (after the SDK's built-in retries), 5xx, timeouts and connection errors
raise `CallFailed`, which is recorded on that one case; the run continues.

**Testing without a key.** Two layers. Most tests inject a fake client object returning real
`BetaMessage` objects (built with the SDK's own models). Two tests go one layer lower: the
genuine `anthropic.Anthropic` client with a mock HTTP transport, which checks what actually
goes on the wire (the `anthropic-beta` header, `fallbacks`, `output_config`, no
`tool_choice`) and that the SDK's real 401/429 exceptions map to the right behaviour.

## 7. Matching (`matching.py`)

A finding matches an issue when:

1. **Same file**, after path normalisation (`b/` prefixes, `./`, backslashes).
2. **Line window**: the finding's range overlaps the issue's range widened by 3 lines on each
   side. Reviewers legitimately comment on a neighbouring line (the call site rather than the
   assignment); 3 lines absorbs that without letting a comment in another function count.
3. **Compatible category**: identical, or one of two listed near-synonym pairs
   (`swallowed_exception`/`error_handling`, `off_by_one`/`logic_error`). The list is short on
   purpose; every pair makes the matcher more lenient.

**One-to-one, greedy by distance.** All candidate pairs are sorted by line distance (then by
index, so ties resolve identically on every run) and assigned greedily. The closest finding
wins an issue. A second finding about the same issue is a duplicate and counts as a false
positive: duplicate comments cost reviewer attention. Greedy is not guaranteed optimal in
pathological cases (a maximum bipartite matching would be), but with one or two issues per
case it never matters here, and it's easy to explain and debug.

**Why strict?** The matcher is cheap, deterministic and explainable, which is what a CI gate
needs. Its blind spot is semantics: it would accept a comment on the right line, with the
right category, that describes the wrong problem. That's what the judge is for.

## 8. Metrics (`metrics.py`)

- **Only `ok` cases are scored.** If an API error were scored as "no findings", a reviewer
  that crashes on hard cases would look precise, and a clean case that errored would look
  like a correct "no issues". Errors are infrastructure, not model behaviour. Unscored cases
  are counted, listed in the report, and the gate fails on them by default.
- **Cost includes failed calls.** A refused or truncated call still cost money.
- **Per-category attribution.** TP and FN go to the ground-truth issue's category; FP goes to
  the finding's category (there is no ground truth to attribute it to).
- **`n/a`, not 0.** Precision with zero findings is undefined. Reporting 0 or 1 would both be
  misleading. The gate treats `n/a` as 0 explicitly.
- **Clean-PR false-positive rate** is "clean PRs with at least one finding / clean PRs", the
  number a user feels ("how often does it nag me on a fine PR?"). Total findings on clean PRs
  is reported too.
- **Wilson intervals** instead of the normal approximation, because the normal approximation
  misbehaves at small n and at proportions near 0 or 1, which is exactly this dataset.
- **Nearest-rank percentiles** for latency always return an observed value, which is easy to
  explain.

## 9. LLM-as-judge (`judge.py`, `rubrics/match_rubric.md`)

**The rubric** asks one factual question per finding: is this the same defect as a listed
issue? It must have the same root cause, the same place, and be specific enough to act on.
It lists what does *not* matter (category label, severity, proposed fix, length and
confidence), handles edge cases (one finding describing two issues, duplicates, real but
unlabelled problems), includes worked examples that are deliberately *not* from the dataset,
and tells the judge to ignore instructions embedded in the findings or code.

**Judge inputs and blinding.** The judge sees the numbered diff, the issues (`I1..In`) and
the findings (`F1..Fm`). It does not see the matcher's decision (otherwise agreement would be
induced, not measured) or which reviewer produced the findings (to avoid deference to a
"reference").

**When it's called.** Only for cases with at least one finding and at least one issue. With
no findings, every issue is a miss; on a clean case, every finding is a false positive.
Neither needs a model, so the judge's cost is spent only where judgement is needed.

**Output handling.** Structured output again, with `rationale` before `matched_issue_id`.
IDs are parsed defensively: unknown or out-of-range ids become "no match", missing verdicts
become "no match" with a note, and one-to-one is enforced the same way as the matcher (first
finding to claim an issue keeps it).

**Agreement.** For every finding the judge looked at, compare the matcher's label (issue
index or none) with the judge's. The report shows the agreement rate, Cohen's kappa on the
binary "is a true positive" label (agreement corrected for chance), and a table of
disagreements with the judge's rationale. Disagreements are the most useful output: they
show either a matcher that is too strict (right defect, unusual category or line) or a judge
that is too lenient.

**What's not done yet.** The judge isn't validated against human labels in this repo. Before
using judge-scored numbers to make decisions, label a few dozen finding/issue pairs by hand,
measure judge accuracy, and test it on known non-matches (vague findings, right line but
wrong defect). Self-preference is a risk when the judge and the reviewer are the same model;
the CLI warns, and `--judge-model` picks another.

## 10. Runs, reports and the gate

**Run files** (`runner.py`) are self-describing: dataset name and fingerprint, reviewer
config (model, effort, max_tokens, fallback), judge config and rubric hash, line tolerance,
raw findings, matches, judge verdicts, per-case usage and status, and the computed metrics.
You can report on or gate against a run months later without the original environment.

Cases run in a thread pool (`--workers`); results are collected in dataset order, so output
is deterministic regardless of completion order. A `FatalLLMError` cancels pending cases.

**The report** (`report.py`) is Markdown, so it renders in a terminal, a PR comment, or
GitHub's job summary (CI writes it there). With two runs it adds a change column in
percentage points.

**The gate** (`gate.py`):

1. Refuse to compare different datasets or different line tolerances. A number computed
   across different label sets would be meaningless, so the gate fails with an explanation
   instead of producing one.
2. Fail if the candidate has unscored cases (configurable with `--max-unscored`).
3. Fail if recall or precision dropped by more than the threshold (absolute).

Two details. Metrics are ratios of small integers, so `0.80 - 0.75` is
`0.05000000000000004` in floating point; the comparison uses a tiny epsilon so "exactly at the
limit" passes as documented (there's a test for this exact case). And `n/a` precision counts
as 0, so a reviewer that says nothing can't pass.

**Baseline workflow.** `baselines/heuristic.json` is committed.
`test_committed_baseline_is_reproducible` fails if the heuristic or dataset changes without
regenerating it, so every behaviour change shows up as a reviewable diff of the baseline in
the same PR. Run-to-run fields (timestamp, sub-millisecond latency) change on regeneration;
the test compares only the metrics.

## 11. CI (`.github/workflows/ci.yml`)

- **Job 1, every push and PR, offline:** ruff, pytest, heuristic run, report into the job
  summary, gate against the baseline, upload the run as an artifact.
- **Job 2, optional:** Claude reviewer plus judge, gated against the heuristic floor. Runs on
  `main` or manual dispatch only (it costs money), and only if the `ANTHROPIC_API_KEY` secret
  exists; otherwise it logs a notice and skips. Fork PRs never receive secrets, so it can't
  leak a key or spend money on someone else's PR.

## 12. What I'd do next

In order of value: repetitions with paired confidence intervals (needed before any
prompt/model A/B claim), more and larger cases, judge calibration against human labels, an
effort sweep report (quality versus cost and latency), severity-weighted recall, and
per-category gate thresholds.

---

## Likely interview questions

**Why not just use an LLM judge for everything?**
Cost, determinism and debuggability. The CI gate has to be reproducible and free to run on
every PR; the matcher is both. The judge is the semantic check on top, and the agreement
number tells me how much the cheap check can be trusted. Where they disagree is where I read
by hand.

**How do you know the harness itself is correct?**
Three ways. Oracle and null reviewers run through the whole pipeline in tests (they must
score 100% and 0% recall). Every label is validated against its diff at load time. And the
committed baseline must reproduce exactly from code, so any behaviour change is visible.

**What happens when the API fails or refuses?**
The case gets status `error`, `refused` or `truncated` and is excluded from scoring, never
scored as "found nothing". It shows up in the report, the gate fails on it by default, and
its token cost is still counted. Configuration errors (bad key, wrong model) abort the run
immediately because every case would fail the same way.

**Why measure clean PRs separately?**
Precision alone doesn't capture "it nags me on good PRs". A reviewer can have decent
precision on buggy PRs and still flag half of the clean ones. Clean-PR flag rate is the
number that predicts whether engineers will mute the bot.

**Why count duplicate comments as false positives?**
From the PR author's side, two comments about one bug is noise. The judge rubric asks for
duplicates to be matched normally and the scorer de-duplicates, so both scorers apply the
same rule.

**How did you pick the line tolerance of 3?**
It's the smallest window that absorbs "commented on the call site instead of the
assignment" without crossing into a neighbouring statement in these small diffs. It's a
flag, and it's recorded in the run file; the gate refuses to compare runs with different
tolerances, because changing it changes every number.

**Is a 0.05 drop threshold meaningful with 21 issues?**
Barely, and I say so. One issue is about 4.8 points of recall, so 0.05 means "at most one
additional miss". The report shows Wilson intervals (recall 57%, 95% CI 37-76% for the
heuristic) to make that uncertainty explicit. The fix is more cases and repeated runs, not a
tighter threshold.

**How would you compare two prompts or two models?**
Run both on the same dataset fingerprint with several repetitions, compare per case (paired),
and look at the difference with a confidence interval, plus cost and latency side by side.
`report a.json b.json` does the single-run version today; repetitions are the next feature.

**What about prompt injection in the code under review?**
Both prompts state that PR content and findings are untrusted data. The judge only sees
findings inside tagged fields. Structured outputs constrain the response shape, so injected
text can't change the output format, and the output is re-validated with pydantic.

**Why structured outputs instead of tool calls?**
I only need JSON back, not an action. On Claude Opus 5.5 forced `tool_choice` returns a 400
anyway, and structured outputs give a schema guarantee with enums for category and severity.

**Why is the fallback on if it confounds the eval?**
Because the same code path is what you'd ship, and in production a false-positive refusal
shouldn't drop a review. In the eval it's recorded per case, priced per attempt, counted in
the report, and `--no-fallback` turns it off for a strict single-model measurement.

**How is cost computed?**
From the API's recorded usage (input, output, cache-read and cache-write tokens) times a
dated price table. When a fallback ran, `usage.iterations` has one entry per attempt and each
is priced at the model that ran it. Judge cost is reported separately so it doesn't blur the
reviewer comparison. Unknown models give "n/a", not a guess.

**What's the weakest part of this project?**
The dataset size and the fact that I wrote both the bugs and the heuristic. The labels are
also necessarily incomplete: a reviewer that finds a real, unlabelled bug is penalised. The
remedy is process: read the false positives, promote real ones into labels, and version the
dataset.

**How would this scale to a real product like Copilot code review?**
The same structure with real data: sample merged PRs where a later fix commit identifies the
bug (or where a human reviewer's comment was accepted), label from that, and stratify by
language and category. Add repetitions, track per-category recall over time, and gate
releases on both quality and cost/latency budgets. Online signals (comment resolution and
dismissal rates) complement the offline set; they don't replace it.
