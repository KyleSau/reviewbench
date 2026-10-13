import pytest

from conftest import BASELINE_PATH
from helpers import pr_for
from reviewbench.reviewers import HeuristicReviewer, NullReviewer, OracleReviewer
from reviewbench.runner import load_run, run_benchmark


def categories(path, lines):
    return [f.category.value for f in HeuristicReviewer().review(pr_for(path, lines)).findings]


@pytest.mark.parametrize(
    ("path", "lines", "expected"),
    [
        ("a.py", ["sql = f\"SELECT * FROM t WHERE name = '{name}'\""], ["sql_injection"]),
        ("a.ts", ["const q = `SELECT * FROM t ORDER BY ${col}`;"], ["sql_injection"]),
        ("a.cs", ['var sql = "SELECT * FROM t WHERE id = " + id;'], ["sql_injection"]),
        ("a.py", ["data = pickle.loads(blob)"], ["insecure_deserialization"]),
        ("a.cs", ["TypeNameHandling = TypeNameHandling.Auto,"], ["insecure_deserialization"]),
        ("a.py", ["f = open(path)"], ["resource_leak"]),
        ("a.cs", ["var conn = new SqlConnection(cs);"], ["resource_leak"]),
        ("a.cs", ["private async void OnTick(object s)"], ["error_handling"]),
        ("a.cs", ["    throw ex;"], ["error_handling"]),
        ("a.ts", ["for (let i = 0; i <= items.length; i++) {"], ["off_by_one"]),
        ("a.ts", ["const city = user.address!.city;"], ["null_handling"]),
        ("a.cs", ["var name = users.FirstOrDefault(u => u.Id == id).Name;"], ["null_handling"]),
    ],
)
def test_line_rules_fire(path, lines, expected):
    assert categories(path, lines) == expected


@pytest.mark.parametrize(
    ("path", "lines"),
    [
        ("a.py", ['cur.execute("SELECT * FROM t WHERE name = %s", (name,))']),
        ("a.py", ["data = yaml.load(text, Loader=yaml.SafeLoader)"]),
        ("a.py", ["with open(path) as f:"]),
        ("a.cs", ["using var conn = new SqlConnection(cs);"]),
        ("a.py", ["conn = new SqlConnection(cs)"]),  # C#-only rule, wrong language
        ("a.ts", ["  throw err;"]),  # rethrowing is fine in JS
        ("a.ts", ["for (let i = 0; i <= items.length - 1; i++) {"]),
        ("a.ts", ["if (a !== b) { x = !ready; }"]),
    ],
)
def test_line_rules_stay_quiet(path, lines):
    assert categories(path, lines) == []


def test_swallowed_exception_python():
    assert categories("a.py", ["try:", "    go()", "except Exception:", "    pass"]) == [
        "swallowed_exception"
    ]
    assert categories("a.py", ["try:", "    go()", "except Exception:", "    log.exception('x')"]) == []


def test_swallowed_exception_brace_languages():
    assert categories("a.ts", ["try {", "  go();", "} catch {", "  // ignore", "}"]) == [
        "swallowed_exception"
    ]
    assert categories("a.ts", ["try {", "  go();", "} catch (err) {", "  next(err);", "}"]) == []
    cs = ["try", "{", "    Go();", "}", "catch (Exception)", "{", "}"]
    assert categories("a.cs", cs) == ["swallowed_exception"]


def test_listener_without_cleanup():
    assert categories("a.ts", ['window.addEventListener("resize", f);']) == ["resource_leak"]
    paired = [
        'window.addEventListener("resize", f);',
        'return () => window.removeEventListener("resize", f);',
    ]
    assert categories("a.ts", paired) == []


def test_query_in_loop_only_inside_the_loop_body():
    loop = ["for order in orders:", "    items = Item.objects.filter(order=order)", "done = True"]
    assert categories("a.py", loop) == ["n_plus_one_query"]
    before = ["items = Item.objects.filter(order__in=orders)", "for item in items:", "    print(item)"]
    assert categories("a.py", before) == []


def test_adjacent_hits_merge_into_one_finding():
    lines = ['a = f"SELECT x FROM t WHERE a = {a}"', 'b = f"SELECT y FROM t WHERE b = {b}"']
    findings = HeuristicReviewer().review(pr_for("a.py", lines)).findings
    assert len(findings) == 1
    assert (findings[0].line_start, findings[0].line_end) == (1, 2)


def test_only_added_lines_are_reviewed():
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n"
        " data = pickle.loads(old)\n+x = 1\n"
    )
    pr = pr_for("a.py", ["x"]).model_copy(update={"diff": diff})
    assert HeuristicReviewer().review(pr).findings == []


def _overall(run):
    o = run.metrics.overall
    return o.tp, o.fp, o.fn


def test_heuristic_end_to_end_is_deterministic(dataset):
    first = run_benchmark(dataset, HeuristicReviewer(), name="a", workers=4)
    second = run_benchmark(dataset, HeuristicReviewer(), name="b", workers=1)
    assert [c.findings for c in first.cases] == [c.findings for c in second.cases]
    assert first.metrics.overall == second.metrics.overall


def test_committed_baseline_is_reproducible(dataset):
    """If this fails, the heuristic or dataset changed: regenerate baselines/heuristic.json
    with `reviewbench run --reviewer heuristic --out baselines/heuristic.json` and review
    the diff in the same PR."""
    baseline = load_run(BASELINE_PATH)
    fresh = run_benchmark(dataset, HeuristicReviewer(), name="fresh")
    assert fresh.dataset.fingerprint == baseline.dataset.fingerprint
    assert fresh.metrics.overall == baseline.metrics.overall
    assert fresh.metrics.per_category == baseline.metrics.per_category


def test_oracle_scores_perfectly(dataset):
    """Harness self-check: ground truth fed back as findings must score 100%."""
    m = run_benchmark(dataset, OracleReviewer(dataset), name="oracle").metrics
    assert (m.overall.precision, m.overall.recall, m.clean_case_fp_rate) == (1.0, 1.0, 0.0)


def test_null_reviewer_scores_zero_recall(dataset):
    """Harness self-check: saying nothing must not look like a good review."""
    m = run_benchmark(dataset, NullReviewer(), name="null").metrics
    assert m.overall.recall == 0.0
    assert m.overall.precision is None
    assert m.clean_case_fp_rate == 0.0
