import json
import shutil
from collections import Counter

import pytest
from pydantic import ValidationError

from conftest import DATASET_DIR
from helpers import issue, new_file_diff
from reviewbench.dataset import DatasetError, load_dataset, validate_case
from reviewbench.models import Case, Category

GROUND_TRUTH_CATEGORIES = {
    Category.OFF_BY_ONE, Category.NULL_HANDLING, Category.SQL_INJECTION,
    Category.MISSING_AUTHORIZATION, Category.RESOURCE_LEAK, Category.RACE_CONDITION,
    Category.SWALLOWED_EXCEPTION, Category.ERROR_HANDLING, Category.N_PLUS_ONE_QUERY,
    Category.INSECURE_DESERIALIZATION,
}  # fmt: skip


def test_v1_shape(dataset):
    assert len(dataset.cases) == 25
    assert sum(c.is_clean for c in dataset.cases) == 6
    assert dataset.issue_count == 21
    assert Counter(c.language for c in dataset.cases) == {"python": 9, "typescript": 8, "csharp": 8}


def test_v1_covers_every_category_at_least_twice(dataset):
    counts = Counter(i.category for c in dataset.cases for i in c.expected_issues)
    assert set(counts) == GROUND_TRUTH_CATEGORIES
    assert min(counts.values()) >= 2


def test_reviewer_view_has_no_labels(dataset):
    pr = dataset.cases[0].to_pull_request()
    assert not hasattr(pr, "expected_issues")
    assert "expected_issues" not in pr.model_dump()


def _case(diff: str, **issue_kwargs) -> Case:
    return Case(
        id="x", language="python", title="t", description="d",
        expected_issues=[issue(**issue_kwargs)], diff=diff,
    )  # fmt: skip


def test_label_must_be_on_an_added_line():
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n context\n+added\n context2\n"
    validate_case(_case(diff, file="a.py", start=2))  # the added line: fine
    with pytest.raises(DatasetError, match="no added line"):
        validate_case(_case(diff, file="a.py", start=1))  # unchanged context line


def test_label_must_reference_a_file_in_the_diff():
    with pytest.raises(DatasetError, match="not in the diff"):
        validate_case(_case(new_file_diff("a.py", ["x"]), file="b.py", start=1))


def test_label_must_be_visible_in_the_diff():
    with pytest.raises(DatasetError, match="not visible"):
        validate_case(_case(new_file_diff("a.py", ["x", "y"]), file="a.py", start=2, end=5))


def test_schema_rejects_unknown_category_and_extra_fields():
    base = {
        "id": "x", "language": "python", "title": "t", "description": "d", "diff": "",
        "expected_issues": [{
            "file": "a.py", "line_start": 1, "line_end": 1, "category": "typo",
            "severity": "high", "description": "long enough description here",
        }],
    }  # fmt: skip
    with pytest.raises(ValidationError):
        Case.model_validate(base)
    base["expected_issues"][0]["category"] = "sql_injection"
    base["surprise"] = True
    with pytest.raises(ValidationError):
        Case.model_validate(base)


def test_schema_rejects_reversed_line_range():
    with pytest.raises(ValidationError):
        issue(start=10, end=5)


def _copy_dataset(tmp_path):
    target = tmp_path / "v1"
    shutil.copytree(DATASET_DIR, target)
    return target


def test_fingerprint_is_stable_and_ignores_line_endings(tmp_path, dataset):
    copy = _copy_dataset(tmp_path)
    for diff in (copy / "cases").glob("*.diff"):
        diff.write_bytes(diff.read_bytes().replace(b"\n", b"\r\n"))
    assert load_dataset(copy).fingerprint == dataset.fingerprint


def test_fingerprint_changes_when_a_label_changes(tmp_path, dataset):
    copy = _copy_dataset(tmp_path)
    meta_path = copy / "cases" / "py-orders-pagination.json"
    meta = json.loads(meta_path.read_text())
    meta["expected_issues"][0]["severity"] = "high"
    meta_path.write_text(json.dumps(meta))
    assert load_dataset(copy).fingerprint != dataset.fingerprint


def test_missing_diff_file_is_an_error(tmp_path):
    copy = _copy_dataset(tmp_path)
    (copy / "cases" / "py-orders-pagination.diff").unlink()
    with pytest.raises(DatasetError, match="missing"):
        load_dataset(copy)


def test_select_unknown_case_is_an_error(dataset):
    assert len(dataset.select(["py-orders-pagination"]).cases) == 1
    with pytest.raises(DatasetError):
        dataset.select(["nope"])
