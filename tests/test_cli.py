import json

import pytest

from conftest import BASELINE_PATH, DATASET_DIR, REPO_ROOT
from reviewbench.cli import main


@pytest.fixture
def in_repo(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # default paths (datasets/v1, rubrics/) are repo-relative


def test_run_report_gate_round_trip(tmp_path, capsys, in_repo):
    out = tmp_path / "heuristic.json"
    assert main(["run", "--reviewer", "heuristic", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["reviewer"]["name"] == "heuristic"
    assert len(data["cases"]) == 25

    assert main(["report", str(out)]) == 0
    report = capsys.readouterr().out
    assert "| Precision (95% CI) |" in report
    assert "## By category" in report

    assert main(["report", str(BASELINE_PATH), str(out), "--out", str(tmp_path / "r.md")]) == 0
    assert "Change" in (tmp_path / "r.md").read_text()

    args = ["gate", "--baseline", str(BASELINE_PATH), "--candidate", str(out)]
    assert main(args) == 0
    assert "gate PASSED" in capsys.readouterr().out


def test_gate_fails_on_regression(tmp_path, capsys, in_repo):
    null_run = tmp_path / "null.json"
    assert main(["run", "--reviewer", "null", "--out", str(null_run)]) == 0
    assert main(["gate", "--baseline", str(BASELINE_PATH), "--candidate", str(null_run)]) == 1
    assert "gate FAILED" in capsys.readouterr().out


def test_claude_without_key_explains_how_to_set_one(capsys, in_repo):
    assert main(["run", "--reviewer", "claude"]) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
    assert "$env:ANTHROPIC_API_KEY" in err  # PowerShell instructions too


def test_judge_without_key_also_needs_one(capsys, in_repo):
    assert main(["run", "--reviewer", "heuristic", "--judge", "claude"]) == 2


def test_bad_dataset_path_is_a_usage_error(tmp_path, capsys):
    assert main(["run", "--reviewer", "heuristic", "--dataset", str(tmp_path)]) == 2
    assert "invalid dataset" in capsys.readouterr().err


def test_case_filter(tmp_path, in_repo):
    out = tmp_path / "one.json"
    args = ["run", "--reviewer", "oracle", "--dataset", str(DATASET_DIR), "--case", "py-orders-pagination"]
    assert main([*args, "--out", str(out)]) == 0
    assert [c["case_id"] for c in json.loads(out.read_text())["cases"]] == ["py-orders-pagination"]


def test_report_rejects_three_runs(capsys):
    assert main(["report", "a", "b", "c"]) == 2
