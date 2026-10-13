from pathlib import Path

import pytest

from reviewbench.dataset import Dataset, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets" / "v1"
BASELINE_PATH = REPO_ROOT / "baselines" / "heuristic.json"


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return load_dataset(DATASET_DIR)


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never reach the real API, even on a machine that has a key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
