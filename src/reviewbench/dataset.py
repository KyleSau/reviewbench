"""Load and validate a dataset directory.

Layout: ``<dataset>/cases/<case-id>.json`` (metadata + ground truth) next to
``<case-id>.diff`` (the pull request as a unified diff).

Validation goes beyond the pydantic schema: every label is cross-checked against its
diff. A label must point at a file in the diff and at lines the PR actually adds,
otherwise a typo in a line number would silently turn a correct finding into a
"false positive".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from reviewbench.diff import DiffParseError, normalize_path, parse_diff
from reviewbench.models import Case


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    cases: list[Case]

    @property
    def issue_count(self) -> int:
        return sum(len(c.expected_issues) for c in self.cases)

    @property
    def fingerprint(self) -> str:
        """Content hash of every case. Two runs are only comparable if this matches.

        Hashes a canonical JSON form (sorted keys, LF line endings), so re-formatting a
        file or a Windows checkout with CRLF does not change it, but editing a label does.
        """
        digest = hashlib.sha256()
        for case in sorted(self.cases, key=lambda c: c.id):
            canonical = json.dumps(case.model_dump(mode="json"), sort_keys=True)
            digest.update(canonical.encode("utf-8"))
        return digest.hexdigest()[:16]

    def select(self, case_ids: list[str]) -> Dataset:
        known = {c.id for c in self.cases}
        unknown = [cid for cid in case_ids if cid not in known]
        if unknown:
            raise DatasetError(f"unknown case id(s): {', '.join(unknown)}")
        wanted = set(case_ids)
        return Dataset(self.name, self.path, [c for c in self.cases if c.id in wanted])


def validate_case(case: Case) -> None:
    """Cross-check a case's labels against its diff. Raises DatasetError."""
    try:
        files = {f.path: f for f in parse_diff(case.diff)}
    except DiffParseError as exc:
        raise DatasetError(f"{case.id}: cannot parse diff: {exc}") from exc
    if not files:
        raise DatasetError(f"{case.id}: diff contains no files")
    for n, issue in enumerate(case.expected_issues):
        where = f"{case.id}: expected_issues[{n}]"
        file_diff = files.get(normalize_path(issue.file))
        if file_diff is None:
            raise DatasetError(f"{where}: file {issue.file!r} is not in the diff")
        visible = file_diff.new_side_lines()
        span = range(issue.line_start, issue.line_end + 1)
        missing = [ln for ln in span if ln not in visible]
        if missing:
            raise DatasetError(f"{where}: lines {missing} are not visible in the diff")
        if not any(ln in file_diff.added_lines() for ln in span):
            raise DatasetError(f"{where}: range touches no added line; labels must be on PR changes")


def load_dataset(path: str | Path) -> Dataset:
    root = Path(path)
    case_dir = root / "cases"
    if not case_dir.is_dir():
        raise DatasetError(f"no cases/ directory under {root}")
    cases: list[Case] = []
    for meta_path in sorted(case_dir.glob("*.json")):
        diff_path = meta_path.with_suffix(".diff")
        if not diff_path.exists():
            raise DatasetError(f"{meta_path.name}: missing {diff_path.name}")
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["diff"] = diff_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        try:
            case = Case.model_validate(raw)
        except ValidationError as exc:
            raise DatasetError(f"{meta_path.name}: {exc}") from exc
        if case.id != meta_path.stem:
            raise DatasetError(f"{meta_path.name}: id {case.id!r} does not match file name")
        validate_case(case)
        cases.append(case)
    orphans = {p.stem for p in case_dir.glob("*.diff")} - {c.id for c in cases}
    if orphans:
        raise DatasetError(f"diff files without metadata: {sorted(orphans)}")
    if not cases:
        raise DatasetError(f"no cases found in {case_dir}")
    return Dataset(name=root.name, path=root.as_posix(), cases=cases)
