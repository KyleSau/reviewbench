"""A small unified-diff parser.

Ground truth, heuristic findings and LLM findings all speak in *new-file line numbers*,
so the one thing this module must get exactly right is mapping each diff line to its
line number in the new file. It also renders a diff with those numbers printed in a
left-hand column, because asking a model to count hunk offsets itself is a reliable way
to get findings that are off by a few lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class DiffParseError(ValueError):
    pass


@dataclass(frozen=True)
class DiffLine:
    kind: Literal["+", "-", " "]
    text: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass
class FileDiff:
    path: str
    is_new: bool = False
    is_deleted: bool = False
    lines: list[DiffLine] = field(default_factory=list)

    def added_lines(self) -> dict[int, str]:
        """New-file line number -> text, for lines this diff adds."""
        return {ln.new_lineno: ln.text for ln in self.lines if ln.kind == "+" and ln.new_lineno}

    def new_side_lines(self) -> dict[int, str]:
        """New-file line number -> text, for every new-file line visible in the diff
        (added lines plus unchanged context)."""
        return {ln.new_lineno: ln.text for ln in self.lines if ln.new_lineno is not None}


def _strip_prefix(path: str) -> str:
    return path[2:] if path.startswith(("a/", "b/")) else path


def normalize_path(path: str) -> str:
    """Canonical form for comparing paths from labels, diffs and model output."""
    path = path.strip().replace("\\", "/")
    path = _strip_prefix(path)
    while path.startswith("./"):
        path = path[2:]
    return path


def parse_diff(text: str) -> list[FileDiff]:
    """Parse a git-style unified diff into per-file line lists."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the final newline is a terminator, not an empty context line
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            parts = line.split()
            current = FileDiff(path=normalize_path(parts[-1]))
            files.append(current)
        elif line.startswith("new file mode") and current:
            current.is_new = True
        elif line.startswith("deleted file mode") and current:
            current.is_deleted = True
        elif line.startswith("--- "):
            if current is None:  # plain unified diff without a "diff --git" header
                current = FileDiff(path="")
                files.append(current)
            if line[4:].strip() == "/dev/null":
                current.is_new = True
        elif line.startswith("+++ ") and current is not None:
            target = line[4:].strip()
            if target == "/dev/null":
                current.is_deleted = True
            else:
                current.path = normalize_path(target)
        elif line.startswith("@@"):
            if current is None:
                raise DiffParseError(f"hunk header before any file header: {line!r}")
            i = _parse_hunk(lines, i, current)
            continue
        i += 1
    for f in files:
        if not f.path:
            raise DiffParseError("file section without a path")
    return files


def _parse_hunk(lines: list[str], i: int, current: FileDiff) -> int:
    match = _HUNK_RE.match(lines[i])
    if not match:
        raise DiffParseError(f"malformed hunk header: {lines[i]!r}")
    old_no, old_count = int(match.group(1)), int(match.group(2) or 1)
    new_no, new_count = int(match.group(3)), int(match.group(4) or 1)
    i += 1
    # Use the counts from the header to know where the hunk ends; this is what makes
    # a line that happens to start with "--- " inside a hunk unambiguous.
    while old_count > 0 or new_count > 0:
        if i >= len(lines):
            raise DiffParseError("diff ended in the middle of a hunk")
        raw = lines[i]
        if raw.startswith("\\"):  # "\ No newline at end of file"
            i += 1
            continue
        kind, body = (raw[0], raw[1:]) if raw else (" ", "")
        if kind == "+":
            current.lines.append(DiffLine("+", body, None, new_no))
            new_no += 1
            new_count -= 1
        elif kind == "-":
            current.lines.append(DiffLine("-", body, old_no, None))
            old_no += 1
            old_count -= 1
        elif kind == " ":
            current.lines.append(DiffLine(" ", body, old_no, new_no))
            old_no += 1
            new_no += 1
            old_count -= 1
            new_count -= 1
        else:
            raise DiffParseError(f"unexpected line inside hunk: {raw!r}")
        i += 1
    while i < len(lines) and lines[i].startswith("\\"):
        i += 1
    return i


def render_numbered(text: str) -> str:
    """Render a diff for a model, with new-file line numbers in a left column.

    Removed lines get no number: reviewers can only comment on the new version.
    """
    out: list[str] = []
    for f in parse_diff(text):
        status = "new file" if f.is_new else "deleted" if f.is_deleted else "modified"
        out.append(f"FILE {f.path} ({status})")
        previous_new: int | None = None
        for ln in f.lines:
            if ln.new_lineno is not None:
                if previous_new is not None and ln.new_lineno != previous_new + 1:
                    out.append("  ...")
                previous_new = ln.new_lineno
            number = "" if ln.new_lineno is None else str(ln.new_lineno)
            out.append(f"{number:>5} {ln.kind} {ln.text}".rstrip())
        out.append("")
    return "\n".join(out)
