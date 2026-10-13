import pytest

from reviewbench.diff import DiffParseError, normalize_path, parse_diff, render_numbered

SAMPLE = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,4 +1,5 @@
 import os
-x = 1
+x = 2
+y = 3

 def f():
@@ -10,2 +11,3 @@ def g():
     a = 1
+    b = 2
     return a
diff --git a/web/new.ts b/web/new.ts
new file mode 100644
--- /dev/null
+++ b/web/new.ts
@@ -0,0 +1,2 @@
+export const a = 1;
+export const b = 2;
\\ No newline at end of file
"""


def test_parses_files_and_flags():
    files = parse_diff(SAMPLE)
    assert [f.path for f in files] == ["app.py", "web/new.ts"]
    assert not files[0].is_new
    assert files[1].is_new


def test_new_file_line_numbers():
    new_ts = parse_diff(SAMPLE)[1]
    assert new_ts.added_lines() == {1: "export const a = 1;", 2: "export const b = 2;"}


def test_modified_file_line_numbers_follow_hunk_headers():
    app = parse_diff(SAMPLE)[0]
    assert app.added_lines() == {2: "x = 2", 3: "y = 3", 12: "    b = 2"}
    visible = app.new_side_lines()
    assert visible[1] == "import os"
    assert visible[11] == "    a = 1"
    assert visible[13] == "    return a"
    assert 6 not in visible  # between hunks: not shown in the diff


def test_render_numbered_shows_new_line_numbers_only():
    text = render_numbered(SAMPLE)
    assert "FILE app.py (modified)" in text
    assert "    2 + x = 2" in text
    assert "      - x = 1" in text  # removed line: no number
    assert "  ..." in text  # gap between hunks is marked
    assert "FILE web/new.ts (new file)" in text


def test_crlf_input_parses_identically():
    assert parse_diff(SAMPLE.replace("\n", "\r\n"))[0].added_lines() == parse_diff(SAMPLE)[0].added_lines()


def test_malformed_hunk_header_raises():
    with pytest.raises(DiffParseError):
        parse_diff("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ nonsense @@\n+a\n")


def test_truncated_hunk_raises():
    with pytest.raises(DiffParseError):
        parse_diff("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1,3 @@\n a\n+b\n")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("b/src/x.py", "src/x.py"), ("./src/x.py", "src/x.py"), ("src\\x.py", "src/x.py"), (" x.py ", "x.py")],
)
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected
