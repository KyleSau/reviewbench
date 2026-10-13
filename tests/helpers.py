"""Small builders shared by the tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from anthropic.types.beta import BetaMessage

from reviewbench.models import Case, ExpectedIssue, Finding, PullRequest


def new_file_diff(path: str, lines: list[str]) -> str:
    body = "".join(f"+{line}\n" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n{body}"
    )


def pr_for(path: str, lines: list[str]) -> PullRequest:
    return PullRequest(id="t", title="t", description="t", diff=new_file_diff(path, lines))


def issue(file="a.py", start=10, end=None, category="sql_injection", severity="high") -> ExpectedIssue:
    return ExpectedIssue(
        file=file,
        line_start=start,
        line_end=end or start,
        category=category,
        severity=severity,
        description="a description that is long enough",
    )


def finding(file="a.py", start=10, end=None, category="sql_injection", severity="high") -> Finding:
    return Finding(
        file=file,
        line_start=start,
        line_end=end or start,
        category=category,
        severity=severity,
        explanation="explanation",
    )


def case(case_id: str, issues: list[ExpectedIssue], path: str = "a.py", n_lines: int = 30) -> Case:
    return Case(
        id=case_id,
        language="python",
        title="t",
        description="d",
        expected_issues=issues,
        diff=new_file_diff(path, [f"line_{i}" for i in range(1, n_lines + 1)]),
    )


def message(
    *,
    text: str | None = None,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    extra: dict[str, Any] | None = None,
    usage_extra: dict[str, Any] | None = None,
) -> BetaMessage:
    """A real SDK response object, so the code under test sees genuine types."""
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    data: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, **(usage_extra or {})},
        **(extra or {}),
    }
    return BetaMessage.model_validate(data)


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    """Stands in for anthropic.Anthropic(); only client.beta.messages.create is used."""

    def __init__(self, *responses: Any) -> None:
        self.messages = FakeMessages(list(responses))
        self.beta = SimpleNamespace(messages=self.messages)
