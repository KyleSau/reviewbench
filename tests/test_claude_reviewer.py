"""The Claude reviewer, offline: a fake client object, and the real SDK over a mock transport."""

import json

import anthropic
import httpx2
import pytest

from helpers import FakeClient, message
from reviewbench.llm import FALLBACK_BETA, FatalLLMError, LLMSettings
from reviewbench.models import PullRequest
from reviewbench.reviewers.claude import FINDINGS_SCHEMA, ClaudeReviewer
from reviewbench.runner import run_benchmark

PR = PullRequest(
    id="pr",
    title="Add search",
    description="desc",
    diff="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n import os\n+q = 1\n",
)
ONE_FINDING = json.dumps(
    {
        "findings": [
            {
                "file": "a.py",
                "line_start": 2,
                "line_end": 2,
                "explanation": "user input reaches SQL",
                "category": "sql_injection",
                "severity": "critical",
            }
        ]
    }
)
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(cls, status):
    return cls("boom", response=httpx2.Response(status, request=_REQUEST), body=None)


def review(*responses, settings=None):
    client = FakeClient(*responses)
    output = ClaudeReviewer(settings or LLMSettings(), client=client).review(PR)
    return output, client.messages.calls


def test_request_follows_the_api_rules():
    _, calls = review(message(text=ONE_FINDING))
    request = calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "medium"  # always explicit
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": FINDINGS_SCHEMA}
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert "tool_choice" not in request  # forced tool use is a 400 on this model
    assert "budget_tokens" not in json.dumps(request)
    assert request["messages"][-1]["role"] == "user"  # no assistant prefill
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "    2 + q = 1" in request["messages"][0]["content"]  # numbered diff


def test_settings_are_passed_through_and_fallback_can_be_disabled():
    settings = LLMSettings(model="claude-sonnet-5-5", effort="high", max_tokens=8000, use_fallback=False)
    _, calls = review(message(text=ONE_FINDING), settings=settings)
    request = calls[0]
    assert (request["model"], request["output_config"]["effort"], request["max_tokens"]) == (
        "claude-sonnet-5-5",
        "high",
        8000,
    )
    assert "betas" not in request and "fallbacks" not in request


def test_findings_parsed_and_cost_estimated():
    output, _ = review(message(text=ONE_FINDING, input_tokens=1000, output_tokens=500))
    assert output.status == "ok"
    assert [(f.file, f.line_start, f.category.value) for f in output.findings] == [
        ("a.py", 2, "sql_injection")
    ]
    assert output.usage.input_tokens == 1000
    assert output.usage.cost_usd == pytest.approx(1000 * 4 / 1e6 + 500 * 20 / 1e6)
    assert output.latency_ms is not None
    assert output.served_model == "claude-opus-5-5"
    assert not output.fallback_used


def test_empty_findings_is_a_valid_answer():
    output, _ = review(message(text='{"findings": []}'))
    assert output.status == "ok"
    assert output.findings == []


def test_refusal_is_recorded_not_parsed():
    refusal = message(
        text=None,
        stop_reason="refusal",
        extra={"stop_details": {"type": "refusal", "category": "cyber", "explanation": None}},
    )
    output, _ = review(refusal)
    assert output.status == "refused"
    assert "cyber" in output.error
    assert output.usage is not None  # the attempt still cost tokens


def test_max_tokens_is_truncated_not_empty():
    output, _ = review(message(text='{"findings": [', stop_reason="max_tokens"))
    assert output.status == "truncated"
    assert output.findings == []


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"findings": [{"file": "a.py"}]}',  # missing fields
        '{"findings": [{"file": "a.py", "line_start": 1, "line_end": 1, "explanation": "x", '
        '"category": "made_up", "severity": "high"}]}',
    ],
)
def test_malformed_output_is_an_error(text):
    output, _ = review(message(text=text))
    assert output.status == "error"


def test_fallback_is_detected_and_each_attempt_priced_at_its_model():
    iterations = [
        {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "output_tokens": 0,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000,
         "output_tokens": 400, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    ]  # fmt: skip
    response = message(text=ONE_FINDING, model="claude-opus-4-8", usage_extra={"iterations": iterations})
    output, _ = review(response)
    assert output.status == "ok"
    assert output.fallback_used
    assert output.served_model == "claude-opus-4-8"
    assert output.usage.input_tokens == 2000
    expected = 1000 * 4 / 1e6 + (1000 * 5 + 400 * 25) / 1e6
    assert output.usage.cost_usd == pytest.approx(expected)


@pytest.mark.parametrize(
    "error",
    [
        api_error(anthropic.RateLimitError, 429),
        api_error(anthropic.InternalServerError, 500),
        anthropic.APITimeoutError(request=_REQUEST),
        anthropic.APIConnectionError(request=_REQUEST),
    ],
)
def test_transient_errors_are_recorded_on_the_case(error):
    output, _ = review(error)
    assert output.status == "error"
    assert output.findings == []


@pytest.mark.parametrize(
    "error",
    [
        api_error(anthropic.AuthenticationError, 401),
        api_error(anthropic.PermissionDeniedError, 403),
        api_error(anthropic.NotFoundError, 404),
        api_error(anthropic.BadRequestError, 400),
    ],
)
def test_configuration_errors_abort(error):
    with pytest.raises(FatalLLMError):
        review(error)


def test_fatal_error_aborts_the_whole_run(dataset):
    client = FakeClient(api_error(anthropic.AuthenticationError, 401))
    reviewer = ClaudeReviewer(LLMSettings(), client=client)
    with pytest.raises(FatalLLMError):
        run_benchmark(dataset, reviewer, name="x", workers=2)


def test_run_with_mixed_outcomes_scores_only_ok_cases(dataset):
    small = dataset.select(["py-admin-user-search", "py-clean-env-config"])
    client = FakeClient(message(text='{"findings": []}'), api_error(anthropic.RateLimitError, 429))
    run = run_benchmark(small, ClaudeReviewer(LLMSettings(), client=client), name="x", workers=1)
    assert [c.status for c in run.cases] == ["ok", "error"]
    assert run.metrics.scored_cases == 1
    assert run.reviewer.model == "claude-opus-5-5"
    assert run.reviewer.settings["effort"] == "medium"


# --- The real SDK client over a mock HTTP transport --------------------------------
# The tests above fake the client object. These go one layer lower: the genuine
# anthropic.Anthropic client builds and sends the HTTP request, and a mock transport
# answers it, so header/body serialization and the SDK's own error mapping are covered.


def sdk_client(status: int, payload: dict, seen: list) -> anthropic.Anthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(status, json=payload)

    transport = httpx2.MockTransport(handler)
    return anthropic.Anthropic(
        api_key="sk-test-not-a-real-key",
        http_client=anthropic.DefaultHttpxClient(transport=transport),
        max_retries=0,
    )


def test_real_sdk_request_on_the_wire():
    seen: list[httpx2.Request] = []
    payload = message(text=ONE_FINDING).model_dump(mode="json", by_alias=True, exclude_none=True)
    client = sdk_client(200, payload, seen)
    output = ClaudeReviewer(LLMSettings(), client=client).review(PR)

    assert output.status == "ok"
    assert len(output.findings) == 1
    request = seen[0]
    assert request.url.path == "/v1/messages"
    assert request.headers["anthropic-beta"] == FALLBACK_BETA
    assert request.headers["x-api-key"] == "sk-test-not-a-real-key"
    body = json.loads(request.content)
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in body


def test_real_sdk_error_mapping():
    error_body = {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
    with pytest.raises(FatalLLMError, match="authentication failed"):
        ClaudeReviewer(LLMSettings(), client=sdk_client(401, error_body, [])).review(PR)

    error_body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    output = ClaudeReviewer(LLMSettings(), client=sdk_client(429, error_body, [])).review(PR)
    assert output.status == "error"
    assert "rate limited" in output.error
