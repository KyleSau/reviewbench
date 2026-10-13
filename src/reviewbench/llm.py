"""One structured-output call to Claude, with the handling every caller needs.

Both the Claude reviewer and the Claude judge go through ``call_structured`` so the
API rules live in exactly one place:

* ``thinking={"type": "adaptive"}``: Claude Opus 5.5 always thinks; ``budget_tokens``
  and ``{"type": "disabled"}`` return 400. ``output_config.effort`` is the dial, and it
  is always sent explicitly (the API default on this model is ``medium``).
* Structured outputs via ``output_config.format``: no forced ``tool_choice`` (400 on
  this model) and no assistant prefill (also 400).
* Server-side refusal fallback (beta ``server-side-fallback-2026-07-01`` with
  ``fallbacks="default"``): if a safety classifier declines, the API re-runs the
  request on Anthropic's recommended fallback model in the same call. We record when
  that happened, because a fallback-served answer measures a different model.
* ``stop_reason`` is checked before content is read: ``refusal`` and ``max_tokens``
  are reported as their own statuses, never parsed as an (empty) answer.
* API errors are caught most-specific-first. Errors that will fail every case the
  same way (bad key, unknown model, malformed request) abort the run; transient ones
  are recorded on the case and the run continues.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from reviewbench.models import CaseStatus, Usage
from reviewbench.pricing import estimate_cost

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 16_000  # thinking counts toward max_tokens, so leave headroom
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MISSING_KEY_HELP = """\
error: this command calls the Claude API, but no API key was found.

Set ANTHROPIC_API_KEY in your environment and run it again:

  PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  bash/zsh:    export ANTHROPIC_API_KEY="sk-ant-..."

You can create a key in the Claude Console. Everything else in reviewbench
(the heuristic reviewer, report, gate, tests) works offline without a key."""


class FatalLLMError(RuntimeError):
    """A configuration problem that would fail every case identically. Abort the run."""


class CallFailed(Exception):
    """A single call produced no usable answer. Recorded on the case; the run continues."""

    def __init__(
        self,
        status: CaseStatus,
        message: str,
        *,
        usage: Usage | None = None,
        latency_ms: float | None = None,
        served_model: str | None = None,
        fallback_used: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.usage = usage
        self.latency_ms = latency_ms
        self.served_model = served_model
        self.fallback_used = fallback_used


@dataclass(frozen=True)
class LLMSettings:
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    use_fallback: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "server_side_fallback": self.use_fallback,
        }


@dataclass
class LLMResult:
    data: dict[str, Any]
    usage: Usage
    latency_ms: float
    served_model: str
    fallback_used: bool


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def make_client() -> anthropic.Anthropic:
    # Credentials come from the environment the normal SDK way; never hardcoded.
    return anthropic.Anthropic()


def build_request(settings: LLMSettings, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": schema},
        },
        # The system prompt is identical for every case, so mark it cacheable.
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
    }
    if settings.use_fallback:
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


def _usage_entry(model: str, item: Any) -> Usage:
    input_tokens = getattr(item, "input_tokens", 0) or 0
    output_tokens = getattr(item, "output_tokens", 0) or 0
    cache_read = getattr(item, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(item, "cache_creation_input_tokens", 0) or 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        cost_usd=estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def usage_from_response(response: Any, requested_model: str) -> Usage:
    """Token usage and cost for a response, including any fallback attempts.

    When a fallback ran, top-level ``usage`` covers only the attempt that produced the
    returned message; ``usage.iterations`` has one entry per attempt (each priced at the
    model that ran it), so prefer it when present.
    """
    iterations = getattr(response.usage, "iterations", None) or []
    if iterations:
        total = Usage(cost_usd=0.0)
        for item in iterations:
            total = total + _usage_entry(getattr(item, "model", None) or requested_model, item)
        return total
    return _usage_entry(response.model or requested_model, response.usage)


def fallback_was_used(response: Any) -> bool:
    iterations = getattr(response.usage, "iterations", None) or []
    return any(getattr(item, "type", None) == "fallback_message" for item in iterations)


def call_structured(
    client: Any, settings: LLMSettings, *, system: str, user: str, schema: dict[str, Any]
) -> LLMResult:
    """Make one structured-output request and return the parsed JSON object.

    Raises FatalLLMError (abort the run) or CallFailed (record and continue).
    """
    request = build_request(settings, system=system, user=user, schema=schema)
    started = time.perf_counter()
    try:
        response = client.beta.messages.create(**request)
    except anthropic.AuthenticationError as exc:
        raise FatalLLMError(f"authentication failed; check ANTHROPIC_API_KEY ({exc.message})") from exc
    except anthropic.PermissionDeniedError as exc:
        raise FatalLLMError(f"this API key cannot use {settings.model} ({exc.message})") from exc
    except anthropic.NotFoundError as exc:
        raise FatalLLMError(f"model {settings.model!r} was not found ({exc.message})") from exc
    except anthropic.BadRequestError as exc:
        raise FatalLLMError(f"the API rejected the request as invalid: {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise CallFailed("error", f"rate limited after SDK retries: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise CallFailed("error", f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APITimeoutError as exc:
        raise CallFailed("error", "request timed out") from exc
    except anthropic.APIConnectionError as exc:
        raise CallFailed("error", f"could not reach the API: {exc}") from exc
    except anthropic.APIError as exc:  # anything else the SDK raises, e.g. response validation
        raise CallFailed("error", f"API call failed: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000

    usage = usage_from_response(response, settings.model)
    served_model = response.model
    fallback_used = fallback_was_used(response)
    context = {
        "usage": usage,
        "latency_ms": latency_ms,
        "served_model": served_model,
        "fallback_used": fallback_used,
    }
    request_id = getattr(response, "_request_id", None)

    # Check why generation stopped *before* touching content.
    if response.stop_reason == "refusal":
        details = response.stop_details
        category = details.category if details else None
        raise CallFailed(
            "refused", f"model declined (category={category}, request_id={request_id})", **context
        )
    if response.stop_reason == "max_tokens":
        raise CallFailed(
            "truncated",
            f"hit max_tokens={settings.max_tokens} before finishing; raise --max-tokens",
            **context,
        )

    # Read blocks by type: thinking (and, after a fallback, a `fallback` block) can come first.
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise CallFailed(
            "error", f"no text block in response (stop_reason={response.stop_reason})", **context
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CallFailed("error", f"response was not valid JSON: {exc}", **context) from exc
    if not isinstance(data, dict):
        raise CallFailed("error", "response JSON was not an object", **context)
    return LLMResult(data=data, **context)
