"""Structured-output probes against a dockerized gateway.

Direct descendant of the hand-run ``probe_structured_output.py``:

  1. /v1/chat/completions  — response_format json_schema  (must be enforced)
  2. /v1/responses         — text.format json_schema       (must be enforced)
  3. /v1/runs              — text.format json_schema       (must be enforced)
  4. /v1/chat/completions  — response_format json_object   (backend-dependent)

json_schema enforcement is a hard requirement on every backend. json_object
has no native Anthropic mapping, so its expected outcome is read from the
provider spec: "reject" → 400 up-front, "accept" → 200, "any" → informational.
"""

from __future__ import annotations

import json
import time

import pytest

from .constants import LOCATION_SCHEMA, MODEL, STEER, WORD_SCHEMA
from .http_client import chat_content, responses_content

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]


def _assert_conforms(content: str | None, status: int, body_text: str, required: list[str]):
    assert status == 200, f"HTTP {status}: {body_text[:300]}"
    assert content, f"empty content (constraint produced nothing): {body_text[:300]}"
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        pytest.fail(f"content was not JSON (constraint ignored?): {content[:300]!r}")
    missing = [k for k in required if k not in obj]
    assert not missing, f"valid JSON but missing {missing}: {json.dumps(obj)}"


def test_chat_json_schema(gateway):
    resp = gateway.post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": STEER},
                {
                    "role": "user",
                    "content": "Extract the city and country from: "
                    "'I climbed the Eiffel Tower last spring.'",
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Location", "schema": LOCATION_SCHEMA, "strict": True},
            },
        },
    )
    _assert_conforms(chat_content(resp.json()), resp.status, resp.text, ["city", "country"])


def test_responses_json_schema(gateway):
    """/v1/responses text.format json_schema — regression for issue #33864."""
    resp = gateway.post(
        "/v1/responses",
        {
            "model": MODEL,
            "stream": False,
            "instructions": STEER,
            "input": "Reply with the word PONG.",
            "text": {
                "format": {"type": "json_schema", "name": "Probe", "schema": WORD_SCHEMA, "strict": True}
            },
        },
    )
    _assert_conforms(responses_content(resp.json()), resp.status, resp.text, ["word"])


def test_runs_json_schema(gateway):
    """Runs must carry the constraint through tool loops and status polling."""
    resp = gateway.post(
        "/v1/runs",
        {
            "model": MODEL,
            "instructions": STEER,
            "input": "Reply with the word PONG.",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Probe",
                    "schema": WORD_SCHEMA,
                    "strict": True,
                }
            },
        },
    )
    assert resp.status == 202, f"HTTP {resp.status}: {resp.text[:300]}"
    run_id = resp.json()["run_id"]
    deadline = time.monotonic() + gateway.timeout
    status = None
    while time.monotonic() < deadline:
        polled = gateway.get(f"/v1/runs/{run_id}")
        assert polled.status == 200, f"HTTP {polled.status}: {polled.text[:300]}"
        status = polled.json()
        if status.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.25)

    assert status is not None and status.get("status") == "completed", status
    _assert_conforms(status.get("output"), 200, json.dumps(status), ["word"])


def test_chat_json_object(gateway):
    """response_format json_object — outcome depends on the backend wire protocol."""
    mode = gateway.provider.spec.json_object
    resp = gateway.post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": STEER},
                {"role": "user", "content": 'Give the capital of France as {"capital": ...}.'},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    if mode == "reject":
        assert resp.status == 400 and "json_object" in resp.text, (
            "expected up-front 400 (no native json_object mapping), "
            f"got HTTP {resp.status}: {resp.text[:300]}"
        )
    elif mode == "accept":
        assert resp.status == 200, f"HTTP {resp.status}: {resp.text[:300]}"
        content = chat_content(resp.json())
        assert content, f"empty content: {resp.text[:300]}"
        json.loads(content)  # must be valid JSON
    else:  # "any" — unverified backend: record, don't fail
        if resp.status == 200:
            content = chat_content(resp.json()) or ""
            print(f"[json_object/{gateway.provider.id}] accepted: {content[:120]!r}")
        elif resp.status == 400:
            print(f"[json_object/{gateway.provider.id}] rejected up-front (HTTP 400)")
        else:
            print(f"[json_object/{gateway.provider.id}] HTTP {resp.status}: {resp.text[:200]}")
