"""Request diagnostics report wire settings without exposing request contents."""

import logging
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import log_openrouter_request_reasoning


@pytest.mark.parametrize("body,expected", [
    ({}, "preset"),
    ({"reasoning": {"enabled": True, "effort": "max"}}, "explicit:max"),
    ({"reasoning": {"enabled": True, "effort": "medium"}}, "explicit:medium"),
    ({"reasoning": {"enabled": False}}, "explicit:disabled"),
    ({"reasoning": {"effort": "none"}}, "explicit:disabled"),
    ({"reasoning": {"max_tokens": 4096}}, "explicit:configured"),
    ({"reasoning": None}, "explicit:configured"),
    ({"reasoning": {"effort": "private-value"}}, "explicit:configured"),
    ({"reasoning_effort": "max"}, "explicit:max"),
])
def test_reports_outgoing_reasoning(caplog, body, expected):
    agent = SimpleNamespace(base_url="https://openrouter.ai/api/v1")
    kwargs = {
        "model": "openai/gpt-5.6-luna@preset/internal",
        "messages": [{"role": "user", "content": "private-message"}],
        "extra_headers": {"Authorization": "private-token"},
        "extra_body": body,
    }
    with caplog.at_level(logging.INFO):
        log_openrouter_request_reasoning(agent, kwargs, "turn:api:2")
    assert f"reasoning={expected}" in caplog.text
    assert "turn:api:2" in caplog.text
    assert "private-message" not in caplog.text
    assert "private-token" not in caplog.text
    assert "private-value" not in caplog.text


def test_extra_body_override_wins(caplog):
    with caplog.at_level(logging.INFO):
        log_openrouter_request_reasoning(
            SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
            {"model": "openai/gpt-5.6-luna@preset/internal",
             "reasoning": {"effort": "medium"},
             "extra_body": {"reasoning": {"effort": "max"}}},
            "turn:api:3",
        )
    assert "reasoning=explicit:max" in caplog.text


def test_no_preset_or_override_is_provider_default(caplog):
    with caplog.at_level(logging.INFO):
        log_openrouter_request_reasoning(
            SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
            {"model": "openai/gpt-5.6-luna"}, "turn:api:1",
        )
    assert "reasoning=provider-default" in caplog.text


def test_other_provider_is_not_labelled_as_openrouter(caplog):
    with caplog.at_level(logging.INFO):
        log_openrouter_request_reasoning(
            SimpleNamespace(base_url="https://openrouter.ai.example.com/api/v1"),
            {"model": "openai/gpt-5.6-luna@preset/internal"}, "turn:api:1",
        )
    assert not caplog.records
