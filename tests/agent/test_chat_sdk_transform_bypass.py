"""Exercise real SDK serialization and the fork's request boundaries."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest
from pydantic import BaseModel

from agent.sdk_transform_bypass import (
    bypass_chat_sdk_request_transform,
    bypass_responses_sdk_request_transform,
    bypass_sdk_request_transform,
)


_CHAT_REPLY: dict[str, Any] = {
    "id": "r1", "object": "chat.completion", "created": 1, "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
}
_CHAT_SSE = (
    b'data: {"id":"r1","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
    b'data: [DONE]\n\n'
)
_RESPONSES_SSE = (
    b'data: {"type":"response.output_text.delta","delta":"ok","item_id":"msg1","output_index":0,"content_index":0,"sequence_number":1}\n\n'
    b'data: {"type":"response.completed","sequence_number":2,"response":{"id":"r1","object":"response","model":"m","status":"completed","output":[],"usage":null}}\n\n'
)


@pytest.fixture
def typed_calls(monkeypatch):
    from openai.resources.chat.completions import Completions
    from openai.resources.responses import Responses

    seen = []
    for resource in (Completions, Responses):
        original = resource.create

        def spy(self, *args, _original=original, **kwargs):
            seen.append(kwargs)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(resource, "create", spy)
    return seen


def _make_agent():
    from run_agent import AIAgent

    return AIAgent(api_key="test", base_url="https://example.com/v1", model="m",
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def _body(family):
    if family == "chat":
        return {
            "model": "m",
            "messages": [
                {"role": "system", "content": "stable prefix"},
                {"role": "user", "content": [{"type": "text", "text": "héllo 日本語"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "done"},
            ],
            "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object", "properties": {}}}}],
            "temperature": 0.7,
            "extra_body": {"reasoning": {"effort": "high"}, "provider": {"order": ["nous"]}},
        }
    return {
        "model": "m", "instructions": "stable prefix",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "héllo 日本語"}]}],
        "tools": [{"type": "function", "name": "terminal", "parameters": {"type": "object", "properties": {}}}],
        "store": False, "extra_body": {"metadata": {"source": "test"}},
    }


def _client(family="chat", *, asynchronous=False, replies=None):
    bodies = []

    def handle(request):
        bodies.append(request.content)
        import json
        body = json.loads(request.content)
        if replies is not None:
            return httpx.Response(200, json=next(replies))
        if body.get("stream"):
            return httpx.Response(200, content=_CHAT_SSE if family == "chat" else _RESPONSES_SSE,
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_CHAT_REPLY if family == "chat" else {
            "id": "r1", "object": "response", "model": "m", "status": "completed", "output": []})

    transport = httpx.MockTransport(handle)
    if asynchronous:
        return openai.AsyncOpenAI(api_key="test", base_url="https://example.com/v1",
                                 http_client=httpx.AsyncClient(transport=transport)), bodies
    return openai.OpenAI(api_key="test", base_url="https://example.com/v1",
                         http_client=httpx.Client(transport=transport)), bodies


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_real_sdk_request_bytes_are_identical(family, stream, override):
    client, bodies = _client(family)
    with client:
        body = _body(family)
        body["stream"] = stream
        bulk = "messages" if family == "chat" else "input"
        if override:
            body["extra_body"][bulk] = [{"role": "user", "content": "explicit override"}]
            body["extra_body"]["tools"] = []
        before = deepcopy(body)
        bypass = bypass_chat_sdk_request_transform if family == "chat" else bypass_responses_sdk_request_transform
        moved = bypass(body, client)
        assert moved[bulk] == [] and moved["tools"] == []
        create = client.chat.completions.create if family == "chat" else client.responses.create
        for kwargs in (body, moved):
            response = create(**kwargs)
            if stream:
                with response:
                    list(response)
        assert bodies[0] == bodies[1]
        assert body == before


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_async_sdk_request_bytes_are_identical(family, stream):
    client, bodies = _client(family, asynchronous=True)
    async with client:
        body = {**_body(family), "stream": stream}
        bypass = bypass_chat_sdk_request_transform if family == "chat" else bypass_responses_sdk_request_transform
        create = client.chat.completions.create if family == "chat" else client.responses.create
        for kwargs in (body, bypass(body, client)):
            response = await create(**kwargs)
            if stream:
                async with response:
                    async for _ in response:
                        pass
        assert bodies[0] == bodies[1]


def test_facades_and_non_json_fields_keep_the_original_path():
    class Model(BaseModel):
        role: str = "user"
        content: str = "hello"

    facade = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()), responses=SimpleNamespace())
    for bypass in (bypass_chat_sdk_request_transform, bypass_responses_sdk_request_transform):
        body = _body("chat")
        assert bypass(body, facade) is body
    for value in ([Model()], iter([{"role": "user"}]), [{"content": b"bytes"}], [{1: "non-string key"}]):
        body = {"input": value}
        assert bypass_sdk_request_transform(body) is body
    body = {"input": [{"role": "user"}], "extra_body": object()}
    assert bypass_sdk_request_transform(body) is body


def test_primary_nonstream_and_stream_paths_reach_real_sdk(typed_calls):

    agent = _make_agent()
    body = _body("chat")
    client, bodies = _client()
    with client, patch.object(agent, "_create_request_openai_client", return_value=client), \
         patch.object(agent, "_close_request_openai_client"):
        assert agent._interruptible_api_call(body).choices[0].message.content == "ok"
        assert agent._interruptible_streaming_api_call(body).choices[0].message.content == "ok"
    import json
    assert len(bodies) == 2
    assert all(c["messages"] == [] and c["tools"] == [] for c in typed_calls)
    assert all(json.loads(b)["messages"] == body["messages"] for b in bodies)


def test_codex_primary_and_auxiliary_paths_reach_real_sdk(typed_calls):
    from agent.auxiliary_client import _CodexCompletionsAdapter

    agent = _make_agent()
    client, bodies = _client("responses")
    with client:
        assert agent._run_codex_stream(_body("responses"), client=client).output_text == "ok"
        adapter = _CodexCompletionsAdapter(client, "m")
        assert adapter.create(model="m", messages=[{"role": "user", "content": "hello"}]).choices[0].message.content == "ok"
    import json
    assert len(bodies) == 2
    assert all(c["input"] == [] for c in typed_calls)
    assert all(json.loads(b)["input"] for b in bodies)


@pytest.mark.parametrize("stream", [False, True])
def test_auxiliary_progress_paths_reach_real_sdk(stream, typed_calls):
    from agent.auxiliary_client import _create_with_progress

    client, bodies = _client()
    with client:
        result = _create_with_progress(client, _body("chat"), force_stream=stream)
        assert result.choices[0].message.content == "ok"
    assert len(bodies) == 1
    assert typed_calls[0]["messages"] == []


def test_summary_initial_and_retry_paths_reach_real_sdk(typed_calls):

    agent = _make_agent()
    agent._cached_system_prompt = "stable prefix"
    empty = deepcopy(_CHAT_REPLY)
    empty["choices"][0]["message"]["content"] = ""
    client, bodies = _client(replies=iter([empty, _CHAT_REPLY]))
    with client, patch.object(agent, "_ensure_primary_openai_client", return_value=client):
        assert agent._handle_max_iterations([{"role": "user", "content": "summarize"}], 60) == "ok"
    assert len(bodies) == 2
    assert all(c["messages"] == [] for c in typed_calls)
