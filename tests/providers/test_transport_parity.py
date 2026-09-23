"""Parity tests: pin the exact current transport behavior per provider.

These tests document the flag-based contract between run_agent.py and
ChatCompletionsTransport.build_kwargs(). When the next PR wires profiles
to replace flags, every assertion here must still pass — any failure is
a behavioral regression.
"""

import json

import httpx
import pytest
from openai import OpenAI

from hermes_constants import parse_reasoning_effort
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _simple_messages():
    return [{"role": "user", "content": "hello"}]


def _max_tokens_fn(n):
    return {"max_completion_tokens": n}


class TestNvidiaParity:
    """NVIDIA NIM: default max_tokens=16384."""

    def test_default_max_tokens(self, transport):
        """NVIDIA default max_tokens=16384 comes from profile, not legacy is_nvidia_nim flag."""
        from providers import get_provider_profile

        profile = get_provider_profile("nvidia")
        kw = transport.build_kwargs(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=_simple_messages(),
            tools=None,
            max_tokens_param_fn=_max_tokens_fn,
            provider_profile=profile,
        )
        assert kw["max_completion_tokens"] == 16384

    def test_user_max_tokens_overrides(self, transport):
        from providers import get_provider_profile

        profile = get_provider_profile("nvidia")
        kw = transport.build_kwargs(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=_simple_messages(),
            tools=None,
            max_tokens=4096,
            max_tokens_param_fn=_max_tokens_fn,
            provider_profile=profile,
        )
        assert kw["max_completion_tokens"] == 4096  # user overrides default


class TestKimiParity:
    """Kimi: OMIT temperature, max_tokens=32000, thinking + reasoning_effort."""

    def test_temperature_omitted(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            omit_temperature=True,
        )
        assert "temperature" not in kw

    def test_default_max_tokens(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            max_tokens_param_fn=_max_tokens_fn,
        )
        assert kw["max_completion_tokens"] == 32000

    def test_thinking_enabled(self, transport):
        # xor contract (fix ce4e74b3): an explicit recognized effort sends
        # reasoning_effort ONLY — never paired with extra_body.thinking.
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kw.get("reasoning_effort") == "high"
        assert "thinking" not in kw.get("extra_body", {})

    def test_thinking_enabled_without_effort(self, transport):
        # enabled but no effort → fall back to the thinking toggle, no effort.
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True},
        )
        assert kw["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in kw

    def test_thinking_disabled(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": False},
        )
        assert kw["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in kw

    def test_reasoning_effort_top_level(self, transport):
        """Kimi reasoning_effort is a TOP-LEVEL api_kwargs key, NOT in extra_body."""
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kw.get("reasoning_effort") == "high"
        assert "reasoning_effort" not in kw.get("extra_body", {})

    def test_reasoning_effort_default_no_effort(self, transport):
        # xor contract: enabled with no effort falls back to thinking-enabled
        # and emits NO top-level reasoning_effort (previously defaulted to
        # "medium" alongside thinking — the pairing this fix removes).
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True},
        )
        assert "reasoning_effort" not in kw
        assert kw["extra_body"]["thinking"] == {"type": "enabled"}


class TestOpenRouterParity:
    """OpenRouter: provider preferences, reasoning in extra_body."""

    def test_provider_preferences(self, transport):
        prefs = {"allow": ["anthropic"], "sort": "price"}
        kw = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("openrouter"),
            provider_preferences=prefs,
        )
        assert kw["extra_body"]["provider"] == prefs

    def test_reasoning_passes_full_config(self, transport):
        """OpenRouter passes the FULL reasoning_config dict, not just effort."""
        rc = {"enabled": True, "effort": "high"}
        kw = transport.build_kwargs(
            model="deepseek/deepseek-chat",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True,
            reasoning_config=rc,
        )
        assert kw["extra_body"]["reasoning"] == rc

    def test_reasoning_omitted_for_mandatory_anthropic(self, transport):
        """Adaptive-thinking Anthropic models (4.6+/fable) get NO reasoning
        field — sending one makes OpenRouter emit thinking.type.disabled on
        tool-replay turns, which the model 400s on."""
        kw = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "reasoning" not in kw.get("extra_body", {})

    def test_default_reasoning_when_no_config(self, transport):
        """When supports_reasoning=True but no config, adds default."""
        kw = transport.build_kwargs(
            model="deepseek/deepseek-chat",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True,
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}

    @pytest.mark.parametrize("profile_loaded", [True, False])
    @pytest.mark.parametrize("model", [
        "@preset/omnio-internal",
        "openai/gpt-5.6-luna@preset/omnio-internal",
    ])
    def test_preset_inherits_reasoning_on_wire(self, transport, model, profile_loaded):
        """An empty local setting must not overwrite preset reasoning in JSON."""
        kw = transport.build_kwargs(
            model=model,
            messages=_simple_messages(),
            provider_profile=get_provider_profile("openrouter") if profile_loaded else None,
            is_openrouter=True,
            supports_reasoning=True,
            reasoning_config=parse_reasoning_effort(""),
        )
        requests = []

        def capture(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "test-response", "object": "chat.completion", "created": 0,
                "model": "openai/gpt-5.6-luna",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
            })

        with OpenAI(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(capture)),
        ) as client:
            client.chat.completions.create(**kw)

        assert len(requests) == 1
        assert requests[0]["model"] == model
        assert "reasoning" not in requests[0]
        assert "reasoning_effort" not in requests[0]

    @pytest.mark.parametrize("reasoning", [
        {"enabled": True, "effort": "max"},
        {"enabled": False},
        {"max_tokens": 4096, "exclude": True},
    ])
    def test_preset_preserves_explicit_reasoning(self, transport, reasoning):
        kw = transport.build_kwargs(
            model="openai/gpt-5.6-luna@preset/omnio-internal",
            messages=_simple_messages(),
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True,
            reasoning_config=reasoning,
        )
        assert kw["extra_body"]["reasoning"] == reasoning

    def test_preset_preserves_request_override(self, transport):
        reasoning = {"enabled": True, "effort": "max"}
        kw = transport.build_kwargs(
            model="openai/gpt-5.6-luna@preset/omnio-internal",
            messages=_simple_messages(),
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True,
            request_overrides={"extra_body": {"reasoning": reasoning}},
        )
        assert kw["extra_body"]["reasoning"] == reasoning

    def test_non_openrouter_preset_name_keeps_default(self, transport):
        kw = transport.build_kwargs(
            model="openai/gpt-5.6-luna@preset/local",
            messages=_simple_messages(),
            supports_reasoning=True,
        )
        assert kw["extra_body"]["reasoning"]["effort"] == "medium"


class TestNousParity:
    """Nous: product tags, reasoning, omit when disabled."""

    def test_tags(self, transport):
        from agent.portal_tags import nous_portal_tags
        kw = transport.build_kwargs(
            model="hermes-3-llama-3.1-405b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nous"),
        )
        assert kw["extra_body"]["tags"] == nous_portal_tags()

    def test_provider_preferences(self, transport):
        preferences = {
            "only": ["deepseek"],
            "ignore": ["deepinfra"],
            "sort": "throughput",
        }
        kw = transport.build_kwargs(
            model="deepseek/deepseek-v4-flash",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nous"),
            provider_preferences=preferences,
        )
        assert kw["extra_body"]["provider"] == preferences

    def test_reasoning_omitted_when_disabled(self, transport):
        """Nous special case: reasoning omitted entirely when disabled."""
        kw = transport.build_kwargs(
            model="hermes-3-llama-3.1-405b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nous"),
            supports_reasoning=True,
            reasoning_config={"enabled": False},
        )
        assert "reasoning" not in kw.get("extra_body", {})

    def test_reasoning_enabled(self, transport):
        rc = {"enabled": True, "effort": "high"}
        kw = transport.build_kwargs(
            model="hermes-3-llama-3.1-405b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nous"),
            supports_reasoning=True,
            reasoning_config=rc,
        )
        assert kw["extra_body"]["reasoning"] == rc


class TestQwenParity:
    """Qwen: max_tokens=65536, vl_high_resolution, metadata top-level."""

    def test_default_max_tokens(self, transport):
        kw = transport.build_kwargs(
            model="qwen3.5-plus",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("qwen-oauth"),
            max_tokens_param_fn=_max_tokens_fn,
        )
        assert kw["max_completion_tokens"] == 65536

    def test_vl_high_resolution(self, transport):
        kw = transport.build_kwargs(
            model="qwen3.5-plus",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("qwen-oauth"),
        )
        assert kw["extra_body"]["vl_high_resolution_images"] is True

    def test_metadata_top_level(self, transport):
        """Qwen metadata goes to top-level api_kwargs, NOT extra_body."""
        meta = {"sessionId": "s123", "promptId": "p456"}
        kw = transport.build_kwargs(
            model="qwen3.5-plus",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("qwen-oauth"),
            qwen_session_metadata=meta,
        )
        assert kw["metadata"] == meta
        assert "metadata" not in kw.get("extra_body", {})


class TestCustomOllamaParity:
    """Custom/Ollama: num_ctx, thinking controls — now tested via profile."""

    def test_ollama_num_ctx(self, transport):
        kw = transport.build_kwargs(
            model="llama3.1",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("custom"),
            ollama_num_ctx=131072,
        )
        assert kw["extra_body"]["options"]["num_ctx"] == 131072

    def test_think_false_when_disabled(self, transport):
        kw = transport.build_kwargs(
            model="qwen3:72b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("custom"),
            reasoning_config={"enabled": False, "effort": "none"},
        )
        assert kw["extra_body"]["think"] is False
