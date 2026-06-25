"""Real provider-reported cost capture — never estimated, absent ≠ zero.

OpenRouter usage accounting returns ``usage.cost`` on the response when the
request carries ``usage: {"include": true}``. ``extract_provider_cost_usd``
reads it (and only it); when the provider reports nothing the result is None —
absent, NOT a fabricated $0.00.
"""

from types import SimpleNamespace

import pytest

from agent.usage_pricing import extract_provider_cost_usd


# ── extract_provider_cost_usd — the per-response REAL cost reader ────────────


class TestExtractProviderCost:
    def test_openrouter_usage_cost_attr(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=0.001234)
        assert extract_provider_cost_usd(usage) == pytest.approx(0.001234)

    def test_dict_shaped_usage(self):
        assert extract_provider_cost_usd({"cost": 0.5}) == pytest.approx(0.5)

    def test_reported_zero_is_real_zero(self):
        # Free-tier models really cost $0 — distinct from "not reported".
        usage = SimpleNamespace(cost=0)
        assert extract_provider_cost_usd(usage) == 0.0

    def test_absent_cost_is_none_not_zero(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        assert extract_provider_cost_usd(usage) is None
        assert extract_provider_cost_usd({"prompt_tokens": 10}) is None

    def test_none_usage_is_none(self):
        assert extract_provider_cost_usd(None) is None

    def test_garbage_cost_values_are_none(self):
        for bad in ("0.01", True, float("nan"), float("inf"), -0.5, [], {}):
            assert extract_provider_cost_usd(SimpleNamespace(cost=bad)) is None, bad


# ── OpenRouter request param — usage accounting must be requested ────────────


class TestOpenRouterUsageParam:
    def test_profile_extra_body_requests_usage_accounting(self):
        import importlib.util
        from pathlib import Path

        from providers import get_provider_profile

        profile = get_provider_profile("openrouter")
        if profile is None:
            # Force plugin discovery in minimal test envs.
            plugin = Path(__file__).resolve().parents[2] / "plugins" / "model-providers" / "openrouter" / "__init__.py"
            spec = importlib.util.spec_from_file_location("_or_plugin", plugin)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            profile = mod.openrouter

        body = profile.build_extra_body(session_id="s-1")
        assert body["usage"] == {"include": True}

    def test_legacy_transport_path_requests_usage_accounting(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        kwargs = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            is_openrouter=True,
        )
        assert kwargs["extra_body"]["usage"] == {"include": True}

    def test_non_openrouter_does_not_send_usage_param(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        kwargs = transport.build_kwargs(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            is_openrouter=False,
        )
        assert "usage" not in (kwargs.get("extra_body") or {})


# ── Auxiliary path requests usage accounting too (compression/vision/etc.) ───


class TestAuxUsageParam:
    def test_aux_openrouter_request_requests_usage_and_price_routing(self):
        from agent.auxiliary_client import _build_call_kwargs

        kw = _build_call_kwargs(
            provider="openrouter",
            model="deepseek/deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert kw["extra_body"]["usage"] == {"include": True}
        assert kw["extra_body"]["provider"] == {"sort": "price"}

    def test_aux_non_openrouter_does_not_request_usage(self):
        from agent.auxiliary_client import _build_call_kwargs

        kw = _build_call_kwargs(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "usage" not in (kw.get("extra_body") or {})


# ── SDK contract: OpenRouter's usage.cost survives the OpenAI SDK model ──────


class TestSDKModelExposesCost:
    def test_openai_completion_usage_exposes_cost_extra_field(self):
        # extract_provider_cost_usd relies on the OpenAI SDK keeping extra="allow"
        # so OpenRouter's `cost` extra field is readable via getattr on the real
        # CompletionUsage model. If a future SDK pin flips this, cost capture
        # silently no-ops to the estimate — this fails loudly instead.
        from openai.types import CompletionUsage

        usage = CompletionUsage.model_validate(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 1.23}
        )
        assert extract_provider_cost_usd(usage) == pytest.approx(1.23)
