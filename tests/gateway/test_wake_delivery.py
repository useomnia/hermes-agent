"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_push_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


def test_explicit_push_capability_is_independent_from_async_capability():
    adapter = ApiServerLikeAdapter()
    adapter.supports_async_delivery = True
    adapter.supports_push_delivery = False

    assert adapter_supports_push(adapter) is False


def test_deliver_wake_push_adapter_uses_handle_message():
    adapter = PushAdapter()
    asyncio.run(deliver_wake(adapter, text="wake up", source=_source()))
    assert len(adapter.handled) == 1
    evt = adapter.handled[0]
    assert evt.text == "wake up"
    assert evt.internal is True
    assert evt.source.chat_id == "chat-1"


def test_deliver_wake_push_adapter_requires_source():
    with pytest.raises(ValueError):
        asyncio.run(deliver_wake(PushAdapter(), text="x", session_id="sid"))


def test_deliver_wake_non_push_requires_session_id():
    with pytest.raises(ValueError):
        asyncio.run(deliver_wake(ApiServerLikeAdapter(), text="x", source=_source()))


def test_deliver_wake_non_push_requires_api_key():
    """Session continuation is 403-gated on API_SERVER_KEY — a missing key
    must fail loudly instead of running the wake in a fresh session."""
    adapter = ApiServerLikeAdapter(key="")
    with pytest.raises(RuntimeError, match="API_SERVER_KEY"):
        asyncio.run(deliver_wake(adapter, text="x", session_id="raw-sid"))


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    return await _serve_at(handler, "/v1/chat/completions")


async def _serve_at(handler, path):
    """Spin an in-process aiohttp server exposing ``handler`` at ``path``."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(adapter, text="task done — wake", session_id="raw-sid-42")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


def test_deliver_wake_raises_on_permanent_http_error(monkeypatch):
    """Auth/validation errors (403/400) are permanent — raise immediately so
    the caller can rewind instead of treating the event as delivered."""
    from aiohttp import web

    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return web.json_response({"error": "forbidden"}, status=403)

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            with pytest.raises(RuntimeError, match="HTTP 403"):
                await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 1


def test_deliver_wake_raises_after_exhausted_retries(monkeypatch):
    """Connection failures raise after bounded retries — never silent."""
    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01,))
    # Nothing is listening on this port.
    adapter = ApiServerLikeAdapter(host="127.0.0.1", port=1, key="k")
    with pytest.raises(RuntimeError, match="gave up"):
        asyncio.run(deliver_wake(adapter, text="x", session_id="sid"))


# ---------------------------------------------------------------------------
# OMNIO_WAKE_HOOK redirect (opt-in, default OFF)
# ---------------------------------------------------------------------------


def test_deliver_wake_unset_env_is_byte_identical_to_self_post(monkeypatch):
    """With OMNIO_WAKE_HOOK unset, passing the new optional kwargs must not
    change the self-post wire contract at all — default-off is sacred."""
    from aiohttp import web

    monkeypatch.delenv("OMNIO_WAKE_HOOK", raising=False)
    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["body"] = await request.json()
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port, key="sekrit")
            await deliver_wake(
                adapter,
                text="task done",
                session_id="raw-sid-99",
                delegation_id="deleg_1",
                origin_turn_id="turn_1",
                subagent_ids=["sa-0-abc"],
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-99"
    assert seen["body"] == {
        "model": "hermes",
        "messages": [{"role": "user", "content": "task done"}],
        "stream": False,
    }


def test_deliver_wake_redirects_to_hook_when_env_set(monkeypatch):
    """OMNIO_WAKE_HOOK + a non-empty origin_turn_id redirects the wake to
    the hook URL instead of self-posting — asserts the exact payload shape
    and the service-token header."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = await request.json()
        return web.json_response({"ok": True})

    async def run():
        runner, port = await _serve_at(handler, "/omnio/wake")
        try:
            monkeypatch.setenv("OMNIO_WAKE_HOOK", f"http://127.0.0.1:{port}/omnio/wake")
            monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "svc-tok-1")
            # host/port point at a port nothing listens on — if the redirect
            # didn't happen, the self-post would fail loudly, not silently
            # produce this test's expected payload.
            adapter = ApiServerLikeAdapter(host="127.0.0.1", port=1, key="k")
            await deliver_wake(
                adapter,
                text="subagent finished",
                session_id="raw-sid-7",
                delegation_id="deleg_42",
                origin_turn_id="turn_777",
                subagent_ids=["sa-0-aaa", "sa-1-bbb"],
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["headers"]["X-Omnio-Service-Token"] == "svc-tok-1"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert seen["body"] == {
        "origin_turn_id": "turn_777",
        "delegation_id": "deleg_42",
        "subagent_ids": ["sa-0-aaa", "sa-1-bbb"],
        "session_id": "raw-sid-7",
        "text": "subagent finished",
    }


def test_deliver_wake_falls_back_to_self_post_without_origin_turn_id(monkeypatch):
    """OMNIO_WAKE_HOOK set but no origin_turn_id (non-Omnio-attributable
    wake) must fall back to the self-post — a wake with no turn identity
    can never become a product turn."""
    from aiohttp import web

    seen = {}

    async def hook_handler(request):
        seen["hook_hit"] = True
        return web.json_response({"ok": True})

    async def self_post_handler(request):
        seen["self_post_hit"] = True
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        return web.json_response({"choices": []})

    async def run():
        hook_runner, hook_port = await _serve_at(hook_handler, "/omnio/wake")
        self_runner, self_port = await _serve(self_post_handler)
        try:
            monkeypatch.setenv(
                "OMNIO_WAKE_HOOK", f"http://127.0.0.1:{hook_port}/omnio/wake"
            )
            monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "svc-tok-1")
            adapter = ApiServerLikeAdapter(port=self_port, key="sekrit")
            await deliver_wake(
                adapter, text="x", session_id="raw-sid-3", delegation_id="deleg_x",
            )
        finally:
            await hook_runner.cleanup()
            await self_runner.cleanup()

    asyncio.run(run())
    assert "hook_hit" not in seen
    assert seen.get("self_post_hit") is True
    assert seen["session_id"] == "raw-sid-3"


def test_deliver_wake_hook_requires_service_token(monkeypatch):
    """A hook set with no OMNIO_INTERNAL_TOKEN is a misconfiguration — fail
    loudly rather than POST an unauthenticated wake."""
    monkeypatch.setenv("OMNIO_WAKE_HOOK", "http://127.0.0.1:1/omnio/wake")
    monkeypatch.delenv("OMNIO_INTERNAL_TOKEN", raising=False)
    adapter = ApiServerLikeAdapter(host="127.0.0.1", port=1, key="k")
    with pytest.raises(RuntimeError, match="OMNIO_INTERNAL_TOKEN"):
        asyncio.run(
            deliver_wake(
                adapter, text="x", session_id="sid", origin_turn_id="turn_1",
            )
        )


def test_deliver_wake_hook_retries_409_then_succeeds(monkeypatch):
    """409 (lock/concurrency contention on the proxy side) is transient —
    retried with the same backoff ladder as the self-post path."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "locked"}, status=409)
        return web.json_response({"ok": True})

    async def run():
        runner, port = await _serve_at(handler, "/omnio/wake")
        try:
            monkeypatch.setenv("OMNIO_WAKE_HOOK", f"http://127.0.0.1:{port}/omnio/wake")
            monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "tok")
            adapter = ApiServerLikeAdapter(host="127.0.0.1", port=1, key="k")
            await deliver_wake(
                adapter, text="x", session_id="sid", origin_turn_id="turn_1",
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


def test_deliver_wake_hook_404_is_permanent_no_retry_loop(monkeypatch):
    """404 (the proxy no longer recognises origin_turn_id — e.g. the
    conversation was deleted) is PERMANENT: a single attempt, a typed
    WakeHookPermanentError, and no retry loop — retrying can never
    succeed."""
    from aiohttp import web

    import gateway.wake as wake_mod
    from gateway.wake import WakeHookPermanentError

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return web.json_response({"error": "unknown turn"}, status=404)

    async def run():
        runner, port = await _serve_at(handler, "/omnio/wake")
        try:
            monkeypatch.setenv("OMNIO_WAKE_HOOK", f"http://127.0.0.1:{port}/omnio/wake")
            monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "tok")
            adapter = ApiServerLikeAdapter(host="127.0.0.1", port=1, key="k")
            with pytest.raises(WakeHookPermanentError) as excinfo:
                await deliver_wake(
                    adapter,
                    text="x",
                    session_id="sid",
                    origin_turn_id="turn-deleted",
                )
            assert excinfo.value.status_code == 404
            assert excinfo.value.origin_turn_id == "turn-deleted"
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 1  # exactly one attempt — no retry loop


def test_deliver_wake_hook_permanent_400_raises_immediately(monkeypatch):
    """Any other permanent 4xx also raises WakeHookPermanentError, not the
    generic RuntimeError the self-post path uses for the same class of
    error — callers need to tell "unwinnable" apart from "exhausted
    retries"."""
    from aiohttp import web

    from gateway.wake import WakeHookPermanentError

    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return web.json_response({"error": "bad request"}, status=400)

    async def run():
        runner, port = await _serve_at(handler, "/omnio/wake")
        try:
            monkeypatch.setenv("OMNIO_WAKE_HOOK", f"http://127.0.0.1:{port}/omnio/wake")
            monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "tok")
            adapter = ApiServerLikeAdapter(host="127.0.0.1", port=1, key="k")
            with pytest.raises(WakeHookPermanentError):
                await deliver_wake(
                    adapter, text="x", session_id="sid", origin_turn_id="turn_1",
                )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 1
