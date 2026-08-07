"""Wake an existing agent session from a background completion event.

Two delivery strategies, selected by the target adapter's
``supports_push_delivery`` capability flag:

* Push-capable adapters (telegram, discord, plugin platforms, ...): inject a
  synthetic ``MessageEvent(internal=True)`` through ``adapter.handle_message``
  — the pre-existing wake path, preserved exactly.

* Stateless request/response adapters (the API server,
  ``supports_push_delivery = False``): by default we self-POST
  ``/v1/chat/completions`` on the in-pod API server with the raw session id
  in the ``X-Hermes-Session-Id`` header — the exact entry point real turns
  use — so the wake turn resumes the REAL session, with full history, and
  its result is visible the next time the client polls/reopens the
  conversation. ``handle_message`` cannot be used here: it would run the
  wake turn under a ``build_session_key()``-derived key
  (``agent:main:api_server:group:<sid>``) that NEVER matches the raw
  ``X-Hermes-Session-Id`` key real gateway/HQ turns run under
  (``_bind_api_server_session``), so the wake would land in a parallel,
  invisible session.

  OPT-IN redirect (Omnio): when env ``OMNIO_WAKE_HOOK`` is set AND the
  completion event carries an ``origin_turn_id``, the wake is instead
  delivered by POSTing that URL — the Omnio proxy runs the wake as a real
  product turn (persisted for the product UI) instead of a same-pod
  self-post the proxy never sees. Falls back to the self-post path when the
  hook is set but there is no turn id to attribute the wake to (a wake
  without turn identity cannot become a product turn). With the env unset,
  behavior is byte-identical to the self-post-only path.

Failures RAISE (after bounded retries on transient errors) so callers can
rewind cursors / retry instead of silently losing the event — WITH ONE
EXCEPTION: a permanent (non-transient) 4xx from the wake hook — e.g. 404
when the Omnio proxy no longer recognises ``origin_turn_id`` (conversation
deleted) — raises :class:`WakeHookPermanentError` instead of a plain
``RuntimeError``, so callers that rewind/retry on any exception can
distinguish "try again" from "this will never succeed, drop it" and avoid
redelivering the same unwinnable wake forever.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A wake self-post runs the entire agent turn synchronously (stream=false);
# generous ceiling so long tool-using turns aren't killed mid-flight.
WAKE_TURN_TIMEOUT_SECONDS = 600.0

# Backoff delays between retries on transient failures (429 concurrency cap,
# connection errors). The API server has no per-session lock — concurrent
# turns on one session are last-writer-wins — but it DOES enforce a global
# max_concurrent_runs cap via HTTP 429, which is worth waiting out.
_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)

# Opt-in redirect target: when set, non-push wakes are delivered by POSTing
# this URL instead of self-posting /v1/chat/completions. Unset (the default)
# preserves today's self-post-only behavior exactly.
OMNIO_WAKE_HOOK_ENV = "OMNIO_WAKE_HOOK"
# Shared service-token header, same coordinate the Omnio turn-finalize hook
# uses (gateway/platforms/api_server.py:_request_turn_finalize_annotations).
_OMNIO_INTERNAL_TOKEN_ENV = "OMNIO_INTERNAL_TOKEN"
_WAKE_HOOK_TIMEOUT_SECONDS = 30.0


class WakeHookPermanentError(RuntimeError):
    """A wake-hook POST failed with a permanent (non-retryable) status.

    Raised for any 4xx from ``OMNIO_WAKE_HOOK`` other than 409/429 (both
    treated as transient — see ``_post_wake_hook``) — most notably 404,
    which the Omnio proxy returns when ``origin_turn_id`` no longer
    resolves to a live conversation (e.g. deleted). Distinct from the plain
    ``RuntimeError`` retries give up with, so a caller that would otherwise
    rewind/requeue on ANY exception can instead drop the event: retrying a
    404 can never succeed and would redeliver the same wake forever.
    """

    def __init__(self, message: str, *, status_code: int, origin_turn_id: str):
        super().__init__(message)
        self.status_code = status_code
        self.origin_turn_id = origin_turn_id


def adapter_supports_push(adapter: Any) -> bool:
    """Whether this adapter can push a message to the user after a turn ends.

    ``supports_async_delivery`` answers whether a later wake is possible;
    ``supports_push_delivery`` answers how it is delivered. Older adapters
    that only declare the former retain their previous behavior.
    """
    if hasattr(adapter, "supports_push_delivery"):
        return bool(getattr(adapter, "supports_push_delivery"))
    return bool(getattr(adapter, "supports_async_delivery", True))


async def deliver_wake(
    adapter: Any,
    *,
    text: str,
    session_id: str = "",
    source: Any = None,
    delegation_id: str = "",
    origin_turn_id: str = "",
    subagent_ids: Optional[list] = None,
) -> None:
    """Deliver a wake turn to the session behind ``adapter``.

    ``session_id`` is the RAW session id (the ``X-Hermes-Session-Id`` value /
    ``state.db`` key) — required for non-push adapters. ``source`` is the
    ``SessionSource`` used to build the synthetic event — required for
    push-capable adapters.

    ``delegation_id`` / ``origin_turn_id`` are the completion event's async-
    delegation id and Omnio product turn id. ``subagent_ids`` is the list of
    the completed children's streamed ``subagentId`` values (see
    ``tools.delegate_tool._run_single_child`` / ``entry["subagent_id"]``) —
    the identity the Omnio proxy actually persisted per-child rows under, as
    opposed to ``delegation_id`` which only identifies the batch. All three
    are optional (callers that don't carry them, e.g. the kanban notifier,
    simply get the self-post path) and are only consulted on the non-push
    branch to decide between the ``OMNIO_WAKE_HOOK`` redirect and the
    self-post fallback / to fill the hook payload.

    Raises on failure (bad arguments, exhausted retries, HTTP error) so the
    caller can rewind/retry instead of treating the wake as delivered.
    """
    if adapter_supports_push(adapter):
        if source is None:
            raise ValueError(
                "deliver_wake: push-capable adapter requires a SessionSource"
            )
        from gateway.platforms.base import MessageEvent, MessageType

        synth_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
        )
        await adapter.handle_message(synth_event)
        return

    if not session_id:
        raise ValueError(
            "deliver_wake: non-push adapter "
            "requires the raw session id to self-post the wake turn"
        )

    hook_url = os.environ.get(OMNIO_WAKE_HOOK_ENV, "").strip()
    if hook_url:
        if origin_turn_id:
            await _post_wake_hook(
                hook_url,
                text=text,
                session_id=session_id,
                delegation_id=delegation_id,
                origin_turn_id=origin_turn_id,
                subagent_ids=list(subagent_ids) if subagent_ids else [],
            )
            return
        # A wake without a turn id cannot become a product turn on the Omnio
        # side — fall back to the self-post rather than dropping the wake.
        logger.warning(
            "OMNIO_WAKE_HOOK is set but this wake has no origin_turn_id "
            "(session %s, delegation %s); falling back to self-post",
            session_id, delegation_id or "<none>",
        )

    await _self_post_chat_completion(adapter, text=text, session_id=session_id)


async def _post_wake_hook(
    hook_url: str,
    *,
    text: str,
    session_id: str,
    delegation_id: str,
    origin_turn_id: str,
    subagent_ids: Optional[list] = None,
) -> None:
    """POST a wake completion to the Omnio wake hook instead of self-posting.

    The hook runs on the Omnio proxy side and turns the wake into a real
    product turn (so it is persisted for the product UI, unlike the same-pod
    self-post the proxy never observes). Retry/error semantics mirror
    ``_self_post_chat_completion``: 429/5xx/connection errors are transient
    and retried with the same backoff ladder before giving up; any other 4xx
    is a permanent (config/validation) failure and raises immediately.
    """
    import aiohttp

    service_token = os.environ.get(_OMNIO_INTERNAL_TOKEN_ENV, "")
    if not service_token:
        raise RuntimeError(
            "OMNIO_WAKE_HOOK is set but OMNIO_INTERNAL_TOKEN is missing: the "
            "wake hook requires the shared service token to authenticate — "
            "refusing to POST an unauthenticated wake"
        )

    headers = {
        "Content-Type": "application/json",
        "X-Omnio-Service-Token": service_token,
    }
    payload = {
        "origin_turn_id": origin_turn_id,
        "delegation_id": delegation_id,
        # The children's streamed subagentId values — the proxy matches
        # persisted per-child rows on THESE, not on delegation_id (a batch
        # id may coincide with a single child's subagent_id, but the proxy
        # must not assume that).
        "subagent_ids": list(subagent_ids) if subagent_ids else [],
        "session_id": session_id,
        "text": text,
    }

    last_err: Optional[BaseException] = None
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            timeout = aiohttp.ClientTimeout(total=_WAKE_HOOK_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(hook_url, json=payload, headers=headers) as resp:
                    if resp.status in (409, 429) or resp.status >= 500:
                        # Transient — concurrency/lock contention (409),
                        # concurrency cap (429), or a server-side hiccup.
                        last_err = RuntimeError(
                            f"wake hook POST got HTTP {resp.status} for "
                            f"turn {origin_turn_id}"
                        )
                        logger.warning(
                            "%s; attempt %d/%d", last_err, attempt + 1, attempts
                        )
                        continue
                    if resp.status >= 400:
                        body = (await resp.text())[:300]
                        # Non-transient (auth/validation/gone) — fail
                        # immediately with a typed error so callers can tell
                        # "unwinnable, drop it" (WakeHookPermanentError) from
                        # "exhausted transient retries" (plain RuntimeError
                        # below) apart. 404 is the expected shape here: the
                        # proxy returns it when origin_turn_id no longer
                        # resolves to a live conversation (e.g. deleted).
                        raise WakeHookPermanentError(
                            f"wake hook POST failed for turn {origin_turn_id}: "
                            f"HTTP {resp.status}: {body}",
                            status_code=resp.status,
                            origin_turn_id=origin_turn_id,
                        )
                    await resp.read()
                    logger.info(
                        "wake hook delivered for turn %s (delegation %s, "
                        "attempt %d)",
                        origin_turn_id, delegation_id or "<none>", attempt + 1,
                    )
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_err = exc
            logger.warning(
                "wake hook POST transient failure for turn %s "
                "(attempt %d/%d): %s",
                origin_turn_id, attempt + 1, attempts, exc,
            )
            continue
    raise RuntimeError(
        f"wake hook POST gave up for turn {origin_turn_id} after "
        f"{attempts} attempts: {last_err}"
    ) from last_err


async def _self_post_chat_completion(
    adapter: Any, *, text: str, session_id: str
) -> None:
    """POST the wake text to the in-pod API server as a normal session turn.

    Uses the adapter's own bind host/port/key (``ApiServerAdapter.__init__``).
    Session continuation via ``X-Hermes-Session-Id`` is 403-gated on
    ``API_SERVER_KEY`` being configured, so a missing key is a hard error —
    raise loudly rather than run the wake in a fresh fingerprint-derived
    session nobody is looking at.
    """
    import aiohttp

    host = str(getattr(adapter, "_host", "") or "127.0.0.1")
    if host in ("0.0.0.0", "::", "*"):
        # Wildcard bind address — connect over loopback.
        host = "127.0.0.1"
    port = int(getattr(adapter, "_port", 0) or 8642)
    api_key = str(getattr(adapter, "_api_key", "") or "")
    if not api_key:
        raise RuntimeError(
            "wake self-post requires API_SERVER_KEY: session continuation via "
            "X-Hermes-Session-Id is rejected (403) on an unauthenticated API "
            "server, so the wake cannot reach the target session"
        )

    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Hermes-Session-Id": session_id,
    }
    payload = {
        "model": str(getattr(adapter, "_model_name", "") or "hermes-agent"),
        "messages": [{"role": "user", "content": text}],
        "stream": False,
    }

    last_err: Optional[BaseException] = None
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            timeout = aiohttp.ClientTimeout(total=WAKE_TURN_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 429:
                        # Global concurrency cap (max_concurrent_runs) —
                        # transient; back off and retry.
                        last_err = RuntimeError(
                            f"wake self-post got HTTP 429 (concurrency cap) "
                            f"for session {session_id}"
                        )
                        logger.warning(
                            "%s; attempt %d/%d", last_err, attempt + 1, attempts
                        )
                        continue
                    if resp.status >= 400:
                        body = (await resp.text())[:300]
                        # Non-transient (auth/validation) — fail immediately.
                        raise RuntimeError(
                            f"wake self-post failed for session {session_id}: "
                            f"HTTP {resp.status}: {body}"
                        )
                    await resp.read()
                    logger.info(
                        "wake self-post delivered for session %s (attempt %d)",
                        session_id,
                        attempt + 1,
                    )
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_err = exc
            logger.warning(
                "wake self-post transient failure for session %s "
                "(attempt %d/%d): %s",
                session_id,
                attempt + 1,
                attempts,
                exc,
            )
            continue
    raise RuntimeError(
        f"wake self-post gave up for session {session_id} after "
        f"{attempts} attempts: {last_err}"
    ) from last_err
