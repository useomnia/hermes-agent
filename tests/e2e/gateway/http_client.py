"""Stdlib HTTP client for probing a running gateway.

Deliberately dependency-free (urllib only) so the probes match the hand-run
``probe_*.py`` scripts they grew out of and can be lifted out of pytest and run
standalone against any reachable gateway.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass(frozen=True)
class Response:
    status: int
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


class GatewayClient:
    """Minimal OpenAI-compatible client against ``base_url`` with a bearer key."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ── transport ──────────────────────────────────────────────────────────
    def _headers(self, *, auth: bool, stream: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    def get(self, path: str, *, auth: bool = True) -> Response:
        req = urllib.request.Request(
            f"{self.base_url}{path}", method="GET", headers=self._headers(auth=auth)
        )
        return self._send(req)

    def post(self, path: str, payload: dict, *, auth: bool = True) -> Response:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._headers(auth=auth),
        )
        return self._send(req)

    def _send(self, req: urllib.request.Request) -> Response:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return Response(resp.status, resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as err:
            return Response(err.code, err.read().decode("utf-8", "replace"))

    def stream_events(
        self, path: str, payload: dict, *, auth: bool = True
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(event, data)`` for each dispatched SSE block.

        ``event`` is the name from the block's ``event:`` line, or the SSE
        default ``"message"`` when the block has none. This is the only way to
        tell apart channels multiplexed on one connection under different event
        names: the default chat-completion chunks carry no ``event:`` line (so
        ``"message"``), while tool progress rides a custom
        ``event: hermes.tool.progress``. ``[DONE]`` sentinels are skipped.
        Raises ``HTTPError`` for non-2xx so callers see the failure rather than
        an empty stream.
        """
        body = {**payload, "stream": True}
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers=self._headers(auth=auth, stream=True),
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            event = "message"
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    event = "message"  # blank line dispatches the block — reset
                elif line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    if data and data != "[DONE]":
                        yield (event, data)

    def stream(self, path: str, payload: dict, *, auth: bool = True) -> Iterator[str]:
        """Yield SSE ``data:`` payload lines (the part after ``data: ``).

        Event names are discarded — every block's data is yielded flat. Use
        :meth:`stream_events` when you need to tell channels apart by event name
        (e.g. ``hermes.tool.progress``). ``[DONE]`` sentinels are skipped.
        """
        for _, data in self.stream_events(path, payload, auth=auth):
            yield data


# ── payload extractors ──────────────────────────────────────────────────────
def chat_content(body: dict) -> Optional[str]:
    """Assistant text from a /v1/chat/completions response body."""
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def responses_content(body: dict) -> Optional[str]:
    """Assistant text from a /v1/responses ``output`` array."""
    for item in body.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    return part.get("text")
    return None


def chat_delta(chunk: dict) -> dict:
    """The ``choices[0].delta`` object from a streaming chat chunk ({} if absent)."""
    try:
        return chunk["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError):
        return {}


# The SSE event name the gateway uses for its tool-progress channel — distinct
# from the unnamed (default ``"message"``) chat-completion chunks. Match it
# against the first element of a :meth:`GatewayClient.stream_events` pair.
TOOL_PROGRESS_EVENT = "hermes.tool.progress"


def tool_progress(data: str) -> Optional[dict]:
    """Decode a ``hermes.tool.progress`` event's ``data`` payload.

    Unlike chat chunks, the progress event's ``data`` *is* the progress object —
    ``tool``/``status``/``preview``, plus for delegated work the child-identity
    fields (``subagent_id``/``parent_id``/``depth``/``task_index``/…). Returns
    the decoded dict, or ``None`` if the payload isn't a JSON object.
    """
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
