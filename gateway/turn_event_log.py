"""Canonical Responses-native event minting and resumable in-memory Turn logs.

The runs API deliberately stores the exact UTF-8 SSE data frame written to the
wire.  Replays resend those bytes; they never rebuild an event from mutable run
state.  Run IDs are process-lifetime identities and MUST NEVER be reused, even
after their retained log has been replaced by a tombstone.

Hermes run IDs are ``run_<uuid hex>``. Their wire response IDs are formed by
replacing that prefix with ``resp_`` while retaining the exact UUID hex. This
prefix substitution is the one documented, bijective identity mapping; Omnio's
database keeps the corresponding UUID without either prefix.

Omnio extensions are native namespaced Responses events. The complete allowed
set is declared below so additions are deliberate contract changes rather than
ad-hoc ``CUSTOM`` payloads.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from hermes_constants import MAX_TODO_ITEMS


TURN_EVENT_LOG_API_VERSION = 3
DEFAULT_RUN_LOG_CAP_BYTES = 8 * 1024 * 1024
DEFAULT_TERMINAL_RETENTION_SECONDS = 5 * 60
DEFAULT_TOMBSTONE_LIMIT = 1000
TERMINAL_FRAME_RESERVE_BYTES = 2048

# A raw per-provider-delta frame costs ~167 bytes of envelope on top of the
# text itself, which is how a 216 KiB turn amplified to 8 MiB and hit the cap.
# Deltas after an item's first one are buffered and coalesced into one frame
# once either bound is crossed, so the log stays close to the byte cost of
# the text it carries. The first delta of every item still flushes on its
# own to protect time-to-first-token.
DELTA_COALESCE_BYTES = 512
DELTA_COALESCE_SECONDS = 0.05

# This is the security boundary for model-authored tool arguments entering a
# client-visible event. Keep it default-deny. The chat-completions projector
# imports the same mapping so both surfaces make the same allowlist decision.
CUSTOM_TOOL_INPUT_KEYS = {
    "request_user_input": "interaction",
    "emit_client_event": "clientEvent",
    "render_component": "genUi",
}

OMNIO_EXTENSION_EVENT_TYPES = frozenset({
    "response.omnio.interaction",
    "response.omnio.interaction_completed",
    "response.omnio.client_event",
    "response.omnio.gen_ui",
    "response.omnio.task_list",
    "response.omnio.warmup",
    # Native replacements for surfaces carried by the former CUSTOM set.
    "response.omnio.subagent_start",
    "response.omnio.subagent_complete",
    "response.omnio.interrupted_history",
    "response.omnio.approval_request",
    "response.omnio.approval_responded",
    "response.omnio.steer_missed",
    "response.omnio.tool_progress",
})

_TOOL_EXTENSION_EVENTS = {
    "request_user_input": ("response.omnio.interaction", "interaction"),
    "emit_client_event": ("response.omnio.client_event", "client_event"),
    "render_component": ("response.omnio.gen_ui", "gen_ui"),
}


def client_projection_withheld(name: str, arguments: Dict[str, Any]) -> bool:
    """Whether a plugin keeps this call's arguments off client-visible events.

    Both projectors ask this before copying an allowlisted tool's inputs to a
    client, so an argument set the owning tool will reject never renders as a
    card or component. The function-call item is unaffected: the call and its
    result stay in the transcript. Fail-open — a hook fault projects as before,
    since a missing card is a worse failure than a stray one.
    """
    if name not in CUSTOM_TOOL_INPUT_KEYS:
        return False
    try:
        from hermes_cli.plugins import resolve_tool_projection_withhold

        return resolve_tool_projection_withhold(name, arguments) is not None
    except Exception:
        return False


def response_id_for_run_id(run_id: str) -> str:
    """Map one Hermes Turn identity to its Responses wire identity."""
    suffix = run_id[4:] if run_id.startswith("run_") else run_id
    return f"resp_{suffix}"


def _bounded_utf8(value: Any, limit: int) -> str:
    """Bound a terminal string by encoded bytes so it fits the cap reserve."""
    encoded = str(value).encode("utf-8")[:limit]
    return encoded.decode("utf-8", errors="ignore")


class UnknownRunError(KeyError):
    """The requested run has no retained proof visible to this profile."""


class CursorExpiredError(LookupError):
    """Retained metadata proves that the requested cursor is too old."""


class InvalidCursorError(ValueError):
    """A live run cannot reach the requested future cursor."""


@dataclass(frozen=True, slots=True)
class StoredTurnEvent:
    """One immutable event in exactly the form sent over SSE."""

    sequence_number: int
    frame: bytes


@dataclass(frozen=True)
class RunTombstone:
    run_id: str
    owner_profile: Optional[str]
    sequence_number_high_water: int
    completed_at: float
    failure_reason: Optional[str]


@dataclass
class RunEventLog:
    run_id: str
    session_id: str
    owner_profile: Optional[str]
    created_at: float
    status: str = "queued"
    completed_at: Optional[float] = None
    failure_reason: Optional[str] = None
    wire_bytes: int = 0
    cap_exceeded: bool = False
    events: List[StoredTurnEvent] = field(default_factory=list)
    _waiters: set[asyncio.Future] = field(default_factory=set, repr=False)

    @property
    def sequence_number_high_water(self) -> int:
        return len(self.events)

    @property
    def terminal(self) -> bool:
        return self.completed_at is not None

    def frames_after(self, after: int) -> List[StoredTurnEvent]:
        # Live logs never advance their floor, so sequence N is at index N-1.
        if after < 0:
            return []
        return list(self.events[after:])

    def wake_waiters(self) -> None:
        waiters = tuple(self._waiters)
        self._waiters.clear()
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def wait_for_change(self, after: int, timeout: float) -> None:
        if self.sequence_number_high_water > after or self.terminal:
            return
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._waiters.add(waiter)
        # No await occurs between the state check and registration. All log
        # mutation is marshalled onto this same event loop, so no wake is lost.
        if self.sequence_number_high_water > after or self.terminal:
            self._waiters.discard(waiter)
            return
        try:
            await asyncio.wait_for(waiter, timeout=timeout)
        finally:
            self._waiters.discard(waiter)


class TurnEventLogStore:
    """Process-local owner of append-only run logs and expiry tombstones."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        run_log_cap_bytes: int = DEFAULT_RUN_LOG_CAP_BYTES,
        terminal_retention_seconds: float = DEFAULT_TERMINAL_RETENTION_SECONDS,
        tombstone_limit: int = DEFAULT_TOMBSTONE_LIMIT,
        on_cap_exceeded: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._clock = clock
        self.run_log_cap_bytes = run_log_cap_bytes
        self.terminal_retention_seconds = terminal_retention_seconds
        self.tombstone_limit = max(0, tombstone_limit)
        self.on_cap_exceeded = on_cap_exceeded
        self._logs: Dict[str, RunEventLog] = {}
        self._tombstones: Dict[str, RunTombstone] = {}
        # Kept for the process lifetime: a run ID is an identity, not a reusable
        # transport slot. UUID generation is not a substitute for this invariant.
        self._seen_run_ids: set[str] = set()

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    def create_run(
        self,
        run_id: str,
        session_id: str,
        *,
        owner_profile: Optional[str] = None,
    ) -> RunEventLog:
        if run_id in self._seen_run_ids:
            raise ValueError(f"run ID must never be reused: {run_id}")
        self._seen_run_ids.add(run_id)
        log = RunEventLog(
            run_id=run_id,
            session_id=session_id,
            owner_profile=owner_profile,
            created_at=self._clock(),
        )
        self._logs[run_id] = log
        return log

    def get_log(self, run_id: str) -> Optional[RunEventLog]:
        return self._logs.get(run_id)

    def set_status(
        self,
        run_id: str,
        status: str,
        *,
        failure_reason: Optional[str] = None,
    ) -> None:
        log = self._logs.get(run_id)
        if log is None:
            return
        log.status = status
        if failure_reason is not None:
            log.failure_reason = failure_reason

    @staticmethod
    def _stored_event(payload: Dict[str, Any], sequence_number: int) -> StoredTurnEvent:
        event = dict(payload)
        event["sequence_number"] = sequence_number
        serialized = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return StoredTurnEvent(
            sequence_number=sequence_number,
            frame=b"data: " + serialized + b"\n\n",
        )

    def _mark_cap_exceeded(self, log: RunEventLog) -> None:
        if log.cap_exceeded:
            return
        log.cap_exceeded = True
        log.failure_reason = "log_cap_exceeded"
        callback = self.on_cap_exceeded
        if callback is not None:
            callback(log.run_id)

    def append_payload(
        self,
        run_id: str,
        payload: Dict[str, Any],
        *,
        force_terminal: bool = False,
    ) -> Optional[StoredTurnEvent]:
        log = self._logs.get(run_id)
        if log is None:
            raise UnknownRunError(run_id)
        if log.terminal:
            return None
        if log.cap_exceeded and not force_terminal:
            return None

        stored = self._stored_event(
            payload,
            log.sequence_number_high_water + 1,
        )

        ordinary_limit = max(0, self.run_log_cap_bytes - TERMINAL_FRAME_RESERVE_BYTES)
        if not force_terminal and log.wire_bytes + len(stored.frame) > ordinary_limit:
            self._mark_cap_exceeded(log)
            return None

        if (
            force_terminal
            and log.wire_bytes + len(stored.frame) > self.run_log_cap_bytes
        ):
            # Production's 2 KiB reserve is comfortably larger than the
            # compact terminal shapes minted below. This guard keeps the byte
            # cap absolute even under a test-only tiny cap or future schema
            # growth; callers must never write more than the configured cap.
            return None

        log.events.append(stored)
        log.wire_bytes += len(stored.frame)
        log.wake_waiters()
        return stored

    def append_payloads(
        self,
        run_id: str,
        payloads: Iterable[Dict[str, Any]],
    ) -> List[StoredTurnEvent]:
        """Atomically append ordinary events or reject the complete batch."""
        log = self._logs.get(run_id)
        if log is None:
            raise UnknownRunError(run_id)
        if log.terminal or log.cap_exceeded:
            return []

        start = log.sequence_number_high_water + 1
        stored_events = [
            self._stored_event(payload, start + index)
            for index, payload in enumerate(payloads)
        ]
        if not stored_events:
            return []

        batch_bytes = sum(len(stored.frame) for stored in stored_events)
        ordinary_limit = max(0, self.run_log_cap_bytes - TERMINAL_FRAME_RESERVE_BYTES)
        if log.wire_bytes + batch_bytes > ordinary_limit:
            self._mark_cap_exceeded(log)
            return []

        log.events.extend(stored_events)
        log.wire_bytes += batch_bytes
        log.wake_waiters()
        return stored_events

    def mark_terminal(
        self,
        run_id: str,
        status: str,
        *,
        failure_reason: Optional[str] = None,
    ) -> None:
        log = self._logs.get(run_id)
        if log is None or log.terminal:
            return
        log.status = status
        log.completed_at = self._clock()
        if failure_reason is not None:
            log.failure_reason = failure_reason
        log.wake_waiters()

    def lookup_for_cursor(
        self,
        run_id: str,
        after: int,
        *,
        owner_profile: Optional[str] = None,
    ) -> RunEventLog | RunTombstone:
        log = self._logs.get(run_id)
        if log is not None:
            if log.owner_profile != owner_profile:
                raise UnknownRunError(run_id)
            if not log.terminal and after > log.sequence_number_high_water:
                raise InvalidCursorError(run_id)
            return log
        tombstone = self._tombstones.get(run_id)
        if tombstone is None or tombstone.owner_profile != owner_profile:
            raise UnknownRunError(run_id)
        if after < tombstone.sequence_number_high_water:
            raise CursorExpiredError(run_id)
        return tombstone

    def sweep(self, now: Optional[float] = None) -> None:
        current = self._clock() if now is None else now
        for run_id, log in list(self._logs.items()):
            if not log.terminal or log.completed_at is None:
                continue
            if current - log.completed_at < self.terminal_retention_seconds:
                continue
            self._tombstones[run_id] = RunTombstone(
                run_id=run_id,
                owner_profile=log.owner_profile,
                sequence_number_high_water=log.sequence_number_high_water,
                completed_at=log.completed_at,
                failure_reason=log.failure_reason,
            )
            del self._logs[run_id]
            log.wake_waiters()

        # Tombstones only exist to distinguish proven cursor expiry from an
        # unknown run. Bound that proof cache for long-lived gateways: once an
        # old tombstone falls out, its run intentionally becomes a 404 again.
        overflow = len(self._tombstones) - self.tombstone_limit
        if overflow > 0:
            oldest = sorted(
                self._tombstones.values(),
                key=lambda item: (item.completed_at, item.run_id),
            )[:overflow]
            for tombstone in oldest:
                self._tombstones.pop(tombstone.run_id, None)

    def recoverable_runs(
        self,
        now: Optional[float] = None,
        *,
        owner_profile: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.sweep(now)
        logs: Iterable[RunEventLog] = (
            log for log in self._logs.values() if log.owner_profile == owner_profile
        )
        return [
            {
                "runId": log.run_id,
                "status": log.status,
                "sessionId": log.session_id,
                "sequence_number": log.sequence_number_high_water,
                "createdAt": log.created_at,
                "completedAt": log.completed_at,
                "failureReason": log.failure_reason,
            }
            for log in sorted(logs, key=lambda item: item.created_at)
        ]


@dataclass
class _PendingDelta:
    """One item's buffered, not-yet-flushed text/reasoning delta frame."""

    event_type: str
    output_index: int
    buffer: str
    started_at: float


class TurnEventEmitter:
    """Mint the Responses-native vocabulary into one run's immutable log."""

    def __init__(
        self,
        store: TurnEventLogStore,
        run_id: str,
        thread_id: str,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.thread_id = thread_id
        self.clock = clock or store.clock
        self.response_id = response_id_for_run_id(run_id)
        self.created_at = int(self.clock())
        self._next_output_index = 0
        self._messages: Dict[str, Dict[str, Any]] = {}
        self._message_output_indexes: Dict[str, int] = {}
        self._reasoning_items: Dict[str, Dict[str, Any]] = {}
        self._function_calls: Dict[str, Dict[str, Any]] = {}
        self._function_call_occurrences: Dict[str, int] = {}
        self._semantic_tool_calls: Dict[str, Dict[str, Any]] = {}
        # Per-item text/reasoning delta buffers awaiting coalesced flush, and
        # the set of item IDs whose first delta has already been flushed.
        self._pending_deltas: Dict[str, _PendingDelta] = {}
        self._delta_started_item_ids: set[str] = set()

    def _emit(
        self,
        event_type: str,
        *,
        force_terminal: bool = False,
        **fields: Any,
    ) -> Optional[StoredTurnEvent]:
        # Every non-delta event must be preceded by any text it logically
        # follows. Flushing here — rather than trusting caller ordering — is
        # what keeps that invariant true regardless of call site.
        self._flush_pending_deltas()
        return self._append(event_type, force_terminal=force_terminal, **fields)

    def _append(
        self,
        event_type: str,
        *,
        force_terminal: bool = False,
        **fields: Any,
    ) -> Optional[StoredTurnEvent]:
        payload = {"type": event_type, **fields}
        return self.store.append_payload(
            self.run_id, payload, force_terminal=force_terminal
        )

    def _buffer_delta(
        self,
        item_id: str,
        event_type: str,
        output_index: int,
        delta: str,
    ) -> None:
        """Coalesce one provider delta into the item's pending frame.

        The first delta for an item flushes immediately so first-token
        latency is unaffected; later deltas accumulate until the buffer
        reaches ``DELTA_COALESCE_BYTES`` or has aged past
        ``DELTA_COALESCE_SECONDS``, whichever comes first.
        """
        is_first_delta = item_id not in self._delta_started_item_ids
        pending = self._pending_deltas.get(item_id)
        if pending is None:
            pending = _PendingDelta(
                event_type=event_type,
                output_index=output_index,
                buffer=delta,
                started_at=self.clock(),
            )
            self._pending_deltas[item_id] = pending
        else:
            pending.buffer += delta

        if is_first_delta:
            self._delta_started_item_ids.add(item_id)
            self._flush_one_pending_delta(item_id)
            return

        buffered_bytes = len(pending.buffer.encode("utf-8"))
        buffered_seconds = self.clock() - pending.started_at
        if (
            buffered_bytes >= DELTA_COALESCE_BYTES
            or buffered_seconds >= DELTA_COALESCE_SECONDS
        ):
            self._flush_one_pending_delta(item_id)

    def _flush_one_pending_delta(self, item_id: str) -> None:
        pending = self._pending_deltas.pop(item_id, None)
        if pending is None or not pending.buffer:
            return
        self._append(
            pending.event_type,
            item_id=item_id,
            output_index=pending.output_index,
            content_index=0,
            delta=pending.buffer,
        )

    def _flush_pending_deltas(self) -> None:
        if not self._pending_deltas:
            return
        pending_items = list(self._pending_deltas.items())
        self._pending_deltas.clear()
        for item_id, pending in pending_items:
            if not pending.buffer:
                continue
            self._append(
                pending.event_type,
                item_id=item_id,
                output_index=pending.output_index,
                content_index=0,
                delta=pending.buffer,
            )

    def _response(self, status: str, **fields: Any) -> Dict[str, Any]:
        return {
            "id": self.response_id,
            "status": status,
            "created_at": self.created_at,
            **fields,
        }

    def _allocate_output_index(self) -> int:
        output_index = self._next_output_index
        self._next_output_index += 1
        return output_index

    def response_started(self) -> None:
        self.store.set_status(self.run_id, "running")
        response = self._response("in_progress")
        self._emit("response.created", response=response)
        self._emit("response.in_progress", response=response)

    def output_text_start(self, item_id: str) -> None:
        output_index = self._allocate_output_index()
        started_at = self.clock()
        item = {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
            "started_at": started_at,
        }
        self._messages[item_id] = {
            "output_index": output_index,
            "text": "",
            "started_at": started_at,
        }
        self._message_output_indexes[item_id] = output_index
        self._emit(
            "response.output_item.added",
            output_index=output_index,
            item=item,
        )

    def output_text_delta(self, item_id: str, delta: str) -> None:
        state = self._messages.get(item_id)
        if state is None:
            return
        state["text"] += delta
        self._buffer_delta(
            item_id,
            "response.output_text.delta",
            state["output_index"],
            delta,
        )

    def output_text_done(self, item_id: str) -> None:
        state = self._messages.pop(item_id, None)
        if state is None:
            return
        self._delta_started_item_ids.discard(item_id)
        output_index = state["output_index"]
        text = state["text"]
        item = {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
            "started_at": state["started_at"],
            "completed_at": self.clock(),
        }
        self._emit(
            "response.output_text.done",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            text=text,
        )
        self._emit(
            "response.output_item.done",
            output_index=output_index,
            item=item,
        )

    def output_text_annotations_added(
        self,
        item_id: str,
        annotations: Iterable[Dict[str, Any]],
    ) -> List[StoredTurnEvent]:
        output_index = self._message_output_indexes.get(item_id)
        if output_index is None:
            return []
        self._flush_pending_deltas()
        return self.store.append_payloads(
            self.run_id,
            (
                {
                    "type": "response.output_text.annotation.added",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "annotation_index": annotation_index,
                    "annotation": annotation,
                }
                for annotation_index, annotation in enumerate(annotations)
            ),
        )

    def output_index_for_message(self, item_id: str) -> Optional[int]:
        return self._message_output_indexes.get(item_id)

    def reasoning_start(self, item_id: str) -> None:
        output_index = self._allocate_output_index()
        started_at = self.clock()
        item = {
            "id": item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
            "content": [],
            "started_at": started_at,
        }
        self._reasoning_items[item_id] = {
            "output_index": output_index,
            "text": "",
            "started_at": started_at,
        }
        self._emit(
            "response.output_item.added",
            output_index=output_index,
            item=item,
        )

    def reasoning_text_delta(self, item_id: str, delta: str) -> None:
        state = self._reasoning_items.get(item_id)
        if state is None:
            return
        state["text"] += delta
        self._buffer_delta(
            item_id,
            "response.reasoning_text.delta",
            state["output_index"],
            delta,
        )

    def reasoning_text_done(self, item_id: str) -> None:
        state = self._reasoning_items.pop(item_id, None)
        if state is None:
            return
        self._delta_started_item_ids.discard(item_id)
        output_index = state["output_index"]
        text = state["text"]
        self._emit(
            "response.reasoning_text.done",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            text=text,
        )
        self._emit(
            "response.output_item.done",
            output_index=output_index,
            item={
                "id": item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": text}],
                "started_at": state["started_at"],
                "completed_at": self.clock(),
            },
        )

    def function_call_start(
        self, call_id: str, name: str, *, early: bool = False
    ) -> None:
        occurrence = self._function_call_occurrences.get(call_id, 0) + 1
        self._function_call_occurrences[call_id] = occurrence
        # A provider may reuse a deterministic call ID on a later retry. The
        # first item keeps the established identity; reopened lifecycles get a
        # distinct Responses item ID while retaining the provider call_id.
        item_id = (
            f"fc_{call_id}"
            if occurrence == 1
            else f"fc_{call_id}_{occurrence}"
        )
        output_index = self._allocate_output_index()
        started_at = self.clock()
        self._function_calls[call_id] = {
            "item_id": item_id,
            "name": name,
            "output_index": output_index,
            "arguments": None,
            "early": early,
            "started_at": started_at,
        }
        self._emit(
            "response.output_item.added",
            output_index=output_index,
            item={
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "name": name,
                "call_id": call_id,
                **({"arguments": ""} if early else {}),
                "started_at": started_at,
            },
        )

    def function_call_arguments(
        self, call_id: str, name: str, arguments: Dict[str, Any]
    ) -> None:
        state = self._function_calls.get(call_id)
        if (
            state is None
            or name not in CUSTOM_TOOL_INPUT_KEYS
            or not isinstance(arguments, dict)
        ):
            return
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        state["arguments"] = serialized
        self._emit(
            "response.function_call_arguments.delta",
            item_id=state["item_id"],
            output_index=state["output_index"],
            delta=serialized,
        )
        if client_projection_withheld(name, arguments):
            return
        extension_type, payload_key = _TOOL_EXTENSION_EVENTS[name]
        self.omnio_event(
            extension_type,
            tool_call_id=call_id,
            **{payload_key: arguments},
        )

    def function_call_done(self, call_id: str) -> None:
        state = self._function_calls.pop(call_id, None)
        if state is None:
            return
        item = {
            "id": state["item_id"],
            "type": "function_call",
            "status": "completed",
            "name": state["name"],
            "call_id": call_id,
            "started_at": state["started_at"],
            "completed_at": self.clock(),
        }
        arguments = state["arguments"]
        if arguments is not None:
            self._emit(
                "response.function_call_arguments.done",
                item_id=state["item_id"],
                output_index=state["output_index"],
                arguments=arguments,
            )
            item["arguments"] = arguments
        elif state["early"]:
            # The early Responses boundary advertised an empty argument
            # string. Keep that native item shape at completion without
            # inventing argument delta/done frames for ordinary tools.
            item["arguments"] = ""
        self._emit(
            "response.output_item.done",
            output_index=state["output_index"],
            item=item,
        )

    def function_call_incomplete(self, call_id: str) -> None:
        """Close a call announced by an abandoned provider stream attempt."""
        state = self._function_calls.pop(call_id, None)
        if state is None:
            return
        self._emit(
            "response.output_item.done",
            output_index=state["output_index"],
            item={
                "id": state["item_id"],
                "type": "function_call",
                "status": "incomplete",
                "name": state["name"],
                "call_id": call_id,
                "arguments": "",
                "started_at": state["started_at"],
                "completed_at": self.clock(),
            },
        )

    def semantic_tool_start(self, source_call_id: str, tool: str) -> None:
        """Expose the executed tool when it differs from the model-facing call."""
        if source_call_id in self._semantic_tool_calls:
            return
        started_at = self.clock()
        self._semantic_tool_calls[source_call_id] = {
            "tool": tool,
            "started_at": started_at,
        }
        self.omnio_event(
            "response.omnio.tool_progress",
            source_call_id=source_call_id,
            tool=tool,
            status="running",
            started_at=started_at,
        )

    def semantic_tool_done(self, source_call_id: str) -> None:
        state = self._semantic_tool_calls.pop(source_call_id, None)
        if state is None:
            return
        self.omnio_event(
            "response.omnio.tool_progress",
            source_call_id=source_call_id,
            tool=state["tool"],
            status="completed",
            started_at=state["started_at"],
            completed_at=self.clock(),
        )

    def task_list(self, todos: list) -> None:
        """Emit the bounded canonical todo snapshot without exposing its result."""
        bounded: List[Dict[str, str]] = []
        if isinstance(todos, list):
            for item in todos[:MAX_TODO_ITEMS]:
                if not isinstance(item, dict):
                    continue
                clean = {
                    key: item[key]
                    for key in ("id", "content", "status")
                    if isinstance(item.get(key), str)
                }
                if clean:
                    bounded.append(clean)
        self.omnio_event("response.omnio.task_list", todos=bounded)

    def omnio_event(self, event_type: str, **fields: Any) -> Optional[StoredTurnEvent]:
        if event_type not in OMNIO_EXTENSION_EVENT_TYPES:
            raise ValueError(f"unknown Omnio response event: {event_type}")
        return self._emit(event_type, **fields)

    def response_completed(self) -> None:
        self._emit(
            "response.completed",
            force_terminal=True,
            response=self._response("completed"),
        )
        self.store.mark_terminal(self.run_id, "completed")

    def response_failed(
        self,
        message: str,
        *,
        code: str,
    ) -> None:
        self._emit(
            "response.failed",
            force_terminal=True,
            response=self._response(
                "failed",
                error={
                    "code": _bounded_utf8(code, 64),
                    "message": _bounded_utf8(message, 256),
                },
            ),
        )
        self.store.mark_terminal(self.run_id, "failed", failure_reason=code)

    def response_incomplete(self) -> None:
        """Map a user/API stop to Responses ``incomplete`` with cancellation."""
        self._emit(
            "response.incomplete",
            force_terminal=True,
            response=self._response(
                "incomplete",
                incomplete_details={"reason": "cancelled"},
            ),
        )
        self.store.mark_terminal(
            self.run_id,
            "cancelled",
            failure_reason="run_cancelled",
        )


__all__ = [
    "client_projection_withheld",
    "CursorExpiredError",
    "CUSTOM_TOOL_INPUT_KEYS",
    "DEFAULT_RUN_LOG_CAP_BYTES",
    "DEFAULT_TERMINAL_RETENTION_SECONDS",
    "DEFAULT_TOMBSTONE_LIMIT",
    "InvalidCursorError",
    "OMNIO_EXTENSION_EVENT_TYPES",
    "RunEventLog",
    "RunTombstone",
    "StoredTurnEvent",
    "TURN_EVENT_LOG_API_VERSION",
    "TurnEventEmitter",
    "TurnEventLogStore",
    "UnknownRunError",
    "response_id_for_run_id",
]
