"""Behaviour of bounded nested calls across a run-owned stream."""

import contextvars
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.code_execution_stream import StreamDispatcher
from tools.code_execution_tool import _NestedToolDispatcher, _open_remote_code_rpc
from tools.interrupt import ToolExecutionScope


class Connection:
    def __init__(self):
        self.incoming = queue.Queue()
        self.outgoing = queue.Queue()
        self.closed = False

    def recv(self, timeout):
        try:
            return self.incoming.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None

    def send(self, raw):
        self.outgoing.put(json.loads(raw))

    def close(self):
        self.closed = True

    def call(self, key, tool="read", **args):
        self.incoming.put(
            json.dumps({"type": "call", "id": key, "tool": tool, "args": args})
        )


def make_stream(dispatch, concurrency=2):
    connection = Connection()
    execution = ToolExecutionScope(threading.Event())
    stream = StreamDispatcher(
        connection,
        dispatch,
        execution,
        concurrency=concurrency,
        read_safe=lambda name: name == "read",
    )
    stream.start()
    return stream, connection


def test_parallel_reads_preserve_response_ids_and_parent_context():
    context = contextvars.ContextVar("stream-test-context")
    token = context.set("session-a")
    both_started = threading.Barrier(2, timeout=5)

    def dispatch(tool, args):
        both_started.wait()
        assert context.get() == "session-a"
        return json.dumps({"value": args["value"]})

    stream, connection = make_stream(dispatch)
    try:
        connection.call("first", value=1)
        connection.call("second", value=2)
        responses = [
            connection.outgoing.get(timeout=5),
            connection.outgoing.get(timeout=5),
        ]
        assert {r["id"]: json.loads(r["result"])["value"] for r in responses} == {
            "first": 1,
            "second": 2,
        }
    finally:
        stream.close()
        context.reset(token)


def test_unsafe_tool_waits_for_reads_and_excludes_later_reads():
    active = set()
    events = []
    lock = threading.Lock()

    def dispatch(tool, args):
        with lock:
            assert "write" not in active
            if tool == "write":
                assert not active
            active.add(args["name"])
            events.append(("start", args["name"]))
        time.sleep(0.03)
        with lock:
            active.remove(args["name"])
            events.append(("end", args["name"]))
        return "{}"

    stream, connection = make_stream(dispatch)
    try:
        connection.call("1", name="a")
        connection.call("2", name="b")
        connection.call("3", tool="write", name="write")
        connection.call("4", name="c")
        for _ in range(4):
            connection.outgoing.get(timeout=5)
        assert events.index(("start", "write")) > events.index(("end", "a"))
        assert events.index(("start", "write")) > events.index(("end", "b"))
        assert events.index(("start", "c")) > events.index(("end", "write"))
    finally:
        stream.close()


def test_atomic_call_budget_cannot_be_exceeded_by_parallel_dispatch(monkeypatch):
    import model_tools

    entered = []
    monkeypatch.setattr(
        model_tools,
        "handle_function_call",
        lambda *a, **kw: entered.append(kw["tool_call_id"]) or "{}",
    )
    calls, count = [], [0]
    dispatcher = _NestedToolDispatcher(
        "task",
        calls,
        count,
        7,
        frozenset({"read"}),
        ToolExecutionScope(threading.Event()),
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: json.loads(dispatcher.dispatch("read", {})), range(40))
        )
    assert len(entered) == len(set(entered)) == count[0] == 7
    assert sum("error" in r for r in results) == 33


def test_cancellation_stops_admission_of_queued_calls():
    entered = []
    started = threading.Event()
    release = threading.Event()

    def dispatch(tool, args):
        entered.append(args["value"])
        started.set()
        release.wait(timeout=5)
        return "{}"

    stream, connection = make_stream(dispatch, concurrency=1)
    try:
        connection.call("1", value=1)
        assert started.wait(timeout=5)
        connection.call("2", value=2)
        stream.execution.cancel()
        release.set()
        stream.close()
        assert entered == [1]
        assert connection.closed
    finally:
        release.set()
        stream.close()


@pytest.mark.parametrize("value", [0, 33, True, "8"])
def test_invalid_concurrency_is_rejected(value):
    with pytest.raises(ValueError):
        make_stream(lambda *_: "{}", concurrency=value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"type": "other"},
        {"type": "call", "id": [], "tool": "read", "args": {}},
        {"type": "call", "id": "x", "tool": "read", "args": []},
    ],
)
def test_invalid_request_closes_channel_and_cancels_script(payload):
    stream, connection = make_stream(lambda *_: "{}")
    try:
        connection.incoming.put(json.dumps(payload))
        stream.thread.join(timeout=5)
        assert stream.execution.is_cancelled()
        assert connection.closed and stream.errors == ["ValueError"]
    finally:
        stream.close()


def test_duplicate_request_cannot_dispatch_twice():
    calls = []
    stream, connection = make_stream(lambda *args: calls.append(args) or "{}")
    try:
        connection.call("one")
        connection.outgoing.get(timeout=5)
        connection.call("one")
        stream.thread.join(timeout=5)
        assert len(calls) == 1
        assert stream.execution.is_cancelled()
    finally:
        stream.close()


def test_stream_capability_negotiation_uses_get_and_keeps_pair_auth_trusted(
    monkeypatch,
):
    import websockets.sync.client
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "http://127.0.0.1:8643/internal/toolbox"
    env.bearer_token = "pair-private"
    env.brand = "brand-a"
    requests = []
    env._request_json = lambda path, **kw: requests.append((path, kw)) or {"rpc": 1}
    connection = Connection()
    connection.incoming.put(
        json.dumps({"type": "ready", "socket": "/tmp/code-rpc-run/rpc.sock"})
    )
    opened = []
    monkeypatch.setattr(
        websockets.sync.client,
        "connect",
        lambda url, **kw: opened.append((url, kw)) or connection,
    )
    result, path = env.open_code_rpc("run-token")
    assert requests == [("/code/capabilities", {"method": "GET"})]
    assert opened[0][0] == "ws://127.0.0.1:8643/internal/toolbox/code/rpc"
    assert opened[0][1]["additional_headers"] == {
        "Authorization": "Bearer pair-private",
        "X-Omnio-Brand": "brand-a",
    }
    assert connection.outgoing.get() == {"token": "run-token"}
    assert result is connection and path == "/tmp/code-rpc-run/rpc.sock"


def test_oversized_result_reports_error_without_losing_other_calls():
    def dispatch(tool, args):
        return (
            json.dumps({"value": "x" * (16 * 1024 * 1024)}) if args["large"] else "{}"
        )

    stream, connection = make_stream(dispatch)
    try:
        connection.call("large", large=True)
        result = connection.outgoing.get(timeout=5)
        assert "16 MiB" in json.loads(result["result"])["error"]
        connection.call("small", large=False)
        assert connection.outgoing.get(timeout=5)["result"] == "{}"
        assert not stream.errors
    finally:
        stream.close()


@pytest.mark.parametrize("policy", [{}, {"enabled": False}, {"enabled": "true"}, {"enabled": 1}])
def test_optional_channel_requires_an_explicit_boolean_rollout(policy):
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    requests = []
    env._request_json = lambda path, **kw: requests.append(path) or policy
    assert env.open_code_rpc("private", optional=True) is None
    assert requests == ["/code/rpc-policy"]


@pytest.mark.parametrize("transport", ["file", "auto"])
def test_compatible_transport_for_an_environment_without_stream_capability(transport):
    assert _open_remote_code_rpc(object(), "private", transport) == ("file", None, None)


@pytest.mark.parametrize("error", [ConnectionError, TimeoutError, ValueError])
def test_auto_fallback_is_confined_to_channel_setup(error):
    class Environment:
        def open_code_rpc(self, token, **kwargs):
            raise error("unavailable")

    assert _open_remote_code_rpc(Environment(), "private", "auto") == ("file", None, None)
    with pytest.raises(error):
        _open_remote_code_rpc(Environment(), "private", "stream")


def test_auto_transport_opens_one_channel_when_rollout_and_capability_allow_it():
    calls = []
    connection = object()

    class Environment:
        def open_code_rpc(self, token, **kwargs):
            calls.append((token, kwargs))
            return connection, "/tmp/code-rpc-owned/rpc.sock"

    assert _open_remote_code_rpc(Environment(), "private", "auto") == (
        "stream", connection, "/tmp/code-rpc-owned/rpc.sock",
    )
    assert calls == [("private", {"optional": True})]


def test_explicit_stream_rejects_an_unsupported_environment():
    with pytest.raises(ValueError, match="compatible Toolbox"):
        _open_remote_code_rpc(object(), "private", "stream")


def test_invalid_transport_fails_before_channel_setup():
    with pytest.raises(ValueError, match="rpc_transport"):
        _open_remote_code_rpc(object(), "private", "invalid")
