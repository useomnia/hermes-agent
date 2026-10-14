"""A bounded, run-scoped tool dispatcher over the Toolbox pair WebSocket."""

import concurrent.futures
import json
import logging
import queue
import threading

from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)


STREAM_HEADER = '''\
"""Generated run-scoped tool stubs. Infrastructure credentials are not present."""
import json, os, socket, shlex, time

def _call(tool_name, args):
    request = {"token": os.environ["HERMES_RPC_TOKEN"], "tool": tool_name, "args": args}
    raw = json.dumps(request).encode() + b"\\n"
    if len(raw) > 1024 * 1024:
        raise ValueError("Tool request exceeds 1 MiB")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(300)
        client.connect(os.environ["HERMES_RPC_SOCKET"])
        client.sendall(raw)
        with client.makefile("rb") as reader:
            response = reader.readline(16 * 1024 * 1024 + 2)
    if not response or not response.endswith(b"\\n") or len(response) > 16 * 1024 * 1024 + 1:
        raise RuntimeError("Tool channel closed or result exceeded 16 MiB")
    envelope = json.loads(response)
    if "error" in envelope:
        raise RuntimeError(envelope["error"])
    return _decode_result(json.loads(envelope["result_json"]))

'''


class StreamDispatcher:
    def __init__(self, connection, dispatch, execution, *, concurrency, read_safe):
        if (
            not isinstance(concurrency, int)
            or isinstance(concurrency, bool)
            or not 1 <= concurrency <= 32
        ):
            raise ValueError("rpc_concurrency must be between 1 and 32")
        self.connection = connection
        self.dispatch = dispatch
        self.execution = execution
        self.concurrency = concurrency
        self.read_safe = read_safe
        self.completed = queue.Queue()
        self.pending = {}
        self.seen = set()
        self.errors = []
        self.thread = threading.Thread(
            target=propagate_context_to_thread(self.run), daemon=True
        )

    def start(self):
        self.thread.start()

    def _finish(self, item):
        request_id, future = item
        self.pending.pop(request_id, None)
        result = future.result()
        if not self.execution.is_cancelled():
            self._send_result(request_id, result)

    def _send_result(self, request_id, result):
        message = {"type": "result", "id": request_id, "result": result}
        raw = json.dumps(message)
        if len(raw.encode()) > 16 * 1024 * 1024:
            message["result"] = json.dumps({
                "error": "Tool result exceeds the 16 MiB channel limit."
            })
            raw = json.dumps(message)
        self.connection.send(raw)

    def _drain(self, *, wait):
        while self.pending:
            try:
                item = self.completed.get(timeout=0.1 if wait else 0)
            except queue.Empty:
                if self.execution.is_cancelled() or not wait:
                    return
                continue
            self._finish(item)
            if not wait:
                continue

    def run(self):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)
        try:
            while not self.execution.is_cancelled():
                self._drain(wait=False)
                if len(self.pending) >= self.concurrency:
                    try:
                        self._finish(self.completed.get(timeout=0.1))
                    except queue.Empty:
                        continue
                try:
                    raw = self.connection.recv(timeout=0.05)
                except TimeoutError:
                    continue
                request = json.loads(raw)
                if not isinstance(request, dict) or request.get("type") != "call":
                    raise ValueError("Invalid stream tool request")
                request_id, tool, args = (
                    request.get("id"),
                    request.get("tool"),
                    request.get("args"),
                )
                if (
                    not isinstance(request_id, str)
                    or len(request_id) > 64
                    or request_id in self.seen
                ):
                    raise ValueError("Invalid or duplicate request id")
                if not isinstance(tool, str) or not isinstance(args, dict):
                    raise ValueError("Invalid tool name or arguments")
                self.seen.add(request_id)
                if len(self.seen) > 10000:
                    raise ValueError("Too many requests on one code channel")
                if not self.read_safe(tool):
                    self._drain(wait=True)
                    if self.execution.is_cancelled():
                        break
                    result = self.dispatch(tool, args)
                    self._send_result(request_id, result)
                    continue
                callback = propagate_context_to_thread(self.dispatch)
                future = pool.submit(callback, tool, args)
                self.pending[request_id] = future
                future.add_done_callback(
                    lambda done, key=request_id: self.completed.put((key, done))
                )
        except Exception as exc:
            if not self.execution.is_cancelled():
                logger.error("execute_code stream failed", exc_info=True)
                self.errors.append(type(exc).__name__)
                self.execution.cancel()
        finally:
            for future in self.pending.values():
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            self.connection.close()

    def close(self):
        self.execution.cancel()
        self.connection.close()
        self.thread.join(timeout=5)
