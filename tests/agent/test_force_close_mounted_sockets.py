"""A stranger-thread abort must reach the socket of a stream on Hermes' own client.

Without a proxy, ``_build_keepalive_http_client`` mounts plain transports for
``http://`` and ``https://``, so live connections sit on ``client._mounts``
rather than ``client._transport``. A socket walk that only reads the default
transport shuts down nothing, and every stale-stream kill and interrupt abort
leaves the stream running until the provider ends it.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from agent.agent_runtime_helpers import force_close_tcp_sockets
from run_agent import AIAgent


class _EndlessStream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        deadline = time.time() + 30
        try:
            while time.time() < deadline:
                self.wfile.write(b"data: {}\n\n")
                self.wfile.flush()
                time.sleep(0.01)
        except OSError:
            pass


def test_abort_shuts_down_the_socket_of_a_stream_on_a_mounted_transport(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EndlessStream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    http_client = AIAgent._build_keepalive_http_client(base_url)
    assert http_client._mounts, "expected the no-proxy client to route through mounts"

    streaming = threading.Event()
    finished = {}

    def read_stream():
        started = time.time()
        try:
            with http_client.stream("GET", base_url) as response:
                for _ in response.iter_bytes():
                    streaming.set()
        except Exception as exc:
            finished["error"] = type(exc).__name__
        finished["elapsed"] = time.time() - started

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    try:
        assert streaming.wait(5)
        assert force_close_tcp_sockets(SimpleNamespace(_client=http_client)) >= 1
        reader.join(5)
        assert not reader.is_alive(), "stream kept reading after the abort"
        assert finished["elapsed"] < 5
    finally:
        server.shutdown()
        http_client.close()
