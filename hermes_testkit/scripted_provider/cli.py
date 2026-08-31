"""Command-line runner for the scripted provider."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

from .schema import SCRIPT_SCHEMA_VERSION, ScriptValidationError
from .server import ScriptedProviderServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hermes_testkit.scripted_provider",
        description="Run a strict, deterministic OpenAI-compatible test provider.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        help="JSON script to arm at startup (stdin when set to '-').",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: loopback)."
    )
    parser.add_argument(
        "--port", type=int, default=0, help="Bind port (0 chooses an ephemeral port)."
    )
    parser.add_argument(
        "--control-token",
        default=None,
        help="Bearer token for the loopback control API (or HERMES_SCRIPTED_PROVIDER_CONTROL_TOKEN).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional inference API key (or HERMES_SCRIPTED_PROVIDER_API_KEY).",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Write a JSON readiness record here without including the control token.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the readiness record to stdout.",
    )
    return parser


def _read_script(path: Path | None) -> Any:
    if path is None:
        return None
    if str(path) == "-":
        return json.load(sys.stdin)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        script = _read_script(args.script)
        # Environment wins so a shell-managed secret cannot be accidentally
        # overridden by a copied command-line argument.  The argument remains
        # for compatibility with existing Sprite launchers.
        token = (
            os.environ.get("HERMES_SCRIPTED_PROVIDER_CONTROL_TOKEN")
            or args.control_token
        )
        api_key = os.environ.get("HERMES_SCRIPTED_PROVIDER_API_KEY") or args.api_key
        server = ScriptedProviderServer(
            script,
            host=args.host,
            port=args.port,
            control_token=token,
            api_key=api_key,
        ).start()
    except (OSError, ValueError, ScriptValidationError, json.JSONDecodeError) as exc:
        print(f"scripted provider: {exc}", file=sys.stderr)
        return 2

    ready = {
        "url": server.url,
        "base_url": server.base_url,
        "port": server.port,
        "healthz": f"{server.url}/healthz",
        "models": f"{server.url}/v1/models",
        "chat_completions": f"{server.url}/v1/chat/completions",
        "control": f"{server.url}/__control",
        "schema_version": SCRIPT_SCHEMA_VERSION,
    }
    encoded_ready = json.dumps(ready, separators=(",", ":"), allow_nan=False)
    if args.ready_file is not None:
        try:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text(encoded_ready + "\n", encoding="utf-8")
        except OSError as exc:
            print(
                f"scripted provider: cannot write --ready-file: {exc}", file=sys.stderr
            )
            server.stop()
            return 2
    if not args.quiet:
        # Deliberately omit the control token.  Callers should supply and retain
        # it themselves instead of relying on logs to carry credentials.
        print(encoded_ready, flush=True)

    stop_event = threading.Event()

    def stop_handler(signum: int, frame: object) -> None:
        stop_event.set()

    previous_int = signal.signal(signal.SIGINT, stop_handler)
    previous_term = signal.getsignal(signal.SIGTERM)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)
    try:
        stop_event.wait()
    finally:
        server.stop()
        signal.signal(signal.SIGINT, previous_int)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, previous_term)
    return 0


__all__ = ["build_parser", "main"]
