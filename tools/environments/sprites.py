"""Omnio toolbox Sprite execution environment."""

import base64
import codecs
import http.client
import json
import logging
import posixpath
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle
from tools.environments.file_sync import (
    FileSyncManager, SPRITES_CACHE_FILE_MAX_BYTES, SPRITES_CACHE_ROOT, iter_sprites_cache_files,
    iter_sprites_sync_files,
)
from tools.file_operations import (
    PatchResult,
    ReadResult,
    SearchMatch,
    SearchResult,
    ShellFileOperations,
    WriteResult,
    _is_write_denied,
    _strip_bom,
)

logger = logging.getLogger(__name__)

_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_ERROR_BYTES = 4096
_MAX_SKILL_FILE_BYTES = 2 * 1024 * 1024
_MAX_SKILL_BATCH_FILES = 200
_MAX_SKILL_BATCH_BYTES = 16 * 1024 * 1024
_MAX_FILE_CONTENT_BYTES = 2 * 1024 * 1024
# Whole-file text reads above the JSON cap go through the raw stream up to
# this ceiling (matches the largest stdout recovery artifact); larger files
# must be paged with read_file offset/limit.
_MAX_RAW_TEXT_READ_BYTES = 5 * 1024 * 1024
_EXEC_PREDISPATCH_RETRY_DELAYS_SECONDS = (2.0, 4.0)
_EXEC_RETRY_MIN_REQUEST_BUDGET_SECONDS = 1.0
# Cache artifacts with these suffixes are text the model or a worker wrote;
# they are secret-redacted before they land on the Toolbox.
_REDACTED_CACHE_SUFFIXES = frozenset({
    ".txt", ".log", ".json", ".md", ".csv", ".html", ".htm", ".xml", ".yaml", ".yml",
})


def _is_toolbox_cache_path(path: str) -> bool:
    """True for paths inside the projected harness cache on the Toolbox."""
    return path == SPRITES_CACHE_ROOT or path.startswith(SPRITES_CACHE_ROOT + "/")


_STREAM_CHUNK_BYTES = 1024 * 1024


def _expand_toolbox_home(path: str) -> str:
    """Expand bare user-home paths in the Toolbox virtual namespace."""
    if path == "~":
        return "/home"
    if path.startswith("~/"):
        return f"/home{path[1:]}"
    return path


def _canonicalize_toolbox_path(path: str) -> str:
    """Normalize absolute Toolbox paths without consulting the host filesystem."""
    expanded = _expand_toolbox_home(path)
    if expanded.startswith("/"):
        return posixpath.normpath(expanded)
    return expanded


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the pair bearer on the configured toolbox origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_URL_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _normalize_toolbox_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OMNIO_TOOLBOX_URL must be an HTTP(S) origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OMNIO_TOOLBOX_URL must not contain credentials, a query, or a fragment")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("OMNIO_TOOLBOX_URL must use HTTPS outside loopback")
    # A base path is allowed so the gateway can point at the proxy's
    # authenticated loopback Toolbox forwarder (e.g. .../internal/toolbox);
    # per-endpoint paths like /exec are appended to it. A bare origin (no path)
    # remains valid, so a gateway paired with a proxy that forwards directly is
    # unaffected. The trailing slash is stripped so f"{base}{path}" never
    # produces a doubled separator.
    base_path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))


class SpritesToolboxError(RuntimeError):
    """Raised when the Omnio toolbox API rejects a request."""

    code: str | None
    phase: str | None
    retryable: bool
    command_started: bool | None
    request_id: str | None
    http_status: int | None
    detail: str | None
    request_cwd: str | None

    def __init__(
        self,
        message: str,
        *,
        detail: str | None = None,
        code: str | None = None,
        phase: str | None = None,
        retryable: bool = False,
        command_started: bool | None = None,
        request_id: str | None = None,
        http_status: int | None = None,
        request_cwd: str | None = None,
    ):
        self.detail = detail
        self.code = code
        self.phase = phase
        self.retryable = retryable
        self.command_started = command_started
        self.request_id = request_id
        self.http_status = http_status
        self.request_cwd = request_cwd

        details = []
        if code:
            details.append(f"code={code}")
        if phase:
            details.append(f"phase={phase}")
        if command_started is not None:
            details.append(f"commandStarted={str(command_started).lower()}")
        if request_id:
            details.append(f"requestId={request_id}")
        if details:
            message = f"{message} ({', '.join(details)})"

        super().__init__(message)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _toolbox_error_from_payload(
    path: str,
    payload: dict[str, Any],
    *,
    http_status: int | None = None,
    fallback_message: str = "Toolbox request failed",
    raw_body: str | None = None,
    request_cwd: str | None = None,
) -> SpritesToolboxError:
    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        message = (
            _optional_string(error_payload.get("message"))
            or _optional_string(payload.get("detail"))
            or fallback_message
        )
        command_started_value = error_payload.get("commandStarted")
        command_started = (
            command_started_value if isinstance(command_started_value, bool) else None
        )
        code = _optional_string(error_payload.get("code"))
        phase = _optional_string(error_payload.get("phase"))
        retryable = error_payload.get("retryable") is True
        request_id = _optional_string(error_payload.get("requestId"))
    else:
        message = (
            _optional_string(error_payload)
            or _optional_string(payload.get("detail"))
            or fallback_message
        )
        command_started = None
        code = None
        phase = None
        retryable = False
        request_id = None

    prefix = f"Toolbox API {path}"
    if http_status is not None:
        prefix += f" failed with HTTP {http_status}"

    return SpritesToolboxError(
        f"{prefix}: {raw_body if raw_body is not None else message}",
        detail=message,
        code=code,
        phase=phase,
        retryable=retryable,
        command_started=command_started,
        request_id=request_id,
        http_status=http_status,
        request_cwd=request_cwd,
    )


def _should_retry_exec_predispatch(error: SpritesToolboxError) -> bool:
    return (
        error.http_status == 503
        and error.retryable is True
        and error.command_started is False
    )


def render_sprites_toolbox_error(
    error: SpritesToolboxError,
    *,
    service: str,
    action: str,
    context: str,
) -> str:
    """Render a concise model-facing error while preserving ``error`` for logs."""
    status = error.http_status
    if status is not None and 400 <= status < 500:
        detail = error.detail or "the toolbox rejected the request"
        return f"{action} not run: {context} - {detail}"

    if status is None or status >= 500:
        code = f", code={error.code}" if error.code else ""
        return (
            f"{service} temporarily unavailable "
            f"(infrastructure issue{code}); retry shortly"
        )

    detail = error.detail or "the toolbox rejected the request"
    return f"{action} not run: {context} - {detail}"


class SpritesEnvironment(BaseEnvironment):
    """Run commands through the paired Omnio toolbox Sprite API."""

    _stdin_mode = "heredoc"

    def __init__(
        self,
        toolbox_url: str,
        bearer_token: str,
        brand: str,
        cwd: str = "/brand",
        timeout: int = 60,
    ):
        if not toolbox_url:
            raise ValueError("Sprites environment requires OMNIO_TOOLBOX_URL")
        if not bearer_token:
            raise ValueError("Sprites environment requires OMNIO_TOOLBOX_BEARER")
        if not brand:
            raise ValueError("Sprites environment requires OMNIO_TOOLBOX_BRAND")

        super().__init__(cwd=cwd, timeout=timeout)
        self.toolbox_url = _normalize_toolbox_url(toolbox_url)
        self.bearer_token = bearer_token
        self.brand = brand

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sprites_sync_files("/skills"),
            upload_fn=self._sprites_upload,
            delete_fn=self._sprites_delete,
            bulk_upload_fn=self._sprites_bulk_upload,
        )
        # The harness cache set (web pages, screenshots, worker artifacts,
        # stdout recovery, ...) is projected under SPRITES_CACHE_ROOT so the
        # paths `to_agent_visible_cache_path` publishes are readable on the
        # Toolbox. Synced before every command and before reads under it.
        self._cache_sync_lock = threading.Lock()
        self._cache_sync_manager = FileSyncManager(
            get_files_fn=iter_sprites_cache_files,
            upload_fn=self._upload_cache_file,
            delete_fn=self._delete_cache_files,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: int | None = None,
        method: str = "POST",
        retry_exec_predispatch: bool = False,
        retry_deadline_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "X-Omnio-Brand": self.brand,
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if request_id is not None:
            headers["X-Request-Id"] = request_id

        retry_count = 0
        retry_deadline = (
            time.monotonic() + retry_deadline_seconds
            if retry_deadline_seconds is not None
            else None
        )
        request_cwd = (
            _optional_string(payload.get("cwd"))
            if path == "/exec" and isinstance(payload, dict)
            else None
        )
        while True:
            request = urllib.request.Request(
                f"{self.toolbox_url}{path}",
                data=data,
                headers=headers,
                method=method,
            )
            try:
                with _URL_OPENER.open(request, timeout=timeout or self.timeout) as response:
                    raw_body = response.read(_MAX_RESPONSE_BYTES + 1)
                break
            except urllib.error.HTTPError as exc:
                body = exc.read(_MAX_ERROR_BYTES).decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    error = _toolbox_error_from_payload(
                        path,
                        parsed,
                        http_status=exc.code,
                        fallback_message=body or str(exc.reason),
                        raw_body=body,
                        request_cwd=request_cwd,
                    )
                else:
                    detail = body or str(exc.reason)
                    error = SpritesToolboxError(
                        f"Toolbox API {path} failed with HTTP {exc.code}: "
                        f"{detail}",
                        detail=detail,
                        http_status=exc.code,
                        request_cwd=request_cwd,
                    )

                if (
                    retry_exec_predispatch
                    and path == "/exec"
                    and retry_count < len(_EXEC_PREDISPATCH_RETRY_DELAYS_SECONDS)
                    and _should_retry_exec_predispatch(error)
                ):
                    delay = _EXEC_PREDISPATCH_RETRY_DELAYS_SECONDS[retry_count]
                    if cancel_event is not None and cancel_event.is_set():
                        raise error from exc
                    if (
                        retry_deadline is not None
                        and time.monotonic()
                        + delay
                        + _EXEC_RETRY_MIN_REQUEST_BUDGET_SECONDS
                        > retry_deadline
                    ):
                        raise error from exc

                    retry_count += 1
                    logger.warning(
                        "Retrying Toolbox API %s after retryable pre-dispatch HTTP 503 "
                        "in %.1fs (retry %d/%d, code=%s, phase=%s, requestId=%s)",
                        path,
                        delay,
                        retry_count,
                        len(_EXEC_PREDISPATCH_RETRY_DELAYS_SECONDS),
                        error.code,
                        error.phase,
                        error.request_id,
                    )
                    if cancel_event is None:
                        time.sleep(delay)
                    elif cancel_event.wait(delay):
                        raise error from exc
                    if cancel_event is not None and cancel_event.is_set():
                        raise error from exc
                    if (
                        retry_deadline is not None
                        and time.monotonic()
                        + _EXEC_RETRY_MIN_REQUEST_BUDGET_SECONDS
                        > retry_deadline
                    ):
                        raise error from exc
                    continue

                raise error from exc
            except urllib.error.URLError as exc:
                raise SpritesToolboxError(
                    f"Toolbox API {path} is unreachable: {exc}"
                ) from exc
            except (OSError, http.client.HTTPException) as exc:
                raise SpritesToolboxError(
                    f"Toolbox API {path} is unreachable: {exc}"
                ) from exc

        if len(raw_body) > _MAX_RESPONSE_BYTES:
            raise SpritesToolboxError(
                f"Toolbox API {path} response exceeded {_MAX_RESPONSE_BYTES} bytes"
            )
        body = raw_body.decode("utf-8")

        if not body:
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SpritesToolboxError(
                f"Toolbox API {path} returned invalid JSON: {body[:200]}"
            ) from exc
        if not isinstance(parsed, dict):
            raise SpritesToolboxError(f"Toolbox API {path} returned a non-object response")
        return parsed

    def file_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a file operation to the toolbox Sprite."""
        return self._request_json("/files", payload)

    def open_code_rpc(self, token: str, *, optional: bool = False):
        """Open one authenticated channel bound to this environment's Brand."""
        from websockets.sync.client import connect

        if optional:
            policy = self._request_json("/code/rpc-policy", method="GET")
            if policy.get("enabled") is not True:
                return None
        capability = self._request_json("/code/capabilities", method="GET")
        if capability.get("rpc") != 1:
            raise SpritesToolboxError("Toolbox does not support code RPC v1")
        url = self.toolbox_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        connection = connect(
            f"{url}/code/rpc",
            additional_headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "X-Omnio-Brand": self.brand,
            },
            open_timeout=10, close_timeout=2, max_size=16 * 1024 * 1024,
            proxy=None,
        )
        try:
            connection.send(json.dumps({"token": token}))
            ready = json.loads(connection.recv(timeout=15))
            path = ready.get("socket", "")
            if ready.get("type") != "ready" or not isinstance(path, str) or not path.startswith("/tmp/code-rpc-"):
                raise SpritesToolboxError("Toolbox returned invalid code RPC readiness")
            return connection, path
        except Exception:
            connection.close()
            raise

    def read_file_bytes(self, path: str, *, max_bytes: int) -> bytes:
        """Read at most ``max_bytes`` from a Toolbox file as raw bytes."""
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        path = _canonicalize_toolbox_path(path)
        self.sync_projected_path(path)
        query = urllib.parse.urlencode({"path": path})
        request = urllib.request.Request(
            f"{self.toolbox_url}/files?{query}",
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "X-Omnio-Brand": self.brand,
            },
            method="GET",
        )
        try:
            with _URL_OPENER.open(request, timeout=self.timeout) as response:
                return response.read(max_bytes)
        except urllib.error.HTTPError as exc:
            detail = exc.read(_MAX_ERROR_BYTES).decode("utf-8", errors="replace")
            raise SpritesToolboxError(
                f"Toolbox API /files failed with HTTP {exc.code}: "
                f"{detail or exc.reason}",
                detail=detail or str(exc.reason),
                http_status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise SpritesToolboxError(
                f"Toolbox API /files is unreachable: {exc}"
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise SpritesToolboxError(
                f"Toolbox API /files is unreachable: {exc}"
            ) from exc

    def stream_file_bytes(
        self, path: str, *, chunk_size: int = _STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        """Yield a Toolbox file's raw bytes in bounded chunks.

        The raw ``GET /files`` route streams any size; consuming it in chunks
        keeps memory bounded to the caller's window, which is what lets file
        tools page through text the JSON read refuses as too large.
        """
        path = _canonicalize_toolbox_path(path)
        self.sync_projected_path(path)
        query = urllib.parse.urlencode({"path": path})
        request = urllib.request.Request(
            f"{self.toolbox_url}/files?{query}",
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "X-Omnio-Brand": self.brand,
            },
            method="GET",
        )
        try:
            with _URL_OPENER.open(request, timeout=self.timeout) as response:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        return
                    yield chunk
        except urllib.error.HTTPError as exc:
            detail = exc.read(_MAX_ERROR_BYTES).decode("utf-8", errors="replace")
            raise SpritesToolboxError(
                f"Toolbox API /files failed with HTTP {exc.code}: "
                f"{detail or exc.reason}",
                detail=detail or str(exc.reason),
                http_status=exc.code,
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise SpritesToolboxError(
                f"Toolbox API /files is unreachable: {exc}"
            ) from exc

    def get_temp_dir(self) -> str:
        return "/tmp/.hermes-session"

    def write_file_content(self, path: str, content: str) -> bool:
        """Write UTF-8 content, using atomic raw upload above the JSON limit."""
        path = _canonicalize_toolbox_path(path)
        if _is_write_denied(path):
            raise SpritesToolboxError(f"Write denied: {path} is a protected path")

        data = content.encode("utf-8")
        if len(data) > SPRITES_CACHE_FILE_MAX_BYTES:
            raise SpritesToolboxError(
                f"Toolbox API /files write content exceeded "
                f"{SPRITES_CACHE_FILE_MAX_BYTES} bytes ({len(data)} bytes)"
            )

        if len(data) > _MAX_FILE_CONTENT_BYTES:
            self._write_raw_artifact(path, data)
            return True

        response = self.file_request(
            {
                "operation": "write",
                "path": path,
                "content": content,
                "encoding": "utf-8",
            }
        )
        return not bool(response.get("error"))

    def _sprites_upload(self, host_path: str, remote_path: str) -> None:
        encoded, _size = self._encoded_skill(host_path, remote_path)
        self.file_request(
            {
                "operation": "writeSkills",
                "path": remote_path,
                "contentBase64": encoded,
                "encoding": "base64",
            }
        )

    def _sprites_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        batch: list[dict[str, str]] = []
        batch_bytes = 0
        for host_path, remote_path in files:
            encoded, size = self._encoded_skill(host_path, remote_path)
            if batch and (
                len(batch) >= _MAX_SKILL_BATCH_FILES
                or batch_bytes + size > _MAX_SKILL_BATCH_BYTES
            ):
                self.file_request({"operation": "writeSkills", "files": batch})
                batch = []
                batch_bytes = 0
            batch.append(
                {
                    "path": remote_path,
                    "contentBase64": encoded,
                    "encoding": "base64",
                }
            )
            batch_bytes += size
        if batch:
            self.file_request({"operation": "writeSkills", "files": batch})

    @staticmethod
    def _encoded_skill(host_path: str, remote_path: str) -> tuple[str, int]:
        if remote_path != "/skills" and not remote_path.startswith("/skills/"):
            raise SpritesToolboxError(
                f"Sprites skills sync refused non-skills path: {remote_path}"
            )
        source = Path(host_path)
        source_stat = source.lstat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise SpritesToolboxError(f"Sprites skills sync refused non-file: {host_path}")
        if source_stat.st_size > _MAX_SKILL_FILE_BYTES:
            raise SpritesToolboxError(
                f"Sprites skill file exceeds {_MAX_SKILL_FILE_BYTES} bytes: {host_path}"
            )
        content = source.read_bytes()
        return base64.b64encode(content).decode("ascii"), len(content)

    def _sprites_delete(self, remote_paths: list[str]) -> None:
        for remote_path in remote_paths:
            if remote_path != "/skills" and not remote_path.startswith("/skills/"):
                raise SpritesToolboxError(
                    f"Sprites skills sync refused to delete non-skills path: {remote_path}"
                )
            self.file_request({"operation": "deleteSkills", "path": remote_path, "missingOk": True})

    def _upload_cache_file(self, host_path: str, remote_path: str) -> None:
        """Project one harness cache file onto the Toolbox.

        Text artifacts (worker transcripts, stored pages, stdout recovery) are
        secret-redacted first: the Toolbox is readable by model-authored code,
        and these files carry exactly the data that tends to hold keys. Media
        crosses byte-for-byte through the raw endpoint.
        """
        remote_path = _canonicalize_toolbox_path(remote_path)
        if not _is_toolbox_cache_path(remote_path):
            raise SpritesToolboxError("Refused artifact outside the Toolbox cache")
        source = Path(host_path)
        if source.is_symlink() or not source.is_file():
            raise SpritesToolboxError(f"Refused non-file cache artifact: {host_path}")
        data = source.read_bytes()
        if source.suffix.lower() in _REDACTED_CACHE_SUFFIXES:
            from tools.delegation_live_log import _redact

            data = _redact(data.decode("utf-8", errors="replace")).encode("utf-8")
        self._write_raw_artifact(remote_path, data)

    def _write_raw_artifact(self, path: str, data: bytes) -> None:
        """The atomic raw-file endpoint: binary-safe, no JSON write cap."""
        query = urllib.parse.urlencode({
            "path": path, "overwrite": "true", "maxBytes": len(data),
        })
        request = urllib.request.Request(
            f"{self.toolbox_url}/files?{query}", data=data, method="PUT",
            headers={"Authorization": f"Bearer {self.bearer_token}",
                     "X-Omnio-Brand": self.brand,
                     "Content-Type": "application/octet-stream"},
        )
        try:
            with _URL_OPENER.open(request, timeout=self.timeout) as response:
                result = json.loads(response.read(_MAX_RESPONSE_BYTES + 1))
            if result.get("bytesWritten") != len(data):
                raise SpritesToolboxError("Artifact transfer was incomplete")
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise SpritesToolboxError("Artifact transfer failed") from exc

    def _delete_cache_files(self, paths: list[str]) -> None:
        for path in paths:
            path = _canonicalize_toolbox_path(path)
            if not _is_toolbox_cache_path(path):
                raise SpritesToolboxError("Refused artifact outside the Toolbox cache")
            result = self.file_request({"operation": "delete", "path": path, "missingOk": True})
            if result.get("error"):
                raise SpritesToolboxError("Artifact removal failed")

    def sync_cache_files(self, *, raise_on_error: bool = True) -> None:
        """Push new or changed harness cache files to the Toolbox now.

        Forced (not rate-limited): callers invoke it right before a read under
        :data:`SPRITES_CACHE_ROOT`, where a stale copy would be a wrong answer.
        """
        manager = getattr(self, "_cache_sync_manager", None)
        if manager is not None:
            with self._cache_sync_lock:
                manager.sync(force=True, raise_on_error=raise_on_error)

    def sync_skill_files(self) -> None:
        """Refresh helper files before use, including inside the sync throttle.

        The manager still uploads only changed files. Serialize its state with
        other projection work and propagate failure rather than executing an
        old helper after an unsuccessful transfer.
        """
        with self._cache_sync_lock:
            self._sync_manager.sync(force=True, raise_on_error=True)

    def sync_projected_path(self, path: str) -> None:
        """Refresh host-owned projections before a Toolbox file consumer."""
        path = _canonicalize_toolbox_path(path)
        if path == "/skills" or path.startswith("/skills/"):
            self.sync_skill_files()
        elif _is_toolbox_cache_path(path):
            self.sync_cache_files()

    def _before_execute(self) -> None:
        self.sync_skill_files()
        self.sync_cache_files(raise_on_error=False)

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
        cwd: str | None = None,
    ):
        request_cwd = cwd or self.cwd
        cancel_event = threading.Event()
        request_id = str(uuid.uuid4())

        def exec_fn() -> tuple[str, int]:
            response = self._request_json(
                "/exec",
                {
                    "command": cmd_string,
                    "cwd": request_cwd,
                    "login": login,
                    "stdin": stdin_data,
                    "timeoutSeconds": timeout,
                },
                timeout=timeout + 5,
                retry_exec_predispatch=True,
                retry_deadline_seconds=timeout,
                cancel_event=cancel_event,
                request_id=request_id,
            )
            output = response.get("output", "")
            exit_code = response.get("returncode", response.get("exitCode", 0))
            if (
                "error" in response
                and response["error"] is not None
                and response["error"] != ""
            ):
                raise _toolbox_error_from_payload(
                    "/exec",
                    response,
                    request_cwd=request_cwd,
                )
            return (str(output), int(exit_code))

        def cancel_fn() -> None:
            cancel_event.set()
            try:
                self._request_json(
                    "/exec/cancel",
                    timeout=5,
                    request_id=request_id,
                )
            except SpritesToolboxError:
                # Toolbox cancellation is additive. Older paired runtimes do
                # not expose the endpoint, so preserve the historical local
                # interrupt while the fleet rolls forward.
                logger.debug(
                    "Toolbox exec cancellation endpoint unavailable",
                    exc_info=True,
                )

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel_fn)

    def _run_bash_with_cwd(
        self,
        cmd_string: str,
        cwd: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        """Pass the per-command cwd through the Toolbox `/exec` boundary.

        Sprites share one persistent environment across Omnio runs.  The
        mutable ``self.cwd`` is therefore the last session's cwd, and may
        name a workspace that has already been cleaned up.  The shell wrapper
        also receives *cwd*, but the Toolbox validates its request cwd before
        running that wrapper, so it must get the same per-command value here.
        """
        return self._run_bash(
            cmd_string,
            cwd=cwd,
            login=login,
            timeout=timeout,
            stdin_data=stdin_data,
        )

    def cleanup(self):
        return None


class SpritesFileOperations(ShellFileOperations):
    """File tools backed by the toolbox Sprite `/files` endpoint."""

    def __init__(self, terminal_env: SpritesEnvironment):
        super().__init__(terminal_env)
        self.env: SpritesEnvironment = terminal_env

    def _expand_path(self, path: str) -> str:
        """Expand user-home paths in the Toolbox's virtual namespace.

        The Toolbox file API speaks the Sprite filesystem, where ``/home`` is
        the virtual home and ``/home/brand`` is the durable brand mount. The
        Sprite process itself runs as ``/home/sprite``; delegating ``~`` to
        ``ShellFileOperations`` would therefore route ``~/brand`` into scratch
        instead of the mounted brand tree. Keep explicit and non-tilde paths
        unchanged, and retain the base implementation for ``~user`` paths.
        """
        expanded = (
            _expand_toolbox_home(path)
            if path == "~" or path.startswith("~/")
            else super()._expand_path(path)
        )
        return _canonicalize_toolbox_path(expanded)

    def _files(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The Toolbox file API does not perform shell expansion. Resolve all
        # path-bearing fields before delegation so single-path and multi-path
        # operations use the same Sprite virtual namespace.
        payload = dict(payload)
        for key in ("path", "src", "dst"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("~"):
                payload[key] = self._expand_path(value)
        try:
            read_path = _canonicalize_toolbox_path(str(payload.get("path", "")))
            if (payload.get("operation") in {"read", "readRaw", "search", "stat"}
                    and (read_path == "/skills" or read_path.startswith("/skills/")
                         or _is_toolbox_cache_path(read_path))):
                self.env.sync_projected_path(read_path)
            return self.env.file_request(payload)
        except SpritesToolboxError as error:
            operation = str(payload.get("operation", "unknown"))
            path = payload.get("path")
            if path is None and ("src" in payload or "dst" in payload):
                path = f"{payload.get('src')} -> {payload.get('dst')}"
            context = (
                f"path {path!r}"
                if path is not None
                else f"cwd {self.env.cwd!r}"
            )
            logger.error(
                "Toolbox file operation failed - Operation: %s - Error: %s: %s",
                operation,
                type(error).__name__,
                error,
            )
            return {
                "error": render_sprites_toolbox_error(
                    error,
                    service="file tools",
                    action="file operation",
                    context=context,
                )
            }

    def read_file(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        from tools.file_operations import normalize_read_pagination

        offset, limit = normalize_read_pagination(offset, limit)
        response = self._files(
            {"operation": "read", "path": path, "offset": offset, "limit": limit}
        )
        if error := response.get("error"):
            if str(error).startswith("File too large"):
                return self._read_large_text_page(path, offset, limit)
            return ReadResult(error=str(error), similar_files=response.get("similarFiles", []))
        content = str(response.get("content", ""))
        if not response.get("lineNumbered", False):
            content = self._add_line_numbers(content, offset)
        return ReadResult(
            content=content,
            total_lines=int(response.get("totalLines", 0)),
            file_size=int(response.get("fileSize", 0)),
            truncated=bool(response.get("truncated", False)),
            hint=response.get("hint"),
            is_binary=bool(response.get("isBinary", False)),
            is_image=bool(response.get("isImage", False)),
        )

    def read_file_raw(self, path: str) -> ReadResult:
        response = self._files({"operation": "readRaw", "path": path})
        if error := response.get("error"):
            if str(error).startswith("File too large"):
                # The JSON read caps at the Toolbox's 2 MiB request size. Text
                # artifacts above it (a stdout recovery file, a long worker
                # transcript) are still whole-readable through the bounded raw
                # stream; beyond that ceiling the caller must page with
                # read_file offset/limit.
                return self._read_large_text(path)
            return ReadResult(error=str(error), similar_files=response.get("similarFiles", []))
        content = str(response.get("content", ""))
        content, _ = _strip_bom(content)
        return ReadResult(content=content, file_size=int(response.get("fileSize", 0)))

    def _read_large_text_page(self, path: str, offset: int, limit: int) -> ReadResult:
        """Page a text file the Toolbox's JSON ``read`` refuses as too large.

        Same window semantics and hint text as the Toolbox's own paged read,
        computed from the raw stream one chunk at a time: only the requested
        window is held in memory, so a file of any size pages, and the
        ``offset/limit`` recipe the whole-read error gives out always works.
        """
        from tools.tool_output_limits import get_max_line_length

        expanded = self._expand_path(path)
        start = max(1, offset)
        end = start + limit - 1
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        window: list[str] = []
        pending = ""
        # Retain only the displayable prefix, including room for a BOM and
        # one over-limit character so _add_line_numbers marks truncation.
        # A minified document must not grow this buffer to the file's size.
        prefix_limit = get_max_line_length() + 2
        line_no = 0
        size = 0

        def take(line: str) -> None:
            nonlocal line_no
            line_no += 1
            if start <= line_no <= end:
                window.append(line.rstrip("\r"))

        try:
            for chunk in self.env.stream_file_bytes(expanded):
                size += len(chunk)
                pieces = decoder.decode(chunk).split("\n")
                for piece in pieces[:-1]:
                    take(pending + piece[:prefix_limit - len(pending)])
                    pending = ""
                pending += pieces[-1][:prefix_limit - len(pending)]
            pending += decoder.decode(b"", final=True)[:prefix_limit - len(pending)]
            if pending:
                take(pending)
        except SpritesToolboxError as error:
            return ReadResult(
                error=render_sprites_toolbox_error(
                    error, service="file tools", action="file read",
                    context=f"path {expanded!r}",
                )
            )
        total = line_no
        if start > total:
            return ReadResult(
                error=f"Offset {start} is beyond the end of the file ({total} lines): {expanded}",
                total_lines=total, file_size=size,
            )
        if start == 1 and window:
            window[0], _ = _strip_bom(window[0])
        shown_end = min(end, total)
        truncated = total > shown_end
        return ReadResult(
            content=self._add_line_numbers("\n".join(window), start),
            total_lines=total,
            file_size=size,
            truncated=truncated,
            hint=f"Use offset={shown_end + 1} to continue reading" if truncated else None,
        )

    def _read_large_text(self, path: str) -> ReadResult:
        """Whole-file text read through the raw stream, bounded by
        ``_MAX_RAW_TEXT_READ_BYTES``; larger files are paged instead."""
        expanded = self._expand_path(path)
        chunks: list[bytes] = []
        size = 0
        try:
            for chunk in self.env.stream_file_bytes(expanded):
                size += len(chunk)
                if size > _MAX_RAW_TEXT_READ_BYTES:
                    return ReadResult(
                        error=(
                            f"File exceeds {_MAX_RAW_TEXT_READ_BYTES} bytes: {expanded}. "
                            "Use read_file with offset/limit to page through it."
                        )
                    )
                chunks.append(chunk)
        except SpritesToolboxError as error:
            return ReadResult(
                error=render_sprites_toolbox_error(
                    error, service="file tools", action="file read",
                    context=f"path {expanded!r}",
                )
            )
        data = b"".join(chunks)
        content, _ = _strip_bom(data.decode("utf-8", errors="replace"))
        return ReadResult(content=content, file_size=len(data))

    def write_file(self, path: str, content: str) -> WriteResult:
        path = self._expand_path(path)
        if _is_write_denied(path):
            return WriteResult(error=f"Write denied: '{path}' is a protected system/credential file.")

        previous = self.read_file_raw(path)
        pre_content = None if previous.error else previous.content
        response = self._files(
            {"operation": "write", "path": path, "content": content, "encoding": "utf-8"}
        )
        if error := response.get("error"):
            return WriteResult(error=str(error))

        lint_result = self._check_lint_delta(path, pre_content=pre_content, post_content=content)
        return WriteResult(
            bytes_written=int(response.get("bytesWritten", len(content.encode("utf-8")))),
            dirs_created=bool(response.get("dirsCreated", False)),
            lint=lint_result.to_dict() if lint_result else None,
        )

    def delete_file(self, path: str) -> WriteResult:
        return self.delete_path(path, recursive=False)

    def delete_path(self, path: str, recursive: bool = False) -> WriteResult:
        path = self._expand_path(path)
        if _is_write_denied(path):
            return WriteResult(error=f"Delete denied: {path} is a protected path")
        response = self._files(
            {"operation": "delete", "path": path, "recursive": recursive, "missingOk": True}
        )
        if error := response.get("error"):
            return WriteResult(error=str(error))
        return WriteResult()

    def move_file(self, src: str, dst: str) -> WriteResult:
        src = self._expand_path(src)
        dst = self._expand_path(dst)
        for path in (src, dst):
            if _is_write_denied(path):
                return WriteResult(error=f"Move denied: {path} is a protected path")
        response = self._files({"operation": "move", "src": src, "dst": dst})
        if error := response.get("error"):
            return WriteResult(error=str(error))
        return WriteResult()

    def patch_v4a(self, patch_content: str) -> PatchResult:
        return super().patch_v4a(patch_content)

    def search(
        self,
        pattern: str,
        path: str = ".",
        target: str = "content",
        file_glob: str | None = None,
        limit: int = 50,
        offset: int = 0,
        output_mode: str = "content",
        context: int = 0,
    ) -> SearchResult:
        from tools.file_operations import normalize_search_pagination

        offset, limit = normalize_search_pagination(offset, limit)
        response = self._files(
            {
                "operation": "search",
                "pattern": pattern,
                "path": path,
                "target": target,
                "fileGlob": file_glob,
                "limit": limit,
                "offset": offset,
                "outputMode": output_mode,
                "context": context,
            }
        )
        if error := response.get("error"):
            return SearchResult(error=str(error), total_count=0)
        matches = [
            SearchMatch(
                path=str(match.get("path", "")),
                line_number=int(match.get("line", match.get("lineNumber", 0))),
                content=str(match.get("content", "")),
            )
            for match in response.get("matches", [])
        ]
        return SearchResult(
            matches=matches,
            files=[str(path) for path in response.get("files", [])],
            counts={str(key): int(value) for key, value in response.get("counts", {}).items()},
            total_count=int(response.get("totalCount", len(matches))),
            truncated=bool(response.get("truncated", False)),
            limit_reason=response.get("limitReason"),
            warning=response.get("warning"),
        )
