"""Omnio toolbox Sprite execution environment."""

import base64
import json
import logging
import stat
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle
from tools.environments.file_sync import FileSyncManager, iter_sprites_sync_files
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
    if parsed.path not in {"", "/"}:
        raise ValueError("OMNIO_TOOLBOX_URL must not contain a path")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("OMNIO_TOOLBOX_URL must use HTTPS outside loopback")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


class SpritesToolboxError(RuntimeError):
    """Raised when the Omnio toolbox API rejects a request."""


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
        self._sync_manager.sync(force=True)
        self.init_session()

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: int | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        data = None
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "X-Omnio-Brand": self.brand,
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            f"{self.toolbox_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with _URL_OPENER.open(request, timeout=timeout or self.timeout) as response:
                raw_body = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            body = exc.read(_MAX_ERROR_BYTES).decode("utf-8", errors="replace")
            message = body
            try:
                parsed = json.loads(body)
                message = str(parsed.get("error") or parsed.get("detail") or body)
            except json.JSONDecodeError:
                pass
            raise SpritesToolboxError(
                f"Toolbox API {path} failed with HTTP {exc.code}: {message}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SpritesToolboxError(f"Toolbox API {path} is unreachable: {exc}") from exc

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

    def get_temp_dir(self) -> str:
        return "/tmp/.hermes-session"

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

    def _before_execute(self) -> None:
        self._sync_manager.sync()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        def exec_fn() -> tuple[str, int]:
            response = self._request_json(
                "/exec",
                {
                    "command": cmd_string,
                    "cwd": self.cwd,
                    "login": login,
                    "stdin": stdin_data,
                    "timeoutSeconds": timeout,
                },
                timeout=timeout + 5,
            )
            output = response.get("output", "")
            exit_code = response.get("returncode", response.get("exitCode", 0))
            return (str(output), int(exit_code))

        return _ThreadedProcessHandle(exec_fn)

    def cleanup(self):
        return None


class SpritesFileOperations(ShellFileOperations):
    """File tools backed by the toolbox Sprite `/files` endpoint."""

    def __init__(self, terminal_env: SpritesEnvironment):
        super().__init__(terminal_env)
        self.env: SpritesEnvironment = terminal_env

    def _files(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.env.file_request(payload)

    def read_file(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        from tools.file_operations import normalize_read_pagination

        offset, limit = normalize_read_pagination(offset, limit)
        response = self._files(
            {"operation": "read", "path": path, "offset": offset, "limit": limit}
        )
        if error := response.get("error"):
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
            return ReadResult(error=str(error), similar_files=response.get("similarFiles", []))
        content = str(response.get("content", ""))
        content, _ = _strip_bom(content)
        return ReadResult(content=content, file_size=int(response.get("fileSize", 0)))

    def write_file(self, path: str, content: str) -> WriteResult:
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
        if _is_write_denied(path):
            return WriteResult(error=f"Delete denied: {path} is a protected path")
        response = self._files(
            {"operation": "delete", "path": path, "recursive": recursive, "missingOk": True}
        )
        if error := response.get("error"):
            return WriteResult(error=str(error))
        return WriteResult()

    def move_file(self, src: str, dst: str) -> WriteResult:
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
                # The toolbox /files model validates offset as 1-based
                # (ge=1, shared with read's line offset) and converts back
                # internally; translate our 0-based result offset at the wire.
                "offset": offset + 1,
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
