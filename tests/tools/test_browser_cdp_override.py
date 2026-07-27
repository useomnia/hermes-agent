import logging
import requests
from unittest.mock import Mock, call, patch


HOST = "example-host"
PORT = 9223
WS_URL = f"ws://{HOST}:{PORT}/devtools/browser/abc123"
HTTP_URL = f"http://{HOST}:{PORT}"
VERSION_URL = f"{HTTP_URL}/json/version"


class TestResolveCdpOverride:
    def test_keeps_full_devtools_websocket_url(self):
        from tools.browser_tool import _resolve_cdp_override

        assert _resolve_cdp_override(WS_URL) == WS_URL

    def test_resolves_http_discovery_endpoint_to_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override(HTTP_URL)

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=1.0)

    def test_resolves_bare_ws_hostport_to_discovery_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override(f"ws://{HOST}:{PORT}")

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=1.0)

    def test_retries_until_late_relay_binds(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with (
            patch(
                "tools.browser_tool.requests.get",
                side_effect=[
                    requests.ConnectionError("connection refused"),
                    requests.ConnectionError("connection refused"),
                    response,
                ],
            ) as mock_get,
            patch("tools.browser_tool.time.sleep") as mock_sleep,
        ):
            resolved = _resolve_cdp_override(HTTP_URL)

        assert resolved == WS_URL
        assert mock_get.call_args_list == [
            call(VERSION_URL, timeout=1.0),
            call(VERSION_URL, timeout=1.0),
            call(VERSION_URL, timeout=1.0),
        ]
        assert mock_sleep.call_args_list == [call(0.1), call(0.2)]

    def test_falls_back_to_raw_url_when_discovery_fails(self):
        from tools.browser_tool import _resolve_cdp_override

        with (
            patch(
                "tools.browser_tool.requests.get",
                side_effect=RuntimeError("boom"),
            ),
            patch("tools.browser_tool.time.sleep"),
        ):
            assert _resolve_cdp_override(HTTP_URL) == HTTP_URL

    def test_logs_one_sanitized_warning_after_all_attempts_fail(self, caplog):
        from tools.browser_tool import _resolve_cdp_override

        secret_url = f"http://token-secret@{HOST}:{PORT}"
        with (
            caplog.at_level(logging.DEBUG, logger="tools.browser_tool"),
            patch(
                "tools.browser_tool.requests.get",
                side_effect=RuntimeError("connection refused"),
            ),
            patch("tools.browser_tool.time.sleep"),
        ):
            assert _resolve_cdp_override(secret_url) == secret_url

        warning_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        debug_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.DEBUG
        ]
        assert len(warning_messages) == 1
        assert "token-secret" not in warning_messages[0]
        assert len(debug_messages) == 5

    def test_normalizes_provider_returned_http_cdp_url_when_creating_session(self, monkeypatch):
        import tools.browser_tool as browser_tool

        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-session",
            "bb_session_id": "bu_123",
            "cdp_url": "https://cdp.browser-use.example/session",
            "features": {"browser_use": True},
        }

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_session_last_activity", {})
        monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(browser_tool, "_update_session_activity", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            session_info = browser_tool._get_session_info("task-browser-use")

        assert session_info["cdp_url"] == WS_URL
        provider.create_session.assert_called_once_with("task-browser-use")
        mock_get.assert_called_once_with(
            "https://cdp.browser-use.example/session/json/version",
            timeout=1.0,
        )


class TestGetCdpOverride:
    def test_prefers_env_var_over_config(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)
        monkeypatch.setattr(
            browser_tool,
            "read_raw_config",
            lambda: {"browser": {"cdp_url": "http://config-host:9222"}},
            raising=False,
        )

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = browser_tool._get_cdp_override()

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=1.0)

    def test_uses_config_browser_cdp_url_when_env_missing(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"cdp_url": HTTP_URL}}), \
             patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = browser_tool._get_cdp_override()

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=1.0)


class TestCdpOverrideProbeBoundary:
    """Discovery is a network probe, so it must happen only where a browser
    connection is actually made — never on a routing or capability check."""

    def test_raw_lookup_returns_configured_url_without_probing(self, monkeypatch):
        from tools.browser_tool import _get_cdp_override_raw

        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)

        with patch("tools.browser_tool.requests.get") as mock_get:
            assert _get_cdp_override_raw() == HTTP_URL

        mock_get.assert_not_called()

    def test_raw_lookup_falls_back_to_config_without_probing(self, monkeypatch):
        from tools.browser_tool import _get_cdp_override_raw

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

        with (
            patch(
                "hermes_cli.config.read_raw_config",
                return_value={"browser": {"cdp_url": HTTP_URL}},
            ),
            patch("tools.browser_tool.requests.get") as mock_get,
        ):
            assert _get_cdp_override_raw() == HTTP_URL

        mock_get.assert_not_called()

    def test_backend_selection_never_probes(self, monkeypatch):
        # Runs during turn init. When the browser host is not listening yet a
        # probe here burns the whole retry ladder and then falls back anyway —
        # latency for an answer the configuration already holds.
        from tools.browser_tool import _is_local_mode

        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)

        with patch("tools.browser_tool.requests.get") as mock_get:
            assert _is_local_mode() is False

        mock_get.assert_not_called()

    def test_capability_check_never_probes(self, monkeypatch):
        # Gating the tool on liveness at registration says nothing about
        # liveness at call time, and stalls registration when the host is down.
        from tools.browser_cdp_tool import _browser_cdp_check

        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)

        with (
            patch("tools.browser_tool.check_browser_requirements", return_value=True),
            patch("tools.browser_tool.requests.get") as mock_get,
        ):
            assert _browser_cdp_check() is True

        mock_get.assert_not_called()
