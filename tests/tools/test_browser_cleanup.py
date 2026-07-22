"""Regression tests for browser session cleanup and screenshot recovery."""

import subprocess
from unittest.mock import MagicMock, patch


class TestScreenshotPathRecovery:
    def test_extracts_standard_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text("Screenshot saved to /tmp/foo.png")
            == "/tmp/foo.png"
        )

    def test_extracts_quoted_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text(
                "Screenshot saved to '/Users/david/.hermes/browser_screenshots/shot.png'"
            )
            == "/Users/david/.hermes/browser_screenshots/shot.png"
        )


class TestBrowserCleanup:
    def setup_method(self):
        from tools import browser_tool

        self.browser_tool = browser_tool
        self.orig_active_sessions = browser_tool._active_sessions.copy()
        self.orig_session_last_activity = browser_tool._session_last_activity.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._recording_sessions.clear()
        self.browser_tool._recording_sessions.update(self.orig_recording_sessions)
        self.browser_tool._cleanup_done = self.orig_cleanup_done

    def test_cleanup_browser_clears_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            browser_tool.cleanup_browser("task-1")

        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity
        mock_stop.assert_called_once_with("task-1")
        mock_run.assert_called_once_with("task-1", "close", [], timeout=10)

    def test_cleanup_camofox_managed_persistence_skips_close(self):
        """When camofox mode + managed persistence, soft_cleanup fires instead of close."""
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch("tools.browser_tool._is_camofox_mode", return_value=True),
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch("tools.browser_tool.os.path.exists", return_value=False),
            patch(
                "tools.browser_camofox.camofox_soft_cleanup",
                return_value=True,
            ) as mock_soft,
            patch("tools.browser_camofox.camofox_close") as mock_close,
        ):
            browser_tool.cleanup_browser("task-1")

        mock_soft.assert_called_once_with("task-1")
        mock_close.assert_not_called()

    def test_cleanup_camofox_no_persistence_calls_close(self):
        """When camofox mode but managed persistence is off, camofox_close fires."""
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch("tools.browser_tool._is_camofox_mode", return_value=True),
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch("tools.browser_tool.os.path.exists", return_value=False),
            patch(
                "tools.browser_camofox.camofox_soft_cleanup",
                return_value=False,
            ) as mock_soft,
            patch("tools.browser_camofox.camofox_close") as mock_close,
        ):
            browser_tool.cleanup_browser("task-1")

        mock_soft.assert_called_once_with("task-1")
        mock_close.assert_called_once_with("task-1")

    def test_emergency_cleanup_clears_all_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._cleanup_done = False
        browser_tool._active_sessions["task-1"] = {"session_name": "sess-1"}
        browser_tool._active_sessions["task-2"] = {"session_name": "sess-2"}
        browser_tool._session_last_activity["task-1"] = 1.0
        browser_tool._session_last_activity["task-2"] = 2.0
        browser_tool._recording_sessions.update({"task-1", "task-2"})

        with patch("tools.browser_tool.cleanup_all_browsers") as mock_cleanup_all:
            browser_tool._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        assert browser_tool._active_sessions == {}
        assert browser_tool._session_last_activity == {}
        assert browser_tool._recording_sessions == set()
        assert browser_tool._cleanup_done is True

    def test_snapshot_uses_at_least_90_second_timeout(self):
        browser_tool = self.browser_tool
        with (
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
            patch("tools.browser_tool._get_command_timeout", return_value=30),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True, "data": {}},
            ) as mock_run,
        ):
            browser_tool.browser_snapshot(task_id="task-1")

        mock_run.assert_called_once_with(
            "task-1", "snapshot", ["-c"], timeout=90
        )

    def test_command_timeout_resets_daemon_and_cached_session(
        self, tmp_path
    ):
        browser_tool = self.browser_tool
        session_info = {
            "session_name": "cdp_timeout123",
            "cdp_url": "ws://relay/devtools/browser/old",
        }
        browser_tool._active_sessions["task-1"] = session_info
        browser_tool._session_last_activity["task-1"] = 123.0

        fake_proc = MagicMock()
        fake_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="agent-browser", timeout=1),
            None,
        ]

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._find_agent_browser", return_value="/bin/true"),
            patch(
                "tools.browser_tool._requires_real_termux_browser_install",
                return_value=False,
            ),
            patch("tools.browser_tool._is_local_mode", return_value=False),
            patch("tools.browser_tool._get_session_info", return_value=session_info),
            patch("tools.browser_tool._write_owner_pid"),
            patch("tools.browser_tool.subprocess.Popen", return_value=fake_proc),
            patch(
                "tools.browser_tool._terminate_timed_out_browser_daemon",
                return_value=True,
            ) as mock_terminate,
            patch("tools.browser_tool._get_browser_engine", return_value="auto"),
        ):
            result = browser_tool._run_browser_command(
                "task-1", "screenshot", [], timeout=1
            )

        assert result == {
            "success": False,
            "error": "Command timed out after 1 seconds",
        }
        fake_proc.kill.assert_called_once_with()
        mock_terminate.assert_called_once()
        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity

        fresh_session = {
            "session_name": "cdp_fresh1234",
            "cdp_url": "ws://relay/devtools/browser/fresh",
        }
        with (
            patch("tools.browser_tool._start_browser_cleanup_thread"),
            patch("tools.browser_tool._update_session_activity"),
            patch("tools.browser_tool._get_cdp_override", return_value="ws://relay"),
            patch("tools.browser_tool._create_cdp_session", return_value=fresh_session) as mock_create,
            patch("tools.browser_tool._ensure_cdp_supervisor"),
        ):
            assert browser_tool._get_session_info("task-1") is fresh_session

        mock_create.assert_called_once_with("task-1", "ws://relay")

    def test_cdp_timeout_terminates_daemon_without_tree_kill(self, tmp_path):
        browser_tool = self.browser_tool
        session_info = {
            "session_name": "cdp_timeout123",
            "cdp_url": "ws://relay/devtools/browser/old",
        }
        socket_dir = tmp_path / "agent-browser-cdp_timeout123"
        socket_dir.mkdir()
        (socket_dir / "cdp_timeout123.pid").write_text("4321")
        daemon = MagicMock()

        with (
            patch(
                "tools.browser_tool._verify_reapable_browser_daemon",
                return_value=True,
            ),
            patch("psutil.Process", return_value=daemon) as mock_process,
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid"
            ) as mock_tree_kill,
        ):
            assert browser_tool._terminate_timed_out_browser_daemon(
                session_info, str(socket_dir)
            ) is True

        mock_process.assert_called_once_with(4321)
        daemon.terminate.assert_called_once_with()
        daemon.wait.assert_called_once_with(timeout=5)
        mock_tree_kill.assert_not_called()
        assert not socket_dir.exists()

    def test_local_timeout_terminates_daemon_tree(self, tmp_path):
        browser_tool = self.browser_tool
        session_info = {"session_name": "h_timeout123", "cdp_url": None}
        socket_dir = tmp_path / "agent-browser-h_timeout123"
        socket_dir.mkdir()
        (socket_dir / "h_timeout123.pid").write_text("9876")

        with (
            patch(
                "tools.browser_tool._verify_reapable_browser_daemon",
                return_value=True,
            ),
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid"
            ) as mock_tree_kill,
        ):
            assert browser_tool._terminate_timed_out_browser_daemon(
                session_info, str(socket_dir)
            ) is True

        mock_tree_kill.assert_called_once_with(9876)
        assert not socket_dir.exists()

    def test_timeout_reset_does_not_drop_replacement_session(self, tmp_path):
        browser_tool = self.browser_tool
        timed_out = {"session_name": "cdp_old", "cdp_url": "ws://old"}
        replacement = {"session_name": "cdp_new", "cdp_url": "ws://new"}
        browser_tool._active_sessions["task-1"] = replacement
        browser_tool._session_last_activity["task-1"] = 456.0

        with patch(
            "tools.browser_tool._terminate_timed_out_browser_daemon",
            return_value=True,
        ):
            browser_tool._reset_browser_session_after_timeout(
                "task-1", timed_out, str(tmp_path)
            )

        assert browser_tool._active_sessions["task-1"] is replacement
        assert browser_tool._session_last_activity["task-1"] == 456.0

    def test_timeout_reset_releases_cloud_provider_session(self, tmp_path):
        browser_tool = self.browser_tool
        session_info = {
            "session_name": "hermes_cloud123",
            "bb_session_id": "bb_123",
            "cdp_url": "ws://provider/devtools/browser/old",
        }
        browser_tool._active_sessions["task-1"] = session_info
        provider = MagicMock()

        with (
            patch(
                "tools.browser_tool._terminate_timed_out_browser_daemon",
                return_value=True,
            ),
            patch("tools.browser_tool._get_cloud_provider", return_value=provider),
        ):
            browser_tool._reset_browser_session_after_timeout(
                "task-1", session_info, str(tmp_path)
            )

        provider.close_session.assert_called_once_with("bb_123")
        assert "task-1" not in browser_tool._active_sessions

    def test_timeout_reset_cloud_release_is_best_effort(self, tmp_path):
        browser_tool = self.browser_tool
        session_info = {
            "session_name": "hermes_cloud456",
            "bb_session_id": "bb_456",
            "cdp_url": "ws://provider/devtools/browser/old",
        }
        browser_tool._active_sessions["task-1"] = session_info
        provider = MagicMock()
        provider.close_session.side_effect = RuntimeError("release failed")

        with (
            patch(
                "tools.browser_tool._terminate_timed_out_browser_daemon",
                return_value=True,
            ),
            patch("tools.browser_tool._get_cloud_provider", return_value=provider),
        ):
            browser_tool._reset_browser_session_after_timeout(
                "task-1", session_info, str(tmp_path)
            )

        provider.close_session.assert_called_once_with("bb_456")
        assert "task-1" not in browser_tool._active_sessions
