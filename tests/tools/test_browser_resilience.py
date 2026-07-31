"""Focused coverage for bounded navigation and Omnio browser recovery."""

import json
from unittest.mock import call, patch

from tools import browser_tool


def _cdp_failure(
    *,
    code: str = "renderer_unresponsive",
    retryable: bool = True,
) -> dict:
    return {
        "success": False,
        "error": {
            "code": code,
            "phase": "renderer",
            "retryable": retryable,
            "commandStarted": True,
        },
    }


def test_navigation_timeout_preserves_cold_start_and_bounds_warm_pages(
    monkeypatch,
) -> None:
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 30)
    monkeypatch.setattr(browser_tool, "_get_navigation_timeout", lambda: 15)

    assert browser_tool._get_open_command_timeout(first_open=True) == 120
    assert browser_tool._get_navigation_timeout() == 15


def test_navigate_returns_partial_success_when_open_times_out_but_page_is_ready(
    monkeypatch,
) -> None:
    session_info = {
        "session_name": "cdp_nav",
        "cdp_url": "ws://relay/devtools/browser",
        "_first_nav": False,
        "features": {"cdp_override": True},
    }
    open_failure = {
        "success": False,
        "error": "Command timed out after 15 seconds",
    }
    ready_probe = {
        "success": True,
        "data": {
            "result": json.dumps({
                "readyState": "complete",
                "url": "https://example.com/",
                "title": "Example Domain",
            }),
        },
    }
    snapshot = {
        "success": True,
        "data": {
            "snapshot": '- heading "Example Domain" [ref=e1]',
            "refs": {"e1": {}},
        },
    }

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda _task: session_info)
    monkeypatch.setattr(browser_tool, "_get_navigation_timeout", lambda: 15)

    with (
        patch(
            "tools.browser_tool._run_browser_command",
            side_effect=[open_failure, ready_probe, snapshot],
        ) as mock_run,
        patch("tools.browser_tool._recover_omnio_browser") as mock_recover,
    ):
        result = json.loads(
            browser_tool.browser_navigate(
                "https://example.com",
                task_id="nav-ready",
            )
        )

    assert result == {
        "success": True,
        "url": "https://example.com/",
        "title": "Example Domain",
        "partial_load": True,
        "network_idle_timeout": True,
        "readiness_state": "complete",
        "note": (
            "Navigation did not report full load completion, but the page is "
            "usable and returned a readiness snapshot."
        ),
        "snapshot": '- heading "Example Domain" [ref=e1]',
        "element_count": 1,
    }
    assert mock_run.call_args_list == [
        call(
            "nav-ready",
            "open",
            ["https://example.com"],
            timeout=15,
            _defer_session_reset_on_timeout=True,
        ),
        call(
            "nav-ready",
            "eval",
            [
                "JSON.stringify({readyState: document.readyState, "
                "url: window.location.href, title: document.title})"
            ],
            timeout=5,
        ),
        call("nav-ready", "snapshot", ["-c"], timeout=5),
    ]
    mock_recover.assert_not_called()
    browser_tool._last_active_session_key.pop("nav-ready", None)


def test_navigate_does_not_accept_ready_about_blank_after_open_failure(
    monkeypatch,
) -> None:
    session_info = {
        "session_name": "cdp_nav",
        "cdp_url": "ws://relay/devtools/browser",
        "_first_nav": False,
    }
    open_failure = {
        "success": False,
        "error": "Command timed out after 15 seconds",
    }
    blank_probe = {
        "success": True,
        "data": {
            "result": json.dumps({
                "readyState": "complete",
                "url": "about:blank",
                "title": "",
            }),
        },
    }
    empty_snapshot = {
        "success": True,
        "data": {"snapshot": "", "refs": {}},
    }

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda _task: session_info)
    monkeypatch.setattr(browser_tool, "_get_navigation_timeout", lambda: 15)

    with (
        patch(
            "tools.browser_tool._run_browser_command",
            side_effect=[open_failure, blank_probe, empty_snapshot],
        ),
        patch(
            "tools.browser_tool._recover_omnio_browser",
            return_value=False,
        ) as mock_recover,
    ):
        result = json.loads(
            browser_tool.browser_navigate(
                "https://unreachable.example",
                task_id="nav-blank",
            )
        )

    assert result == {
        "success": False,
        "error": "Command timed out after 15 seconds",
    }
    mock_recover.assert_not_called()
    browser_tool._last_active_session_key.pop("nav-blank", None)


def test_snapshot_renderer_failure_recovers_without_retrying_blank_page(
    monkeypatch,
) -> None:
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda _task: "brand-a")
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 30)
    monkeypatch.setattr(browser_tool, "_session_is_cdp_backed", lambda _task: True)

    with (
        patch(
            "tools.browser_tool._run_browser_command",
            return_value=_cdp_failure(),
        ) as mock_run,
        patch(
            "tools.browser_tool._recover_omnio_browser",
            return_value=True,
        ) as mock_recover,
    ):
        result = json.loads(browser_tool.browser_snapshot(task_id="brand-a"))

    assert result == {
        "success": False,
        "error": (
            "The browser renderer became unresponsive and the session was "
            "reset; re-open the page with browser_navigate to continue."
        ),
        "code": "browser_session_reset",
        "retryable": True,
        "session_reset": True,
        "next_action": "browser_navigate",
    }
    mock_run.assert_called_once_with(
        "brand-a",
        "snapshot",
        ["-c"],
        timeout=90,
    )
    mock_recover.assert_called_once_with("brand-a")


def test_vision_renderer_failure_uses_same_scoped_recovery_contract(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda _task: "conversation-a")
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(browser_tool, "_session_is_cdp_backed", lambda _task: True)
    monkeypatch.setattr("hermes_constants.get_hermes_dir", lambda *_args: tmp_path)

    with (
        patch("tools.browser_tool._run_browser_command", return_value=_cdp_failure()),
        patch(
            "tools.browser_tool._recover_omnio_browser",
            return_value=True,
        ) as mock_recover,
    ):
        result = json.loads(
            browser_tool.browser_vision("what is visible?", task_id="conversation-a")
        )

    assert result["code"] == "browser_session_reset"
    assert result["next_action"] == "browser_navigate"
    mock_recover.assert_called_once_with("conversation-a")


def test_snapshot_bare_timeout_does_not_recover(monkeypatch) -> None:
    failure = {
        "success": False,
        "error": "Command timed out after 90 seconds",
    }
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda _task: "brand-a")
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 30)
    monkeypatch.setattr(browser_tool, "_session_is_cdp_backed", lambda _task: True)

    with (
        patch(
            "tools.browser_tool._run_browser_command",
            return_value=failure,
        ) as mock_run,
        patch("tools.browser_tool._recover_omnio_browser") as mock_recover,
    ):
        result = json.loads(browser_tool.browser_snapshot(task_id="brand-a"))

    assert result == failure
    mock_run.assert_called_once_with(
        "brand-a",
        "snapshot",
        ["-c"],
        timeout=90,
    )
    mock_recover.assert_not_called()


def test_snapshot_other_retryable_structured_error_does_not_recover(
    monkeypatch,
) -> None:
    failure = _cdp_failure(code="relay_unavailable")
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda _task: "brand-a")
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 30)
    monkeypatch.setattr(browser_tool, "_session_is_cdp_backed", lambda _task: True)

    with (
        patch(
            "tools.browser_tool._run_browser_command",
            return_value=failure,
        ),
        patch("tools.browser_tool._recover_omnio_browser") as mock_recover,
    ):
        result = json.loads(browser_tool.browser_snapshot(task_id="brand-a"))

    assert result == failure
    mock_recover.assert_not_called()


def test_navigate_retryable_failure_recovers_and_retries_once(
    monkeypatch,
) -> None:
    session_info = {
        "session_name": "cdp_nav",
        "cdp_url": "ws://relay/devtools/browser",
        "_first_nav": False,
    }
    failed_probe = {"success": False, "error": "renderer unavailable"}
    successful_open = {
        "success": True,
        "data": {
            "url": "https://example.com/",
            "title": "Example Domain",
        },
    }
    successful_snapshot = {
        "success": True,
        "data": {
            "snapshot": '- heading "Example Domain" [ref=e1]',
            "refs": {"e1": {}},
        },
    }

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda _task: session_info)
    monkeypatch.setattr(browser_tool, "_get_navigation_timeout", lambda: 15)
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 30)

    with (
        patch(
            "tools.browser_tool._run_browser_command",
            side_effect=[
                _cdp_failure(),
                failed_probe,
                failed_probe,
                successful_open,
                successful_snapshot,
            ],
        ) as mock_run,
        patch(
            "tools.browser_tool._recover_omnio_browser",
            return_value=True,
        ) as mock_recover,
    ):
        result = json.loads(
            browser_tool.browser_navigate(
                "https://example.com",
                task_id="nav-recover",
            )
        )

    assert result == {
        "success": True,
        "url": "https://example.com/",
        "title": "Example Domain",
        "browser_recovered": True,
        "snapshot": '- heading "Example Domain" [ref=e1]',
        "element_count": 1,
    }
    assert mock_run.call_args_list == [
        call(
            "nav-recover",
            "open",
            ["https://example.com"],
            timeout=15,
            _defer_session_reset_on_timeout=True,
        ),
        call(
            "nav-recover",
            "eval",
            [
                "JSON.stringify({readyState: document.readyState, "
                "url: window.location.href, title: document.title})"
            ],
            timeout=5,
        ),
        call("nav-recover", "snapshot", ["-c"], timeout=5),
        call(
            "nav-recover",
            "open",
            ["https://example.com"],
            timeout=15,
            _defer_session_reset_on_timeout=True,
        ),
        call("nav-recover", "snapshot", ["-c"], timeout=90),
    ]
    mock_recover.assert_called_once_with("nav-recover")
    browser_tool._last_active_session_key.pop("nav-recover", None)


def test_snapshot_preserves_failure_when_recover_is_unsupported(
    monkeypatch,
) -> None:
    failure = _cdp_failure()
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda _task: "brand-a")
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 30)
    monkeypatch.setattr(browser_tool, "_session_is_cdp_backed", lambda _task: True)

    with (
        patch(
            "tools.browser_tool._run_browser_command",
            return_value=failure,
        ) as mock_run,
        patch(
            "tools.browser_tool._recover_omnio_browser",
            return_value=False,
        ) as mock_recover,
    ):
        result = json.loads(browser_tool.browser_snapshot(task_id="brand-a"))

    assert result == {
        "success": False,
        "error": failure["error"],
    }
    mock_run.assert_called_once_with(
        "brand-a",
        "snapshot",
        ["-c"],
        timeout=90,
    )
    mock_recover.assert_called_once_with("brand-a")


def test_toolbox_v5_does_not_receive_recover_operation(monkeypatch) -> None:
    health_response = type(
        "HealthResponse",
        (),
        {"json": lambda self: {"version": 5}},
    )()
    monkeypatch.setenv("OMNIO_TOOLBOX_URL", "https://toolbox.test")
    monkeypatch.setenv("OMNIO_TOOLBOX_BEARER", "pair-bearer")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-a")

    with (
        patch(
            "tools.browser_tool.requests.get",
            return_value=health_response,
        ) as mock_get,
        patch("tools.browser_tool.requests.post") as mock_post,
    ):
        recovered = browser_tool._recover_omnio_browser("conversation-a")

    assert recovered is False
    mock_get.assert_called_once_with(
        "https://toolbox.test/health",
        headers={
            "Authorization": "Bearer pair-bearer",
            "Content-Type": "application/json",
            "X-Omnio-Brand": "brand-a",
            "X-Hermes-Session-Id": "conversation-a",
        },
        timeout=3,
    )
    mock_post.assert_not_called()


def test_toolbox_recover_404_degrades_gracefully(monkeypatch) -> None:
    health_response = type(
        "HealthResponse",
        (),
        {"json": lambda self: {"version": 6}},
    )()
    unsupported_response = type(
        "UnsupportedResponse",
        (),
        {"status_code": 404},
    )()
    monkeypatch.setenv("OMNIO_TOOLBOX_URL", "https://toolbox.test")
    monkeypatch.setenv("OMNIO_TOOLBOX_BEARER", "pair-bearer")
    monkeypatch.setenv("OMNIO_TOOLBOX_BRAND", "brand-a")

    with (
        patch(
            "tools.browser_tool.requests.get",
            return_value=health_response,
        ),
        patch(
            "tools.browser_tool.requests.post",
            return_value=unsupported_response,
        ) as mock_post,
    ):
        recovered = browser_tool._recover_omnio_browser("conversation-a")

    assert recovered is False
    mock_post.assert_called_once_with(
        "https://toolbox.test/browser",
        headers={
            "Authorization": "Bearer pair-bearer",
            "Content-Type": "application/json",
            "X-Omnio-Brand": "brand-a",
            "X-Hermes-Session-Id": "conversation-a",
        },
        json={"operation": "recover"},
        timeout=20,
    )
