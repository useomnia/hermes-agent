"""Targeted MCP reload: only the named servers are refreshed or reconnected."""

import asyncio

import pytest

import tools.mcp_tool as mcp_tool


class _Server:
    def __init__(self, name: str, config: dict, *, session: object = object()) -> None:
        self.name = name
        self._config = config
        self.session = session
        self.refreshes = 0
        self.stopped = False
        self.refresh_error: Exception | None = None

    async def _refresh_tools(self) -> None:
        if self.refresh_error is not None:
            raise self.refresh_error
        self.refreshes += 1

    async def shutdown(self) -> None:
        self.stopped = True


@pytest.fixture
def mcp(monkeypatch):
    configs: dict[str, dict] = {}
    servers: dict[str, _Server] = {}
    registered: list[str] = []

    def register(selected: dict) -> list:
        for name, config in selected.items():
            registered.append(name)
            if config.get("fails") is None:
                servers[name] = _Server(name, config)
        return []

    monkeypatch.setattr(mcp_tool, "_servers", servers)
    monkeypatch.setattr(mcp_tool, "_load_mcp_config", lambda: dict(configs))
    monkeypatch.setattr(mcp_tool, "register_mcp_servers", register)
    monkeypatch.setattr(
        mcp_tool, "_run_on_mcp_loop", lambda factory, timeout=30: asyncio.run(factory())
    )
    monkeypatch.setattr(mcp_tool, "_server_connect_retry_after", {})
    monkeypatch.setattr(mcp_tool, "_server_connect_failures", {})
    return configs, servers, registered


def test_unchanged_connected_server_is_refreshed_in_place(mcp):
    configs, servers, _ = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}
    servers["connectors"] = server = _Server("connectors", configs["connectors"])

    assert mcp_tool.reload_mcp_servers(["connectors"]) == {"connectors": "refreshed"}
    assert server.refreshes == 1


def test_in_place_refresh_keeps_the_server_connected(mcp):
    configs, servers, _ = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}
    servers["connectors"] = server = _Server("connectors", configs["connectors"])

    mcp_tool.reload_mcp_servers(["connectors"])

    assert servers["connectors"] is server and not server.stopped


def test_servers_not_named_are_left_untouched(mcp):
    configs, servers, _ = mcp
    configs.update(connectors={"url": "https://a.test/mcp"}, omnia={"url": "https://b.test/mcp"})
    servers["connectors"] = _Server("connectors", configs["connectors"])
    servers["omnia"] = omnia = _Server("omnia", configs["omnia"])

    mcp_tool.reload_mcp_servers(["connectors"])

    assert (omnia.refreshes, omnia.stopped) == (0, False)


def test_changed_config_reconnects_only_that_server(mcp):
    configs, servers, registered = mcp
    servers["connectors"] = old = _Server("connectors", {"url": "https://old.test/mcp"})
    configs["connectors"] = {"url": "https://new.test/mcp"}

    assert mcp_tool.reload_mcp_servers(["connectors"]) == {"connectors": "reconnected"}
    assert (old.stopped, registered) == (True, ["connectors"])


def test_server_without_a_live_session_is_reconnected(mcp):
    configs, servers, _ = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}
    servers["connectors"] = _Server("connectors", configs["connectors"], session=None)

    assert mcp_tool.reload_mcp_servers(["connectors"]) == {"connectors": "reconnected"}


def test_failed_in_place_refresh_falls_back_to_a_reconnect(mcp):
    configs, servers, _ = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}
    servers["connectors"] = server = _Server("connectors", configs["connectors"])
    server.refresh_error = RuntimeError("tools/list failed")

    assert mcp_tool.reload_mcp_servers(["connectors"]) == {"connectors": "reconnected"}


def test_configured_but_disconnected_server_is_connected(mcp):
    configs, _, registered = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}

    assert mcp_tool.reload_mcp_servers(["connectors"]) == {"connectors": "reconnected"}
    assert registered == ["connectors"]


def test_server_removed_from_config_is_shut_down(mcp):
    _, servers, _ = mcp
    servers["connectors"] = server = _Server("connectors", {"url": "https://a.test/mcp"})

    assert mcp_tool.reload_mcp_servers(["connectors"]) == {"connectors": "removed"}
    assert server.stopped and "connectors" not in servers


def test_unknown_server_is_reported_absent(mcp):
    assert mcp_tool.reload_mcp_servers(["nope"]) == {"nope": "absent"}


def test_server_that_cannot_reconnect_is_reported_failed(mcp):
    configs, _, _ = mcp
    configs["connectors"] = {"url": "https://a.test/mcp", "fails": True}

    assert mcp_tool.reload_mcp_servers(["connectors"]) == {"connectors": "failed"}


def test_reload_clears_the_named_servers_connect_backoff(mcp):
    configs, servers, _ = mcp
    servers["connectors"] = _Server("connectors", {"url": "https://old.test/mcp"})
    configs["connectors"] = {"url": "https://new.test/mcp"}
    mcp_tool._server_connect_retry_after["connectors"] = 1e12

    mcp_tool.reload_mcp_servers(["connectors"])

    assert "connectors" not in mcp_tool._server_connect_retry_after


def test_live_reload_still_refreshes_in_place(mcp):
    configs, servers, _ = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}
    servers["connectors"] = server = _Server("connectors", configs["connectors"])

    assert mcp_tool.reload_mcp_servers(["connectors"], live=True) == {"connectors": "refreshed"}
    assert (server.refreshes, server.stopped) == (1, False)


def test_live_reload_defers_a_config_change_without_closing_the_connection(mcp):
    configs, servers, registered = mcp
    servers["connectors"] = old = _Server("connectors", {"url": "https://old.test/mcp"})
    configs["connectors"] = {"url": "https://new.test/mcp"}

    assert mcp_tool.reload_mcp_servers(["connectors"], live=True) == {"connectors": "deferred"}
    assert (old.stopped, servers["connectors"], registered) == (False, old, [])


def test_live_reload_defers_when_the_in_place_refresh_fails(mcp):
    configs, servers, registered = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}
    servers["connectors"] = server = _Server("connectors", configs["connectors"])
    server.refresh_error = RuntimeError("tools/list failed")

    assert mcp_tool.reload_mcp_servers(["connectors"], live=True) == {"connectors": "deferred"}
    assert (server.stopped, registered) == (False, [])


def test_live_reload_defers_a_server_without_a_live_session(mcp):
    configs, servers, _ = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}
    servers["connectors"] = server = _Server("connectors", configs["connectors"], session=None)

    assert mcp_tool.reload_mcp_servers(["connectors"], live=True) == {"connectors": "deferred"}
    assert not server.stopped


def test_live_reload_defers_removing_a_server_dropped_from_config(mcp):
    _, servers, _ = mcp
    servers["connectors"] = server = _Server("connectors", {"url": "https://a.test/mcp"})

    assert mcp_tool.reload_mcp_servers(["connectors"], live=True) == {"connectors": "deferred"}
    assert (server.stopped, "connectors" in servers) == (False, True)


def test_live_reload_connects_a_configured_server_with_no_connection(mcp):
    configs, _, registered = mcp
    configs["connectors"] = {"url": "https://a.test/mcp"}

    assert mcp_tool.reload_mcp_servers(["connectors"], live=True) == {"connectors": "reconnected"}
    assert registered == ["connectors"]


def test_live_reload_reports_an_unknown_server_absent(mcp):
    assert mcp_tool.reload_mcp_servers(["nope"], live=True) == {"nope": "absent"}
