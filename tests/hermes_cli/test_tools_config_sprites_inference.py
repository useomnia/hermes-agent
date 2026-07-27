"""Regression coverage for static toolset inference on paired Sprites."""


def test_api_server_keeps_terminal_toolset_for_sprites(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    monkeypatch.setenv("OMNIO_TOOLBOX_URL", "https://toolbox.example")
    monkeypatch.setenv("OMNIO_TOOLBOX_BEARER", "pair-secret")
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)

    from hermes_cli.tools_config import _get_platform_tools
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()

    assert "terminal" in _get_platform_tools(
        {}, "api_server", include_default_mcp_servers=False
    )
