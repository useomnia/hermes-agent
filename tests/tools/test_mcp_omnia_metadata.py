"""Approval metadata follows the lifetime of its registered MCP tool."""

from tools import mcp_tool


def test_omnia_approval_metadata_is_registered_and_forgotten():
    tool_name = "mcp_connectors_send_email"
    credits = {"creditsPerUnit": 2, "unit": "engine"}

    mcp_tool._track_mcp_tool_metadata(
        tool_name,
        read_only_hint=False,
        credits=credits,
    )

    assert mcp_tool.mcp_tool_has_read_only_hint(tool_name) is True
    assert mcp_tool.mcp_tool_is_read_only(tool_name) is False
    assert mcp_tool.mcp_tool_credits_meta(tool_name) == credits

    mcp_tool._forget_mcp_tool_server(tool_name)

    assert mcp_tool.mcp_tool_has_read_only_hint(tool_name) is False
    assert mcp_tool.mcp_tool_credits_meta(tool_name) is None
