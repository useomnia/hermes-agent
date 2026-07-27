"""Behavior checks for Omnia extensions on the upstream API route table."""

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def test_omnia_extensions_share_the_upstream_profile_aware_route_table():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    routes = {
        (method, path): handler.__name__
        for method, path, handler in adapter._http_route_table()
    }

    assert routes[("POST", "/api/sessions/{session_id}/model")] == (
        "_handle_session_model_lock"
    )
    assert routes[("PUT", "/api/sessions/{session_id}/messages")] == (
        "_handle_replace_session_messages"
    )
    assert routes[("DELETE", "/api/sessions/{session_id}/messages")] == (
        "_handle_delete_session_messages"
    )
    assert routes[("POST", "/v1/omnio/tool-approval")] == (
        "_handle_omnio_tool_approval"
    )
    assert routes[("POST", "/v1/omnio/user-input")] == (
        "_handle_omnio_user_input"
    )
    assert routes[("POST", "/v1/mcp/reload")] == "_handle_mcp_reload"
    assert routes[("POST", "/v1/skills/reload")] == "_handle_skills_reload"
