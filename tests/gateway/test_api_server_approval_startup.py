"""The API approval startup seam must not import the heavy approval gate."""

import sys

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def test_approval_source_uses_lightweight_state_import(monkeypatch):
    # Other approval tests in the same pytest process may already have loaded
    # the heavy gate. Remove only its import-cache entry so this assertion is
    # independent of collection/execution order while still exercising the
    # runtime startup path instead of inspecting source text.
    monkeypatch.delitem(sys.modules, "tools.tool_approval", raising=False)
    assert "tools.tool_approval" not in sys.modules

    monkeypatch.setenv("OMNIA_BASE_URL", "https://omnia.test")
    monkeypatch.setenv("OMNIA_API_TOKEN", "test-token")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-1")
    monkeypatch.delenv("OMNIO_TOOL_APPROVAL_DURABLE_DISABLED", raising=False)

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._omnia_approval_source(clear_snapshot=True) == (
        "https://omnia.test",
        "test-token",
        "brand-1",
    )
    assert "tools.tool_approval" not in sys.modules
