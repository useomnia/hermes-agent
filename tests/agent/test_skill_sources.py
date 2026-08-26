"""Tests for split skill source identity and path boundaries."""

from unittest.mock import patch

from agent.skill_sources import direct_skill_path


def test_source_qualified_path_cannot_cross_its_source_boundary(tmp_path):
    with patch(
        "hermes_cli.config.load_config",
        return_value={"skills": {"source_layout": "split"}},
    ):
        assert direct_skill_path("custom:../system/shipped", tmp_path) is None
        assert direct_skill_path("system:../custom/learned", tmp_path) is None
        assert direct_skill_path("custom:toolkit/learned", tmp_path) == (
            tmp_path / "custom/toolkit/learned"
        ).resolve()
