"""Unit tests for ``slugify_skill_name`` — the single source of truth for the
slash-command slug shared between the command registry and the /v1/skills listing.
"""

import pytest

from agent.skill_commands import slugify_skill_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("site-audit", "site-audit"),
        ("Site Audit", "site-audit"),
        ("create_pdf", "create-pdf"),
        ("My Cool Skill", "my-cool-skill"),
        ("Foo + Bar / Baz", "foo-bar-baz"),  # invalid chars dropped, runs collapsed
        ("  spaced  ", "spaced"),
        ("Trailing-", "trailing"),  # leading/trailing hyphens trimmed
        ("-Leading", "leading"),
        ("a--b", "a-b"),  # multi-hyphen collapse
        ("ALLCAPS", "allcaps"),
    ],
)
def test_slugify_produces_expected_slug(name, expected):
    assert slugify_skill_name(name) == expected


def test_slugify_returns_empty_when_nothing_usable_remains():
    # A name made entirely of invalid characters reduces to "".
    assert slugify_skill_name("+++") == ""
    assert slugify_skill_name("   ") == ""
