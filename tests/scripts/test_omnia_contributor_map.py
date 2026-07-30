"""Omnia contributors remain attributed by the 0.19 release pipeline."""

from scripts import release


def test_omnia_contributor_emails_resolve_to_github_logins():
    assert release.AUTHOR_MAP["miguel@mff.io"] == "miguelff"
    assert release.AUTHOR_MAP["pablopazosp3@gmail.com"] == "ppazosp"
    assert release.AUTHOR_MAP["pablo@useomnia.com"] == "ppazosp"
    assert release.AUTHOR_MAP["uesteibar@gmail.com"] == "uesteibar"
