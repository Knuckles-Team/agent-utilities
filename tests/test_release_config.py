"""Release configuration keeps changelog headings across version bumps."""

from configparser import RawConfigParser
from datetime import datetime
from pathlib import Path


def test_bump2version_changelog_replacement_preserves_release_history():
    config = RawConfigParser()
    config.read(Path(__file__).resolve().parents[1] / ".bumpversion.cfg")
    section = config["bumpversion:file:CHANGELOG.md"]
    context = {"new_version": "4.0.0", "now": datetime(2026, 10, 25), "#": "#"}
    search = section["search"].format(**context)
    replacement = section["replace"].format(**context)
    history = "## [3.2.0] - 2026-10-24\n\nExisting release notes.\n"
    changelog = "## [Unreleased]\n\n" + history

    updated = changelog.replace(search, replacement)

    assert updated == "## [Unreleased]\n\n## [4.0.0] - 2026-10-25\n\n" + history
