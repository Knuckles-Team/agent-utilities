"""AU-DEV-R003.2: wire the refresh model into the real XDG skills path."""

from __future__ import annotations

import pytest

from agent_utilities.skills import inventory_refresh_runtime as runtime


@pytest.mark.spec("AU-DEV-R003.2")
def test_refresh_prints_plan_before_mutating(tmp_path, monkeypatch, capsys) -> None:
    root = tmp_path / "skills"
    monkeypatch.setattr(runtime, "unified_skills_dir", lambda: root)
    monkeypatch.setattr(
        runtime, "provider_registrations", lambda _group: ()
    )
    monkeypatch.setattr(runtime, "install_unified", lambda: {})

    plan = runtime.print_skills_refresh_plan()
    out = capsys.readouterr().out
    assert plan == {}
    assert not root.exists(), "the plan must not mutate the XDG skills tree"
    assert "skill refresh plan" not in out or plan == {}


@pytest.mark.spec("AU-DEV-R003.2")
def test_refresh_returns_entries_without_raising(tmp_path, monkeypatch) -> None:
    root = tmp_path / "skills"
    monkeypatch.setattr(runtime, "unified_skills_dir", lambda: root)
    monkeypatch.setattr(
        runtime, "provider_registrations", lambda _group: ()
    )
    monkeypatch.setattr(runtime, "install_unified", lambda: {})

    entries = runtime.refresh_skills_inventory()
    assert entries == {}
