"""CONCEPT:OS-5.30 — declarative skill/skill-workflow scheduler."""

from __future__ import annotations

from datetime import datetime

from agent_utilities.core import skill_scheduler as s


def _at(h: int, m: int) -> datetime:
    return datetime(2026, 6, 15, h, m)  # 2026-06-15 is a Monday


def test_cron_matcher_fields() -> None:
    assert s.cron_matches("0 */4 * * *", _at(4, 0))
    assert not s.cron_matches("0 */4 * * *", _at(5, 0))
    assert not s.cron_matches("0 */4 * * *", _at(4, 30))
    assert s.cron_matches("* * * * *", _at(4, 30))
    assert s.cron_matches("30 9 * * 1", _at(9, 30))  # Monday 09:30
    assert not s.cron_matches("30 9 * * 0", _at(9, 30))  # Sunday only
    assert s.cron_matches("0 0,12 * * *", _at(12, 0))
    assert s.cron_matches("0 9-17 * * *", _at(13, 0))


def test_run_due_dispatches_matching_and_does_not_double_fire(
    monkeypatch, tmp_path
) -> None:
    fired: list[str] = []

    def _handler(engine, entry):  # noqa: ANN001
        fired.append(entry["name"])
        return {"status": "ok"}

    # mock the registry + the deterministic handler + isolate state file
    monkeypatch.setattr(
        s,
        "load_schedules",
        lambda: [
            {
                "name": "job-a",
                "cron": "0 */4 * * *",
                "kind": "skill",
                "ref": "code-enhancer",
                "action": "liveness",
                "enabled": True,
            },
            {
                "name": "job-b",
                "cron": "0 5 * * *",
                "kind": "skill",
                "ref": "code-enhancer",
                "action": "liveness",
                "enabled": True,
            },
        ],
    )
    monkeypatch.setitem(s._SKILL_HANDLERS, ("code-enhancer", "liveness"), _handler)
    monkeypatch.setattr(s, "_STATE_FILE", tmp_path / "state.json")

    # 04:00 fires only job-a (job-b is 05:00)
    res = s.run_due_schedules(object(), now=_at(4, 0))
    assert res["fired"] == ["job-a"]
    # same minute again → no double-fire (state persisted)
    assert s.run_due_schedules(object(), now=_at(4, 0))["fired"] == []
    # 05:00 fires job-b
    assert s.run_due_schedules(object(), now=_at(5, 0))["fired"] == ["job-b"]
    assert fired == ["job-a", "job-b"]


def test_load_schedules_filters_disabled(monkeypatch, tmp_path) -> None:
    reg = tmp_path / "schedules.yml"
    reg.write_text(
        "schedules:\n"
        "  - name: on-job\n    cron: '* * * * *'\n    enabled: true\n"
        "  - name: off-job\n    cron: '* * * * *'\n    enabled: false\n"
    )
    monkeypatch.setattr(s, "_registry_path", lambda: reg)
    names = [e["name"] for e in s.load_schedules()]
    assert names == ["on-job"]  # enabled=false is filtered out
