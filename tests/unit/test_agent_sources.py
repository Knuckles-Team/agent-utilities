"""Parser registry + collector tests (CONCEPT:ECO-4.38 / ECO-4.42)."""

from __future__ import annotations

import json

import pytest

from agent_utilities.ingestion.agent_sources import (
    detect_installed,
    ensure_parsers_loaded,
    get_source,
)
from agent_utilities.ingestion.agent_sources.parsers.claude import parse as claude_parse
from agent_utilities.ingestion.collector import collect_paths, iter_local_bundles


def test_registry_has_all_36(monkeypatch):
    ensure_parsers_loaded()
    from agent_utilities.ingestion.agent_sources import all_sources

    sources = all_sources()
    assert len(sources) >= 36
    assert get_source("claude") is not None
    assert get_source("codex") is not None
    assert get_source("cursor") is not None


def test_auto_detection_probes_dirs(tmp_path, monkeypatch):
    ensure_parsers_loaded()
    # Point claude at an empty temp dir → not "installed".
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(tmp_path / "nope"))
    src = get_source("claude")
    assert src.root() is None
    # Create the dir → now detected.
    (tmp_path / "yes").mkdir()
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(tmp_path / "yes"))
    assert src.root() is not None
    assert any(s.agent_type == "claude" for s in detect_installed())


def _write_claude_log(path):
    lines = [
        {"type": "user", "timestamp": "2026-06-10T09:00:00Z",
         "message": {"role": "user", "content": "fix the parser"}},
        {"type": "assistant", "timestamp": "2026-06-10T09:00:05Z",
         "message": {
             "role": "assistant", "model": "claude-opus-4-8", "id": "m1",
             "content": [
                 {"type": "thinking", "thinking": "let me look"},
                 {"type": "text", "text": "done"},
                 {"type": "tool_use", "name": "Edit", "id": "t1", "input": {}},
             ],
             "usage": {"input_tokens": 1000, "output_tokens": 200,
                       "cache_read_input_tokens": 50},
         }},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))


def test_claude_parser_real_shape(tmp_path):
    proj = tmp_path / "-home-apps-foo"
    proj.mkdir()
    log = proj / "sess-abc.jsonl"
    _write_claude_log(log)
    src = get_source("claude")
    ensure_parsers_loaded()
    bundles = list(claude_parse(log, src))
    assert len(bundles) == 1
    b = bundles[0]
    assert b.session.id == "sess-abc"
    assert b.session.project == "/home/apps/foo"
    assert b.session.message_count == 2
    assert b.session.user_message_count == 1
    assert b.messages[1].has_tool_use is True
    assert b.tool_calls[0].tool_name == "Edit"
    assert b.tool_calls[0].category == "edit"
    e = b.usage_events[0]
    assert e.input_tokens == 1000 and e.output_tokens == 200
    assert e.cache_read_input_tokens == 50


def test_iter_local_bundles_detects_and_parses(tmp_path, monkeypatch):
    ensure_parsers_loaded()
    proj = tmp_path / "claude" / "-home-x"
    proj.mkdir(parents=True)
    _write_claude_log(proj / "s1.jsonl")
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(tmp_path / "claude"))
    # Isolate detection to just claude by pointing others nowhere is overkill;
    # just assert our session shows up.
    found = [b for _p, _m, _s, b in iter_local_bundles(only_changed=False)]
    assert any(b.session.id == "s1" for b in found)


def test_collect_paths_into_store(tmp_path, monkeypatch):
    monkeypatch.setenv("USAGE_DB_PATH", str(tmp_path / "u.db"))
    from agent_utilities.usage import backends, get_usage_backend
    import agent_utilities.usage.recorder as rec_mod

    backends.reset_usage_backend_for_tests()
    rec_mod._recorder = None

    proj = tmp_path / "-home-y"
    proj.mkdir()
    _write_claude_log(proj / "s2.jsonl")
    result = collect_paths([proj / "s2.jsonl"])
    assert result["ingested"] == 1
    backend = get_usage_backend()
    assert backend.summary().session_count == 1
    backends.reset_usage_backend_for_tests()
