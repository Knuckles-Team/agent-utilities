"""Parser registry + collector tests (CONCEPT:AU-ECO.connector.agent-source-ingestion / ECO-4.42)."""

from __future__ import annotations

import json

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
        {
            "type": "user",
            "timestamp": "2026-06-10T09:00:00Z",
            "message": {"role": "user", "content": "fix the parser"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-06-10T09:00:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "id": "m1",
                "content": [
                    {"type": "thinking", "thinking": "let me look"},
                    {"type": "text", "text": "done"},
                    {"type": "tool_use", "name": "Edit", "id": "t1", "input": {}},
                ],
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 50,
                },
            },
        },
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


def _write_antigravity_session(base, uuid: str):
    """Lay out one Antigravity IDE session: encrypted .pb anchor + brain artifacts."""
    (base / "conversations").mkdir(parents=True, exist_ok=True)
    (base / "conversations" / f"{uuid}.pb").write_bytes(b"\x72\x15encrypted-bytes")
    brain = base / "brain" / uuid
    brain.mkdir(parents=True, exist_ok=True)
    (brain / "task.md").write_text("- [x] Build the antigravity parser\n")
    (brain / "task.md.metadata.json").write_text(
        json.dumps(
            {
                "artifactType": "ARTIFACT_TYPE_TASK",
                "summary": "Parser task",
                "updatedAt": "2026-06-01T01:00:00Z",
            }
        )
    )
    (brain / "implementation_plan.md").write_text("# Plan\nRead brain artifacts.\n")
    (brain / "implementation_plan.md.metadata.json").write_text(
        json.dumps(
            {
                "artifactType": "ARTIFACT_TYPE_IMPLEMENTATION_PLAN",
                "updatedAt": "2026-06-01T01:05:00Z",
            }
        )
    )
    # Snapshot/sidecar files that MUST be ignored (not *.md exactly).
    (brain / "task.md.resolved").write_text("ignored snapshot")
    (brain / "task.md.resolved.0").write_text("ignored snapshot")


def test_antigravity_parser_reads_brain_artifacts(tmp_path, monkeypatch):
    ensure_parsers_loaded()
    base = tmp_path / "antigravity"
    uuid = "64f4ba90-ac5e-4536-bc88-ed7f3ebe87bc"
    _write_antigravity_session(base, uuid)
    monkeypatch.setenv("ANTIGRAVITY_DIR", str(base / "conversations"))
    src = get_source("antigravity")
    assert src.root() is not None
    files = src.discover()
    assert [p.name for p in files] == [f"{uuid}.pb"]
    bundles = list(src.parse(files[0]))
    assert len(bundles) == 1
    b = bundles[0]
    assert b.session.id == uuid
    assert b.session.agent == "antigravity"
    # task.md + implementation_plan.md = 2 messages; the .resolved snapshots ignored.
    assert b.session.message_count == 2
    assert b.session.user_message_count == 1  # task.md → user turn
    assert b.session.first_message == "Parser task"  # metadata summary preferred
    roles = [m.role for m in b.messages]
    assert roles == ["user", "assistant"]  # ordered by updatedAt


def test_antigravity_encrypted_only_session_yields_nothing(tmp_path, monkeypatch):
    """A session with only the encrypted .pb (no brain dir) yields no bundle."""
    ensure_parsers_loaded()
    base = tmp_path / "antigravity"
    (base / "conversations").mkdir(parents=True)
    (base / "conversations" / "deadbeef-0000-0000-0000-000000000000.pb").write_bytes(
        b"\x00encrypted"
    )
    monkeypatch.setenv("ANTIGRAVITY_DIR", str(base / "conversations"))
    src = get_source("antigravity")
    files = src.discover()
    assert len(files) == 1
    assert list(src.parse(files[0])) == []


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


def test_upload_local_sessions_pushes_bundles_to_remote(monkeypatch):
    """Remote-engine path: client parses local logs and pushes via MCP upload."""
    import agent_utilities.ingestion.collector as col
    import agent_utilities.protocols.source_connectors.connectors.mcp_tool as mcp_tool
    from agent_utilities.usage.models import ParsedSessionBundle, UsageSession

    calls: list[int] = []

    async def _fake_call(**kw):
        assert kw["tool"] == "ingest_sessions"
        assert kw["action"] == "upload"
        assert kw["params_style"] == "args"
        bundles = json.loads(kw["params"]["bundles_json"])
        calls.append(len(bundles))
        return {"received": len(bundles), "ingested": len(bundles)}

    def _fake_iter(only_changed=True, backend=None):
        for i in range(3):
            agent = "antigravity" if i else "claude"
            yield (
                f"/p{i}",
                1,
                2,
                ParsedSessionBundle(session=UsageSession(id=f"s{i}", agent=agent)),
            )

    monkeypatch.setattr(mcp_tool, "call_tool_once", _fake_call)
    monkeypatch.setattr(col, "iter_local_bundles", _fake_iter)
    res = col.upload_local_sessions(server="graph-os", batch=2)
    assert res["mode"] == "upload" and res["transport"] == "mcp"
    assert res["received"] == 3 and res["ingested"] == 3
    assert res["agents"] == ["antigravity", "claude"]
    assert calls == [2, 1]  # batched 2 + 1


def test_collect_paths_into_store(tmp_path, monkeypatch):
    monkeypatch.setenv("USAGE_DB_PATH", str(tmp_path / "u.db"))
    import agent_utilities.usage.recorder as rec_mod
    from agent_utilities.usage import backends, get_usage_backend

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
