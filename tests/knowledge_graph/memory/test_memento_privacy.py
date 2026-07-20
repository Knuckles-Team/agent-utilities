"""Privacy and current-schema contract for durable Memento records."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.memory import memento_compressor as mc


@pytest.fixture
def engine() -> MagicMock:
    value = MagicMock()
    value.backend = MagicMock()
    return value


def _enable_encrypted_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMENTO_RAW_RETENTION_ENABLED", "true")
    monkeypatch.setenv("MEMENTO_RAW_RETENTION_POLICY", mc.MEMENTO_RAW_RETENTION_POLICY)
    monkeypatch.setenv("MEMENTO_RAW_ENCRYPTION_KEY_REF", "secret://tests/memento-key")
    monkeypatch.setattr(
        mc, "_resolve_secret_reference", lambda _reference: "unit-test-key-material"
    )


def test_default_retention_never_writes_raw_or_source_identity(
    monkeypatch: pytest.MonkeyPatch, engine: MagicMock
) -> None:
    monkeypatch.delenv("MEMENTO_RAW_RETENTION_ENABLED", raising=False)
    monkeypatch.delenv("MEMENTO_RAW_RETENTION_POLICY", raising=False)
    monkeypatch.delenv("MEMENTO_RAW_ENCRYPTION_KEY_REF", raising=False)
    raw = "private runtime conversation"
    source = "runtime-session-key"

    memento_id = mc._persist_memento(
        engine,
        "Contact synthetic.person@example.invalid from C:\\Users\\agent-user\\notes.txt",
        source=source,
        raw_block=raw,
    )

    assert memento_id
    assert [call.args[1] for call in engine.add_node.call_args_list] == ["Memento"]
    props = engine.add_node.call_args.kwargs["properties"]
    assert props["recoverable"] is False
    assert props["source"] == mc.memento_source_reference(source)
    assert source not in repr(props)
    assert raw not in repr(props)
    assert "synthetic.person@example.invalid" not in props["content"]
    assert "C:\\Users\\agent-user" not in props["content"]
    engine.link_nodes.assert_not_called()


def test_wrong_policy_fails_closed(
    monkeypatch: pytest.MonkeyPatch, engine: MagicMock
) -> None:
    monkeypatch.setenv("MEMENTO_RAW_RETENTION_ENABLED", "true")
    monkeypatch.setenv("MEMENTO_RAW_RETENTION_POLICY", "unapproved")
    monkeypatch.setenv("MEMENTO_RAW_ENCRYPTION_KEY_REF", "secret://tests/memento-key")

    mc._persist_memento(
        engine, "safe memento", source="runtime-source", raw_block="raw transcript"
    )

    labels = [call.args[1] for call in engine.add_node.call_args_list]
    assert labels == ["Memento"]
    assert engine.add_node.call_args.kwargs["properties"]["recoverable"] is False


def test_approved_raw_retention_is_authenticated_and_secret_backed(
    monkeypatch: pytest.MonkeyPatch, engine: MagicMock
) -> None:
    _enable_encrypted_retention(monkeypatch)
    raw = "exact private transcript"

    memento_id = mc._persist_memento(
        engine, "safe state", source="runtime-source", raw_block=raw
    )
    block_call = next(
        call
        for call in engine.add_node.call_args_list
        if call.args[1] == "EvictedBlock"
    )
    block_id = block_call.args[0]
    props = block_call.kwargs["properties"]

    assert raw not in repr(props)
    assert "secret://" not in repr(props)
    assert props["content"] == mc._ENCRYPTED_CONTENT_MARKER
    assert props["encryption_algorithm"] == mc.MEMENTO_RAW_ENCRYPTION_ALGORITHM
    assert props["key_reference"].startswith("pref_memento_raw_key_")

    engine.backend.execute.return_value = [{"id": block_id, **props}]
    assert mc.recover_evicted_block(engine, str(memento_id)) == raw


def test_current_source_lookup_sanitizes_returned_content(engine: MagicMock) -> None:
    source = "runtime-session"
    source_ref = mc.memento_source_reference(source)
    engine.backend.execute.return_value = [
        {
            "id": "memento-1",
            "content": "Contact synthetic.person@example.invalid",
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    result = mc.get_recent_mementos(engine, source, limit=1)

    assert result == ["Contact [REDACTED_EMAIL]"]
    assert engine.backend.execute.call_args.args[1]["source"] == source_ref
    sanitized = engine.add_node.call_args.kwargs["properties"]
    assert sanitized["source"] == source_ref
    assert "synthetic.person@example.invalid" not in repr(sanitized)


def test_plaintext_record_recovery_fails_closed(engine: MagicMock) -> None:
    engine.backend.execute.return_value = [
        {"id": "plaintext-block-id", "content": "private transcript"}
    ]

    assert mc.recover_evicted_block(engine, "memento-1") is None
    engine.add_node.assert_not_called()
