"""BUG-PE-055 (Python half): never mint an engine encryption-at-rest key
silently.

``_load_or_create_engine_encryption_key`` previously minted a fresh key under
``AGENT_UTILITIES_DATA_DIR`` -- an emptyDir on the deployed pod -- with no log
line at all, and with no check of whether the durable store resolved from
``GRAPH_SERVICE_PERSIST_DIR`` already holds data (in which case a new key
means that data becomes unreachable, or the engine refuses to encrypt over
existing plaintext). This pins the loud, privacy-safe warning added on the
branch that actually mints a key -- never on the branch that reads an
existing one.
"""

from __future__ import annotations

import logging

from agent_utilities.knowledge_graph.core import graph_compute as gc


def _key_warnings(caplog):
    return [
        record
        for record in caplog.records
        if "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF" in record.getMessage()
    ]


class TestWarnNewEngineEncryptionKey:
    def test_minting_a_new_key_warns_exactly_once_and_reading_it_back_is_silent(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
        # An explicitly-set, empty persist dir keeps `_resolve_engine_persist_dir`
        # silent (it only warns when the setting is UNSET) and keeps
        # `_engine_persist_dir_holds_data()` False, isolating this test to the
        # single warning under test.
        empty_persist_dir = tmp_path / "graph_snapshots"
        empty_persist_dir.mkdir()
        monkeypatch.setenv("GRAPH_SERVICE_PERSIST_DIR", str(empty_persist_dir))

        with caplog.at_level(logging.WARNING, logger=gc.__name__):
            first = gc._load_or_create_engine_encryption_key()

        warnings = _key_warnings(caplog)
        assert len(warnings) == 1
        assert "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF" in warnings[0].getMessage()
        assert "ALREADY HOLDS DATA" not in warnings[0].getMessage()
        # Privacy: the message must carry the setting NAME, never a filesystem
        # location the log-privacy boundary would otherwise have to redact.
        assert "/" not in warnings[0].getMessage()

        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=gc.__name__):
            second = gc._load_or_create_engine_encryption_key()

        assert second == first
        assert _key_warnings(caplog) == []

    def test_minting_over_an_already_populated_persist_dir_escalates_the_message(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "data"))
        populated_persist_dir = tmp_path / "graph_snapshots"
        populated_persist_dir.mkdir()
        (populated_persist_dir / "existing.snapshot").write_bytes(b"x")
        monkeypatch.setenv("GRAPH_SERVICE_PERSIST_DIR", str(populated_persist_dir))

        with caplog.at_level(logging.WARNING, logger=gc.__name__):
            gc._load_or_create_engine_encryption_key()

        warnings = _key_warnings(caplog)
        assert len(warnings) == 1
        assert "ALREADY HOLDS DATA" in warnings[0].getMessage()
        assert "/" not in warnings[0].getMessage()


class TestEnginePersistDirHoldsData:
    def test_false_when_unresolvable(self, monkeypatch) -> None:
        monkeypatch.setattr(gc, "_resolve_engine_persist_dir", lambda: None)
        assert gc._engine_persist_dir_holds_data() is False

    def test_false_when_empty(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(gc, "_resolve_engine_persist_dir", lambda: str(tmp_path))
        assert gc._engine_persist_dir_holds_data() is False

    def test_true_when_populated(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "snapshot").write_bytes(b"x")
        monkeypatch.setattr(gc, "_resolve_engine_persist_dir", lambda: str(tmp_path))
        assert gc._engine_persist_dir_holds_data() is True

    def test_false_on_missing_directory(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            gc, "_resolve_engine_persist_dir", lambda: str(tmp_path / "nope")
        )
        assert gc._engine_persist_dir_holds_data() is False
