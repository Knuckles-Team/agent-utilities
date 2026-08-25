"""Argv construction for the auto-started ``epistemic-graph-server`` (BUG-PE-003).

Covers the single-container collapse fix in
``GraphComputeEngine._autostart_engine``: UDS and TCP transports must be
independent (not mutually exclusive), the new TLS/metrics flags must be
settings-gated and optional, and the unset-``--persist-dir`` fallback must
warn loudly instead of silently choosing a path that may not be durable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_utilities.knowledge_graph.core import graph_compute as gc


class TestBuildEngineTransportArgv:
    def test_uds_only_is_byte_identical_to_prior_behavior(self, monkeypatch) -> None:
        """Today's default: only ``GRAPH_SERVICE_ENDPOINTS=unix://...`` set."""

        monkeypatch.delenv("GRAPH_SERVICE_TCP_ADDR", raising=False)
        monkeypatch.delenv("GRAPH_SERVICE_TLS_CERT", raising=False)
        monkeypatch.delenv("GRAPH_SERVICE_TLS_KEY", raising=False)
        monkeypatch.delenv("GRAPH_SERVICE_METRICS_ADDR", raising=False)

        cmd = gc._build_engine_transport_argv(
            "/usr/local/bin/epistemic-graph-server",
            "/run/epistemic-graph/epistemic-graph.sock",
            {},
        )

        assert cmd == [
            "/usr/local/bin/epistemic-graph-server",
            "--socket-path",
            "/run/epistemic-graph/epistemic-graph.sock",
        ]

    def test_collapse_configuration_matches_sidecar_argv(self, monkeypatch) -> None:
        """UDS + TCP + TLS + metrics: the exact argv the sidecar hardcodes today."""

        monkeypatch.setenv("GRAPH_SERVICE_TCP_ADDR", "0.0.0.0:9100")
        monkeypatch.setenv("GRAPH_SERVICE_TLS_CERT", "/etc/eg-tls/eng-tls.crt")
        monkeypatch.setenv("GRAPH_SERVICE_TLS_KEY", "/etc/eg-tls/eng-tls.key")
        monkeypatch.setenv("GRAPH_SERVICE_METRICS_ADDR", "127.0.0.1:9101")

        cmd = gc._build_engine_transport_argv(
            "/usr/local/bin/epistemic-graph-server",
            "/run/epistemic-graph/epistemic-graph.sock",
            {},
        )

        assert cmd == [
            "/usr/local/bin/epistemic-graph-server",
            "--socket-path",
            "/run/epistemic-graph/epistemic-graph.sock",
            "--tcp-addr",
            "0.0.0.0:9100",
            "--tcp-tls-cert",
            "/etc/eg-tls/eng-tls.crt",
            "--tcp-tls-key",
            "/etc/eg-tls/eng-tls.key",
            "--metrics-addr",
            "127.0.0.1:9101",
        ]

    def test_tcp_addr_only_windows_loopback_path_unchanged(self, monkeypatch) -> None:
        """No UDS (``sock`` unset): only the resolver-supplied ``tcp_addr`` wins."""

        monkeypatch.delenv("GRAPH_SERVICE_TCP_ADDR", raising=False)
        monkeypatch.delenv("GRAPH_SERVICE_TLS_CERT", raising=False)
        monkeypatch.delenv("GRAPH_SERVICE_TLS_KEY", raising=False)
        monkeypatch.delenv("GRAPH_SERVICE_METRICS_ADDR", raising=False)

        cmd = gc._build_engine_transport_argv(
            "/usr/local/bin/epistemic-graph-server",
            None,
            {"tcp_addr": "127.0.0.1:54732"},
        )

        assert cmd == [
            "/usr/local/bin/epistemic-graph-server",
            "--tcp-addr",
            "127.0.0.1:54732",
        ]

    def test_tls_and_metrics_flags_are_independently_optional(self, monkeypatch) -> None:
        """Only the cert is set: only ``--tcp-tls-cert`` is emitted, nothing else."""

        monkeypatch.delenv("GRAPH_SERVICE_TCP_ADDR", raising=False)
        monkeypatch.setenv("GRAPH_SERVICE_TLS_CERT", "/etc/eg-tls/eng-tls.crt")
        monkeypatch.delenv("GRAPH_SERVICE_TLS_KEY", raising=False)
        monkeypatch.delenv("GRAPH_SERVICE_METRICS_ADDR", raising=False)

        cmd = gc._build_engine_transport_argv(
            "/usr/local/bin/epistemic-graph-server",
            "/run/epistemic-graph/epistemic-graph.sock",
            {},
        )

        assert cmd == [
            "/usr/local/bin/epistemic-graph-server",
            "--socket-path",
            "/run/epistemic-graph/epistemic-graph.sock",
            "--tcp-tls-cert",
            "/etc/eg-tls/eng-tls.crt",
        ]


class TestResolveEnginePersistDir:
    def test_explicit_setting_wins_and_is_silent(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("GRAPH_SERVICE_PERSIST_DIR", "/data/graph_snapshots")

        with caplog.at_level(logging.WARNING, logger=gc.__name__):
            persist_dir = gc._resolve_engine_persist_dir()

        assert persist_dir == "/data/graph_snapshots"
        assert not caplog.records

    def test_unset_falls_back_and_warns_loudly(self, monkeypatch, caplog) -> None:
        """The warning must carry the full risk signal while leaking no path.

        The process-wide log privacy boundary
        (``agent_utilities.core.log_privacy``) strips filesystem paths from
        every ``agent_utilities.*`` record, so interpolating the resolved
        directory would render as ``<path>`` and tell an operator nothing
        actionable. Naming the controlling env var in the STATIC message text
        instead conveys exactly what the operator needs — which setting
        decides this, and that it was not set explicitly — while emitting
        nothing the boundary would have to redact. Asserts both halves: the
        env var name is present, and the resolved path is absent.
        """

        monkeypatch.delenv("GRAPH_SERVICE_PERSIST_DIR", raising=False)
        monkeypatch.setenv(
            "AGENT_UTILITIES_DATA_DIR", "/tmp/.local/share/agent-utilities"
        )

        with caplog.at_level(logging.WARNING, logger=gc.__name__):
            persist_dir = gc._resolve_engine_persist_dir()

        expected = str(Path("/tmp/.local/share/agent-utilities") / "graph_snapshots")
        assert persist_dir == expected
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        assert "GRAPH_SERVICE_PERSIST_DIR is not set" in message
        # The controlling setting is named, so the operator knows what to fix.
        assert "AGENT_UTILITIES_DATA_DIR" in message
        # ...and no filesystem path is emitted, so nothing needs redacting.
        assert expected not in message
        assert "<path>" not in message
