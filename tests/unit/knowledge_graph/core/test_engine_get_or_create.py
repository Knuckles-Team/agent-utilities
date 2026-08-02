"""D-03: IntelligenceGraphEngine.get_or_create() observability guard.

``get_or_create()`` is the one process-owned engine acquisition path. The MCP
server's own boot (``kg_server.py``'s ``_start_engine_bootstrap``) always
passes ``factory=`` with ``defer_background_start=True``, closing the
materialization-wait race for that entrypoint — but ~25 OTHER call sites
across the package call ``get_or_create()`` with no ``factory=`` at all, which
constructs with ``defer_background_start`` defaulting to ``False``. That is
correct for those callers today (standalone CLI tools/workers with no later
un-defer step), but would be a real, previously-invisible bug if any of them
ever won the process-wide singleton race ahead of the MCP server's own
bootstrap. This module tests the D-03 hardening: a ``logger.warning`` fires
exactly when a caller constructs the singleton without
``defer_background_start=True`` and without a ``factory=`` seam, and stays
silent for the two paths that ARE covered (an explicit ``factory=``, or
``defer_background_start=True`` passed directly).

Uses a lightweight ``IntelligenceGraphEngine`` subclass with a no-op
``__init__`` so the guard's LOGGING behavior is tested in isolation, without
constructing a real engine (backend/socket/threads) — the guard fires before
any of that heavy setup runs.
"""

from __future__ import annotations

import logging

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

_LOGGER_NAME = "agent_utilities.knowledge_graph.core.engine"


class _NoOpEngine(IntelligenceGraphEngine):
    """A subclass whose ``__init__`` skips all real engine construction.

    Intentionally does NOT call ``super().__init__()`` — that would build a
    real backend/graph/schema pack, which this test has no need for; it only
    exercises ``get_or_create()``'s own logic, not engine internals. Uses its
    own ``_ACTIVE_ENGINE`` class attribute (set per-test below), isolated from
    the real ``IntelligenceGraphEngine._ACTIVE_ENGINE`` singleton.
    """

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_get_or_create_warns_when_constructed_without_deferred_start(caplog):
    _NoOpEngine._ACTIVE_ENGINE = None
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _NoOpEngine.get_or_create()
        assert isinstance(result, _NoOpEngine)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "WITHOUT defer_background_start" in m and "D-03" in m for m in warnings
        )
        # The caller's own frame (this test function) is what should be named.
        assert any("test_get_or_create_warns" in m for m in warnings)
    finally:
        _NoOpEngine._ACTIVE_ENGINE = None


def test_get_or_create_silent_with_explicit_deferred_start(caplog):
    _NoOpEngine._ACTIVE_ENGINE = None
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _NoOpEngine.get_or_create(defer_background_start=True)
        assert isinstance(result, _NoOpEngine)
        assert result.kwargs == {"defer_background_start": True}
        assert not any(
            "WITHOUT defer_background_start" in r.message for r in caplog.records
        )
    finally:
        _NoOpEngine._ACTIVE_ENGINE = None


def test_get_or_create_silent_with_factory_seam(caplog):
    """The one production direct-construction call site (kg_server.py) uses
    ``factory=`` — the guard must never fire on that path, matching the
    D-03 investigation's finding that it is the sole covered call site."""
    _NoOpEngine._ACTIVE_ENGINE = None
    sentinel = _NoOpEngine.__new__(_NoOpEngine)
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _NoOpEngine.get_or_create(factory=lambda: sentinel)
        assert result is sentinel
        assert not any(
            "WITHOUT defer_background_start" in r.message for r in caplog.records
        )
    finally:
        _NoOpEngine._ACTIVE_ENGINE = None


def test_get_or_create_only_warns_on_the_winning_construction(caplog):
    """A second call after the singleton already exists must return the
    cached instance without constructing again — and without re-warning."""
    _NoOpEngine._ACTIVE_ENGINE = None
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            first = _NoOpEngine.get_or_create()
            second = _NoOpEngine.get_or_create()
        assert first is second
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "WITHOUT defer_background_start" in r.message
        ]
        assert len(warnings) == 1
    finally:
        _NoOpEngine._ACTIVE_ENGINE = None
