"""Unit tests for the GRAPH_MIRROR_TARGETS parser (CONCEPT:KG-2.203).

Regression for the live bug where ``create_backend(backend_type="fanout")``
naively comma-split a JSON-array string, so ``["prod-neo4j","team-falkor"]``
became fragments (``'["prod-neo4j"'`` / ``'"team-falkor"]'``) — each then
misread as a backend type ("Unknown graph backend type") and every mirror was
silently dropped ("fanout: no mirrors configured").

The parser must be tolerant of every shape the value arrives in: a JSON-array
string (single- or double-quoted), a comma-separated string (spaces / trailing
commas ok), a single bare value, an already-native list, and empty.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends import _parse_mirror_targets

EXPECTED = ["prod-neo4j", "team-falkor"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The live, failing shape: a JSON-array string (config.json injects list
        # settings into the env as JSON).
        ('["prod-neo4j","team-falkor"]', EXPECTED),
        # JSON array with interior whitespace.
        ('[ "prod-neo4j" , "team-falkor" ]', EXPECTED),
        # Single-quoted "JSON" array (Python-repr style) — falls to comma-split.
        ("['prod-neo4j','team-falkor']", EXPECTED),
        # Plain comma-separated.
        ("prod-neo4j,team-falkor", EXPECTED),
        # Comma-separated with spaces and a trailing comma.
        ("prod-neo4j, team-falkor, ", EXPECTED),
        # A single bare value.
        ("prod-neo4j", ["prod-neo4j"]),
        # Single value as a one-element JSON array.
        ('["prod-neo4j"]', ["prod-neo4j"]),
        # Already a native list (config.json native shape).
        (["prod-neo4j", "team-falkor"], EXPECTED),
        # A list with blank/whitespace items is cleaned.
        (["prod-neo4j", "", "  ", "team-falkor"], EXPECTED),
        # Empty / whitespace / None all yield [].
        ("", []),
        ("   ", []),
        ("[]", []),
        (None, []),
        ([], []),
    ],
)
def test_parse_mirror_targets(raw, expected) -> None:
    assert _parse_mirror_targets(raw) == expected


def test_json_array_does_not_leak_bracket_fragments() -> None:
    """No parsed item may carry stray JSON punctuation (the root of the bug)."""
    parsed = _parse_mirror_targets('["prod-neo4j","team-falkor"]')
    for item in parsed:
        assert "[" not in item and "]" not in item
        assert '"' not in item and "'" not in item


def test_fanout_json_array_targets_attempts_right_mirrors(monkeypatch, caplog) -> None:
    """A fanout backend built with a JSON-array GRAPH_MIRROR_TARGETS resolves the
    correct mirror names — no "Unknown graph backend type" noise from format
    fragments.

    We stub ``_build_member`` (so no live engine/driver is needed in the unit
    env) and assert on the PARSED backend_type each call receives: the authority
    plus the two real mirror names, and never a bracket/quote fragment.
    """
    import logging

    from agent_utilities.knowledge_graph import backends as backends_mod

    settings = {
        "GRAPH_BACKEND": "fanout",
        "GRAPH_AUTHORITY": "epistemic_graph",
        "GRAPH_MIRROR_TARGETS": '["neo4j","falkordb"]',
    }
    monkeypatch.setattr(
        backends_mod, "setting", lambda key, default=None: settings.get(key, default)
    )

    captured: list[str] = []

    class _FakeMember(backends_mod.GraphBackend):
        """Inert, fully-concrete GraphBackend so no live engine/driver is needed."""

        def execute(self, query, params=None):
            return [{"backend": "fake"}]

        def execute_batch(self, query, batch):
            return [{"backend": "fake"}]

        def create_schema(self):
            return None

        def add_embedding(self, node_id, embedding):
            return None

        def semantic_search(self, query_embedding, n_results=5):
            return []

        def prune(self, criteria):
            return None

        def close(self):
            return None

    def _fake_build_member(spec):
        captured.append(str(spec.get("backend_type") or ""))
        return _FakeMember()

    monkeypatch.setattr(backends_mod, "_build_member", _fake_build_member)

    with caplog.at_level(logging.ERROR, logger="agent_utilities.knowledge_graph"):
        backend = backends_mod.create_backend(backend_type="fanout")

    # The parser yielded the two real mirror names (plus the authority), NOT
    # bracket/quote fragments.
    assert "epistemic_graph" in captured  # authority
    assert "neo4j" in captured
    assert "falkordb" in captured
    assert not any("[" in c or "]" in c or '"' in c for c in captured)

    # No "Unknown graph backend type: '[\"neo4j\"'" style error from format noise.
    fragment_errors = [
        r.getMessage()
        for r in caplog.records
        if "Unknown graph backend type" in r.getMessage()
        and ("[" in r.getMessage() or '"' in r.getMessage())
    ]
    assert not fragment_errors, fragment_errors

    # With both mirrors built, a real FanOutBackend is returned.
    assert backend is not None
    # Stop the outbox drainer threads the composite spun up.
    close = getattr(backend, "close", None)
    if callable(close):
        close()
