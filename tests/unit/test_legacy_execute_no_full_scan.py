"""Hardening (CONCEPT:AU-ORCH.session.invoker-agent-handoff): an unscoped query must NOT silently return the whole graph.

This originally guarded the `graph_context list` "garbage" over-match: an
unparsed WHERE on the legacy client-side Cypher reader (``EpistemicGraphBackend
._legacy_execute``, a params-dict mini-DSL) fell through to a full-graph scan
gated only by a retired boolean opt-in environment variable.

That reader is retired outright (CONCEPT:AU-P0-2, "kg: route general Cypher in
EpistemicGraphBackend to the native engine", commit ``213fcdce``): ``execute()``
now hands every query straight to the native engine's own Cypher parser
(``GraphComputeEngine.query_cypher``) instead of a client-side regex
scan-and-eval, and there is no more ``_legacy_execute``/full-scan opt-in
escape hatch at all —
``tests/unit/test_native_cypher_routing.py``'s ``retired`` tuple independently
enforces that this and its sibling client-side interpreter helpers never
reappear in ``execute``/``execute_read``/``execute_write``/``execute_batch``'s
source. The invariant these tests guard is unchanged and, if anything,
stronger: a query the native parser cannot make sense of now RAISES
(``CypherEngineError``) rather than falling through to ANY scan, legacy or
otherwise — there is no silent path to "return everything" left to gate at
all. A caller that genuinely wants every node still gets it, but only by
writing that scan explicitly (``MATCH (n) RETURN n``), never by omission.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    CypherEngineError,
    EpistemicGraphBackend,
)


class _FakeGraph:
    """A minimal native-Cypher stand-in covering exactly the two shapes these
    tests need — an id-anchored point lookup and a deliberate, explicit full
    scan — and REJECTING (not silently scanning) anything else, the same way
    the real engine's parser rejects a shape it does not recognize."""

    def __init__(self):
        self._nodes = {
            "ctx:s1:a": {
                "id": "ctx:s1:a",
                "node_type": "ContextBlob",
                "session_id": "s1",
                "content": "x",
            },
            "doc:9": {
                "id": "doc:9",
                "node_type": "Document",
                "content": "unrelated textbook chunk",
            },
        }

    def query_cypher(self, query: str):
        stripped = query.strip()
        if stripped == "MATCH (n) RETURN n":
            # A deliberate, EXPLICIT full scan — the caller wrote exactly this,
            # so getting every node back is correct, not a garbage over-match.
            return [{"n": dict(props)} for props in self._nodes.values()]
        if stripped.startswith("MATCH (n) WHERE n.id = ") and stripped.endswith(
            "RETURN n"
        ):
            literal = stripped[len("MATCH (n) WHERE n.id = ") : -len("RETURN n")]
            literal = literal.strip().strip("'")
            props = self._nodes.get(literal)
            return [{"n": dict(props)}] if props else []
        # Anything else (an unparsed/unsupported predicate, an empty/garbled
        # WHERE, ...) is a shape this fake native parser does not understand —
        # raise, exactly like the real engine's parser does, instead of
        # degrading to any kind of scan.
        raise NotImplementedError(f"unsupported test query shape: {query!r}")


def _backend() -> EpistemicGraphBackend:
    b = object.__new__(EpistemicGraphBackend)  # bypass engine-connecting __init__
    b._graph = _FakeGraph()
    return b


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_unscoped_query_returns_empty_not_all_nodes():
    """An unparseable/unscoped predicate RAISES — it must never silently
    degrade to a full-graph scan, legacy or otherwise."""
    b = _backend()
    with pytest.raises(CypherEngineError) as exc_info:
        b.execute_read("MATCH (n) WHERE n.bogus_predicate RETURN n")
    assert exc_info.value.mode == "read"


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_id_lookup_still_precise():
    b = _backend()
    rows = b.execute_read("MATCH (n) WHERE n.id = $id RETURN n", {"id": "ctx:s1:a"})
    assert len(rows) == 1 and rows[0]["n"]["id"] == "ctx:s1:a"


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_explicit_opt_in_allows_full_scan():
    """The old boolean env-var opt-in is retired along with the
    legacy reader it gated — a full scan is opted into by WRITING one
    explicitly now, not by an environment flag."""
    b = _backend()
    assert len(b.execute_read("MATCH (n) RETURN n")) == 2
