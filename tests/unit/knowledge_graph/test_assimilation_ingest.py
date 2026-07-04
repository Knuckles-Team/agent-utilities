#!/usr/bin/python
"""Multi-source ingest adapters + granular idempotency (VU-6).

CONCEPT:AU-KG.query.vendor-agnostic-traversal
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    canonical_source_id,
    content_fingerprint,
    ingest_conversations,
    ingest_documents,
)

pytestmark = pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")


class _Graph:
    def __init__(self):
        self._n: dict = {}

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)


class _Engine:
    def __init__(self):
        self.graph = _Graph()
        self.add_calls = 0

    def add_node(self, node_id, node_type, properties=None, ephemeral=False):
        self.add_calls += 1
        self.graph._n[node_id] = {**(properties or {}), "type": node_type}


# --- canonicalization -------------------------------------------------------
def test_canonical_arxiv_collapses_abs_pdf_version():
    a = canonical_source_id("https://arxiv.org/abs/2605.07069")
    b = canonical_source_id("https://arxiv.org/pdf/2605.07069v3")
    assert a == b == "arxiv:2605.07069"


def test_canonical_doi_and_file():
    assert canonical_source_id("https://doi.org/10.1234/xyz") == "doi:10.1234/xyz"
    assert canonical_source_id("/papers/foo.pdf") == "file:/papers/foo.pdf"


def test_fingerprint_ignores_whitespace_and_case():
    assert content_fingerprint("Hello  World") == content_fingerprint("hello world")
    assert content_fingerprint("a") != content_fingerprint("b")


# --- idempotency ------------------------------------------------------------
def test_unchanged_reingest_is_skipped():
    engine = _Engine()
    docs = [{"uri": "/prd/auth.md", "text": "must support OIDC", "title": "Auth PRD"}]
    r1 = ingest_documents(engine, docs)
    assert r1.ingested == 1 and r1.skipped == 0
    r2 = ingest_documents(engine, docs)  # same content
    assert r2.skipped == 1 and r2.ingested == 0
    assert engine.add_calls == 1  # second pass wrote nothing


def test_changed_content_updates_in_place():
    engine = _Engine()
    ingest_documents(engine, [{"uri": "/prd/auth.md", "text": "v1"}])
    r2 = ingest_documents(engine, [{"uri": "/prd/auth.md", "text": "v2 changed"}])
    assert r2.updated == 1 and r2.ingested == 0
    assert len(engine.graph.nodes()) == 1  # same node, updated


def test_same_source_two_uris_collapses():
    engine = _Engine()
    ingest_documents(engine, [{"uri": "https://arxiv.org/abs/2605.07069", "text": "x"}])
    ingest_documents(
        engine, [{"uri": "https://arxiv.org/pdf/2605.07069v2", "text": "x"}]
    )
    assert len(engine.graph.nodes()) == 1  # one canonical node


def test_documents_are_requirement_nodes_and_chat_are_decision():
    engine = _Engine()
    ingest_documents(engine, [{"uri": "/sow/scope.md", "text": "deliver X"}])
    ingest_conversations(
        engine, [{"id": "thread-1", "text": "we chose KG-native dedup"}]
    )
    types = {d["type"] for _, d in engine.graph.nodes(data=True)}
    assert types == {"requirement", "decision"}
