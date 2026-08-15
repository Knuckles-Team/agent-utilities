"""Unit-test fixtures shared across ``tests/unit``.

Keeps the unit suite hermetic: it must never open a real network connection.
The embedding factory (``create_embedding_model``) otherwise defaults to a live
``OpenAIEmbedding`` client whenever the ``openai`` extra is installed (as it is
under ``uv run --all-extras`` in pre-commit), which makes embedding-dependent
"unit" tests hang on a refused TCP connection until the pytest timeout fires.

We neutralize it the same way a provider-less environment does: the factory
raises, and :class:`HybridRetriever` transparently falls back to its lexical
path (``embed_model is None``). Tests that need a functioning embedder still
``@patch`` the factory themselves, and that per-test patch overrides this one.
"""

import pytest


@pytest.fixture(autouse=True)
def _hermetic_embeddings(monkeypatch):
    """Block the live embedding client so unit tests never touch the network."""

    def _no_network_embeddings(*args, **kwargs):
        raise RuntimeError(
            "create_embedding_model is disabled in the unit suite to keep it "
            "hermetic; patch it explicitly in tests that need an embedder."
        )

    # Patch the canonical factory plus every module that imported it by name,
    # so already-bound references are intercepted too.
    for target in (
        "agent_utilities.core.embedding_utilities.create_embedding_model",
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model",
    ):
        monkeypatch.setattr(target, _no_network_embeddings, raising=False)


@pytest.fixture(autouse=True)
def _isolate_correlation_id():
    """Reset the ambient correlation-id contextvar around every test.

    ``correlation.ensure_correlation_id()`` (agent_utilities/observability/
    correlation.py) generates and permanently ``.set()``s a value with no
    token/reset of its own by design (it is meant to persist for the whole
    request/call scope in production) -- but pytest-xdist workers are reused
    across many test items in one process/thread, so any test that reaches
    that path (directly, or via ``persist_event``/``current_carrier``/
    ``inject``) leaves the contextvar set for every later test in the same
    worker, regardless of file/collection order. Force a clean ``None``
    baseline before each test and restore it after, so
    ``get_correlation_id() is None`` is a reliable precondition rather than
    one that depends on which other tests already ran in this worker.
    """
    from agent_utilities.observability import correlation

    token = correlation._correlation_id.set(None)
    try:
        yield
    finally:
        correlation._correlation_id.reset(token)
