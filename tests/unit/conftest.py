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
