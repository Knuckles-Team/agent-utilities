"""The serving/host plane must ship the CANONICAL embedder (CONCEPT:KG-2.3).

Regression for the e2e north-star ingestion profiler findings:

  * BUG 1 — every ``document`` ingest dead-lettered with
    ``ModuleNotFoundError: No module named 'llama_index.embeddings'`` (the document
    worker calls ``create_embedding_model()``).
  * BUG 2 — ``embed_calls: 0`` fleet-wide / ZERO embeddings (``make_embed_fn()`` /
    the backfill could not construct an embedder).

Root cause: the ``serving`` optional-dependency extra (graph-os MCP + host daemon +
messaging — the plane that RUNS ingestion + enrichment) pulled the bare
``embeddings`` extra, which provides only ``llama-index-core`` — NO embedding
provider. The canonical embedder is the OpenAI-compatible bge-m3 vLLM endpoint, so
the host needs ``llama-index-embeddings-openai`` (the ``embeddings-openai`` extra).

This test locks the closure so the serving plane can never again be built unable to
embed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _pyproject() -> dict:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _resolve_extra(
    extras: dict[str, list[str]], name: str, _seen: set[str]
) -> set[str]:
    """Flatten an extra to the set of self-referential ``agent-utilities[...]`` extras it pulls."""
    if name in _seen:
        return set()
    _seen.add(name)
    out: set[str] = {name}
    for dep in extras.get(name, []):
        if dep.startswith("agent-utilities["):
            inner = dep.split("[", 1)[1].split("]", 1)[0]
            for part in inner.split(","):
                out |= _resolve_extra(extras, part.strip(), _seen)
    return out


def test_serving_plane_pulls_the_canonical_embeddings_openai_extra():
    extras = _pyproject()["project"]["optional-dependencies"]
    closure = _resolve_extra(extras, "serving", set())
    assert "embeddings-openai" in closure, (
        "serving plane must pull 'embeddings-openai' (the bge-m3 vLLM provider); "
        f"got closure={sorted(closure)}"
    )


def test_embeddings_openai_extra_ships_the_openai_provider_package():
    extras = _pyproject()["project"]["optional-dependencies"]
    joined = " ".join(extras["embeddings-openai"])
    assert "llama-index-embeddings-openai" in joined


def test_serving_does_not_rely_on_bare_embeddings_only():
    """The bare ``embeddings`` extra alone cannot embed — guard against regressing to it."""
    extras = _pyproject()["project"]["optional-dependencies"]
    # 'embeddings' provides only llama-index-core (no provider) — never the sole embed dep.
    assert extras["embeddings"] == ["llama-index-core>=0.14.22"]
