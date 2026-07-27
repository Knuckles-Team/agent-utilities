"""Unit tests for the prompt-ingestion content-hash checkpoint (Phase C).

CONCEPT:EG-KG.storage.nonblocking-checkpoint — ``ingest_prompts_to_graph``
(``agent_utilities/agent/registry_builder.py``) previously full-re-upserted
every packaged/fleet/overlay prompt on every run. These tests prove the new
per-file ``DeltaManifest`` checkpoint (category ``"prompt_base"``): the first
ingest upserts every prompt, an unchanged second ingest skips every
``_upsert_node`` call, and a changed prompt is re-upserted (its new hash
re-recorded) while its unchanged sibling stays skipped.

Fully hermetic / service-free (no live engine, no graph-os pod):
``IntelligenceGraphEngine.get_active`` is monkeypatched to a minimal fake
engine, ``_iter_prompt_sources`` to two temp JSON files, and ``DeltaManifest``
to a SQLite-fallback instance pinned under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import agent_utilities.agent.registry_builder as registry_builder
import agent_utilities.knowledge_graph.ingestion.manifest as manifest_mod
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


class _FakeBackend:
    """Truthy backend stub exposing only ``create_schema()``."""

    def create_schema(self) -> None:
        pass


class _FakeEngine:
    """Minimal duck-typed stand-in for ``IntelligenceGraphEngine`` covering
    only the surface ``ingest_prompts_to_graph`` touches."""

    def __init__(self) -> None:
        self.backend = _FakeBackend()
        self.upsert_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.edge_calls: list[tuple[str, str, str]] = []

    def _serialize_node(self, node: Any, label: str | None = None) -> dict[str, Any]:
        return {"id": node.id, "name": getattr(node, "name", None)}

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]) -> None:
        self.upsert_calls.append((label, node_id, data))

    def add_edge(self, src: str, dst: str, rel: str) -> None:
        self.edge_calls.append((src, dst, rel))

    def prompt_names_upserted(self) -> list[str]:
        """Names of every ``Prompt``-labeled node upserted so far (order preserved)."""
        return [c[2]["name"] for c in self.upsert_calls if c[0] == "Prompt"]


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeEngine:
    engine = _FakeEngine()
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", classmethod(lambda cls: engine)
    )
    return engine


@pytest.fixture
def pinned_manifest_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin every ``DeltaManifest(...)`` construction to one tmp SQLite file.

    ``ingest_prompts_to_graph`` constructs its own ``DeltaManifest`` each
    call (no injection seam), so the class itself is monkeypatched at its
    defining module — the checkpoint then durably survives across the
    multiple sequential ``ingest_prompts_to_graph()`` calls below, exactly
    like ``DeltaManifest``'s own ``test_durable_across_restart`` proves for a
    fresh instance over the same store.
    """
    db_path = str(tmp_path / "prompt_manifest.db")

    class _PinnedManifest(manifest_mod.DeltaManifest):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__(backend=None, db_path=db_path)

    monkeypatch.setattr(manifest_mod, "DeltaManifest", _PinnedManifest)
    return db_path


def _write_prompt(path: Path, name: str, directive: str) -> None:
    path.write_text(
        json.dumps(
            {
                "name": name,
                "description": f"{name} test prompt",
                "instructions": {"core_directive": directive},
            }
        )
    )


@pytest.fixture
def two_prompt_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Two packaged-base-labeled prompt files, wired in place of real discovery."""
    alpha = tmp_path / "alpha.json"
    beta = tmp_path / "beta.json"
    _write_prompt(alpha, "alpha", "do alpha things")
    _write_prompt(beta, "beta", "do beta things")

    sources = [
        (registry_builder._BASE_SOURCE, alpha),
        (registry_builder._BASE_SOURCE, beta),
    ]
    monkeypatch.setattr(registry_builder, "_iter_prompt_sources", lambda: sources)
    return alpha, beta


@pytest.mark.asyncio
async def test_first_ingest_upserts_every_prompt(
    fake_engine: _FakeEngine,
    pinned_manifest_db: str,
    two_prompt_sources: tuple[Path, Path],
) -> None:
    await registry_builder.ingest_prompts_to_graph()

    assert sorted(fake_engine.prompt_names_upserted()) == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_second_ingest_skips_unchanged_prompts(
    fake_engine: _FakeEngine,
    pinned_manifest_db: str,
    two_prompt_sources: tuple[Path, Path],
) -> None:
    await registry_builder.ingest_prompts_to_graph()
    assert len(fake_engine.upsert_calls) > 0  # sanity: first run did real work
    fake_engine.upsert_calls.clear()

    await registry_builder.ingest_prompts_to_graph()

    assert fake_engine.prompt_names_upserted() == [], (
        "an unchanged prompt file must not trigger _upsert_node on a later ingest"
    )


@pytest.mark.asyncio
async def test_changed_prompt_reingested_unchanged_sibling_stays_skipped(
    fake_engine: _FakeEngine,
    pinned_manifest_db: str,
    two_prompt_sources: tuple[Path, Path],
) -> None:
    alpha, beta = two_prompt_sources
    await registry_builder.ingest_prompts_to_graph()
    fake_engine.upsert_calls.clear()

    _write_prompt(beta, "beta", "do COMPLETELY NEW beta things")  # content changed

    await registry_builder.ingest_prompts_to_graph()

    assert fake_engine.prompt_names_upserted() == ["beta"], (
        "only the changed prompt should be re-upserted; 'alpha' is unchanged "
        "and must stay skipped"
    )
    assert alpha.read_text()  # sibling file itself untouched by the test


@pytest.mark.asyncio
async def test_checkpoint_recorded_under_prompt_base_category(
    fake_engine: _FakeEngine,
    pinned_manifest_db: str,
    two_prompt_sources: tuple[Path, Path],
) -> None:
    alpha, _beta = two_prompt_sources

    await registry_builder.ingest_prompts_to_graph()

    dm = manifest_mod.DeltaManifest(backend=None, db_path=pinned_manifest_db)
    recorded = dm.get("__commons__", "prompt_base", str(alpha.resolve()))
    assert recorded == hashlib.sha256(alpha.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_manifest_unavailable_falls_back_to_always_upsert(
    fake_engine: _FakeEngine,
    two_prompt_sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the checkpoint can't be constructed, ingestion must still run in
    full every time (the prior full-re-upsert-every-run behavior) rather
    than silently ingesting nothing."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("manifest backend unavailable")

    monkeypatch.setattr(manifest_mod, "DeltaManifest", _boom)

    await registry_builder.ingest_prompts_to_graph()
    assert sorted(fake_engine.prompt_names_upserted()) == ["alpha", "beta"]
    fake_engine.upsert_calls.clear()

    # No manifest to persist a checkpoint into, so a second run re-upserts too.
    await registry_builder.ingest_prompts_to_graph()
    assert sorted(fake_engine.prompt_names_upserted()) == ["alpha", "beta"]
