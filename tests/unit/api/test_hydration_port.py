"""The public hydration port delegates mutations to manifest-gated sync."""

from __future__ import annotations

from agent_utilities.api import hydration


def test_hydration_mutations_use_canonical_source_sync(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core import source_sync

    calls: list[tuple[object, str, str]] = []
    engine = object()

    def sync(target: object, source: str, *, mode: str) -> dict[str, str]:
        calls.append((target, source, mode))
        return {"status": "queued", "source": source}

    monkeypatch.setattr(source_sync, "sync_source", sync)
    assert hydration.hydrate_source(engine, "github") == {
        "status": "queued",
        "source": "github",
    }
    assert hydration.hydrate_all(engine) == {"status": "queued", "source": "all"}
    assert calls == [(engine, "github", "full"), (engine, "all", "full")]


def test_hydration_status_uses_existing_source_registry(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core import hydration as core_hydration

    monkeypatch.setattr(
        core_hydration.HydrationManager,
        "get_status",
        lambda self: {"github": {"configured": False}},
    )
    assert hydration.hydration_status() == {"github": {"configured": False}}
