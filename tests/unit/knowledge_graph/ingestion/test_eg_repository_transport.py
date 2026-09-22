"""EH-280 Phase 1 adapter tests (CONCEPT:AU-KG.ingest.branch-aware-repository-delta).

Covers ``agent_utilities.knowledge_graph.ingestion.eg_repository_transport`` —
the AU-side seam over ``agent_connector_sdk.repository``'s branch-aware
transport. Full context:
``/var/tmp/l9/finish/au-deletion/EH-280-CODEBASE-INGEST-CUTOVER-DESIGN.md``.

This module is NOT wired into the live codebase-ingest default path (see the
design doc §5/§6), so these tests exercise it standalone: the fail-closed
behavior needs no real SDK (and this environment genuinely lacks
``epistemic_graph.generated``, so that path is exercised for real, not
mocked); the success path needs the real SDK types and is skipped cleanly via
``pytest.importorskip`` when they are unavailable, per this lane's EH-345
precedent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.ingestion import eg_repository_transport as ert
from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

pytestmark = pytest.mark.concept("AU-KG.ingest.branch-aware-repository-delta")


def _manifest(tmp_path: Path) -> DeltaManifest:
    return DeltaManifest(backend=None, db_path=str(tmp_path / "manifest.db"))


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    (root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "a.py"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "initial"], check=True
    )


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("KG_CODEBASE_DELTA_VIA_SDK_TRANSPORT", raising=False)
    assert ert.codebase_delta_via_sdk_transport_enabled() is False


def test_flag_reads_setting(monkeypatch):
    monkeypatch.setenv("KG_CODEBASE_DELTA_VIA_SDK_TRANSPORT", "true")
    assert ert.codebase_delta_via_sdk_transport_enabled() is True


def test_index_codebase_fails_closed_without_the_sdk(tmp_path):
    """The real, current state of this environment: ``agent_connector_sdk.
    repository`` fails to import (its own ``epistemic_graph.generated``
    dependency is absent) -- not mocked, this is what actually happens today.
    """
    manifest = _manifest(tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)

    async def _run():
        return await ert.index_codebase_via_sdk_transport(
            manifest,
            async_client=SimpleNamespace(),
            graph_name="g1",
            source_path=str(repo_root),
            repository_id="local-git:repo",
        )

    with pytest.raises(ert.RepositoryTransportUnavailable):
        import asyncio

        asyncio.run(_run())


def test_load_prior_manifest_missing_is_none(tmp_path):
    pytest.importorskip("agent_connector_sdk.repository")
    manifest = _manifest(tmp_path)
    assert ert._load_prior_manifest(manifest, "g1", "local-git:repo") is None


def test_load_prior_manifest_malformed_is_none_not_a_crash(tmp_path):
    pytest.importorskip("agent_connector_sdk.repository")
    manifest = _manifest(tmp_path)
    manifest.record("g1", ert._MANIFEST_CATEGORY, "local-git:repo", "{not json")
    assert ert._load_prior_manifest(manifest, "g1", "local-git:repo") is None


def test_manifest_round_trip(tmp_path):
    """Store then reload a real ``RepositoryIndexManifest`` -- proves the
    JSON round trip matches the real generated type's shape, not an assumed
    one."""
    pytest.importorskip("agent_connector_sdk.repository")
    from agent_connector_sdk.repository import RepositoryIndexManifest

    manifest = _manifest(tmp_path)
    empty = RepositoryIndexManifest(repository_id="local-git:repo", refs=())
    ert._store_manifest(manifest, "g1", "local-git:repo", empty)
    reloaded = ert._load_prior_manifest(manifest, "g1", "local-git:repo")
    assert reloaded is not None
    assert reloaded.repository_id == "local-git:repo"
    assert reloaded.refs == ()


def test_index_codebase_via_sdk_transport_end_to_end(tmp_path):
    """Full success path against a real temp git repo and a fake async EG
    client -- proves the provider/plan/submit/manifest-persist wiring, not
    just that imports resolve."""
    pytest.importorskip("agent_connector_sdk.repository")
    from epistemic_graph.generated.index_repository import IndexResult

    manifest = _manifest(tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)

    empty_result = IndexResult(
        calls_resolved=0,
        calls_scope_resolved=0,
        calls_type_resolved=0,
        calls_unresolved=0,
        edges=[],
        file_outcomes=[
            {
                "content_digest": "sha256:" + "0" * 64,
                "diagnostics": [],
                "file_path": "a.py",
                "parser_capability_digest": "sha256:" + "1" * 64,
                "status": "success",
            }
        ],
        files_parsed=1,
        imports_resolved=0,
        imports_unresolved=0,
        inherits_edges=0,
        nodes=[],
        realizes_edges=0,
        similar_edges=0,
        symbols_extracted=0,
    )

    class _FakeGraph:
        async def index_repository(self, payload, *, scope):
            digests = {p: "sha256:" + "0" * 64 for p, _content in payload}
            outcomes = [
                {
                    "content_digest": digests[p],
                    "diagnostics": [],
                    "file_path": p,
                    "parser_capability_digest": "sha256:" + "1" * 64,
                    "status": "success",
                }
                for p, _content in payload
            ]
            return empty_result.model_copy(
                update={"file_outcomes": outcomes, "files_parsed": len(payload)}
            )

    fake_client = SimpleNamespace(graph=_FakeGraph())

    async def _run():
        return await ert.index_codebase_via_sdk_transport(
            manifest,
            fake_client,
            graph_name="g1",
            source_path=str(repo_root),
            repository_id="local-git:repo",
        )

    import asyncio

    receipt = asyncio.run(_run())
    assert receipt.blobs_fetched == 1
    assert receipt.manifest.repository_id == "local-git:repo"

    # The manifest is now persisted as the next run's prior.
    reloaded = ert._load_prior_manifest(manifest, "g1", "local-git:repo")
    assert reloaded is not None
    assert reloaded.repository_id == "local-git:repo"
