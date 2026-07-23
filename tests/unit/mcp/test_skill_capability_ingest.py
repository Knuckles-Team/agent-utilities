"""Privacy contract for startup skill-capability ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_utilities.core import providers
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionRequiredError,
    suspend_session,
    use_session,
)
from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
    ingest_runnable_skill,
    runnable_skill_digest,
)
from agent_utilities.mcp import kg_server
from agent_utilities.models.company_brain import ActorType
from agent_utilities.orchestration.agent_runner import _resolve_agent_from_kg
from agent_utilities.security.brain_context import ActorContext
from agent_utilities.security.persistence_privacy import persistence_reference


class RecordingEngine:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.backend = self

    def _upsert_node(self, node_type, node_id, properties) -> None:
        self.nodes[node_id] = {"type": node_type, **properties}

    def add_node(self, *_args, **_kwargs) -> None:
        raise AssertionError("runnable skills must use the typed authority writer")

    def link_nodes(self, source, target, relationship, **_kwargs) -> None:
        self.edges.append((source, target, relationship))

    def execute(self, query, params=None):
        params = params or {}
        if "MATCH (s:Server)" in query or "MATCH (at:AgentTemplate)" in query:
            return []
        if "MATCH (n:CallableResource) WHERE n.name IN $names" in query:
            names = set(params.get("names") or [])
            return [
                {
                    "id": node_id,
                    "name": node.get("name", ""),
                    "rtype": node.get("resource_type"),
                    "system_prompt": node.get("system_prompt", ""),
                    "instruction_digest": node.get("instruction_digest", ""),
                    "source_ref": node.get("source_ref", ""),
                    "runnable_bound": node.get("runnable_bound"),
                }
                for node_id, node in self.nodes.items()
                if node.get("type") == "CallableResource" and node.get("name") in names
            ]
        if "MATCH (r:CallableResource) WHERE r.name" in query:
            return [
                {
                    "rid": node_id,
                    "rtype": node.get("resource_type"),
                    "description": node.get("description", ""),
                    "system_prompt": node.get("system_prompt", ""),
                    "instruction_digest": node.get("instruction_digest", ""),
                    "source_ref": node.get("source_ref", ""),
                }
                for node_id, node in self.nodes.items()
                if node.get("type") == "CallableResource"
                and node.get("name") == params.get("name")
            ]
        if "USES_TOOL" in query:
            return []
        return []

    def query_cypher(self, query, params=None):
        return self.execute(query, params)

    def search_hybrid(self, *_args, **_kwargs):
        return []


@pytest.fixture(autouse=True)
def _verified_write_session():
    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.SYSTEM,
        roles=(),
        tenant_id="tenant:opaque:synthetic",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read", "kg:write"}),
        audience="epistemic-graph",
        policy_version="policy:synthetic",
    )
    with use_session(session):
        yield


def test_runnable_skill_requires_verified_write_authority() -> None:
    engine = RecordingEngine()

    with suspend_session(), pytest.raises(SessionRequiredError):
        ingest_runnable_skill(
            engine,
            name="synthetic-skill",
            description="Synthetic.",
            instructions="Perform a bounded synthetic action.",
            provider="synthetic-provider",
        )

    assert engine.nodes == {}
    assert engine.edges == []


def test_boot_skill_ingest_persists_uri_not_local_path(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    skill = root / "sample-runtime-audit"
    skill.mkdir(parents=True)
    local_hint = str(tmp_path / "operator-workspace")
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: sample-runtime-audit\n"
        "description: Audit a synthetic runtime.\n"
        f"local_hint: {local_hint}\n"
        "operator: synthetic-operator\n"
        "---\n\n# Sample\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kg_server, "get_existing_disabled", lambda *_args: False)
    engine = RecordingEngine()

    assert kg_server._ingest_skill_capabilities(engine, "synthetic-provider", root) == 1

    node = engine.nodes["resource:skill:sample-runtime-audit"]
    rendered = json.dumps(node, sort_keys=True)
    assert node["source_ref"] == "skill://sample-runtime-audit"
    assert node["provider_ref"] == "provider://synthetic_provider"
    assert node["system_prompt"] == "# Sample"
    assert node["instruction_digest"] == runnable_skill_digest("# Sample")
    assert "path" not in node and "local_hint" not in node and "operator" not in node
    assert str(tmp_path) not in rendered
    assert "synthetic-operator" not in rendered


def test_boot_skill_failure_log_uses_neutral_reference(tmp_path, caplog, monkeypatch):
    root = tmp_path / "skills"
    skill = root / "broken-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: [\n---\n", encoding="utf-8")
    monkeypatch.setattr(kg_server, "get_existing_disabled", lambda *_args: False)

    assert (
        kg_server._ingest_skill_capabilities(
            RecordingEngine(), "synthetic-provider", root
        )
        == 0
    )
    assert (
        persistence_reference(
            "skill", "broken-skill", namespace="skill-provider-ingest"
        )
        in caplog.text
    )
    # Deliberate contract update: the failure log format is
    # "stage=%s %s: %s" (stage, exception type, exception MESSAGE) — it now
    # preserves the (sanitized) exception message instead of collapsing it to
    # just the exception type, so an operator can see WHY a skill failed to
    # ingest, not only that it did.
    assert "stage=declaration ParserError:" in caplog.text
    assert str(tmp_path) not in caplog.text


def test_boot_skill_ingest_sanitizes_instruction_body(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    skill = root / "privacy-audit"
    skill.mkdir(parents=True)
    local_path = str(tmp_path / "private" / "notes.md")
    synthetic_token = "".join(
        ("sk", "-proj-", "abcdefghijklmnopqrstuvwxyz", "0123456789")
    )
    (skill / "SKILL.md").write_text(
        "---\nname: privacy-audit\ndescription: Synthetic.\n---\n"
        f"Read {local_path} and notify person@example.test with token "
        f"{synthetic_token}.",
        encoding="utf-8",
    )
    monkeypatch.setattr(kg_server, "get_existing_disabled", lambda *_args: False)
    engine = RecordingEngine()

    assert kg_server._ingest_skill_capabilities(engine, "synthetic", root) == 1

    node = engine.nodes["resource:skill:privacy-audit"]
    rendered = json.dumps(node, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "person@example.test" not in rendered
    assert synthetic_token not in rendered
    assert node["privacy_redactions"] >= 3
    assert node["instruction_digest"] == runnable_skill_digest(node["system_prompt"])


def test_all_bundled_skills_are_runnable_and_resolve_by_body_digest(monkeypatch):
    from agent_utilities.skills import BUNDLED_SKILLS

    root = Path(kg_server.__file__).resolve().parents[1] / "skills"
    monkeypatch.setattr(kg_server, "get_existing_disabled", lambda *_args: False)
    engine = RecordingEngine()

    # Scoped to exactly the packaged/boot-critical skills: `agent_utilities/skills/`
    # also holds atomic skills that are NOT bundled (e.g. agent-utilities-self-
    # evolution, autonomous-contribution) and the `skill_graphs` reference corpus
    # (excluded by `is_skill_graph_reference_path`) — an unscoped call ingests all
    # of the former too, which is correct for a real boot but not what this test
    # is asserting about the 10 BUNDLED_SKILLS specifically.
    assert (
        kg_server._ingest_skill_capabilities(
            engine, "agent-utilities", root, include_names=frozenset(BUNDLED_SKILLS)
        )
        == 10
    )

    resources = {
        node["name"]: (node_id, node)
        for node_id, node in engine.nodes.items()
        if node.get("type") == "CallableResource"
    }
    assert len(resources) == 10
    for name, (resource_id, node) in resources.items():
        assert resource_id == f"resource:skill:{name}"
        assert node["instruction_digest"] == runnable_skill_digest(
            node["system_prompt"]
        )
        assert node["tenant_id"] == "tenant:opaque:synthetic"
        assert node["classification"] == "public"
        assert node["external_access"] == {
            "group_ids": [],
            "is_public": True,
            "markings": [],
            "read_roles": [],
            "user_emails": [],
        }
        resolved = _resolve_agent_from_kg(engine, name)
        assert resolved["type"] == "skill"
        assert resolved["skill_id"] == resource_id
        assert resolved["skill_instruction_digest"] == node["instruction_digest"]
        assert node["system_prompt"] in resolved["system_prompt"]


def test_bundled_skill_readiness_is_idempotent(monkeypatch):
    from agent_utilities.skills import BUNDLED_SKILLS

    root = Path(kg_server.__file__).resolve().parents[1] / "skills"
    monkeypatch.setattr(kg_server, "get_existing_disabled", lambda *_args: False)
    engine = RecordingEngine()
    assert (
        kg_server._ingest_skill_capabilities(
            engine, "agent-utilities", root, include_names=frozenset(BUNDLED_SKILLS)
        )
        == 10
    )
    ingest = MagicMock(wraps=kg_server._ingest_skill_capabilities)
    monkeypatch.setattr(kg_server, "_ingest_skill_capabilities", ingest)

    readiness = kg_server._ensure_bundled_skills_ready(engine)

    assert readiness == {
        "required": 10,
        "already_ready": 10,
        "ingested": 0,
        "ready": 10,
        "not_ready": [],
    }
    ingest.assert_not_called()


def test_bundled_skill_readiness_uses_governed_query_boundary(monkeypatch):
    from agent_utilities.skills import BUNDLED_SKILLS

    root = Path(kg_server.__file__).resolve().parents[1] / "skills"
    monkeypatch.setattr(kg_server, "get_existing_disabled", lambda *_args: False)
    engine = RecordingEngine()
    assert (
        kg_server._ingest_skill_capabilities(
            engine, "agent-utilities", root, include_names=frozenset(BUNDLED_SKILLS)
        )
        == 10
    )
    governed_query = MagicMock(wraps=engine.query_cypher)
    engine.query_cypher = governed_query
    engine.backend = MagicMock()
    engine.backend.execute.side_effect = AssertionError(
        "startup readiness must not bypass the governed graph query boundary"
    )

    ready = kg_server._ready_bundled_skill_names(
        engine, kg_server._bundled_skill_contract()[1]
    )

    assert len(ready) == 10
    governed_query.assert_called_once()
    engine.backend.execute.assert_not_called()


def test_bundled_skill_readiness_repairs_only_missing_resource(monkeypatch):
    from agent_utilities.skills import BUNDLED_SKILLS

    root = Path(kg_server.__file__).resolve().parents[1] / "skills"
    monkeypatch.setattr(kg_server, "get_existing_disabled", lambda *_args: False)
    engine = RecordingEngine()
    assert (
        kg_server._ingest_skill_capabilities(
            engine, "agent-utilities", root, include_names=frozenset(BUNDLED_SKILLS)
        )
        == 10
    )
    missing = BUNDLED_SKILLS[-1]
    engine.nodes.pop(f"resource:skill:{missing}")
    ingest = MagicMock(wraps=kg_server._ingest_skill_capabilities)
    monkeypatch.setattr(kg_server, "_ingest_skill_capabilities", ingest)

    readiness = kg_server._ensure_bundled_skills_ready(engine)

    assert readiness == {
        "required": 10,
        "already_ready": 9,
        "ingested": 1,
        "ready": 10,
        "not_ready": [],
    }
    ingest.assert_called_once()
    assert ingest.call_args.kwargs["include_names"] == frozenset({missing})


def test_bundled_skill_readiness_failure_is_controlled(monkeypatch, caplog):
    """A packaged-skill readiness shortfall is a CAPABILITY gap, not a reason
    to refuse serving. It must not raise (that would take the whole server
    down over some skills failing to ingest — the same class of bug as the
    ``DuplicateSkillIdentity`` collision aborting the whole sweep); it is
    logged loudly and reported back so ``/health`` and the caller can see
    exactly which packaged skills are not ready.
    """
    monkeypatch.setattr(kg_server, "_ingest_skill_capabilities", lambda *_a, **_k: 0)

    with caplog.at_level("ERROR", logger="agent_utilities.mcp.kg_server"):
        report = kg_server._ensure_bundled_skills_ready(RecordingEngine())

    from agent_utilities.skills import BUNDLED_SKILLS

    assert report["ready"] == 0
    assert report["required"] == len(BUNDLED_SKILLS)
    assert sorted(report["not_ready"]) == sorted(BUNDLED_SKILLS)
    assert "readiness incomplete" in caplog.text


def test_provider_resolution_ignores_unmarked_nested_xdg_root(tmp_path, monkeypatch):
    direct = tmp_path / "direct-skill"
    nested = tmp_path / "provider-a" / "nested-skill"
    direct.mkdir(parents=True)
    nested.mkdir(parents=True)
    (direct / "SKILL.md").write_text("---\nname: direct-skill\n---\nbody")
    (nested / "SKILL.md").write_text("---\nname: nested-skill\n---\nbody")
    monkeypatch.setattr("agent_utilities.core.paths.skills_dir", lambda: tmp_path)
    monkeypatch.setattr(providers, "current_provider_assets", lambda _group: ())

    roots = providers.resolve_skill_provider_dirs()
    selected = [(name, root) for name, root in roots if root.is_relative_to(tmp_path)]
    files = [path for _name, root in selected for path in root.rglob("SKILL.md")]

    assert files.count(direct / "SKILL.md") == 1
    assert nested / "SKILL.md" not in files


def test_provider_resolution_accepts_marked_current_nested_provider(
    tmp_path, monkeypatch
):
    from agent_utilities.core import unified_install
    from agent_utilities.core.provider_materialization import build_asset_manifest

    source = tmp_path / "source"
    nested = source / "nested-skill"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nested-skill\n---\nbody", encoding="utf-8"
    )
    manifest = build_asset_manifest(source, leg="skills")
    registration = providers.ProviderRegistration(
        name="provider-a",
        group=providers.SKILL_PROVIDER_GROUP,
        target="provider.skills",
        owner_name="provider",
        owner_version="1",
        digest="a" * 64,
        source_root=source,
        owned_paths=frozenset({"nested-skill/SKILL.md"}),
    )
    assets = providers.ProviderAssets(registration, source, manifest)
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    unified_install._materialize_provider(
        root=xdg,
        provider="provider-a",
        leg="skills",
        registration=registration.digest,
        source=source,
        manifest=manifest,
    )
    monkeypatch.setattr("agent_utilities.core.paths.skills_dir", lambda: xdg)
    monkeypatch.setattr(providers, "current_provider_assets", lambda _group: (assets,))

    matches = [
        (provider, root)
        for provider, root in providers.resolve_skill_provider_dirs()
        if root.name == "nested-skill"
    ]

    assert len(matches) == 1
    assert matches[0][0] == "provider-a"
    assert matches[0][1].is_relative_to(xdg)


def test_flat_xdg_copy_conflicting_with_provider_identity_fails_closed(
    tmp_path, monkeypatch, caplog
):
    """An untrusted flat xdg-local skill claiming a provider-owned identity

    must never be allowed to shadow or coexist with the trusted provider
    skill ("fails closed" on the identity itself) — but it must also not take
    down every OTHER unrelated skill with it. The provider-owned copy wins
    (added first), the untrusted duplicate is dropped, and the collision is
    logged loudly rather than raised, so one bad/malicious local file can no
    longer deny the whole skill sweep (CONCEPT: sweep resilience).
    """
    xdg_root = tmp_path / "xdg"
    xdg_skill = xdg_root / "runtime-audit"
    package_root = tmp_path / "package"
    package_skill = package_root / "runtime-audit"
    xdg_skill.mkdir(parents=True)
    package_skill.mkdir(parents=True)
    body = "---\nname: runtime-audit\n---\nAuthoritative instructions."
    (xdg_skill / "SKILL.md").write_text(body)
    (package_skill / "SKILL.md").write_text(body)
    monkeypatch.setattr("agent_utilities.core.paths.skills_dir", lambda: xdg_root)
    manifest = providers.build_asset_manifest(package_root, leg="skills")
    registration = providers.ProviderRegistration(
        name="canonical-provider",
        group=providers.SKILL_PROVIDER_GROUP,
        target="canonical.skills",
        owner_name="canonical-provider",
        owner_version="1",
        digest="a" * 64,
        source_root=package_root,
        owned_paths=frozenset({"runtime-audit/SKILL.md"}),
    )
    monkeypatch.setattr(
        providers,
        "current_provider_assets",
        lambda _group: (
            providers.ProviderAssets(registration, package_root, manifest),
        ),
    )

    with caplog.at_level("ERROR", logger="agent_utilities.core.providers"):
        roots = providers.resolve_skill_provider_dirs()

    matches = [(provider, root) for provider, root in roots if root.name == "runtime-audit"]
    assert matches == [("canonical-provider", package_skill)]
    assert "runtime-audit" in caplog.text
    assert "canonical-provider" in caplog.text


def test_resolve_skill_provider_dirs_exempts_skill_graph_reference_pages(
    tmp_path, monkeypatch
):
    """A ``skill_type: graph`` reference page legitimately reuses the exact
    ``name:`` of the atomic skill it documents — e.g. the packaged
    ``agent-utilities`` skill-graph's "deployment" page
    (``skill_graphs/agent-utilities/deployment/SKILL.md``) is itself named
    ``agent-utilities-deployment``, matching the real atomic skill it
    documents. This must never be flagged as a fleet-wide identity collision
    (it is a KG-ingestion reference corpus, not an installable skill — the
    same scoping as ``scripts/check_skill_name_collision.py``'s
    ``skill_graphs``/``skill-graphs`` exclusion and the fleet harness's
    ``_exempt_from_uniqueness``).

    Regression test for the ``DuplicateSkillIdentity`` bug that took
    ``agent-utilities-deployment``/``agent-utilities-development`` down at
    graph-os boot ("SERVING DEGRADED: 8/10 packaged skills ready").
    """
    xdg_root = tmp_path / "xdg"
    xdg_root.mkdir()
    provider_root = tmp_path / "provider"
    atomic_dir = provider_root / "shared-topic"
    atomic_dir.mkdir(parents=True)
    (atomic_dir / "SKILL.md").write_text(
        "---\nname: shared-topic\nskill_type: skill\n---\nbody"
    )
    graph_page_dir = provider_root / "skill_graphs" / "some-graph" / "shared-topic"
    graph_page_dir.mkdir(parents=True)
    (graph_page_dir / "SKILL.md").write_text(
        "---\nname: shared-topic\nskill_type: graph\n---\nbody"
    )
    monkeypatch.setattr("agent_utilities.core.paths.skills_dir", lambda: xdg_root)
    manifest = providers.build_asset_manifest(provider_root, leg="skills")
    registration = providers.ProviderRegistration(
        name="topic-provider",
        group=providers.SKILL_PROVIDER_GROUP,
        target="topic.skills",
        owner_name="topic-provider",
        owner_version="1",
        digest="c" * 64,
        source_root=provider_root,
        owned_paths=frozenset(
            {"shared-topic/SKILL.md", "skill_graphs/some-graph/shared-topic/SKILL.md"}
        ),
    )
    monkeypatch.setattr(
        providers,
        "current_provider_assets",
        lambda _group: (
            providers.ProviderAssets(registration, provider_root, manifest),
        ),
    )

    roots = providers.resolve_skill_provider_dirs()

    matches = [(provider, root) for provider, root in roots if root.name == "shared-topic"]
    assert matches == [("topic-provider", atomic_dir)]


def test_resolve_skill_provider_dirs_sweep_survives_one_collision(
    tmp_path, monkeypatch, caplog
):
    """One colliding skill must not abort resolution of every OTHER skill.

    Regression test for the bug where a single ``DuplicateSkillIdentity``
    escaping the resolution loop aborted the WHOLE sweep, so graph-os went
    from "10/10 packaged skills ready" to "SERVING DEGRADED: 8/10 ready" —
    two unrelated, non-colliding skills were also reported not_ready because
    the collision on two OTHER skills took the entire function down.
    """
    xdg_root = tmp_path / "xdg"
    xdg_root.mkdir()
    provider_a_root = tmp_path / "provider-a"
    provider_b_root = tmp_path / "provider-b"
    for root, name in (
        (provider_a_root, "shared-name"),
        (provider_b_root, "shared-name"),
    ):
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody")
    unrelated_dir = provider_b_root / "unrelated-skill"
    unrelated_dir.mkdir()
    (unrelated_dir / "SKILL.md").write_text("---\nname: unrelated-skill\n---\nbody")

    monkeypatch.setattr("agent_utilities.core.paths.skills_dir", lambda: xdg_root)

    def _asset(name: str, root: Path, owned: frozenset[str]):
        manifest = providers.build_asset_manifest(root, leg="skills")
        registration = providers.ProviderRegistration(
            name=name,
            group=providers.SKILL_PROVIDER_GROUP,
            target=f"{name}.skills",
            owner_name=name,
            owner_version="1",
            digest=("d" if name == "provider-a" else "e") * 64,
            source_root=root,
            owned_paths=owned,
        )
        return providers.ProviderAssets(registration, root, manifest)

    asset_a = _asset("provider-a", provider_a_root, frozenset({"shared-name/SKILL.md"}))
    asset_b = _asset(
        "provider-b",
        provider_b_root,
        frozenset({"shared-name/SKILL.md", "unrelated-skill/SKILL.md"}),
    )
    monkeypatch.setattr(
        providers, "current_provider_assets", lambda _group: (asset_a, asset_b)
    )

    with caplog.at_level("ERROR", logger="agent_utilities.core.providers"):
        roots = providers.resolve_skill_provider_dirs()

    by_name = {root.name: (provider, root) for provider, root in roots}
    assert by_name["unrelated-skill"][0] == "provider-b"  # survived the collision
    assert by_name["shared-name"][0] == "provider-a"  # first in sorted order wins
    assert "shared-name" in caplog.text
    assert "provider-a" in caplog.text
    assert "provider-b" in caplog.text
