"""Skill-workflow → KG WorkflowDefinition ingestion (CONCEPT:AU-KG.ingest.skill-workflow-corpus).

Unit + live-path coverage for
``agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest``:
a fixture workflow corpus is parsed and upserted into a fake / in-memory engine
as a ``WorkflowDefinition`` DAG, asserting the node/edge shape
``execute_workflow`` reads, idempotent re-ingest, and discoverability by name.

@pytest.mark.concept("AU-KG.ingest.skill-workflow-corpus")
"""

from __future__ import annotations

import json
import textwrap

import pytest

from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
    discover_atomic_skill_files,
    discover_workflow_skill_files,
    ingest_atomic_skills,
    ingest_one,
    ingest_skill_workflows,
    parse_workflow_skill,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

pytestmark = pytest.mark.concept("AU-KG.ingest.skill-workflow-corpus")


# --------------------------------------------------------------------------- #
# Fixtures: a tiny two-workflow corpus exercising BOTH heading dialects.       #
# --------------------------------------------------------------------------- #

# Dialect A — kebab heading IS the atomic skill name, numeric depends_on.
_INFRA_WF = textwrap.dedent(
    """\
    ---
    name: tiny-infra-deploy
    description: Stand up a tiny stack.
    domain: infra
    tags: [infra, deploy]
    team_config:
      specialist_ids: [infra-bot, dns-bot]
      tool_assignments:
        infra-bot: [container_manager]
    ---

    # tiny-infra-deploy Workflow

    ## Steps

    ### Step 1: network-topology-sweep
    Discover the network.

    ### Step 2: dns-record-manager [depends_on: Step 1]
    Register DNS.

    ### Step 3: swarm-mesh-provisioner [depends_on: Step 1, Step 2]
    Provision swarm.
    """
)

# Dialect B — Title-Case heading + **Agent** body field, name-based depends_on.
_FINANCE_WF = textwrap.dedent(
    """\
    ---
    name: tiny_pnl
    description: Compute pnl.
    domain: finance
    tags: [finance]
    team_config:
      specialist_ids: [data-fetcher, report-generator]
    concept: CONCEPT:AU-AHE.assimilation.skill-workflow-ingest
    ---

    # Tiny Pnl Workflow

    ## Steps

    ### Step 1: Fetch Trades
    **Agent**: `data-fetcher`
    **Tools**: `graph_query, sx_search`

    Execute fetch trades.

    ### Step 2: Report [depends_on: fetch_trades]
    **Agent**: `report-generator`
    **Tools**: `graph_write`

    Write the report.
    """
)


# An atomic, MCP-tool-backed skill — mirrors the real
# ``servicenow-incident-management`` SKILL.md shape (declares itself
# ``skill_type: skill``, has NO ``### Step N:`` headings). D-SNV-1: mixed into
# an explicit-root corpus sweep, this must NOT be minted as a WorkflowDefinition.
_ATOMIC_SKILL = textwrap.dedent(
    """\
    ---
    name: servicenow-incident-management
    skill_type: skill
    description: ITSM incident operations on the ServiceNow Incident API.
    tags: [servicenow, incident, itsm]
    ---

    # servicenow-incident-management

    ## When to use
    List, read, and create incident records.

    ## Tools & actions
    servicenow_get_incidents, servicenow_create_incident
    """
)


@pytest.fixture
def corpus(tmp_path):
    """A ``workflows/<domain>/<name>/SKILL.md`` tree under ``tmp_path``."""
    wf = tmp_path / "workflows"
    a = wf / "infra" / "tiny-infra-deploy"
    a.mkdir(parents=True)
    (a / "SKILL.md").write_text(_INFRA_WF, encoding="utf-8")
    b = wf / "finance" / "tiny_pnl"
    b.mkdir(parents=True)
    (b / "SKILL.md").write_text(_FINANCE_WF, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Fake engine (records node/edge writes, answers content_hash + count queries) #
# --------------------------------------------------------------------------- #


class FakeEngine:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str, dict]] = []

    def add_node(self, node_id, node_type, properties=None, **props):
        self.nodes[node_id] = {"type": node_type, **(properties or props or {})}

    def link_nodes(self, source, target, rel_type, properties=None):
        self.edges.append((source, target, rel_type, properties or {}))

    def query_cypher(self, query, params=None):
        params = params or {}
        if "content_hash" in query and "$wid" in query:
            n = self.nodes.get(params.get("wid"))
            return (
                [
                    {
                        "h": n["content_hash"],
                        "tenant_id": n.get("tenant_id"),
                        "classification": n.get("classification"),
                        "external_access": n.get("external_access"),
                    }
                ]
                if n
                else []
            )
        return []

    # convenience accessors for assertions
    def of_type(self, t):
        return {k: v for k, v in self.nodes.items() if v.get("type") == t}

    def edges_of(self, rel):
        return [e for e in self.edges if e[2] == rel]


# --------------------------------------------------------------------------- #
# Parser tests                                                                  #
# --------------------------------------------------------------------------- #


def test_parse_kebab_dialect(corpus):
    skill_md = corpus / "workflows" / "infra" / "tiny-infra-deploy" / "SKILL.md"
    parsed = parse_workflow_skill(skill_md)
    assert parsed["name"] == "tiny-infra-deploy"
    assert parsed["source_ref"] == "skill://tiny-infra-deploy"
    assert "path" not in parsed
    assert parsed["domain"] == "infra"
    assert parsed["specialist_ids"] == ["infra-bot", "dns-bot"]
    assert len(parsed["steps"]) == 3
    # kebab heading → atomic skill name is the component itself.
    assert parsed["steps"][0]["skill_name"] == "network-topology-sweep"
    # numeric multi-dep.
    assert parsed["steps"][2]["depends_on"] == ["Step 1", "Step 2"]


def test_parse_titlecase_agent_dialect(corpus):
    skill_md = corpus / "workflows" / "finance" / "tiny_pnl" / "SKILL.md"
    parsed = parse_workflow_skill(skill_md)
    assert parsed["concept"] == "CONCEPT:AU-AHE.assimilation.skill-workflow-ingest"
    # Title-Case heading → atomic skill comes from the **Agent** body field.
    assert parsed["steps"][0]["skill_name"] == "data-fetcher"
    assert parsed["steps"][0]["tools"] == ["graph_query", "sx_search"]
    # name-based depends_on (slug of "Fetch Trades").
    assert parsed["steps"][1]["depends_on"] == ["fetch_trades"]


# --------------------------------------------------------------------------- #
# Ingestion shape + DAG + skill-link tests                                      #
# --------------------------------------------------------------------------- #


def test_ingest_creates_definition_steps_and_dag(corpus):
    eng = FakeEngine()
    report = ingest_skill_workflows(eng, root=str(corpus))

    assert report["workflows"] == 2
    assert report["steps"] == 5  # 3 + 2
    assert report["errors"] == 0

    defs = eng.of_type("WorkflowDefinition")
    assert "skill_workflow:tiny_infra_deploy" in defs
    assert "skill_workflow:tiny_pnl" in defs
    d = defs["skill_workflow:tiny_infra_deploy"]
    assert d["source"] == "universal-skills"
    assert d["domain"] == "infra"
    assert d["step_count"] == 3
    assert d["name"] == "tiny-infra-deploy"  # lookup key execute_workflow uses
    assert d["source_ref"] == "skill://tiny-infra-deploy"
    assert "source_path" not in d
    assert str(corpus) not in json.dumps(eng.nodes, sort_keys=True)

    # HAS_STEP edges from the definition.
    has_step = [
        e
        for e in eng.edges_of("HAS_STEP")
        if e[0] == "skill_workflow:tiny_infra_deploy"
    ]
    assert len(has_step) == 3

    # depends_on → TRANSITION_TO edges (Step 3 depends on Steps 1 & 2).
    s3 = "skill_workflow:tiny_infra_deploy:step:3"
    preds = {e[0] for e in eng.edges_of("TRANSITION_TO") if e[1] == s3}
    assert preds == {
        "skill_workflow:tiny_infra_deploy:step:1",
        "skill_workflow:tiny_infra_deploy:step:2",
    }
    # Step 1 has no deps → parallel.
    s1 = eng.nodes["skill_workflow:tiny_infra_deploy:step:1"]
    assert s1["is_parallel"] is True


def test_ingest_links_atomic_skills(corpus):
    eng = FakeEngine()
    ingest_skill_workflows(eng, root=str(corpus))
    skills = eng.of_type("Skill")
    # Ids are slug-normalised; the original name is preserved as a property.
    assert "skill:network_topology_sweep" in skills
    assert skills["skill:network_topology_sweep"]["name"] == "network-topology-sweep"
    assert (
        skills["skill:network_topology_sweep"]["source_ref"]
        == "skill://network-topology-sweep"
    )
    assert "skill:data_fetcher" in skills  # resolved from **Agent**
    assert skills["skill:data_fetcher"]["name"] == "data-fetcher"
    uses = {(e[0], e[1]) for e in eng.edges_of("USES_SKILL")}
    assert ("skill_workflow:tiny_pnl:step:1", "skill:data_fetcher") in uses


def test_ingest_stamps_governance_on_every_node_and_edge(corpus):
    """KG-2.97 §4 regression (the ACL-invisibility gap): every node/edge
    ``ingest_one`` writes must carry ``classification``/``external_access``/
    ``tenant_id`` or ``secured_reads.permit()`` denies it forever — the write
    durably lands but is invisible to every ``query_cypher``/``graph_search``
    read, even though it "succeeded". Mirrors the stamping
    ``ingest_runnable_skill`` already does in this same module (which is why
    its nodes were visible while these were not).
    """
    eng = FakeEngine()
    ingest_skill_workflows(eng, root=str(corpus))

    def _assert_governed(props, where):
        assert props.get("classification") == "public", where
        access = props.get("external_access")
        assert isinstance(access, dict) and access.get("is_public") is True, where
        assert props.get("tenant_id"), where

    assert eng.nodes, "fixture must have produced nodes to check"
    for node_id, props in eng.nodes.items():
        _assert_governed(props, node_id)
    assert eng.edges, "fixture must have produced edges to check"
    for source, target, rel, props in eng.edges:
        _assert_governed(props, f"{source}-[{rel}]->{target}")


def test_ingest_is_idempotent(corpus):
    eng = FakeEngine()
    first = ingest_skill_workflows(eng, root=str(corpus))
    assert first["workflows"] == 2 and first["skipped"] == 0
    # Re-run on the SAME engine → content_hash matches → all skipped, no churn.
    nodes_before = dict(eng.nodes)
    second = ingest_skill_workflows(eng, root=str(corpus))
    assert second["workflows"] == 0
    assert second["skipped"] == 2
    assert eng.nodes == nodes_before


def test_changed_content_reingests(corpus):
    eng = FakeEngine()
    ingest_skill_workflows(eng, root=str(corpus))
    # Mutate one workflow → its hash changes → it re-ingests (not skipped).
    skill_md = corpus / "workflows" / "finance" / "tiny_pnl" / "SKILL.md"
    skill_md.write_text(_FINANCE_WF.replace("Compute pnl.", "Compute pnl v2."), "utf-8")
    rep = ingest_skill_workflows(eng, root=str(corpus))
    assert rep["workflows"] == 1
    assert rep["skipped"] == 1


def test_ingest_one_returns_skipped_on_repeat(corpus):
    eng = FakeEngine()
    skill_md = corpus / "workflows" / "infra" / "tiny-infra-deploy" / "SKILL.md"
    parsed = parse_workflow_skill(skill_md)
    assert ingest_one(eng, parsed) == "ingested"
    assert ingest_one(eng, parsed) == "skipped"


def test_ingest_one_repairs_governance_drift_even_when_content_is_unchanged(corpus):
    eng = FakeEngine()
    skill_md = corpus / "workflows" / "infra" / "tiny-infra-deploy" / "SKILL.md"
    parsed = parse_workflow_skill(skill_md)
    assert ingest_one(eng, parsed) == "ingested"
    workflow = eng.nodes["skill_workflow:tiny_infra_deploy"]
    expected_tenant = workflow["tenant_id"]
    workflow["tenant_id"] = "stale-local-tenant"

    assert ingest_one(eng, parsed) == "ingested"
    assert eng.nodes["skill_workflow:tiny_infra_deploy"]["tenant_id"] == expected_tenant


class CollisionSensitiveEngine(FakeEngine):
    """Mimics the real native backend's ``add_node(node_id, **properties)``
    shape (``BrainGuardedBackend``/``EpistemicGraphBackend``), where a literal
    ``"node_id"`` key inside the properties dict collides with the positional
    ``node_id`` argument of the same name — the exact production crash
    (KG-2.97 §5b: ``TypeError: add_node() got multiple values for argument
    'node_id'``), reproduced here without needing a real backend.
    """

    def add_node(self, node_id, node_type, properties=None, **props):
        merged = dict(properties or props or {})
        if "node_id" in merged:
            raise TypeError("add_node() got multiple values for argument 'node_id'")
        super().add_node(node_id, node_type, properties=merged)


def test_ingest_step_props_do_not_collide_with_node_id_argument(corpus):
    """KG-2.97 §5b regression: a WorkflowStep's properties must never carry a
    literal ``"node_id"`` key — ``step_id`` is already the node's identity via
    the positional argument. Before the fix this crashed on the FIRST step of
    every workflow that has one, orphaning the parent WorkflowDefinition with
    zero real steps and zero USES_SKILL links.
    """
    eng = CollisionSensitiveEngine()
    report = ingest_skill_workflows(eng, root=str(corpus))
    assert report["errors"] == 0
    steps = eng.of_type("WorkflowStep")
    assert len(steps) == 5  # 3 (infra) + 2 (finance), matches the DAG test above
    assert all(step.get("step_id") for step in steps.values())
    skills = eng.of_type("Skill")
    assert skills  # USES_SKILL-linked Skill nodes also require steps to land


def test_delegated_ingest_failure_report_does_not_leak_local_path(corpus):
    """A background/delegated corpus failure retains only a skill URI and error
    class, even when the underlying exception contains a local path."""

    class ExplodingEngine(FakeEngine):
        def add_node(self, node_id, node_type, properties=None, **props):
            raise RuntimeError(str(corpus))

    report = ingest_skill_workflows(ExplodingEngine(), root=str(corpus))
    rendered = json.dumps(report, sort_keys=True)
    assert report["errors"] == 2
    assert str(corpus) not in rendered
    assert "skill://tiny-infra-deploy" in rendered
    assert "RuntimeError" in rendered


# --------------------------------------------------------------------------- #
# Live-path / integration: a real in-memory IntelligenceGraphEngine            #
# --------------------------------------------------------------------------- #


def test_live_ingest_into_memory_engine_discoverable_by_name(corpus):
    """The ingested workflow is retrievable from a real engine the way
    ``execute_workflow`` / ``graph-orchestration-and-automation`` look it up: a
    ``WorkflowDefinition`` queryable by ``name`` with its ``WorkflowStep`` DAG.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    # Construct the process-owned compute engine FIRST so it (not a bare
    # ``IntelligenceGraphEngine(db_path=...)``'s own tenant-routed default)
    # claims the isolated per-test graph the ``isolate_graph_compute_engine``
    # autouse fixture provisions (tests/conftest.py) — the same
    # GraphComputeEngine-first idiom used elsewhere (e.g.
    # tests/unit/test_kg_native_orchestration.py) to avoid the shared
    # ``tenant__..____commons__`` default graph, whose racy durable-lifecycle
    # registration is a documented pre-existing flake (D-OTR-3/D-W2X-3). The
    # backend's own ``_graph`` is then rebound onto that already-isolated
    # compute engine (D-OTR-2's fix idiom) — otherwise ``EpistemicGraphBackend``
    # independently resolves its own tenant-routed graph and opens a
    # ``for_graph()`` view that cannot retarget the verified GraphSession.
    isolated = GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(db_path=":memory:")
    # ``engine.backend`` is a ``BrainGuardedBackend`` proxy — the real
    # ``EpistemicGraphBackend`` (whose ``_graph`` needs rebinding) is its
    # ``_inner``.
    getattr(engine.backend, "_inner", engine.backend)._graph = isolated
    report = ingest_skill_workflows(engine, root=str(corpus))
    assert report["workflows"] == 2
    assert report["errors"] == 0

    # Lookup by name (the WorkflowStore.load_workflow lookup key).
    rows = engine.query_cypher(
        "MATCH (w:WorkflowDefinition) WHERE w.name = $name "
        "RETURN w.id AS id, w.step_count AS sc, w.source AS source",
        {"name": "tiny-infra-deploy"},
    )
    assert rows, "ingested WorkflowDefinition must be findable by name"
    assert rows[0]["id"] == "skill_workflow:tiny_infra_deploy"
    assert rows[0]["source"] == "universal-skills"

    # The HAS_STEP subgraph is traversable (what WorkflowStore.load reads).
    # ``s.id`` must be projected alongside ``step_order`` — public graph reads
    # are governed by a per-row node id (secured_reads.row_node_ids): a
    # projection with no identifiable node id is denied outright, regardless
    # of how benign the returned scalar is.
    steps = engine.query_cypher(
        "MATCH (w:WorkflowDefinition {id: $wid})-[:HAS_STEP]->(s:WorkflowStep) "
        "RETURN s.id AS id, s.step_order AS o ORDER BY s.step_order",
        {"wid": "skill_workflow:tiny_infra_deploy"},
    )
    assert [r["o"] for r in steps] == [1, 2, 3]


def test_discover_accepts_explicit_root(corpus):
    files = discover_workflow_skill_files(root=str(corpus))
    names = {f.parent.name for f in files}
    assert names == {"tiny-infra-deploy", "tiny_pnl"}


def test_discover_explicit_root_does_not_redirect_into_nested_workflows_dir(tmp_path):
    """KG-2.97 §5a regression: an explicit root must be searched AS-IS. The old
    code silently redirected into ``root/workflows/`` whenever that subdir
    happened to exist, discarding every sibling SKILL.md — exactly the real
    ``agent_utilities/skills`` shape (13 domain skills + skill_graphs/ + ONE
    real workflow that happens to live under ``skills/workflows/
    agent-os-genesis``), which made the explicit-root call find only 1 of 20
    real files.
    """
    root = tmp_path / "skills"
    # A real workflow living under a nested "workflows/" dir...
    decoy = root / "workflows" / "agent-os-genesis"
    decoy.mkdir(parents=True)
    (decoy / "SKILL.md").write_text(_INFRA_WF, encoding="utf-8")
    # ...sibling to an atomic skill that must NOT be shadowed by that redirect.
    sibling = root / "graph-query-and-explanation"
    sibling.mkdir(parents=True)
    (sibling / "SKILL.md").write_text(_FINANCE_WF, encoding="utf-8")

    files = discover_workflow_skill_files(root=str(root))
    names = {f.parent.name for f in files}
    assert names == {"agent-os-genesis", "graph-query-and-explanation"}


def test_discover_default_matches_domain_workflows_convention(monkeypatch, tmp_path):
    """KG-2.97 §5a regression: the shipped convention is
    ``<domain>-workflows/<name>/SKILL.md`` (e.g. ``finance-workflows/``) —
    there is no directory literally named ``workflows/`` anywhere in the real
    corpus, so the default (no-``root``) call found 0 workflows before the
    fix. It must now find the ``-workflows`` categories and must NOT sweep in
    atomic skills from a sibling plain ``<domain>/`` category (that corpus is
    covered by the separate boot-time ``ingest_runnable_skill`` path).
    """
    import sys
    import types

    pkg_root = tmp_path / "universal_skills"
    wf_dir = pkg_root / "finance-workflows" / "tiny_pnl"
    wf_dir.mkdir(parents=True)
    (wf_dir / "SKILL.md").write_text(_FINANCE_WF, encoding="utf-8")
    atomic_dir = pkg_root / "finance" / "some-atomic-skill"
    atomic_dir.mkdir(parents=True)
    (atomic_dir / "SKILL.md").write_text(
        "---\nname: some-atomic-skill\n---\nbody", encoding="utf-8"
    )

    fake_pkg = types.ModuleType("universal_skills")
    fake_pkg.__path__ = [str(pkg_root)]
    fake_skill_utilities = types.ModuleType("universal_skills.skill_utilities")
    fake_skill_utilities.get_universal_skills_path = lambda: [
        str(wf_dir),
        str(atomic_dir),
    ]
    fake_pkg.skill_utilities = fake_skill_utilities
    monkeypatch.setitem(sys.modules, "universal_skills", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "universal_skills.skill_utilities", fake_skill_utilities
    )

    files = discover_workflow_skill_files()
    names = {f.parent.name for f in files}
    assert names == {"tiny_pnl"}


def test_explicit_root_does_not_mint_workflowdefinition_for_declared_atomic_skill(
    tmp_path,
):
    """D-SNV-1 regression: a fleet package's own ``skills/`` tree mixes atomic
    skills (``skill_type: skill``, MCP-tool-backed, no ``### Step N:`` DAG)
    alongside real workflows. An explicit-root sweep of that tree must
    discover BOTH SKILL.md files (directory-shape filtering stays off — see
    ``test_discover_explicit_root_does_not_redirect_into_nested_workflows_dir``)
    but must NOT mint a WorkflowDefinition for the one that self-declares it is
    not a workflow — that used to happen unconditionally and is exactly how
    ``servicenow-incident-management`` (real prod skill, identical frontmatter
    shape) landed in the graph as a 0-step WorkflowDefinition instead of a
    CallableResource, making it undelegatable (only describable).
    """
    root = tmp_path / "skills"
    wf = root / "tiny-infra-deploy"
    wf.mkdir(parents=True)
    (wf / "SKILL.md").write_text(_INFRA_WF, encoding="utf-8")
    atomic = root / "servicenow-incident-management"
    atomic.mkdir(parents=True)
    (atomic / "SKILL.md").write_text(_ATOMIC_SKILL, encoding="utf-8")

    # Discovery itself stays directory-shape-agnostic (both files found)...
    files = discover_workflow_skill_files(root=str(root))
    assert {f.parent.name for f in files} == {
        "tiny-infra-deploy",
        "servicenow-incident-management",
    }

    # ...but ingestion must gate on the declared skill_type before writing.
    eng = FakeEngine()
    report = ingest_skill_workflows(eng, root=str(root))
    assert report["workflows"] == 1
    assert report["not_workflow"] == 1
    assert (
        "skill://servicenow-incident-management: skill_type='skill'"
        in (report["not_workflow_detail"])
    )
    assert report["errors"] == 0

    defs = eng.of_type("WorkflowDefinition")
    assert "skill_workflow:tiny_infra_deploy" in defs
    assert "skill_workflow:servicenow_incident_management" not in defs
    assert not any(
        d.get("name") == "servicenow-incident-management" for d in defs.values()
    )


# --------------------------------------------------------------------------- #
# Atomic-skill leg: the sibling closing the "left for its own ingester" gap.   #
# --------------------------------------------------------------------------- #


class _RunnableEngine:
    """Records typed writes the way ``ingest_runnable_skill`` needs
    (``_upsert_node`` + ``link_nodes``) — mirrors ``_Engine`` in
    ``test_fleet_skill_harvest.py``, the sibling primitive's own test double.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.backend = self

    def _upsert_node(self, node_type: str, node_id: str, properties: dict) -> None:
        self.nodes[node_id] = {"type": node_type, **properties}

    def link_nodes(self, source: str, target: str, relationship: str, **_kw) -> None:
        self.edges.append((source, target, relationship))

    def of_type(self, t: str) -> dict[str, dict]:
        return {k: v for k, v in self.nodes.items() if v.get("type") == t}


@pytest.fixture()
def authority():
    """A verified session/actor — ``ingest_runnable_skill`` requires
    ``resolve_session(required_scope="kg:write")`` (fail-closed identity)."""
    actor = ActorContext(
        actor_id="skill-ingest-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="tenant_test",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant_test",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="g",
        policy_version="v1",
        audience="test",
    )
    with use_actor(actor), use_session(session):
        yield


def test_discover_atomic_skill_files_walks_the_whole_package(tmp_path):
    """Unlike ``discover_workflow_skill_files``' default sweep (confined to
    ``<domain>-workflows/`` categories), the atomic discoverer must find a
    plain ``<domain>/<name>/SKILL.md`` — that is where every atomic skill
    actually lives, so a default-root call that missed it would silently
    reproduce the exact gap this leg exists to close.
    """
    root = tmp_path / "skills"
    atomic_dir = root / "infra" / "some-atomic-skill"
    atomic_dir.mkdir(parents=True)
    (atomic_dir / "SKILL.md").write_text(
        "---\nname: some-atomic-skill\nskill_type: skill\n---\nbody",
        encoding="utf-8",
    )
    files = discover_atomic_skill_files(root=str(root))
    assert {f.parent.name for f in files} == {"some-atomic-skill"}


def test_ingest_atomic_skills_creates_a_runnable_callable_resource(
    tmp_path, authority
):
    """The exact real-prod shape (``servicenow-incident-management``): a
    ``skill_type: skill`` file must land as a ``CallableResource`` carrying
    ``resource_type='AGENT_SKILL'`` — the field the registry UI's
    KG-authoritative classification reads
    (``agent_webui.api_extensions._fetch_kg_skill_classification``) — not be
    left with no KG node at all.
    """
    root = tmp_path / "skills"
    atomic = root / "servicenow-incident-management"
    atomic.mkdir(parents=True)
    (atomic / "SKILL.md").write_text(_ATOMIC_SKILL, encoding="utf-8")

    eng = _RunnableEngine()
    report = ingest_atomic_skills(eng, root=str(root))

    assert report["skills"] == 1
    assert report["not_skill"] == 0
    assert report["errors"] == 0

    resources = eng.of_type("CallableResource")
    assert len(resources) == 1
    (resource,) = resources.values()
    assert resource["resource_type"] == "AGENT_SKILL"
    assert resource["name"] == "servicenow-incident-management"


def test_ingest_atomic_skills_skips_declared_workflows_and_graphs(tmp_path, authority):
    """D-SNV-1 mirror of the workflow leg's own gate: a file that declares
    ``skill_type: workflow`` (or ``graph``, or is silent -- "assume workflow"
    by the same convention ``ingest_skill_workflows`` uses) must NOT be
    minted as a ``CallableResource`` here — that would double-classify it
    against whatever its own ingester lands, or claim a workflow/skill-graph
    is atomically runnable when it never declared itself so.
    """
    root = tmp_path / "skills"
    wf = root / "tiny-infra-deploy"
    wf.mkdir(parents=True)
    (wf / "SKILL.md").write_text(_INFRA_WF, encoding="utf-8")
    graph_dir = root / "some-skill-graph"
    graph_dir.mkdir(parents=True)
    (graph_dir / "SKILL.md").write_text(
        "---\nname: some-skill-graph\nskill_type: graph\n---\nbody",
        encoding="utf-8",
    )

    eng = _RunnableEngine()
    report = ingest_atomic_skills(eng, root=str(root))

    assert report["skills"] == 0
    assert report["not_skill"] == 2
    assert not eng.of_type("CallableResource")


def test_ingest_atomic_skills_is_idempotent(tmp_path, authority):
    """A re-run (the recurring ``package_install`` schedule tick) must upsert
    in place, not duplicate — matches ``ingest_runnable_skill``'s own stable,
    content-addressed id contract."""
    root = tmp_path / "skills"
    atomic = root / "servicenow-incident-management"
    atomic.mkdir(parents=True)
    (atomic / "SKILL.md").write_text(_ATOMIC_SKILL, encoding="utf-8")

    eng = _RunnableEngine()
    ingest_atomic_skills(eng, root=str(root))
    ingest_atomic_skills(eng, root=str(root))

    assert len(eng.of_type("CallableResource")) == 1


# --------------------------------------------------------------------------- #
# Background-job path: the worker dispatch branch (CONCEPT:AU-KG.ingest.skill-workflow-corpus)            #
# --------------------------------------------------------------------------- #
def test_background_job_branch_ingests_corpus(corpus):
    """The ``skill_workflows`` task-worker branch (what ``submit_task`` enqueues)
    runs the ingest off the request path and lands the WorkflowDefinitions —
    so the MCP action returns a job_id immediately and never blocks the call.
    """
    import asyncio
    from pathlib import Path

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    # See test_live_ingest_into_memory_engine_discoverable_by_name for why the
    # compute engine is constructed first and the backend rebound onto it
    # (isolated-graph idiom, D-OTR-2/D-OTR-3/D-W2X-3).
    isolated = GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(db_path=":memory:")
    getattr(engine.backend, "_inner", engine.backend)._graph = isolated

    # ``_run_background_task`` commits its terminal outcome through the
    # in-memory WorkItem claim a real worker's ``_claim_next_task()`` leaves
    # behind (``_update_task_status``/``_fail_or_retry_task`` both require
    # one, via ``_wi.heartbeat``/``_wi.commit_result`` against a REAL
    # backing WorkItem). Standing one up for real would route through
    # ``submit_task``, which rejects any target outside the configured
    # agent workspace — incompatible with a pytest ``tmp_path`` corpus. This
    # test's unit is the ``skill_workflows`` ingest branch inside
    # ``_run_background_task``, not the WorkItem claim/commit lifecycle
    # (covered separately by ``test_ingest_task_workitem_lifecycle.py``), so
    # a minimal in-memory claim plus stubbed lease/commit calls isolate that
    # unit the same way ``test_ingest_tail_optimization.py`` stubs the
    # inverse (mocking OUT ``_run_background_task`` to test claiming).
    from unittest.mock import patch

    from agent_utilities.orchestration import work_item as _wi

    job_id = "job-skilltest"
    fake_claim = {
        "work_item_id": _wi.ingest_task_work_item_id(job_id),
        "lease_owner": "test-worker",
        "lease_epoch": 1,
    }
    engine._remember_work_item_claim(job_id, fake_claim)
    with (
        patch.object(_wi, "heartbeat", return_value=True),
        patch.object(_wi, "commit_result", return_value="committed"),
    ):
        # Drive the exact branch the background worker dispatches for the job.
        asyncio.run(
            engine._run_background_task(
                job_id, Path(str(corpus)), False, "skill_workflows"
            )
        )
    rows = engine.query_cypher(
        "MATCH (w:WorkflowDefinition) WHERE w.source = $s RETURN count(w) AS c",
        {"s": "universal-skills"},
    )
    assert rows and rows[0]["c"] >= 2


def test_skill_workflows_is_a_heavy_background_task_type():
    """skill_workflows must be registered heavy so it runs on the worker, not inline."""
    import inspect

    from agent_utilities.knowledge_graph.core import engine_tasks

    src = inspect.getsource(engine_tasks)
    assert '"skill_workflows"' in src and "_HEAVY_TASK_TYPES" in src


# --------------------------------------------------------------------------- #
# Chunk/embed side-write gating (KG-2.97 §5d)                                   #
# --------------------------------------------------------------------------- #


def test_chunk_workflow_body_self_gates_after_first_failure(monkeypatch):
    """KG-2.97 §5d regression: a systemic chunk/embed failure (observed live:
    near-100% ``STALE_GRAPH_VERSION`` OCC retry-budget exhaustion, ~30-40s
    wall clock per file across 8 doomed retries) must not silently repeat a
    doomed multi-retry write on every remaining file in a bulk run — it
    disables itself after the first failure and logs clearly instead.
    """
    import agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest as swi

    monkeypatch.setattr(swi, "_chunk_body_disabled", False)
    calls = {"n": 0}

    class _ExplodingProcessor:
        def __init__(self, *_args, **_kwargs):
            pass

        def process(self, *_args, **_kwargs):
            calls["n"] += 1
            raise RuntimeError(
                "native ChangeEnvelope OCC retry budget exhausted "
                "(conflict_sequence=STALE_GRAPH_VERSION*8)"
            )

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
        _ExplodingProcessor,
    )

    swi._chunk_workflow_body(object(), "skill_workflow:x", "some body text", "x")
    assert calls["n"] == 1
    assert swi._chunk_body_disabled is True

    # A second call (a later file in the same bulk run) must be a no-op — no
    # repeated doomed attempt.
    swi._chunk_workflow_body(object(), "skill_workflow:y", "more body text", "y")
    assert calls["n"] == 1


def test_chunk_workflow_body_failure_never_blocks_workflow_registration(corpus):
    """Best-effort by design: even with chunking permanently disabled,
    ``ingest_one``/``ingest_skill_workflows`` must still land the
    WorkflowDefinition/WorkflowStep/Skill DAG cleanly.
    """
    import agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest as swi

    original = swi._chunk_body_disabled
    swi._chunk_body_disabled = True
    try:
        eng = FakeEngine()
        report = ingest_skill_workflows(eng, root=str(corpus))
        assert report["workflows"] == 2
        assert report["errors"] == 0
        assert eng.of_type("WorkflowStep")
    finally:
        swi._chunk_body_disabled = original
