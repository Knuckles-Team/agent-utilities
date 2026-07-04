"""Skill-workflow → KG WorkflowDefinition ingestion (CONCEPT:AU-KG.ingest.skill-workflow-corpus).

Unit + live-path coverage for
``agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest``:
a fixture workflow corpus is parsed and upserted into a fake / in-memory engine
as a ``WorkflowDefinition`` DAG, asserting the node/edge shape
``execute_workflow`` reads, idempotent re-ingest, and discoverability by name.

@pytest.mark.concept("AU-KG.ingest.skill-workflow-corpus")
"""

from __future__ import annotations

import textwrap

import pytest

from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
    discover_workflow_skill_files,
    ingest_one,
    ingest_skill_workflows,
    parse_workflow_skill,
)

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
            return [{"h": n["content_hash"]}] if n else []
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
    assert "skill:data_fetcher" in skills  # resolved from **Agent**
    assert skills["skill:data_fetcher"]["name"] == "data-fetcher"
    uses = {(e[0], e[1]) for e in eng.edges_of("USES_SKILL")}
    assert ("skill_workflow:tiny_pnl:step:1", "skill:data_fetcher") in uses


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


# --------------------------------------------------------------------------- #
# Live-path / integration: a real in-memory IntelligenceGraphEngine            #
# --------------------------------------------------------------------------- #


def test_live_ingest_into_memory_engine_discoverable_by_name(corpus):
    """The ingested workflow is retrievable from a real engine the way
    ``execute_workflow`` / ``kg-delegate`` look it up: a
    ``WorkflowDefinition`` queryable by ``name`` with its ``WorkflowStep`` DAG.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine(db_path=":memory:")
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
    steps = engine.query_cypher(
        "MATCH (w:WorkflowDefinition {id: $wid})-[:HAS_STEP]->(s:WorkflowStep) "
        "RETURN s.step_order AS o ORDER BY s.step_order",
        {"wid": "skill_workflow:tiny_infra_deploy"},
    )
    assert [r["o"] for r in steps] == [1, 2, 3]


def test_discover_accepts_explicit_root(corpus):
    files = discover_workflow_skill_files(root=str(corpus))
    names = {f.parent.name for f in files}
    assert names == {"tiny-infra-deploy", "tiny_pnl"}


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

    engine = IntelligenceGraphEngine(db_path=":memory:")
    # Drive the exact branch the background worker dispatches for the job.
    asyncio.run(
        engine._run_background_task(
            "job-skilltest", Path(str(corpus)), False, "skill_workflows"
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
