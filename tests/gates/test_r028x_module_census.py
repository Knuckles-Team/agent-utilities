"""Importer census for the still-blocked AU-BOUNDARY-R028.x children
(R028.2 kg/infra, R028.6 observability/{trace_ontology,self_ingest,
audit_logger}, R028.7 governance/relational_authority, R028.8
kg/research/placement_mining).

Each of these four children is SPECIFIED, not LANDED: every one names a
production caller with no epistemic-graph client equivalent yet, so the
module cannot be deleted. This is each child's `.1` slice: a pinned
production-importer census that can only shrink as callers are redirected
to an EG client ahead of the module's own deletion. A grown set fails the
test; a shrunk or equal set passes.

R028.2's requirements.md text claims "no production caller anywhere" for
`kg/infra/{placement_optimizer,inventory_collector}.py`; this census found
one real production caller (`mcp/tools/analysis_tools.py`, via
`optimize_from_graph`/`collect_and_persist`) that the prose missed, so the
pinned set below corrects that and the row text is updated alongside.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import files_importing

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "agent_utilities"


def _modules_importing(
    needle_substrings: tuple[str, ...], exclude: tuple[str, ...]
) -> frozenset[str]:
    """Production files (posix, relative to agent_utilities/) importing a
    needle, excluding the defining modules. Tests are out of scope."""
    return files_importing((PACKAGE_ROOT,), needle_substrings, PACKAGE_ROOT, exclude)


# AU-BOUNDARY-R028.2.1 -- kg/infra/{placement_optimizer,inventory_collector}
PINNED_INFRA_IMPORTERS: frozenset[str] = frozenset(
    {
        "mcp/tools/analysis_tools.py",
    }
)

# AU-BOUNDARY-R028.6.1 -- observability/{trace_ontology,self_ingest,audit_logger}
PINNED_OBSERVABILITY_TRIAD_IMPORTERS: frozenset[str] = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "gateway/daemon.py",
        "models/knowledge_graph.py",
        "capabilities/kg_audit_sink.py",
        "capabilities/hooks.py",
        "capabilities/eg_history_source.py",
        "capabilities/governed_dynamic_workflow.py",
        "knowledge_graph/workflow_store.py",
        "knowledge_graph/durable_execution_kg.py",
        "knowledge_graph/etl/lineage.py",
        "knowledge_graph/etl/openlineage_consumer.py",
        "knowledge_graph/research/evidence.py",
        "knowledge_graph/research/trace_pattern_miner.py",
        "knowledge_graph/research/placement_mining.py",
        "knowledge_graph/research/loop_controller.py",
        "knowledge_graph/retrieval/troubleshoot_context.py",
        "knowledge_graph/retrieval/context_compiler.py",
        "knowledge_graph/core/engine.py",
        "knowledge_graph/orchestration/engine_ahe.py",
        "knowledge_graph/core/maintainer.py",
        "knowledge_graph/ontology/functions/runtime.py",
        "knowledge_graph/ontology/edits/ledger.py",
        "knowledge_graph/ontology/derived_properties.py",
        "knowledge_graph/actions/executor.py",
        "knowledge_graph/actions/__init__.py",
        "knowledge_graph/research/auto_merge.py",
        "knowledge_graph/research/change_publisher.py",
        "messaging/router.py",
        "server/routers/runtime.py",
        "observability/self_ingest.py",
        "observability/repository_provenance.py",
        "observability/error_detail_sink.py",
        "observability/escalation_matrix.py",
        "runtime/provenance.py",
        "runtime/run_vcs/kernel.py",
        "harness/trace_examples.py",
        "harness/agentic_evolution_engine.py",
        "harness/variant_pool.py",
        "harness/run_outcome_prompt_evolution.py",
        "tools/self_improvement_tools.py",
        "workflows/runner.py",
        "orchestration/session_continuity.py",
        "orchestration/durable_execution.py",
        "orchestration/agent_digital_twin.py",
        "orchestration/manager.py",
        "orchestration/agent_runner.py",
        "orchestration/engine.py",
        "graph/integration.py",
        "orchestration/agent_dispatch_worker.py",
        "security/guardrails.py",
        "mcp/remote_oauth_broker.py",
    }
)

# AU-BOUNDARY-R028.7.1 -- governance/relational_authority
PINNED_RELATIONAL_AUTHORITY_IMPORTERS: frozenset[str] = frozenset()
# `scripts/security/check_relational_authority.py` is the one non-test
# caller, and it lives outside `agent_utilities/` (in `scripts/`), so the
# package-scoped census above is legitimately empty; the live CI gate is
# tracked separately and is why the row stays SPECIFIED.

# AU-BOUNDARY-R028.8.1 -- kg/research/placement_mining
PINNED_PLACEMENT_MINING_IMPORTERS: frozenset[str] = frozenset(
    {
        "knowledge_graph/research/candidate_insight.py",
        "knowledge_graph/research/loop_controller.py",
        "knowledge_graph/core/engine_tasks.py",
        "mcp/tools/state_tools.py",
        "core/schedule_engine.py",
        "orchestration/action_policy.py",
    }
)


@pytest.mark.spec("AU-BOUNDARY-R028.2.1")
def test_r028_2_1_infra_importer_set_has_not_grown() -> None:
    found = _modules_importing(
        ("knowledge_graph.infra", "knowledge_graph import infra"),
        exclude=("knowledge_graph/infra/",),
    )
    assert found <= PINNED_INFRA_IMPORTERS, (
        f"new caller(s) of knowledge_graph/infra not yet recorded: {found - PINNED_INFRA_IMPORTERS}"
    )


@pytest.mark.spec("AU-BOUNDARY-R028.6.1")
def test_r028_6_1_observability_triad_importer_set_has_not_grown() -> None:
    found = _modules_importing(
        (
            "observability.trace_ontology",
            "observability import trace_ontology",
            "observability.self_ingest",
            "observability import self_ingest",
            "observability.audit_logger",
            "observability import audit_logger",
        ),
        exclude=(
            "observability/trace_ontology.py",
            "observability/self_ingest.py",
            "observability/audit_logger.py",
        ),
    )
    assert found <= PINNED_OBSERVABILITY_TRIAD_IMPORTERS, (
        "new caller(s) of observability/{trace_ontology,self_ingest,audit_logger} not yet recorded: "
        f"{found - PINNED_OBSERVABILITY_TRIAD_IMPORTERS}"
    )


@pytest.mark.spec("AU-BOUNDARY-R028.7.1")
def test_r028_7_1_relational_authority_importer_set_has_not_grown() -> None:
    found = _modules_importing(
        ("governance.relational_authority", "governance import relational_authority"),
        exclude=("governance/relational_authority.py",),
    )
    assert found <= PINNED_RELATIONAL_AUTHORITY_IMPORTERS, (
        "new caller(s) of governance/relational_authority not yet recorded: "
        f"{found - PINNED_RELATIONAL_AUTHORITY_IMPORTERS}"
    )


@pytest.mark.spec("AU-BOUNDARY-R028.8.1")
def test_r028_8_1_placement_mining_importer_set_has_not_grown() -> None:
    found = _modules_importing(
        ("research.placement_mining", "research import placement_mining"),
        exclude=("knowledge_graph/research/placement_mining.py",),
    )
    assert found <= PINNED_PLACEMENT_MINING_IMPORTERS, (
        "new caller(s) of knowledge_graph/research/placement_mining not yet recorded: "
        f"{found - PINNED_PLACEMENT_MINING_IMPORTERS}"
    )
