"""AU-P1-6 CI contract test: the 12 high-value named connectors MUST each ship a

``connector_manifest.yml`` that compiles, hash-matches, and is wired.

This is what makes "mandatory manifests" real at the repo/CI level (independent
of whether a given connector is dispatched through ``sync_source`` today — see
:mod:`connector_manifest_gate`'s ``MANDATORY_NAMED_CONNECTOR_SOURCES`` docstring
for the honest scope note on which of the 12 have a live ``sync_source`` call
site vs. not). Every one of the 12 has a manifest bundled directly in
agent-utilities (:func:`connector_manifest_gate.bundled_manifests_root`) — this
test fails the build the moment any of the 12 goes missing or drifts out of
sync with its own compiled/re-hashed content, closing exactly the gap Codex P1
flagged (manifests existed for 17 fleet connectors but none of these 12).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate

# The 12 named connectors this workstream (AU-P1-6) makes mandatory, keyed by
# their ``agents/<pkg>`` directory name (the bundled staging dir uses the same
# names — see agent_utilities/knowledge_graph/ontology/connector_manifests/).
NAMED_CONNECTOR_PACKAGES: tuple[str, ...] = (
    "atlassian-agent",
    "gitlab-api",
    "servicenow-api",
    "microsoft-agent",
    "leanix-agent",
    "container-manager-mcp",
    "documentdb-mcp",
    "langfuse-agent",
    "tunnel-manager",
    "repository-manager",
    "systems-manager",
    "vector-mcp",
)


def test_exactly_twelve_named_connectors_tracked():
    """Guards against silently dropping (or forgetting to add) one of the 12."""
    assert len(NAMED_CONNECTOR_PACKAGES) == 12
    assert len(set(NAMED_CONNECTOR_PACKAGES)) == 12


@pytest.mark.parametrize("package", NAMED_CONNECTOR_PACKAGES)
def test_named_connector_has_bundled_manifest(package: str):
    path = gate.bundled_manifests_root() / package / "connector_manifest.yml"
    assert path.exists(), (
        f"{package} is one of the AU-P1-6 mandatory named connectors but has no "
        f"bundled connector_manifest.yml at {path} — generate one via "
        "scripts/generate_connector_manifests.py."
    )


@pytest.mark.parametrize("package", NAMED_CONNECTOR_PACKAGES)
def test_named_connector_manifest_passes_gate(package: str):
    path = gate.bundled_manifests_root() / package / "connector_manifest.yml"
    violations = gate.check_manifest_bytes(path)
    assert violations == [], f"{package}: {violations}"


def test_all_named_connectors_resolve_through_precheck_source():
    """Every ``MANDATORY_NAMED_CONNECTOR_SOURCES`` identifier finds ITS bundled

    manifest and passes the gate via the exact same ``precheck_source`` path
    ``sync_source`` uses in production — not just via a direct file check.
    """
    for source in sorted(gate.MANDATORY_NAMED_CONNECTOR_SOURCES):
        result = gate.precheck_source(source)
        assert result["checked"] is True, source
        assert result["ok"] is True, (source, result["violations"])


def test_mandatory_set_is_a_superset_of_named_connector_packages():
    """Every package in :data:`NAMED_CONNECTOR_PACKAGES` is reachable from at

    least one identifier in :data:`gate.MANDATORY_NAMED_CONNECTOR_SOURCES`.
    """
    root = gate.bundled_manifests_root()
    resolved_packages = {
        gate.resolve_connector_package(source, agents_root=root)
        for source in gate.MANDATORY_NAMED_CONNECTOR_SOURCES
    }
    assert resolved_packages == set(NAMED_CONNECTOR_PACKAGES)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
