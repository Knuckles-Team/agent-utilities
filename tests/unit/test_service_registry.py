#!/usr/bin/python
"""Tests for the Unified Service Registry (CONCEPT:ORCH-1.4)."""

import pytest


class TestServiceRegistry:
    """Validate the central service registry wires all modules."""

    def test_initialize(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        count = registry.initialize()
        assert count >= 50  # At least 50 services should be registered

    def test_discover_by_domain(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        registry.initialize()
        finance = registry.discover(domain="finance")
        assert len(finance) > 10  # All finance + general services

    def test_discover_by_layer(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        registry.initialize()
        security = registry.discover(layer="security")
        assert len(security) >= 6

    def test_get_capability(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        registry.initialize()
        svc = registry.get("team_composition")
        assert svc is not None
        assert svc.module_path == "agent_utilities.graph.team_composer"

    def test_list_capabilities(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        registry.initialize()
        caps = registry.list_capabilities()
        assert "team_composition" in caps
        assert "prompt_scanning" in caps
        assert "alpha_factors" in caps

    def test_layer_summary(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        registry.initialize()
        summary = registry.get_layer_summary()
        assert "orchestration" in summary
        assert "security" in summary
        assert "domain" in summary

    def test_validate_loadable(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        registry.initialize()
        loadable, failed = registry.validate_loadable()
        # Majority should load successfully
        assert len(loadable) > 30
        if failed:
            print(f"Failed to load: {failed}")

    def test_singleton_pattern(self):
        from agent_utilities.graph.service_registry import ServiceRegistry

        r1 = ServiceRegistry.instance()
        r2 = ServiceRegistry.instance()
        assert r1 is r2

    def test_register_with_kg(self):
        import networkx as nx
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
        from agent_utilities.graph.service_registry import ServiceRegistry

        g = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(g, backend=None)
        registry = ServiceRegistry()
        registry.initialize()
        # No backend = register_with_kg will gracefully skip (backend=None)
        count = registry.register_with_kg(engine)
        assert count >= 50  # Services registered (via NX fallback)


class TestDomainRegistry:
    """Validate the domain routing registry."""

    def test_list_domains(self):
        from agent_utilities.domains import list_domains

        domains = list_domains()
        assert "finance" in domains

    def test_get_finance_capabilities(self):
        from agent_utilities.domains import get_domain_capabilities

        caps = get_domain_capabilities("finance")
        assert "alpha_factors" in caps
        assert "risk_management" in caps
        assert "kronos_forecaster" in caps
        assert len(caps) >= 13

    def test_unknown_domain(self):
        from agent_utilities.domains import get_domain_capabilities

        caps = get_domain_capabilities("nonexistent")
        assert caps == []
