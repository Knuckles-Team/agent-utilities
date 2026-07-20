from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _module():
    source = Path(__file__).parents[3] / "scripts" / "check_ontology.py"
    spec = importlib.util.spec_from_file_location("check_ontology", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _provider(root: Path, name: str = "sample-agent") -> Path:
    project = root / name
    ontology = project / "sample_agent" / "ontology"
    shapes = ontology / "shapes"
    shapes.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{name}"\n'
        'version = "1.0.0"\n'
        '[project.entry-points."agent_utilities.ontology_providers"]\n'
        f'{name} = "sample_agent.ontology"\n',
        encoding="utf-8",
    )
    (ontology / "sample.ttl").write_text(
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "<http://knuckles.team/kg/sample> a owl:Ontology ;\n"
        "  owl:imports <http://knuckles.team/kg> .\n",
        encoding="utf-8",
    )
    (shapes / "connector.shacl.ttl").write_text(
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
        "<http://knuckles.team/kg/sample/shape#Resource> a sh:NodeShape .\n",
        encoding="utf-8",
    )
    return project


def test_source_fleet_discovery_is_declared_bounded_and_owner_labeled(tmp_path):
    module = _module()
    agents = tmp_path / "agents"
    _provider(agents)
    hidden = agents / ".worktrees" / "copy" / "sample_agent" / "ontology"
    hidden.mkdir(parents=True)
    (hidden / "duplicate.ttl").write_text("not turtle", encoding="utf-8")

    assets = module._source_provider_ttls(agents)

    assert [path.name for path in assets] == ["sample.ttl", "connector.shacl.ttl"]
    assert [module._rel(path).as_posix() for path in assets] == [
        "provider-assets/sample-agent/sample.ttl",
        "provider-assets/sample-agent/shapes/connector.shacl.ttl",
    ]


def test_source_fleet_discovery_rejects_linked_ontology_root(tmp_path):
    module = _module()
    agents = tmp_path / "agents"
    project = _provider(agents)
    ontology = project / "sample_agent" / "ontology"
    replacement = tmp_path / "replacement"
    ontology.rename(replacement)
    ontology.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(module.FleetScanError, match="provider-ontology-root"):
        module._source_provider_ttls(agents)


def test_source_fleet_discovery_fails_closed_on_malformed_metadata(tmp_path):
    module = _module()
    agents = tmp_path / "agents"
    project = _provider(agents)
    (project / "pyproject.toml").write_text("[project\n", encoding="utf-8")

    with pytest.raises(module.FleetScanError, match="provider-metadata-parse"):
        module._source_provider_ttls(agents)


def test_import_connectivity_accepts_child_chain_and_bounds_cycles():
    module = _module()
    graph = {
        "urn:child": {"urn:extension"},
        "urn:extension": {"urn:root"},
        "urn:cycle-a": {"urn:cycle-b"},
        "urn:cycle-b": {"urn:cycle-a"},
    }

    assert module._has_import_path("urn:child", graph, {"urn:root"})
    assert not module._has_import_path("urn:cycle-a", graph, {"urn:root"})


def test_canonical_imports_every_registered_federated_iri():
    module = _module()
    canonical = module._parse(module.CANONICAL)

    assert module._federated_iris() <= set(module._imports(canonical))

