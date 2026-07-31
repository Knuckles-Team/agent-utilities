"""A signed manifest may never name a server the fleet registry does not run.

CONCEPT:AU-KG.ontology.registry-derived-server-alias. The regression pinned here is
D-OB-7: providers hard-coded ``server`` in ``mcp_source_presets.json`` and the
manifest generator copied it verbatim with no registry validation, so 27 signed
manifests named the provider distribution (``github-agent``) rather than the
registered service (``github-mcp``), and 9 more named services the committed
registry had never contained.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import secrets
import sys
from pathlib import Path
from types import ModuleType

import pytest

from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.orchestration.fleet_reconciler import (
    FleetRegistryError,
    registry_server_alias,
    registry_server_aliases,
)

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> ModuleType:
    alias = f"{name}_alias_drift"
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so module-level ``@dataclass`` can resolve its
    # own module namespace (Python 3.14 dereferences ``cls.__module__``).
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


generator = _load("generate_connector_manifests")
preset_gate = _load("check_connector_source_presets")
registry_generator = _load("gen_mcp_fleet_registry")

_ONTOLOGY = """\
@prefix : <http://knuckles.team/kg#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://knuckles.team/kg/fixture> a owl:Ontology .
:Widget a owl:Class ; rdfs:label "Widget" .
"""

_REGISTRY = """\
version: 1
services:
  - name: widget-mcp
    package: widget-agent
    console_script: widget-mcp
    host_port: 8200
  - name: gadget-mcp
    package: gadget-agent
    console_script: gadget-mcp
    host_port: 8201
"""


@pytest.fixture
def signer(
    monkeypatch: pytest.MonkeyPatch,
) -> ontology_integrity.ReleaseSigner:
    """Manifest generation signs; the drift under test is orthogonal to the key.

    CONCEPT:AU-KG.ontology.release-key-rotation — publication signing now
    requires versioned secret custody (``vault://``/``secret://``), which this
    fixture's ``env://`` material intentionally is not. Return an explicit
    signer built from that material (``ReleaseSigner.from_runtime`` has no such
    restriction) so callers bypass the publication resolver entirely, the same
    pattern used throughout the connector-manifest test suite.
    """

    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    return ontology_integrity.ReleaseSigner.from_runtime()


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "mcp-fleet.registry.yml"
    path.write_text(_REGISTRY, encoding="utf-8")
    return path


def _provider(agents_root: Path, name: str, declared_server: str) -> Path:
    repo = agents_root / name
    module = repo / name.replace("-", "_")
    (module / "ontology").mkdir(parents=True)
    (module / "ontology" / "fixture.ttl").write_text(_ONTOLOGY, encoding="utf-8")
    (module / "connectors").mkdir(parents=True)
    (module / "connectors" / "mcp_source_presets.json").write_text(
        json.dumps(
            {
                "widgets": {
                    "server": declared_server,
                    "tool": "list_widgets",
                    "action": "list",
                    "doc_type": "widget",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    return repo


def test_the_alias_is_derived_from_the_registry(
    tmp_path: Path, registry: Path, signer: ontology_integrity.ReleaseSigner
) -> None:
    repo = _provider(tmp_path / "agents", "widget-agent", "widget-mcp")

    manifest = generator.build_manifest(
        repo, registry_path=registry, release_signer=signer
    )

    assert [sync.server for sync in manifest.sync] == ["widget-mcp"]
    assert [sync.raw["server"] for sync in manifest.sync] == ["widget-mcp"]


def test_a_preset_that_restates_a_wrong_alias_cannot_be_signed(
    tmp_path: Path, registry: Path
) -> None:
    """The exact D-OB-7 shape: the distribution name where the service belongs."""

    repo = _provider(tmp_path / "agents", "widget-agent", "widget-agent")

    with pytest.raises(RuntimeError):
        generator.build_manifest(repo, registry_path=registry)


def test_an_unregistered_provider_cannot_be_signed(
    tmp_path: Path, registry: Path
) -> None:
    repo = _provider(tmp_path / "agents", "unlisted-agent", "unlisted-mcp")

    with pytest.raises(FleetRegistryError):
        generator.build_manifest(repo, registry_path=registry)


def test_an_omitted_server_field_still_resolves(
    tmp_path: Path, registry: Path, signer: ontology_integrity.ReleaseSigner
) -> None:
    repo = _provider(tmp_path / "agents", "widget-agent", "")

    manifest = generator.build_manifest(
        repo, registry_path=registry, release_signer=signer
    )

    assert [sync.server for sync in manifest.sync] == ["widget-mcp"]


def test_the_gate_reports_then_repairs_alias_drift(
    tmp_path: Path, registry: Path
) -> None:
    agents_root = tmp_path / "agents"
    _provider(agents_root, "widget-agent", "widget-agent")
    _provider(agents_root, "gadget-agent", "gadget-mcp")

    findings = preset_gate.audit(agents_root, registry)
    assert [(f.kind, f.provider) for f in findings] == [
        ("server-alias-drift", "widget-agent")
    ]

    preset_gate.repair(findings)

    assert preset_gate.audit(agents_root, registry) == []


def test_the_gate_reports_an_unregistered_provider_it_cannot_repair(
    tmp_path: Path, registry: Path
) -> None:
    agents_root = tmp_path / "agents"
    _provider(agents_root, "unlisted-agent", "unlisted-mcp")

    findings = preset_gate.audit(agents_root, registry)

    assert [f.kind for f in findings] == ["unregistered-provider"]
    assert preset_gate.repair(findings) == []


def test_the_committed_registry_registers_every_certified_connector() -> None:
    """The 9 'absent from the registry' providers must all resolve now."""

    aliases = registry_server_aliases()
    catalog = json.loads(
        (ROOT / "deploy/release/connector-bundles.catalog.json").read_text(
            encoding="utf-8"
        )
    )
    certified = sorted(entry["connector"] for entry in catalog["entries"])

    assert certified
    assert [name for name in certified if name not in aliases] == []


def test_regenerating_the_registry_never_renumbers_a_published_port(
    tmp_path: Path,
) -> None:
    """Port churn is why the registry was left stale; regeneration must be a no-op."""

    services = [
        {"name": "widget-mcp", "package": "widget-agent"},
        {"name": "gadget-mcp", "package": "gadget-agent"},
    ]
    allocated = registry_generator.allocated_host_ports(
        _write_registry(tmp_path, [("widget-mcp", 8200), ("gadget-mcp", 8207)])
    )

    registry_generator.assign_host_ports(services, allocated)

    assert [svc["host_port"] for svc in services] == [8200, 8207]


def test_a_new_provider_takes_the_next_free_port(tmp_path: Path) -> None:
    services = [
        {"name": "widget-mcp", "package": "widget-agent"},
        {"name": "brand-new-mcp", "package": "brand-new-agent"},
    ]
    allocated = registry_generator.allocated_host_ports(
        _write_registry(tmp_path, [("widget-mcp", 8200)])
    )

    registry_generator.assign_host_ports(services, allocated)

    assert [svc["host_port"] for svc in services] == [8200, 8201]


def test_registry_lookup_fails_closed_for_an_unknown_package(registry: Path) -> None:
    with pytest.raises(FleetRegistryError):
        registry_server_alias("nothing-like-this", registry)


def _write_registry(tmp_path: Path, entries: list[tuple[str, int]]) -> Path:
    lines = ["version: 1", "services:"]
    for name, port in entries:
        lines += [
            f"  - name: {name}",
            f"    package: {name.replace('-mcp', '-agent')}",
            f"    host_port: {port}",
        ]
    path = tmp_path / "existing-registry.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
