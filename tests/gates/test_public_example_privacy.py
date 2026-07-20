"""Keep public configuration examples independent of any private environment."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_EXAMPLES = (
    ".env.example",
    "AGENTS.head.md",
    "AGENTS.md",
    "CHANGELOG.md",
    "README.md",
    "docs/architecture/graph_backends_architecture.md",
    "docs/pillars/4_ecosystem_peripherals/ECO-4.5-Messaging_Configuration_Guide.md",
    "agent_utilities/core/config.py",
    "agent_utilities/gateway/widgets/langfuse.py",
    "agent_utilities/knowledge_graph/core/hydration.py",
    "agent_utilities/knowledge_graph/core/source_sync.py",
    "agent_utilities/knowledge_graph/ontology/connector_manifest_gate.py",
    "agent_utilities/mcp/kg_server.py",
    "agent_utilities/mcp/multiplexer.py",
    "agent_utilities/models/knowledge_graph.py",
    "tests/unit/core/test_config.py",
    "tests/unit/knowledge_graph/core/test_source_sync.py",
    "tests/unit/test_fuseki_publish_tick.py",
)
PRIVATE_IPV4 = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
ENVIRONMENT_DNS = re.compile(
    r"(?i)\b(?:[A-Za-z0-9-]+\.)+(?:arpa|local)(?=[:/\s\"'])"
)
MACHINE_HOME = re.compile(
    r"(?i)(?:[A-Z]:[\\/]Users[\\/](?![<%$])[^\\/\s]+|"
    r"/(?:home|Users)/(?![<%$])[^/\s]+|"
    r"/mnt/[A-Z]/Users/(?![<%$])[^/\s]+)"
)
ENVIRONMENT_DERIVED_HOST_ALIAS = re.compile(r"(?i)\b(?:rw?|gr)\d{3,}\b")
FIXTURE_SUFFIXES = frozenset({".json", ".py", ".toml", ".yaml", ".yml"})
INTENTIONAL_NEGATIVE_LITERALS = {
    # Narrow, counted exceptions for tests that prove the corresponding guard fails.
    "tests/gates/test_docs_contract_gate.py": ("rw" + "123",),
}


def _without_intentional_negatives(relative: str | Path, content: str) -> str:
    for literal in INTENTIONAL_NEGATIVE_LITERALS.get(str(relative), ()):
        assert content.count(literal) == 1, (relative, literal)
        content = content.replace(literal, "[INTENTIONAL_NEGATIVE_FIXTURE]")
    return content


def test_public_examples_are_environment_neutral() -> None:
    for relative in PUBLIC_EXAMPLES:
        content = (ROOT / relative).read_text(encoding="utf-8")
        content = _without_intentional_negatives(relative, content)
        assert PRIVATE_IPV4.search(content) is None, relative
        assert ENVIRONMENT_DNS.search(content) is None, relative
        assert MACHINE_HOME.search(content) is None, relative
        assert ENVIRONMENT_DERIVED_HOST_ALIAS.search(content) is None, relative


def test_tracked_fixtures_do_not_encode_environment_host_aliases() -> None:
    """Fixture identities must describe roles, never mirror an operator fleet."""
    for path in (ROOT / "tests").rglob("*"):
        if not path.is_file() or path.suffix not in FIXTURE_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        content = _without_intentional_negatives(relative, content)
        assert ENVIRONMENT_DERIVED_HOST_ALIAS.search(content) is None, relative


def test_private_network_gate_keeps_negative_and_documentation_ranges_distinct() -> None:
    private_examples = (
        "10" + ".0.0.1",
        "172" + ".16.0.1",
        "192" + ".168.0.1",
    )
    documentation_examples = ("192.0.2.1", "198.51.100.1", "203.0.113.1")

    assert all(PRIVATE_IPV4.search(value) for value in private_examples)
    assert all(PRIVATE_IPV4.search(value) is None for value in documentation_examples)


def test_tool_catalog_exposes_only_neutral_resource_references() -> None:
    source = (ROOT / "agent_utilities/mcp/kg_server.py").read_text(encoding="utf-8")

    assert '"file_path": f"skill://{name}"' in source
    assert '"file_path": f"tool://{f.stem}"' in source
    assert '"command": "[configured]"' in source
    assert '"args": ["[configured]"]' in source
