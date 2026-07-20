from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    source = Path(__file__).parents[3] / "scripts" / "check_provider_fleet_contract.py"
    spec = importlib.util.spec_from_file_location(
        "check_provider_fleet_contract", source
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workspace(path: Path, providers: tuple[str, ...]) -> Path:
    entries = "\n".join(
        f'          - url: "https://example.invalid/providers/{name}.git"'
        for name in providers
    )
    path.write_text(
        "subdirectories:\n"
        "  agent-packages:\n"
        "    subdirectories:\n"
        "      agents:\n"
        "        repositories:\n"
        f"{entries}\n",
        encoding="utf-8",
    )
    return path


def _provider(root: Path, name: str, *, requirement: str) -> Path:
    provider = root / name
    provider.mkdir(parents=True)
    (provider / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{name}"\n'
        'version = "1.0.0"\n'
        'license = "MIT"\n'
        'authors = [{ name = "Project Maintainers", email = "maintainers@example.invalid" }]\n'
        f'dependencies = ["{requirement}"]\n',
        encoding="utf-8",
    )
    (provider / "requirements.txt").write_text(f"{requirement}\n", encoding="utf-8")
    (provider / "README.md").write_text(
        "The connector-focused `[mcp]` surface carries "
        "`epistemic-graph[full]`; `[agent]` adds model orchestration.\n",
        encoding="utf-8",
    )
    docker = provider / "docker"
    docker.mkdir()
    (docker / "Dockerfile").write_text(
        "# agent-utilities[agent-runtime] adds the interactive model runtime; "
        "the full engine is a base dependency.\n"
        "# agent-utilities[mcp] adds the MCP server surface.\n",
        encoding="utf-8",
    )
    return provider


def test_validates_declared_provider_contract(tmp_path):
    module = _module()
    providers_root = tmp_path / "agents"
    requirement = "agent-utilities[mcp]>=1.27.1,<2.0.0"
    _provider(providers_root, "alpha-agent", requirement=requirement)
    _provider(providers_root, "beta-mcp", requirement=requirement)
    workspace = _workspace(tmp_path / "workspace.yml", ("alpha-agent", "beta-mcp"))

    findings, stats = module.validate_fleet(
        workspace, providers_root, expected_provider_count=2
    )

    assert findings == []
    assert stats.providers == 2
    assert stats.pyproject_requirements == 2
    assert stats.requirements_file_requirements == 2


def test_rejects_stale_bounds_source_and_documentation(tmp_path):
    module = _module()
    providers_root = tmp_path / "agents"
    provider = _provider(
        providers_root,
        "sample-agent",
        requirement="agent-utilities[mcp]>=1.20.0",
    )
    pyproject = provider / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n[tool.uv.sources]\n"
        + 'agent-utilities = { path = "../different-source", editable = false }\n',
        encoding="utf-8",
    )
    (provider / "README.md").write_text(
        "The slim `[mcp]` server does **not** require the database.\n",
        encoding="utf-8",
    )
    workspace = _workspace(tmp_path / "workspace.yml", ("sample-agent",))

    findings, _ = module.validate_fleet(
        workspace, providers_root, expected_provider_count=1
    )
    rules = {finding.rule.split(":", 1)[0] for finding in findings}

    assert rules >= {
        "dependency_bounds",
        "local_source_forbidden",
        "mcp_declared_without_database",
    }
    assert all(str(tmp_path) not in finding.path for finding in findings)


def test_rejects_retired_extras_in_provider_dependency_surfaces(tmp_path):
    module = _module()
    providers_root = tmp_path / "agents"
    provider = _provider(
        providers_root,
        "sample-agent",
        requirement="agent-utilities[agent,logfire]>=1.27.1,<2.0.0",
    )
    (provider / "requirements.txt").write_text(
        "agent-utilities[engine,mcp]>=1.27.1,<2.0.0\n",
        encoding="utf-8",
    )
    workspace = _workspace(tmp_path / "workspace.yml", ("sample-agent",))

    findings, _ = module.validate_fleet(
        workspace, providers_root, expected_provider_count=1
    )

    rules = {finding.rule for finding in findings}
    assert any(
        rule.startswith("retired_agent_utilities_extra:agent:") for rule in rules
    )
    assert "retired_agent_utilities_extra:engine" in rules


def test_rejects_retired_dockerfile_runtime_claims(tmp_path):
    module = _module()
    providers_root = tmp_path / "agents"
    provider = _provider(
        providers_root,
        "sample-agent",
        requirement="agent-utilities[mcp]>=1.27.1,<2.0.0",
    )
    (provider / "docker" / "Dockerfile").write_text(
        "# agent-utilities[agent] includes pydantic-ai and DSPy.\n"
        "# agent-utilities[mcp] has NO engine.\n"
        "# agent-utilities[engine] selects the graph runtime.\n",
        encoding="utf-8",
    )
    workspace = _workspace(tmp_path / "workspace.yml", ("sample-agent",))

    findings, _ = module.validate_fleet(
        workspace, providers_root, expected_provider_count=1
    )

    docker_findings = {
        finding.rule
        for finding in findings
        if finding.path == "docker/Dockerfile"
    }
    assert docker_findings >= {
        "agent_utilities_dspy_claim",
        "mcp_claims_no_engine",
        "retired_agent_utilities_extra",
    }


def test_rejects_misnested_license_in_author_metadata(tmp_path):
    module = _module()
    providers_root = tmp_path / "agents"
    provider = _provider(
        providers_root,
        "sample-agent",
        requirement="agent-utilities[mcp]>=1.27.1,<2.0.0",
    )
    pyproject = provider / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        .replace('license = "MIT"\n', "")
        .replace(
            'email = "maintainers@example.invalid"',
            'email = "maintainers@example.invalid", license = "MIT"',
        ),
        encoding="utf-8",
    )
    workspace = _workspace(tmp_path / "workspace.yml", ("sample-agent",))

    findings, _ = module.validate_fleet(
        workspace, providers_root, expected_provider_count=1
    )
    rules = {finding.rule for finding in findings}

    assert "project_authors_invalid" in rules
    assert "project_license_invalid" in rules
