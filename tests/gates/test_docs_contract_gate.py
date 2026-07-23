"""Meta-tests proving the documentation and privacy gates can fail."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_privacy_gate_detects_persisted_machine_path_without_echoing_value():
    privacy = _load_script("check_tracked_privacy.py")
    line = 'workspace_path: "/home/local-account/project"'
    categories = privacy.classify_line(
        line,
        identifiers=frozenset(),
        deployment_doc=True,
    )
    assert "persisted machine path" in categories
    rendered = privacy.Violation("docs/sample.md", 7, next(iter(categories))).render()
    assert "local-account" not in rendered
    assert "/home/" not in rendered


def test_privacy_gate_requires_neutral_uri_for_persisted_path_field():
    privacy = _load_script("check_tracked_privacy.py")
    unsafe = privacy.classify_line(
        'source_path: "package/docs/source.md"',
        identifiers=frozenset(),
        deployment_doc=False,
    )
    safe = privacy.classify_line(
        'source_path: "repo://package/docs/source.md"',
        identifiers=frozenset(),
        deployment_doc=False,
    )
    assert "persisted machine path" in unsafe
    assert "persisted machine path" not in safe


def test_privacy_gate_allows_repo_relative_runtime_env_path():
    privacy = _load_script("check_tracked_privacy.py")
    categories = privacy.classify_line(
        '"WORKSPACE_PATH": "./runtime-workspace"',
        identifiers=frozenset(),
        deployment_doc=False,
    )
    assert "persisted machine path" not in categories


def test_privacy_gate_detects_runtime_identifier_without_echoing_value():
    privacy = _load_script("check_tracked_privacy.py")
    categories = privacy.classify_line(
        "deployed by runtime-identity",
        identifiers=frozenset({"runtime-identity"}),
        deployment_doc=False,
    )
    assert categories == {"local account or host identifier"}


def test_privacy_gate_detects_machine_host_id_pattern():
    privacy = _load_script("check_tracked_privacy.py")
    categories = privacy.classify_line(
        "deployment host rw123",
        identifiers=frozenset(),
        deployment_doc=False,
    )
    assert categories == {"machine-specific host identifier"}


def test_privacy_gate_detects_internal_deployment_endpoint():
    privacy = _load_script("check_tracked_privacy.py")
    categories = privacy.classify_line(
        "broker.apps.svc.cluster.local:9092",
        identifiers=frozenset(),
        deployment_doc=True,
    )
    assert categories == {"hard-coded internal endpoint"}


def test_privacy_gate_rejects_non_neutral_package_author(tmp_path):
    privacy = _load_script("check_tracked_privacy.py")
    manifest = tmp_path / "Cargo.toml"
    lines = ['authors = ["Package Author <author@example.invalid>"]']
    assert privacy._author_metadata_lines(manifest, lines) == [1]
    neutral = ['authors = ["Repository Maintainers <maintainers@example.invalid>"]']
    assert privacy._author_metadata_lines(manifest, neutral) == []


def test_privacy_gate_checks_pep621_author_table(tmp_path):
    privacy = _load_script("check_tracked_privacy.py")
    manifest = tmp_path / "pyproject.toml"
    non_neutral = [
        "[[project.authors]]",
        'name = "Package Author"',
        'email = "author@package.example"',
    ]
    assert privacy._author_metadata_lines(manifest, non_neutral) == [2, 3]
    neutral = [
        "[[project.authors]]",
        'name = "Repository Maintainers"',
        'email = "maintainers@example.invalid"',
    ]
    assert privacy._author_metadata_lines(manifest, neutral) == []


def test_privacy_gate_includes_public_metadata_but_excludes_bundled_skills():
    privacy = _load_script("check_tracked_privacy.py")
    assert privacy._is_public_artifact("pyproject.toml")
    assert privacy._is_public_artifact(".github/workflows/guardrails.yml")
    assert not privacy._is_public_artifact("agent_utilities/skills/demo/SKILL.md")


def test_privacy_gate_checks_changed_source_for_environment_material():
    privacy = _load_script("check_tracked_privacy.py")
    synthetic_endpoint = "https://service.private." + "internal/api"
    categories = privacy.classify_changed_source_line(
        f'ENDPOINT = "{synthetic_endpoint}"',
        identifiers=frozenset(),
    )
    assert categories == {"hard-coded internal endpoint in changed source"}


def test_privacy_gate_checks_changed_source_for_local_identity_without_echoing_it():
    privacy = _load_script("check_tracked_privacy.py")
    categories = privacy.classify_changed_source_line(
        "owner = runtime-identity",
        identifiers=frozenset({"runtime-identity"}),
    )
    category = next(iter(categories))
    rendered = privacy.Violation("agent_utilities/sample.py", 2, category).render()
    assert "local account or host identifier" in rendered
    assert "runtime-identity" not in rendered


def test_privacy_gate_rejects_bundled_connector_profiles():
    privacy = _load_script("check_tracked_privacy.py")
    assert privacy._is_bundled_connector_profile(
        Path("agent_utilities/protocols/source_connectors/profiles/site.py")
    )
    assert not privacy._is_bundled_connector_profile(
        Path(
            "agent_utilities/protocols/source_connectors/connectors/graphql_document.py"
        )
    )


def test_privacy_gate_changed_source_scope_excludes_adversarial_tests():
    privacy = _load_script("check_tracked_privacy.py")
    assert privacy._is_runtime_source_path(Path("agent_utilities/core/config.py"))
    assert privacy._is_runtime_source_path(
        Path("agent_utilities/skills/graph-query-and-explanation/SKILL.md")
    )
    assert not privacy._is_runtime_source_path(Path("tests/test_privacy.py"))
    assert not privacy._is_runtime_source_path(Path("scripts/check_tracked_privacy.py"))


def test_privacy_gate_scans_immutable_no_git_snapshot(tmp_path):
    privacy = _load_script("check_tracked_privacy.py")
    public = tmp_path / "docs" / "configuration.md"
    public.parent.mkdir()
    public.write_text("# Configuration\n", encoding="utf-8")
    runtime = tmp_path / "agent_utilities" / "core" / "sample.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("SETTING = 'neutral'\n", encoding="utf-8")
    ignored = tmp_path / ".pytest_cache" / "leaked.md"
    ignored.parent.mkdir()
    ignored.write_text("ignored\n", encoding="utf-8")

    assert privacy._tracked_artifacts(tmp_path) == [public]
    assert privacy._changed_source_artifacts(tmp_path) == [runtime]


def test_docs_link_gate_trips_on_missing_target(tmp_path, monkeypatch):
    contract = _load_script("docs_contract.py")
    docs = tmp_path / "docs"
    docs.mkdir()
    page = docs / "index.md"
    page.write_text("# Index\n\n[missing](missing.md)\n", encoding="utf-8")
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "DOCS", docs)
    monkeypatch.setattr(contract, "_site_pages", lambda: [page])
    assert any("unresolved local link" in error for error in contract.check_links())


def test_docs_anchor_gate_trips_on_missing_anchor(tmp_path, monkeypatch):
    contract = _load_script("docs_contract.py")
    docs = tmp_path / "docs"
    docs.mkdir()
    page = docs / "index.md"
    page.write_text("# Index\n\n[section](#absent)\n", encoding="utf-8")
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "DOCS", docs)
    monkeypatch.setattr(contract, "_site_pages", lambda: [page])
    assert any("missing anchor" in error for error in contract.check_links())


def test_docs_link_gate_rejects_machine_local_file_uri(tmp_path, monkeypatch):
    contract = _load_script("docs_contract.py")
    docs = tmp_path / "docs"
    docs.mkdir()
    page = docs / "index.md"
    page.write_text(
        "# Index\n\n[local](file:///workspace/docs/note.md)\n", encoding="utf-8"
    )
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "DOCS", docs)
    monkeypatch.setattr(contract, "_site_pages", lambda: [page])
    assert any("non-portable file URI" in error for error in contract.check_links())


def test_docs_gate_rejects_retired_graphos_tool_surface_claim():
    contract = _load_script("docs_contract.py")
    current = contract.GRAPHOS_SURFACE_DOC.read_text(encoding="utf-8")
    assert contract.check_graphos_surface_contract(current) == []

    stale = current + "\nThe graph-os MCP server exposes **25 tools**.\n"
    errors = contract.check_graphos_surface_contract(stale)
    assert any("retired 25-tool claim" in error for error in errors)


def test_docs_gate_binds_graphos_surface_to_generated_capability_count():
    contract = _load_script("docs_contract.py")
    current = contract.GRAPHOS_SURFACE_DOC.read_text(encoding="utf-8")
    assert contract.check_graphos_surface_contract(current) == []
    assert contract.check_graphos_surface_contract(
        current,
        capability_count=116,
    ) == ["GraphOS surface documentation is stale"]


def test_docs_gate_requires_installed_release_skill_attestation():
    contract = _load_script("docs_contract.py")
    current = contract.SKILL_CERTIFICATION_DOC.read_text(encoding="utf-8")
    assert contract.check_skill_certification_docs_contract(current) == []

    stale = current.replace('"agentUtilitiesSha256": "sha256:<digest>"', "", 1)
    assert contract.check_skill_certification_docs_contract(stale) == [
        "skill certification documentation is stale"
    ]


def test_generated_agent_tree_contains_only_tracked_top_level_paths():
    """Ignored runtime artifacts must not leak into the generated project tree."""
    generator = _load_script("gen_agents_md.py")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked_top = {
        Path(raw.decode("utf-8", errors="strict")).parts[0]
        for raw in result.stdout.split(b"\0")
        if raw
    }
    rendered_top = {
        line[4:].split("/", 1)[0]
        for line in generator.project_tree_section().splitlines()
        if line.startswith(("├── ", "└── "))
    }

    assert rendered_top
    assert rendered_top <= tracked_top
