"""Meta-tests proving the documentation and privacy gates can fail."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _isolated_git_env() -> dict[str, str]:
    """D-LGI-1: an explicit, belt-and-suspenders env for git calls that
    target a throwaway fixture repo (``tmp_path``), not this repository.

    ``tests/conftest.py``'s ``_strip_inherited_git_repository_env()`` already
    does this once, session-wide, at collection time -- this is the specific
    site the incident was found at: a real ``git commit`` exports
    ``GIT_DIR``/``GIT_INDEX_FILE`` into the hooks it runs (including
    ``guardrail-gate-meta-tests``, which runs this test), and ``git -C
    <tmp_path> ...`` does **not** override them. Without this, `add -A`
    below wrote its fixture's file list into the REAL repository's index
    instead of the throwaway one it was pointed at.
    """
    env = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
    ):
        env.pop(name, None)
    return env


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
    rendered = privacy.Violation(
        "docs/sample.md", 7, next(iter(categories)), privacy._content_hash(line), 0
    ).render()
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


def test_privacy_gate_checks_runtime_source_for_environment_material():
    privacy = _load_script("check_tracked_privacy.py")
    synthetic_endpoint = "https://service.private." + "internal/api"
    categories = privacy.classify_runtime_source_line(
        f'ENDPOINT = "{synthetic_endpoint}"',
        identifiers=frozenset(),
    )
    assert categories == {"hard-coded internal endpoint in runtime source"}


def test_privacy_gate_checks_runtime_source_for_local_identity_without_echoing_it():
    privacy = _load_script("check_tracked_privacy.py")
    categories = privacy.classify_runtime_source_line(
        "owner = runtime-identity",
        identifiers=frozenset({"runtime-identity"}),
    )
    category = next(iter(categories))
    rendered = privacy.Violation(
        "agent_utilities/sample.py",
        2,
        category,
        privacy._content_hash("owner = runtime-identity"),
        0,
    ).render()
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
    assert privacy._runtime_source_artifacts(tmp_path) == [runtime]


def test_privacy_baseline_excuses_only_listed_leaks_and_never_a_new_one(tmp_path):
    """The D-CIP-10 baseline must be a ratchet, not an exemption.

    The scope fix took the gate from 7 reported leaks to 37, all of them
    pre-existing but previously invisible. They are baselined so `main` stays
    green while the debt is tracked — which is only defensible if a *new* leak
    still fails. That is what this pins, in both directions.

    D-W2P: the baseline key used to be ``(path, line, category)``. A leak
    whose LINE moved because unrelated code was inserted above it (the
    ordinary case — nothing about the leak itself changed) then reported as
    a phantom NEW finding, indistinguishable from a genuinely new leak. The
    key is now ``(path, category, content_hash, ordinal)`` — this pins both
    halves: line motion of the SAME content must stay baselined, and
    genuinely DIFFERENT content must still fail even at an already-baselined
    line number.
    """
    privacy = _load_script("check_tracked_privacy.py")
    same_content = "path: /home/some-account/build/state"
    baselined = privacy.Violation(
        "docker/job.yaml",
        12,
        "machine-specific home path in runtime source",
        privacy._content_hash(same_content),
        0,
    )
    # The identical leak, unchanged, just pushed to a later line by unrelated
    # edits earlier in the file — must resolve to the SAME baseline key.
    shifted = privacy.Violation(
        "docker/job.yaml",
        99,
        "machine-specific home path in runtime source",
        privacy._content_hash(same_content),
        0,
    )
    # A different leak, coincidentally landing on the previously-baselined
    # line number — must NOT be excused by a line-number match alone.
    different_content = "path: /home/a-different-account/other/state"
    fresh = privacy.Violation(
        "docker/job.yaml",
        12,
        "machine-specific home path in runtime source",
        privacy._content_hash(different_content),
        0,
    )

    baseline_file = tmp_path / "tracked_privacy_baseline.txt"
    baseline_file.write_text(
        "# header comment is skipped\n"
        f"{baselined.path}\t{baselined.line}\t{baselined.category}"
        f"\t{baselined.content_hash}\t{baselined.ordinal}\n",
        encoding="utf-8",
    )
    privacy.BASELINE = baseline_file
    loaded = privacy._load_baseline()

    assert privacy._baseline_key(baselined) in loaded
    # Pure line motion of the SAME leak must still read as already-baselined.
    assert privacy._baseline_key(shifted) in loaded
    assert privacy._baseline_key(baselined) == privacy._baseline_key(shifted)
    # A genuinely different leak at the SAME old line number is still new.
    assert privacy._baseline_key(fresh) not in loaded

    # A missing baseline must read as EMPTY, i.e. stricter — never laxer.
    privacy.BASELINE = tmp_path / "does_not_exist.txt"
    assert privacy._load_baseline() == set()


def test_privacy_gate_ordinal_disambiguates_duplicate_lines_in_one_file(tmp_path):
    """Two genuinely identical leak lines in the same file/category must not
    collapse into one baseline entry — fixing one occurrence and leaving the
    other must still show exactly one remaining violation, not zero."""
    privacy = _load_script("check_tracked_privacy.py")
    runtime_dir = tmp_path / "agent_utilities" / "core"
    runtime_dir.mkdir(parents=True)
    leak_line = 'ENDPOINT = "https://svc.internal.example.svc.cluster.local/api"'
    (runtime_dir / "sample.py").write_text(
        f"{leak_line}\nX = 1\n{leak_line}\n", encoding="utf-8"
    )
    violations = privacy.scan(tmp_path)
    matches = [v for v in violations if v.path.endswith("sample.py")]
    assert len(matches) == 2
    assert matches[0].content_hash == matches[1].content_hash
    assert {v.ordinal for v in matches} == {0, 1}
    # Distinct ordinals -> distinct baseline keys, so each site is tracked
    # independently.
    assert privacy._baseline_key(matches[0]) != privacy._baseline_key(matches[1])


def test_privacy_gate_scans_unchanged_runtime_source_not_only_the_diff(tmp_path):
    """D-CIP-10: a leak that landed in an earlier commit must still be caught.

    The regression this pins: ``_runtime_source_artifacts`` was scoped to
    ``git diff``/untracked files, so a machine path already committed under
    ``docker/`` was never re-examined and stayed public forever — while
    ``_is_public_artifact``'s location filter (``docs/``, ``.github/``,
    top-level, ``*.toml``) meant the whole-tree pass never looked at
    ``docker/`` either.

    This **must** run inside a real git repository with the file *committed and
    unmodified*. In a plain temp directory both the old and the new
    implementation fall back to :func:`_filesystem_files` and return the same
    answer, so a non-git fixture would pass against the very bug it claims to
    pin — the vacuous-gate shape this suite exists to prevent. With a real repo,
    the old diff-scoped code sees an empty ``git diff`` and returns ``[]``.
    """
    git_env = _isolated_git_env()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=git_env)
    leaked = tmp_path / "docker" / "build-job.yaml"
    leaked.parent.mkdir(parents=True)
    leaked.write_text("            path: /home/someone/state/tree\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True, env=git_env)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=gate@example.invalid",
            "-c",
            "user.name=gate",
            "commit",
            "-q",
            "-m",
            "committed leak",
        ],
        check=True,
        env=git_env,
    )
    # Nothing is staged or modified now: `git diff` and `git diff --cached` are
    # both empty, and `ls-files --others` lists nothing.
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "diff", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
            env=git_env,
        ).stdout.strip()
        == ""
    )

    privacy = _load_script("check_tracked_privacy.py")
    assert privacy._runtime_source_artifacts(tmp_path) == [leaked]

    categories = privacy.classify_runtime_source_line(
        leaked.read_text(encoding="utf-8").strip(), identifiers=frozenset()
    )
    assert "machine-specific home path in runtime source" in categories


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


def test_privacy_gate_identifiers_vary_with_ambient_identity_when_unpinned(
    monkeypatch,
):
    """D-ORC-57 (reproduction): with NO declared override, two different
    ambient identities (simulating two different lanes' sandboxes running
    the same scan against the same tree) derive DIFFERENT identifier sets --
    this is the non-determinism the item reports, and it must still
    reproduce when nothing has opted in to the fix.
    """
    privacy = _load_script("check_tracked_privacy.py")
    monkeypatch.delenv("AGENT_UTILITIES_PRIVACY_IDENTIFIERS", raising=False)

    monkeypatch.setattr(privacy.getpass, "getuser", lambda: "sandbox-one")
    monkeypatch.setattr(
        privacy.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        privacy.pwd, "getpwuid", lambda _uid: type("_Pw", (), {"pw_name": "sandbox-one"})()
    )
    first = privacy.derive_local_identifiers()

    monkeypatch.setattr(privacy.getpass, "getuser", lambda: "sandbox-two")
    monkeypatch.setattr(
        privacy.pwd, "getpwuid", lambda _uid: type("_Pw", (), {"pw_name": "sandbox-two"})()
    )
    second = privacy.derive_local_identifiers()

    assert first != second
    assert "sandbox-one" in first
    assert "sandbox-two" in second


def test_privacy_gate_identifiers_are_deterministic_with_declared_override(
    monkeypatch,
):
    """D-ORC-57 (fix): with a DECLARED override set, the same two ambient
    identities from the test above collapse to the identical identifier set
    -- the gate's verdict no longer depends on who/where it runs. Revert the
    ``AGENT_UTILITIES_PRIVACY_IDENTIFIERS`` short-circuit in
    ``derive_local_identifiers`` and this goes red (falls back to the
    ambient, sandbox-dependent set proven to differ above).
    """
    privacy = _load_script("check_tracked_privacy.py")

    monkeypatch.setenv("AGENT_UTILITIES_PRIVACY_IDENTIFIERS", "declared-identity")
    monkeypatch.setattr(privacy.getpass, "getuser", lambda: "sandbox-one")
    monkeypatch.setattr(
        privacy.pwd, "getpwuid", lambda _uid: type("_Pw", (), {"pw_name": "sandbox-one"})()
    )
    first = privacy.derive_local_identifiers()

    monkeypatch.setattr(privacy.getpass, "getuser", lambda: "sandbox-two")
    monkeypatch.setattr(
        privacy.pwd, "getpwuid", lambda _uid: type("_Pw", (), {"pw_name": "sandbox-two"})()
    )
    second = privacy.derive_local_identifiers()

    assert first == second == frozenset({"declared-identity"})
