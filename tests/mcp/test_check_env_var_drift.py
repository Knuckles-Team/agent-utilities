"""Tests for the env-var drift detector (CONCEPT:AU-OS.config.env-var-drift-guard)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.mcp import check_env_var_drift as drift


def _make_pkg(tmp_path: Path, *, env_example: str, mcp_config: dict, code: str) -> Path:
    root = tmp_path / "demo-agent"
    (root / "demo_agent").mkdir(parents=True)
    (root / ".env.example").write_text(env_example, encoding="utf-8")
    (root / "mcp_config.json").write_text(json.dumps(mcp_config), encoding="utf-8")
    (root / "demo_agent" / "auth.py").write_text(code, encoding="utf-8")
    return root


def _types(report: dict, kind: str) -> set[str]:
    return {f["var"] for f in report["findings"] if f["type"] == kind}


def test_dead_var_flagged(tmp_path: Path) -> None:
    """A var in mcp_config that no code reads is DEAD."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {
                        "DEMO_BASE_URL": "x",
                        "DEMO_TOKEN": "x",  # read by nothing -> DEAD
                        "MCP_TOOL_MODE": "condensed",
                    }
                }
            }
        },
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    report = drift.analyze(root)
    assert "DEMO_TOKEN" in _types(report, "DEAD")
    assert "DEMO_BASE_URL" not in _types(report, "DEAD")


def test_runtime_allowlist_not_dead(tmp_path: Path) -> None:
    """Generic process vars (TERM, NO_COLOR) are not flagged dead."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {"TERM": "xterm", "NO_COLOR": "1", "MCP_TOOL_MODE": "both"}
                }
            }
        },
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    report = drift.analyze(root)
    assert _types(report, "DEAD") == set()


def test_platform_runtime_inputs_are_not_provider_configuration(
    tmp_path: Path,
) -> None:
    """Locale and Windows platform inputs need no provider env documentation."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=https://service.example.invalid\n",
        mcp_config={
            "mcpServers": {
                "demo": {"env": {"MCP_TOOL_MODE": "intent"}},
            }
        },
        code=(
            "import os\n"
            'os.getenv("LC_ALL")\n'
            'os.getenv("SYSTEMROOT")\n'
            'os.getenv("PROGRAMFILES")\n'
            'setting("DEMO_BASE_URL", "")\n'
        ),
    )
    report = drift.analyze(root)
    platform_inputs = {"LC_ALL", "SYSTEMROOT", "PROGRAMFILES"}
    assert platform_inputs.isdisjoint(_types(report, "UNDOCUMENTED"))
    declared = (root / ".env.example").read_text(encoding="utf-8")
    assert all(name not in declared for name in platform_inputs)


def test_parent_injected_provider_profile_is_not_public_configuration(
    tmp_path: Path,
) -> None:
    """The GraphOS child selector is internal and needs no provider env entry."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=https://service.example.invalid\n",
        mcp_config={
            "mcpServers": {
                "demo": {"env": {"MCP_TOOL_MODE": "intent"}},
            }
        },
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "")\n'
            'setting("AGENT_PROVIDER_PROFILE", "")\n'
        ),
    )
    report = drift.analyze(root)
    assert "AGENT_PROVIDER_PROFILE" not in _types(report, "UNDOCUMENTED")
    assert "AGENT_PROVIDER_PROFILE" not in (root / ".env.example").read_text(
        encoding="utf-8"
    )


def test_missing_tool_mode_flagged(tmp_path: Path) -> None:
    """An mcp_config env block without MCP_TOOL_MODE is flagged."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"DEMO_BASE_URL": "x"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    report = drift.analyze(root)
    assert "MCP_TOOL_MODE" in _types(report, "MISSING_TOOL_MODE")


def test_upstream_host_not_undocumented(tmp_path: Path) -> None:
    """An upstream SDK host input is covered by its documented service host."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "") or setting("DEMO_HOST", "d")\n'
        ),
    )
    report = drift.analyze(root)
    assert "DEMO_HOST" not in _types(report, "UNDOCUMENTED")


def test_os_getenv_read_not_dead(tmp_path: Path) -> None:
    """A var read via bare os.getenv / os.environ is a real read, not DEAD."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {
                        "DEMO_BASE_URL": "x",
                        "HARVEST_HOST": "h",  # read via os.getenv below
                        "HARVEST_PORT": "1",  # read via os.environ[] below
                        "MCP_TOOL_MODE": "condensed",
                    }
                }
            }
        },
        code='import os\nos.getenv("HARVEST_HOST")\nos.environ["HARVEST_PORT"]\n',
    )
    report = drift.analyze(root)
    assert "HARVEST_HOST" not in _types(report, "DEAD")
    assert "HARVEST_PORT" not in _types(report, "DEAD")


def test_multiline_and_wrapped_reads_are_discovered(tmp_path: Path) -> None:
    """AST discovery covers multiline and package-local config wrappers."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_TOKEN=\nDEMO_GATE=true\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "intent"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            "def _setting(name, default):\n"
            "    return setting(name, default)\n"
            "setting(\n"
            "    'DEMO_TOKEN',\n"
            "    '',\n"
            ")\n"
            "_setting('DEMO_GATE', 'true')\n"
        ),
    )
    report = drift.analyze(root)
    assert "DEMO_TOKEN" not in _types(report, "DEAD")
    assert "DEMO_GATE" not in _types(report, "DEAD")


def test_test_only_reads_are_not_deployment_configuration(tmp_path: Path) -> None:
    """Test fixtures never expand a package's public runtime configuration."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "intent"}}}},
        code='setting("DEMO_BASE_URL", "")\n',
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_fixture.py").write_text(
        'setting("TEST_ONLY_RUNTIME_INPUT", "")\n', encoding="utf-8"
    )
    assert "TEST_ONLY_RUNTIME_INPUT" not in _types(drift.analyze(root), "UNDOCUMENTED")


def test_env_write_not_a_read(tmp_path: Path) -> None:
    """os.environ["X"] = ... is a write (cross-process signal), not a documentable read."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "import os\n"
            'setting("DEMO_BASE_URL", "")\n'
            'os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"\n'
            'os.environ["DEMO_SIGNAL"] = "1"\n'
        ),
    )
    flagged = {f["var"] for f in drift.analyze(root)["findings"]}
    assert "DEMO_SIGNAL" not in flagged
    assert "FASTMCP_LOG_LEVEL" not in flagged


def test_read_inside_string_literal_ignored(tmp_path: Path) -> None:
    """An os.environ.get("X") nested in a codegen template string is not a real read."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "import os\n"
            'setting("DEMO_BASE_URL", "")\n'
            "lines = []\n"
            "lines.append('    token = os.environ.get(\"GENERATED_TOKEN\")')\n"
        ),
    )
    assert "GENERATED_TOKEN" not in {f["var"] for f in drift.analyze(root)["findings"]}


def test_malformed_substitution_flagged(tmp_path: Path) -> None:
    """A whitespace-padded substitution "${ VAR:-True }" in a config value is MALFORMED_VALUE."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {
                        "DEMO_BASE_URL": "${ DEMO_BASE_URL:-http://x }",  # spaces -> malformed
                        "MCP_TOOL_MODE": "condensed",
                    }
                }
            }
        },
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    assert "DEMO_BASE_URL" in _types(drift.analyze(root), "MALFORMED_VALUE")


def test_agent_var_in_mcp_flagged(tmp_path: Path) -> None:
    """An agent-only var in an MCP-server config env block is AGENT_VAR_IN_MCP."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {
                        "DEMO_BASE_URL": "x",
                        "AGENT_DESCRIPTION": "y",  # agent-only
                        "SYSTEM_TOOLS_ENABLE": "True",  # companion suite
                        "MCP_TOOL_MODE": "condensed",
                    }
                }
            }
        },
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    flagged = _types(drift.analyze(root), "AGENT_VAR_IN_MCP")
    assert {"AGENT_DESCRIPTION", "SYSTEM_TOOLS_ENABLE"} <= flagged


def test_stale_readme_example_flagged(tmp_path: Path) -> None:
    """A README mcp_config example env key not in the code-read surface is STALE_EXAMPLE,
    and an example block without MCP_TOOL_MODE is flagged MISSING_TOOL_MODE."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    (root / "README.md").write_text(
        "## MCP Configuration Examples\n\n"
        "```json\n"
        '{"mcpServers": {"demo": {"env": {"SYSTEM_TOOLS_ENABLE": "x"}}}}\n'
        "```\n"
        "<!-- BEGIN GENERATED: additional-deployment-options -->\n",
        encoding="utf-8",
    )
    report = drift.analyze(root)
    assert "SYSTEM_TOOLS_ENABLE" in _types(report, "STALE_EXAMPLE")
    assert "MCP_TOOL_MODE" in _types(report, "MISSING_TOOL_MODE")


def test_scripts_read_suppresses_dead(tmp_path: Path) -> None:
    """Blind spot 1: a var declared in .env.example/mcp_config but read ONLY by the
    package's own scripts/*.py (not the main package tree) must not be reported DEAD —
    scripts/ is genuine first-party code, e.g. scripts/validate_falkordb.py reading
    FALKORDB_URI in the real genius-agent repo."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\nFALKORDB_URI=\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {"DEMO_BASE_URL": "x", "MCP_TOOL_MODE": "condensed"}
                }
            }
        },
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "validate_falkordb.py").write_text(
        'import os\nos.environ.get("FALKORDB_URI", "redis://127.0.0.1:6379")\n',
        encoding="utf-8",
    )
    report = drift.analyze(root)
    assert "FALKORDB_URI" not in _types(report, "DEAD")


def test_scripts_read_does_not_force_undocumented(tmp_path: Path) -> None:
    """A var read ONLY by a scripts/*.py dev/CI tool — never declared anywhere — is NOT
    flagged UNDOCUMENTED. scripts/ in this fleet is scaffolded maintainer-only tooling
    (validation harnesses, local gate runners) shared byte-identical across 60+ packages,
    each reading it with a hardcoded fallback default at the call site (e.g. AGENT_UTILITIES_
    ROOT in scripts/run_agent_utilities_gate.py). Folding scripts/ reads into the
    documentable surface would force every package's PUBLIC .env.example to document
    dev-only gate-script knobs a deployer never sets — an UNDOCUMENTED false positive
    observed identically across the fleet while fixing this blind spot."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "intent"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "run_agent_utilities_gate.py").write_text(
        'import os\nos.environ.get("AGENT_UTILITIES_ROOT")\n', encoding="utf-8"
    )
    report = drift.analyze(root)
    assert "AGENT_UTILITIES_ROOT" not in _types(report, "UNDOCUMENTED")
    assert "AGENT_UTILITIES_ROOT" not in _types(report, "DEAD")


def test_dynamic_tls_family_read_suppresses_dead(tmp_path: Path) -> None:
    """Blind spot 2: MEALIE_TLS_PROFILE / MEALIE_TLS_PROFILE_REF are composed at runtime by
    ``resolve_configured_tls_profile(service="mealie")`` — no literal of either name appears
    anywhere in the calling package, so a purely static scan cannot see the read. The
    checker instead reads the CONCEPT:AU-OS.config.dynamic-env-family declaration published
    on the real ``resolve_configured_tls_profile``/``resolve_tls_profile`` functions in
    ``agent_utilities.core.transport_security`` (dynamic_env_prefix_arg="service",
    dynamic_env_suffixes=("TLS_PROFILE", "TLS_PROFILE_REF"))."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\nMEALIE_TLS_PROFILE=\nMEALIE_TLS_PROFILE_REF=\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "")\n'
            "from agent_utilities.core.transport_security import (\n"
            "    resolve_configured_tls_profile,\n"
            ")\n"
            'resolve_configured_tls_profile("mealie")\n'
        ),
    )
    report = drift.analyze(root)
    assert "MEALIE_TLS_PROFILE" not in _types(report, "DEAD")
    assert "MEALIE_TLS_PROFILE_REF" not in _types(report, "DEAD")


def test_dynamic_tls_family_read_flags_undocumented(tmp_path: Path) -> None:
    """The dynamic-family family is narrowly scoped to the two PRIMARY per-service TLS
    selectors precisely so it is safe to also enforce documentation of them: a service
    literal used in code but whose derived vars are absent from .env.example is genuinely
    UNDOCUMENTED, same as any other real code read."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "")\n'
            "from agent_utilities.core.transport_security import resolve_tls_profile\n"
            'resolve_tls_profile(service="newsvc")\n'
        ),
    )
    report = drift.analyze(root)
    undocumented = _types(report, "UNDOCUMENTED")
    assert "NEWSVC_TLS_PROFILE" in undocumented
    assert "NEWSVC_TLS_PROFILE_REF" in undocumented


def test_dynamic_family_non_literal_prefix_is_silently_skipped(tmp_path: Path) -> None:
    """A runtime-computed (non-literal) service argument cannot be resolved statically —
    same limit as every other static scan here — and must not raise or produce a bogus
    var name."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "")\n'
            "from agent_utilities.core.transport_security import resolve_tls_profile\n"
            "svc = compute_service_name()\n"
            "resolve_tls_profile(svc)\n"
        ),
    )
    report = drift.analyze(root)  # must not raise
    assert report["drift"] >= 0


def test_compose_image_substitution_suppresses_dead(tmp_path: Path) -> None:
    """Blind spot 3: image: ${VAR:?...} in a compose file is a genuine read (docker compose
    substitutes it at `docker compose up` time) that only ``environment:`` blocks were
    previously scanned for."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\nDEMO_MCP_IMAGE=\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    (root / "docker").mkdir()
    (root / "docker" / "mcp.compose.yml").write_text(
        "services:\n"
        "  demo:\n"
        "    image: ${DEMO_MCP_IMAGE:?set-DEMO_MCP_IMAGE-to-image@sha256-digest}\n",
        encoding="utf-8",
    )
    report = drift.analyze(root)
    assert "DEMO_MCP_IMAGE" not in _types(report, "DEAD")


def test_compose_command_and_entrypoint_substitution_suppresses_dead(
    tmp_path: Path,
) -> None:
    """The same scan covers multi-line YAML list values under command:/entrypoint:/args:,
    not just an inline scalar like image:."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\nDEMO_FLAG=\nDEMO_ENTRY=\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    (root / "docker").mkdir()
    (root / "docker" / "mcp.compose.yml").write_text(
        "services:\n"
        "  demo:\n"
        "    entrypoint: [\"${DEMO_ENTRY}\"]\n"
        "    command:\n"
        '      - "--flag=${DEMO_FLAG:-default}"\n'
        '      - "--static"\n'
        "    ports:\n"
        '      - "8000:8000"\n',
        encoding="utf-8",
    )
    report = drift.analyze(root)
    assert "DEMO_FLAG" not in _types(report, "DEAD")
    assert "DEMO_ENTRY" not in _types(report, "DEAD")


def test_compose_image_substitution_flags_undocumented(tmp_path: Path) -> None:
    """A compose image:/command:/entrypoint:/args: substitution var that is genuinely
    undeclared anywhere is UNDOCUMENTED, not silently accepted."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    (root / "docker").mkdir()
    (root / "docker" / "mcp.compose.yml").write_text(
        "services:\n"
        "  demo:\n"
        "    image: ${DEMO_AGENT_IMAGE:?set-DEMO_AGENT_IMAGE-to-image@sha256-digest}\n",
        encoding="utf-8",
    )
    report = drift.analyze(root)
    assert "DEMO_AGENT_IMAGE" in _types(report, "UNDOCUMENTED")


def test_compose_environment_block_still_not_treated_as_subst_read(
    tmp_path: Path,
) -> None:
    """The new image:/command:/entrypoint:/args: substitution scan must not change the
    pre-existing environment: block handling (a LHS key there is a DECLARATION source,
    like .env.example — see test_dead_var_flagged — not folded into the read surface)."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    (root / "docker").mkdir()
    (root / "docker" / "mcp.compose.yml").write_text(
        "services:\n"
        "  demo:\n"
        "    environment:\n"
        "      - DEMO_ENV_ONLY_VAR=${DEMO_ENV_ONLY_VAR}\n",
        encoding="utf-8",
    )
    report = drift.analyze(root)
    # declared via the environment: block (as before) -> DEAD, since nothing reads it
    assert "DEMO_ENV_ONLY_VAR" in _types(report, "DEAD")


def test_known_bad_dead_var_still_caught_after_widening(tmp_path: Path) -> None:
    """PROOF (a): a genuinely DEAD var — declared but read by nothing at all, not by
    scripts/, not by a compose image/command/entrypoint/args substitution, not by a dynamic
    TLS family call — is still reported DEAD even with every blind-spot fix from this change
    present and active in the same package."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\nTOTALLY_ORPHANED_VAR=\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "")\n'
            "from agent_utilities.core.transport_security import resolve_tls_profile\n"
            'resolve_tls_profile(service="mealie")\n'
        ),
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "gate.py").write_text(
        'import os\nos.environ.get("SCRIPT_ONLY_VAR")\n', encoding="utf-8"
    )
    (root / "docker").mkdir()
    (root / "docker" / "mcp.compose.yml").write_text(
        "services:\n"
        "  demo:\n"
        "    image: ${DEMO_MCP_IMAGE:?set-image}\n",
        encoding="utf-8",
    )
    report = drift.analyze(root)
    assert "TOTALLY_ORPHANED_VAR" in _types(report, "DEAD")


def test_known_bad_undocumented_var_still_caught_after_widening(tmp_path: Path) -> None:
    """PROOF (b): a genuinely UNDOCUMENTED var — read by ordinary package code, absent from
    .env.example — is still reported UNDOCUMENTED with every blind-spot fix active in the
    same package (scripts/, compose substitution, and a dynamic TLS family call all present
    alongside it, to prove none of the widening accidentally swallows a real finding)."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\nMEALIE_TLS_PROFILE=\nMEALIE_TLS_PROFILE_REF=\nDEMO_MCP_IMAGE=\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "")\n'
            'setting("BRAND_NEW_UNDOCUMENTED_VAR", "")\n'
            "from agent_utilities.core.transport_security import resolve_tls_profile\n"
            'resolve_tls_profile(service="mealie")\n'
        ),
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "gate.py").write_text(
        'import os\nos.environ.get("SCRIPT_ONLY_VAR")\n', encoding="utf-8"
    )
    (root / "docker").mkdir()
    (root / "docker" / "mcp.compose.yml").write_text(
        "services:\n"
        "  demo:\n"
        "    image: ${DEMO_MCP_IMAGE:?set-image}\n",
        encoding="utf-8",
    )
    report = drift.analyze(root)
    assert "BRAND_NEW_UNDOCUMENTED_VAR" in _types(report, "UNDOCUMENTED")
    # and the fixes correctly keep the others quiet/clean alongside it
    assert "MEALIE_TLS_PROFILE" not in _types(report, "DEAD")
    assert "DEMO_MCP_IMAGE" not in _types(report, "DEAD")
    assert "SCRIPT_ONLY_VAR" not in _types(report, "UNDOCUMENTED")


def test_derived_toggle_undocumented(tmp_path: Path) -> None:
    """A register_<tag>_tools registrar implies <TAG>TOOL; undocumented if absent from .env.example."""
    root = tmp_path / "demo-agent"
    (root / "demo_agent").mkdir(parents=True)
    (root / ".env.example").write_text("DEMO_BASE_URL=http://x\n", encoding="utf-8")
    (root / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}}),
        encoding="utf-8",
    )
    (root / "demo_agent" / "tools.py").write_text(
        "def register_demo_reports_tools(mcp):\n    pass\n", encoding="utf-8"
    )
    report = drift.analyze(root)
    assert "DEMO_REPORTSTOOL" in _types(report, "UNDOCUMENTED")
