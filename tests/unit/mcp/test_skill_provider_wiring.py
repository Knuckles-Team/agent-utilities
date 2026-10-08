"""Server-side Skills-over-MCP wiring (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider).

Covers ``_register_skill_providers`` (called from ``create_mcp_server``): it
must call ``mcp.add_provider(...)`` once per directory
``resolve_skill_provider_dirs`` resolves, and must never crash server
construction when a provider cannot be registered.

These are ISOLATION tests — they stand a ``MagicMock`` in for the server and a
fake module in for ``fastmcp.server.providers.skills``, so they say nothing
about whether the registration works against the real fastmcp-4
``SkillProvider``. That live proof is
``tests/integration/mcp/test_skill_provider_live_path.py``; keep both.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent_utilities.mcp import server_factory
from agent_utilities.mcp.server_factory import (
    _register_skill_providers,
    create_mcp_server,
)


def test_register_skill_providers_never_breaks_server_construction(caplog) -> None:
    """A server object that cannot take providers degrades to a log, not a raise.

    The old ``hasattr(mcp, "add_provider")`` gate is gone with the fastmcp-3
    default it guarded (the ``[mcp]`` extra now floors on ``fastmcp>=4.0.0b1``,
    where ``add_provider`` always exists), so what still needs guarding is only
    the never-raise contract.
    """
    fake_mcp = MagicMock(spec=[])  # no attributes at all, incl. no add_provider

    with caplog.at_level("DEBUG", logger="agent_utilities.mcp.server_factory"):
        _register_skill_providers(fake_mcp)  # must not raise

    assert "skill" in caplog.text.casefold()


def test_create_mcp_server_construction_succeeds() -> None:
    """``create_mcp_server`` builds a working server with providers wired in."""
    args, mcp, _middlewares = create_mcp_server("Skills Wiring Test", command_args=[])
    assert mcp is not None
    assert args is not None


def test_register_skill_providers_calls_add_provider_per_resolved_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """When the server exposes ``add_provider`` (fastmcp 4), one SkillProvider
    is registered per :func:`resolve_skill_provider_dirs` entry — the SAME
    discovery the in-loop SkillsToolset/installer use, not a re-implementation.
    """
    import sys
    import types

    skill_root = tmp_path / "some-provider" / "a-skill"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text("---\ndescription: x\n---\nbody", "utf-8")

    calls: list[Path] = []

    class _FakeSkillProvider:
        def __init__(self, path):
            calls.append(Path(path))

    fake_pkg = types.ModuleType("fastmcp.server.providers.skills")
    fake_pkg.SkillProvider = _FakeSkillProvider
    monkeypatch.setitem(sys.modules, "fastmcp.server.providers.skills", fake_pkg)
    # Ensure the parent packages resolve too (importlib needs them registered).
    monkeypatch.setitem(sys.modules, "fastmcp", types.ModuleType("fastmcp"))
    monkeypatch.setitem(
        sys.modules, "fastmcp.server", types.ModuleType("fastmcp.server")
    )
    monkeypatch.setitem(
        sys.modules,
        "fastmcp.server.providers",
        types.ModuleType("fastmcp.server.providers"),
    )

    monkeypatch.setattr(
        "agent_utilities.core.providers.resolve_skill_provider_dirs",
        lambda: [("some-provider", skill_root.parent)],
    )

    fake_mcp = MagicMock()  # has add_provider by default (MagicMock auto-attrs)
    _register_skill_providers(fake_mcp)

    assert calls == [skill_root.parent]
    fake_mcp.add_provider.assert_called_once()
    registered_provider = fake_mcp.add_provider.call_args.args[0]
    assert isinstance(registered_provider, _FakeSkillProvider)


def test_register_skill_providers_one_bad_dir_does_not_sink_the_rest(
    monkeypatch,
) -> None:
    """A provider dir that fails to register must not prevent the others."""
    import sys
    import types

    class _FakeSkillProvider:
        def __init__(self, path):
            if str(path) == "bad":
                raise RuntimeError("boom")

    fake_pkg = types.ModuleType("fastmcp.server.providers.skills")
    fake_pkg.SkillProvider = _FakeSkillProvider
    monkeypatch.setitem(sys.modules, "fastmcp.server.providers.skills", fake_pkg)
    monkeypatch.setitem(sys.modules, "fastmcp", types.ModuleType("fastmcp"))
    monkeypatch.setitem(
        sys.modules, "fastmcp.server", types.ModuleType("fastmcp.server")
    )
    monkeypatch.setitem(
        sys.modules,
        "fastmcp.server.providers",
        types.ModuleType("fastmcp.server.providers"),
    )
    monkeypatch.setattr(
        "agent_utilities.core.providers.resolve_skill_provider_dirs",
        lambda: [("bad-provider", "bad"), ("good-provider", "good")],
    )

    fake_mcp = MagicMock()
    _register_skill_providers(fake_mcp)

    # The bad dir's SkillProvider() constructor raised and was skipped; the
    # good dir still registered — one failure must not sink the whole sweep.
    assert fake_mcp.add_provider.call_count == 1


def test_fleet_server_serves_only_its_own_package_skills() -> None:
    """A fleet server advertises its own package's skills, never a sibling's."""
    from agent_utilities.mcp.server_factory import _owned_skill_dirs

    dirs = [
        ("ansible-tower-mcp", Path("a/inventory")),
        ("gitlab-api", Path("g/gitlab-vulnerabilities")),
        ("langfuse-agent", Path("l/langfuse-prompt-management")),
        ("agent-utilities", Path("au/core")),
    ]
    assert _owned_skill_dirs(dirs, "ansible_tower_mcp") == [
        ("ansible-tower-mcp", Path("a/inventory"))
    ]
    assert _owned_skill_dirs(dirs, "gitlab_api") == [
        ("gitlab-api", Path("g/gitlab-vulnerabilities"))
    ]


def test_hub_and_unknown_owner_keep_the_full_catalogue() -> None:
    from agent_utilities.mcp.server_factory import _owned_skill_dirs

    dirs = [("gitlab-api", Path("g")), ("agent-utilities", Path("au"))]
    assert _owned_skill_dirs(dirs, "agent_utilities") == dirs
    assert _owned_skill_dirs(dirs, None) == dirs


def _build_from_fleet_module() -> None:
    """Rebound onto a fleet module's globals to simulate a fleet caller."""
    server_factory.create_mcp_server("Scoped", command_args=[])


def test_create_mcp_server_scopes_skills_to_the_calling_package(
    monkeypatch,
) -> None:
    """The factory infers the owner from the module that called it."""
    import agent_utilities.mcp.server_factory as factory

    seen: list[str | None] = []
    monkeypatch.setattr(
        factory, "_register_skill_providers", lambda mcp, owner=None: seen.append(owner)
    )
    import types

    fleet_globals = {
        "__name__": "ansible_tower_mcp.mcp.mcp_server",
        "server_factory": factory,
    }
    types.FunctionType(_build_from_fleet_module.__code__, fleet_globals)()
    assert seen == ["ansible_tower_mcp"]
