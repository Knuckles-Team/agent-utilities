"""Server-side Skills-over-MCP wiring (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider).

Covers ``_register_skill_providers`` (called from ``create_mcp_server``): it
must degrade cleanly (never raise, never crash server construction) when the
live ``fastmcp`` has no ``SkillProvider``/``add_provider`` (the default
``[mcp]`` extra, fastmcp 3.x), and it must call ``mcp.add_provider(...)`` once
per directory ``resolve_skill_provider_dirs`` resolves when the capability is
present (the opt-in ``[mcp-v4]`` extra, fastmcp 4.x).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent_utilities.mcp.server_factory import (
    _register_skill_providers,
    create_mcp_server,
)


def test_register_skill_providers_noop_when_add_provider_absent(caplog) -> None:
    """A fastmcp-3 FastMCP instance has no ``add_provider`` — degrade silently."""
    fake_mcp = MagicMock(spec=[])  # no attributes at all, incl. no add_provider
    assert not hasattr(fake_mcp, "add_provider")

    with caplog.at_level("DEBUG", logger="agent_utilities.mcp.server_factory"):
        _register_skill_providers(fake_mcp)

    assert "add_provider" in caplog.text or "fastmcp" in caplog.text.casefold()


def test_create_mcp_server_construction_unaffected_without_mcp_v4() -> None:
    """The live [mcp] (fastmcp 3.x) extra must still build a working server."""
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
