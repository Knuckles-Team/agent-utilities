"""WIRING — ``create_agent`` governs every MCP toolset through a REAL kernel.

CONCEPT:AU-AHE.evaluation.wiring-test-taxonomy

Taxonomy (``AGENTS.md`` → *Wire-First*): a **wiring** test. It proves the edge
``agent.factory.create_agent`` → real ``resolve_permission_context`` → real
``PermissionsKernel.verify_identity`` → ``flag_mcp_tool_definitions`` carrying
that same kernel. It says nothing about whether the kernel's policy decisions are
correct (``tests/test_permissions_kernel.py`` is the unit test for that) and
nothing about a deployed agent enforcing them against a live MCP server.

Why this seam
-------------
This is the authorization boundary for every MCP tool an agent can call: a
toolset that reaches the agent without being flagged by a verified identity is
ungoverned. It is also the cleanest live illustration of *never mock the seam you
are validating* — ``tests/unit/core/test_agent_factory.py`` carries an **autouse**
fixture that monkeypatches ``resolve_permission_context`` to a ``MagicMock`` and
``flag_mcp_tool_definitions`` to a guard returning ``MagicMock(spec=[])``. Every
``create_agent`` test in the repo therefore runs with the security chokepoint
replaced by a mock of itself. That is defensible for tests about parsing and
prompts, but it means *nothing in the suite proved the real kernel is ever
constructed or ever reaches the toolset*. This file is the missing half, and it
deliberately does not reuse that fixture.

The environment flag that hid it
--------------------------------
Writing this test surfaced a **fourth** green-signal factory beyond the three the
program catalogued: ``pytest.ini`` sets ``AGENT_UTILITIES_TESTING=true``, which
makes ``DEFAULT_VALIDATION_MODE`` true, which makes ``create_agent`` skip its
entire ``mcp_toolsets`` block (``VALIDATION_MODE: Skipping external mcp_toolsets
connection``). Under the default suite the governed-boundary branch is therefore
**unreachable by construction** — no test could have proved or disproved it. A
wiring test has to restore production conditions for the branch it validates,
which is what :func:`_production_mode` does. Turning off a global test flag is not
mocking the seam; it is refusing to accept a branch the flag deleted.

What is injected and what is real
---------------------------------
The kernel and identity are built by the test and passed *into* ``create_agent``
— its documented DI seam ("dependency injection for bounded tests"). That is
**input**, not the seam: ``resolve_permission_context``, ``verify_permission_context``,
the real signature verification and ``flag_mcp_tool_definitions`` all execute for
real. The self-provisioning branch (durable signing key out of the engine's secret
store) is not reachable from a unit test today — see ``D-WS-2`` — so its edge is
left to live validation rather than faked here.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.agent import factory as agent_factory
from agent_utilities.security import permissions_kernel as pk
from tests.wiring import assert_not_faked, observe, observe_all

#: 32 bytes — the kernel refuses anything weaker.
TEST_SIGNING_KEY = "wiring-standard-test-signing-key"


class _Toolset:
    """A minimal stand-in for an MCP toolset.

    Note what is faked and what is not: the *input* to the seam is a stub (no live
    MCP server is needed to ask "was this toolset governed?"), while the seam
    itself runs for real. Faking the input is fine; faking the seam is the bug
    this file exists for.
    """

    def __init__(self) -> None:
        self.tool_definitions: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the production branch that ``AGENT_UTILITIES_TESTING`` deletes.

    Scoped to this module only — the flag earns its keep elsewhere by keeping the
    suite off the network. The ``_Toolset`` stub keeps *this* module offline
    without the flag's help, so nothing is lost by turning it off here.
    """
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)


@pytest.fixture()
def governed_pair() -> tuple[pk.PermissionsKernel, pk.AgentIdentity]:
    """A real kernel and a real identity it actually signed."""
    kernel = pk.PermissionsKernel(signing_key=TEST_SIGNING_KEY)
    identity = kernel.issue_identity("wiring-probe-agent", pk.AgentRole.SPECIALIST)
    return kernel, identity


def _create_agent(
    toolset: _Toolset | None,
    pair: tuple[pk.PermissionsKernel, pk.AgentIdentity] | None = None,
) -> Any:
    kernel, identity = pair if pair else (None, None)
    return agent_factory.create_agent(
        name="WiringProbeAgent",
        system_prompt="probe",
        mcp_toolsets=[toolset] if toolset else None,
        permissions_kernel=kernel,
        agent_identity=identity,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )


class TestPermissionContextWiring:
    def test_create_agent_reaches_the_real_resolver_and_verifies_the_identity(
        self, governed_pair
    ) -> None:
        """The live entrypoint reaches the real resolver — no mock in between."""
        kernel, identity = governed_pair

        with observe_all(
            (pk, "resolve_permission_context", "resolve"),
            (pk.PermissionsKernel, "verify_identity", "verify"),
        ) as seen:
            _create_agent(_Toolset(), governed_pair)

        call = seen["resolve"].assert_called(
            why=(
                "create_agent must resolve a verified permission context before any "
                "MCP toolset reaches the agent"
            )
        )
        context = call.result
        assert_not_faked(context, name="PermissionContext")
        assert isinstance(context, pk.PermissionContext)
        assert context.kernel is kernel
        assert context.identity is identity
        # The signature check genuinely ran — not merely "an object came back".
        seen["verify"].assert_called(
            why="an injected identity is cryptographically verified, not trusted"
        )

    def test_the_live_caller_demands_the_context_rather_than_allowing_none(
        self, governed_pair
    ) -> None:
        """Wire-First step 2: the entrypoint passes the value that turns it ON.

        ``resolve_permission_context(required=False)`` returns ``None`` and the
        governed path degrades silently. The whole guarantee rests on
        ``create_agent`` passing ``required=True`` whenever a toolset is present,
        so assert the *argument the live caller passed*, not just that it called.
        This is the KV-fork failure mode exactly — plumbing that was complete and
        a caller that never passed the parameter switching it on.
        """
        with observe(pk, "resolve_permission_context") as resolved:
            _create_agent(_Toolset(), governed_pair)

        call = resolved.assert_called()
        assert call.arg("required") is True
        assert call.arg("agent_subject") == "WiringProbeAgent"

    def test_the_toolset_is_flagged_with_that_same_real_kernel(
        self, governed_pair
    ) -> None:
        """The context is not merely built — it is what governs the toolset.

        "A kernel was constructed" and "the constructed kernel reached the guard"
        are different claims, and every capability this program found dead had the
        first without the second.
        """
        kernel, identity = governed_pair
        toolset = _Toolset()

        with observe(agent_factory, "flag_mcp_tool_definitions") as flagged_seam:
            _create_agent(toolset, governed_pair)

        flagged = flagged_seam.assert_called(
            why="every MCP toolset is flagged by the verified identity before use"
        )
        assert flagged.arg("permissions_kernel") is kernel
        assert flagged.arg("agent_identity") is identity
        assert toolset in flagged.arg("toolsets")

    def test_a_tampered_identity_fails_the_agent_closed(self) -> None:
        """The gate must *stop* the call, not merely sit on the path.

        Reaching a security seam proves nothing if it always says yes. A toolset
        must never reach the agent behind an identity its kernel did not sign.
        """
        kernel = pk.PermissionsKernel(signing_key=TEST_SIGNING_KEY)
        identity = kernel.issue_identity("wiring-probe-agent", pk.AgentRole.SPECIALIST)
        identity.role = pk.AgentRole.ADMIN  # forged privilege escalation

        with pytest.raises(pk.PermissionBootstrapError):
            _create_agent(_Toolset(), (kernel, identity))

    def test_no_toolsets_means_no_context_is_forced(self) -> None:
        """The negative half of the edge — the gate is scoped, not unconditional.

        With nothing to govern the bootstrap must not run (it would provision a
        durable signing authority for no reason). Pinning the negative keeps a
        future "just always resolve it" change honest.
        """
        with observe(pk, "resolve_permission_context") as resolved:
            _create_agent(None)

        resolved.assert_not_called(
            why="no MCP toolsets means no governed boundary to bootstrap"
        )

    def test_the_factory_suite_still_mocks_this_seam(self) -> None:
        """A standing pointer at why this file has to exist.

        If the autouse mock is ever removed from ``test_agent_factory.py`` this
        fails — at which point delete this test, not the file. Until then it
        records, executably, that the ``create_agent`` tests prove nothing about
        the security chokepoint.
        """
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "core" / "test_agent_factory.py"
        text = source.read_text(encoding="utf-8")
        assert "resolve_permission_context" in text
        assert "monkeypatch.setattr" in text
