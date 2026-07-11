"""Skills bind their owning server's tools (F7) + all-tools-errored degradation.

CONCEPT:AU-ORCH.execution.skill-bound-server-tools / AU-ORCH.execution.all-tool-calls-errored — a
package-bundled skill (under .../agents/<pkg>/.../skills/...) must drive its MCP
server's tools, not run prompt-only; and a run whose every tool call returned an error
is degraded, not a clean success.
"""

from __future__ import annotations

from agent_utilities.orchestration.agent_runner import (
    _bind_skill_to_owning_server,
    _delegation_degraded,
    _result_looks_like_error,
    _tool_call_errored,
)


class _FakeBackend:
    """Minimal backend: answers the Server lookup + PROVIDES tool fetch."""

    def __init__(self, server_name, url, tools):
        self._server_name = server_name
        self._url = url
        self._tools = tools

    def execute(self, cypher, params):
        if "MATCH (s:Server) WHERE s.name" in cypher:
            if params.get("name") == self._server_name:
                return [
                    {
                        "sid": f"srv:{self._server_name}",
                        "name": self._server_name,
                        "url": self._url,
                        "env": "{}",
                    }
                ]
            return []
        if "PROVIDES" in cypher:
            return [{"name": n, "description": d} for n, d in self._tools]
        return []


class _FakeEngine:
    def __init__(self, backend):
        self.backend = backend


def test_skill_binds_to_owning_server_with_mcp_suffix():
    # package dir 'tunnel-manager' -> server 'tunnel-manager-mcp' (the -mcp mismatch).
    eng = _FakeEngine(
        _FakeBackend(
            "tunnel-manager-mcp",
            "http://tunnel-manager-mcp.arpa/mcp",
            [
                ("tm_ssh_exec", "run a command over ssh"),
                ("tm_list_hosts", "list hosts"),
            ],
        )
    )
    meta = {
        "type": "skill",
        "system_prompt": "You are the 'tunnel-manager-remote-execution' skill. Do X.",
    }
    _bind_skill_to_owning_server(
        eng,
        meta,
        "/home/apps/workspace/agent-packages/agents/tunnel-manager/tunnel_manager/skills/tunnel-manager-remote-execution/SKILL.md",
        "tunnel-manager-remote-execution",
    )
    assert meta["type"] == "server"  # now routes single-server (F1 selection applies)
    assert meta["url"] == "http://tunnel-manager-mcp.arpa/mcp"
    assert meta["skill_of_server"] == "tunnel-manager-mcp"
    assert len(meta["tools"]) == 2
    # the skill's instructions are preserved as the system prompt
    assert "tunnel-manager-remote-execution" in meta["system_prompt"]


def test_skill_binding_noop_when_no_owning_server():
    eng = _FakeEngine(_FakeBackend("other-mcp", "http://x/mcp", []))
    meta = {"type": "skill", "system_prompt": "SOP"}
    _bind_skill_to_owning_server(
        eng, meta, "/some/universal-skills/foo/SKILL.md", "foo"
    )
    assert meta["type"] == "skill"  # unchanged — not a package-bundled skill


def test_skill_binding_noop_for_unknown_package():
    eng = _FakeEngine(_FakeBackend("known-mcp", "http://x/mcp", []))
    meta = {"type": "skill", "system_prompt": "SOP"}
    _bind_skill_to_owning_server(
        eng, meta, "/x/agents/nonexistent-pkg/skills/s/SKILL.md", "s"
    )
    assert meta["type"] == "skill"  # server not found -> stays prompt-only


def test_result_looks_like_error():
    assert _result_looks_like_error(
        "Error executing list_namespaces: 'MultiContextManager' object has no attribute 'list_namespaces'"
    )
    assert _result_looks_like_error("Traceback (most recent call last): ...")
    assert not _result_looks_like_error("web_server, cache_service")
    assert not _result_looks_like_error("")


def test_tool_call_errored():
    assert _tool_call_errored({"result": "Error executing X: has no attribute 'X'"})
    assert _tool_call_errored({"error": "boom", "result": "ok"})
    assert not _tool_call_errored(
        {"result": "3 namespaces: default, kube-system, apps"}
    )


def test_all_tools_errored_is_degraded():
    # mirrors the live k8s run: 13 calls, all 'has no attribute' -> degraded, not completed.
    result = {
        "results": {"output": "I could not list namespaces; the tools are failing."},
        "tool_calls": [
            {
                "tool_name": "cm_k8s_config__list_namespaces",
                "result": "Error executing list_namespaces: 'MultiContextManager' object has no attribute 'list_namespaces'",
            },
            {
                "tool_name": "cm_k8s_cluster",
                "result": "Error executing get_cluster_info: 'MultiContextManager' object has no attribute 'get_cluster_info'",
            },
        ],
    }
    assert _delegation_degraded(result) is True


def test_partial_tool_success_is_not_degraded():
    result = {
        "results": {"output": "namespaces: default, apps"},
        "tool_calls": [
            {
                "tool_name": "cm_k8s_config__list_namespaces",
                "result": "default, apps, kube-system",
            },
            {
                "tool_name": "cm_k8s_cluster",
                "result": "Error executing get_cluster_info: has no attribute",
            },
        ],
    }
    assert _delegation_degraded(result) is False
