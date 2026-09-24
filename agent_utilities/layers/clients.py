"""Typed cross-layer clients L0-L5 over EG's generated contract (RF-ADR-010 §3).

Each layer is reached through exactly one typed client, bound to a verified
session's graph and EG client; no layer is reached by raw query text:

* L0 knowledge -- provenance of the facts a run used (``ExplainProvenanceByIds``);
* L1 components -- ``AgentComponent`` search / content / current;
* L2 agents -- agent components (A2A cards, library agents) by typed search;
* L3 agent graphs -- ``AgentAssemble`` (``graph.assemble()``: the Decide ladder
  replaces LLM composition, DECISIONS 2026-09-17 addendum A1) and
  ``DecisionCommit``;
* L4 harnesses -- the local :class:`~agent_utilities.layers.execution.HarnessRegistry`;
* L5 runs -- the committed outcome read (``GetWorkItemOutcome``); the one
  writer is :class:`~agent_utilities.layers.l5_writer.RunOutcomeWriter`.

A method whose EG surface the installed client does not generate fails closed
with :class:`LayerUnavailable`; nothing is reconstructed from untyped calls.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.session import GraphSession


class LayerUnavailable(RuntimeError):
    """The connected/installed EG does not serve this layer operation."""


def generated(module: str, name: str) -> Any:
    """One generated EG sender/model, or :class:`LayerUnavailable`."""
    try:
        loaded = importlib.import_module(f"epistemic_graph.generated.{module}")
    except ImportError as exc:
        raise LayerUnavailable(f"EG generated module {module!r} is absent") from exc
    found = getattr(loaded, name, None)
    if found is None:
        raise LayerUnavailable(f"EG does not generate {module}.{name}")
    return found


@dataclass(frozen=True, slots=True)
class _Bound:
    client: Any
    session: GraphSession

    @property
    def graph(self) -> str:
        graph = str(self.session.graph or "").strip()
        if not graph:
            raise LayerUnavailable("the verified session is not bound to a graph")
        return graph


class KnowledgeClient(_Bound):
    """L0: provenance of graph facts."""

    async def explain_provenance(self, ids: list[str]) -> Any:
        send = generated("query", "send_explain_provenance_by_ids")
        return await send(self.client, {"ids": list(ids)}, self.graph)


class ComponentClient(_Bound):
    """L1: tenant-scoped agent components."""

    async def search(self, request: Any) -> Any:
        send = generated("storage", "send_agent_component_search")
        return await send(self.client, request, self.graph)

    async def content(self, request: Any) -> Any:
        send = generated("storage", "send_agent_component_content")
        return await send(self.client, request, self.graph)

    async def current(self, request: Any) -> Any:
        send = generated("storage", "send_agent_component_current")
        return await send(self.client, request, self.graph)


class AgentClient(_Bound):
    """L2: agents are components of an agent kind, found by typed search."""

    async def agents(self, *, limit: int = 64, cursor: str | None = None) -> Any:
        kind = generated("agent_component", "AgentComponentKind")
        request_type = generated("agent_component", "AgentComponentSearchRequest")
        request = request_type(
            tenant_id=self.session.tenant,
            kinds=[kind.A2A_AGENT_CARD],
            limit=limit,
            cursor=cursor,
        )
        return await ComponentClient(self.client, self.session).search(request)


class AgentGraphClient(_Bound):
    """L3: decision-ladder assembly, the decision commit and the graph publish."""

    async def assemble(self, request: Any) -> Any:
        send = generated("storage", "send_agent_assemble")
        return await send(self.client, {"request": _json(request)}, self.graph)

    async def commit_decision(
        self, request: Any, *, idempotency_key: str | None = None
    ) -> Any:
        send = generated("storage", "send_decision_commit")
        return await send(
            self.client,
            {"request": _json(request)},
            self.graph,
            idempotency_key=idempotency_key,
        )

    async def publish_graph(
        self,
        draft: Any,
        context: Any,
        *,
        evidence: Any = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Publish one agent graph (``AgentGraph.publish``).

        ``evidence`` -- a ``ComponentDependency`` pinning the committed
        ``DecisionRecord`` -- becomes the draft's ``synthesis_evidence``, which is
        inside the graph's definition digest, so a delegation of the graph
        resolves to the decision that chose it (EH-044, DECIDE §4.6).
        ``context`` is the ``AgentLibraryMutationContext`` the policy owner minted.
        """
        send = generated("storage", "send_agent_graph")
        graph = dict(_json(draft))
        if evidence is not None:
            graph["synthesis_evidence"] = dict(_json(evidence))
        request = {"context": _json(context), "graph": graph}
        return await send(
            self.client,
            {"op": {"op": "publish", "request": request}},
            self.graph,
            idempotency_key=idempotency_key,
        )


class RunClient(_Bound):
    """L5: the committed outcome of a WorkItem run, as EG verified it."""

    async def outcome(self, work_item_id: str) -> dict[str, Any] | None:
        work_items = getattr(self.client, "work_items", None)
        reader = getattr(work_items, "get_outcome", None)
        if not callable(reader):
            raise LayerUnavailable("EG does not serve GetWorkItemOutcome")
        return await reader(tenant=self.session.tenant, work_item_id=work_item_id)


def _json(model: Any) -> Any:
    dump = getattr(model, "model_dump", None)
    return dump(mode="json", exclude_none=True) if callable(dump) else model


@dataclass(frozen=True, slots=True)
class LayerClients:
    """The L0/L1/L2/L3/L5 clients for one verified session."""

    knowledge: KnowledgeClient
    components: ComponentClient
    agents: AgentClient
    graphs: AgentGraphClient
    runs: RunClient

    @classmethod
    def for_session(cls, eg_client: Any, session: GraphSession) -> LayerClients:
        if eg_client is None:
            raise LayerUnavailable("an epistemic-graph client is required")
        return cls(
            knowledge=KnowledgeClient(eg_client, session),
            components=ComponentClient(eg_client, session),
            agents=AgentClient(eg_client, session),
            graphs=AgentGraphClient(eg_client, session),
            runs=RunClient(eg_client, session),
        )


__all__ = [
    "AgentClient",
    "AgentGraphClient",
    "ComponentClient",
    "KnowledgeClient",
    "LayerClients",
    "LayerUnavailable",
    "RunClient",
    "generated",
]
