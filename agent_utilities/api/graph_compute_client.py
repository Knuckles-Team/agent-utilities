"""The single typed composition of the generated EG client for AU-SEMANTIC-R010.

``AU-SEMANTIC-R010`` replaces AU's graph-compute and session facades
(``kg/core/graph_compute.py``, ``session.py``, ``epistemic_row.py``, ``ogm.py``,
the ``company_brain*`` modules, and ``core/registry/kg_adapter.py``) with calls
into the generated EG client. This module is the ``.1`` slice: the typed model
those replacement call sites will compose against, plus the refusal behavior
when the installed EG client does not serve the needed surface. The call-site
migrations themselves (``.2``+) land separately.

Nothing here reconstructs graph-compute/session behavior locally; the protocol
below only names the generated-client surface a caller is allowed to depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class GraphComputeClientUnavailable(RuntimeError):
    """The connected/installed EG client does not serve graph-compute/session ops.

    Raised instead of falling back to any local reconstruction of the removed
    ``graph_compute``/``session``/``epistemic_row``/``ogm``/``kg_adapter`` logic.
    """


@runtime_checkable
class GeneratedGraphComputeSurface(Protocol):
    """The generated-EG-client methods the graph-compute/session facade needs.

    Each attribute name is the EG-side method the facade calls; this is a
    structural check against the installed client, never a grep over AU's own
    internal names (AU-SEMANTIC-R027).
    """

    def compute_graph(self, *args: Any, **kwargs: Any) -> Any: ...

    def get_session_row(self, *args: Any, **kwargs: Any) -> Any: ...

    def resolve_object_mapping(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class GraphComputeClient:
    """AU's one typed composition of the generated EG client for this surface.

    Construct only via :meth:`for_client`, which validates the installed
    client actually implements :class:`GeneratedGraphComputeSurface` before
    handing back a usable composition.
    """

    client: GeneratedGraphComputeSurface

    @classmethod
    def for_client(cls, eg_client: Any) -> GraphComputeClient:
        if eg_client is None:
            raise GraphComputeClientUnavailable(
                "an epistemic-graph client is required for graph-compute/session ops"
            )
        if not isinstance(eg_client, GeneratedGraphComputeSurface):
            missing = [
                name
                for name in (
                    "compute_graph",
                    "get_session_row",
                    "resolve_object_mapping",
                )
                if not callable(getattr(eg_client, name, None))
            ]
            raise GraphComputeClientUnavailable(
                "the installed EG client does not serve the graph-compute/session "
                f"surface (missing: {', '.join(missing) or 'unknown'})"
            )
        return cls(client=eg_client)

    def compute_graph(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.compute_graph(*args, **kwargs)

    def get_session_row(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.get_session_row(*args, **kwargs)

    def resolve_object_mapping(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.resolve_object_mapping(*args, **kwargs)
