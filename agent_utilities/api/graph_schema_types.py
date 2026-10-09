"""The single typed composition of EG-generated graph-schema DTOs for AU-SEMANTIC-R022.

``AU-SEMANTIC-R022`` replaces AU's hand-maintained graph-schema DTOs
(``models/knowledge_graph.py``/``schema_definition.py``/``evidence_bundle.py``/
``knowledge_pack.py``/``codemap.py``/``graph.py``/``knowledge_base.py``) with
EG-generated types. This module is the ``.1`` slice: the typed composition
those replacement call sites will build DTO instances through, plus the
refusal behavior when the installed EG client does not serve the needed
generated-type constructors. The call-site migrations themselves (``.2``+)
land separately.

Nothing here reconstructs the hand-written DTOs locally; the protocol below
only names the generated-client surface a caller is allowed to depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class GraphSchemaTypesUnavailable(RuntimeError):
    """The connected/installed EG client does not serve graph-schema DTO types.

    Raised instead of falling back to any local reconstruction of the removed
    ``knowledge_graph``/``schema_definition``/``evidence_bundle``/``knowledge_pack``
    hand-written models.
    """


@runtime_checkable
class GeneratedGraphSchemaSurface(Protocol):
    """The generated-EG-client constructors the graph-schema DTO callers need.

    Each attribute name is the EG-side constructor the caller invokes; this is
    a structural check against the installed client, never a grep over AU's
    own internal names (AU-SEMANTIC-R027).
    """

    def build_knowledge_graph(self, *args: Any, **kwargs: Any) -> Any: ...

    def build_schema_definition(self, *args: Any, **kwargs: Any) -> Any: ...

    def build_evidence_bundle(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class GraphSchemaTypes:
    """AU's one typed composition of the generated EG client for this surface.

    Construct only via :meth:`for_client`, which validates the installed
    client actually implements :class:`GeneratedGraphSchemaSurface` before
    handing back a usable composition.
    """

    client: GeneratedGraphSchemaSurface

    @classmethod
    def for_client(cls, eg_client: Any) -> GraphSchemaTypes:
        if eg_client is None:
            raise GraphSchemaTypesUnavailable(
                "an epistemic-graph client is required for graph-schema DTO types"
            )
        if not isinstance(eg_client, GeneratedGraphSchemaSurface):
            missing = [
                name
                for name in (
                    "build_knowledge_graph",
                    "build_schema_definition",
                    "build_evidence_bundle",
                )
                if not callable(getattr(eg_client, name, None))
            ]
            raise GraphSchemaTypesUnavailable(
                "the installed EG client does not serve the graph-schema DTO "
                f"surface (missing: {', '.join(missing) or 'unknown'})"
            )
        return cls(client=eg_client)

    def build_knowledge_graph(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.build_knowledge_graph(*args, **kwargs)

    def build_schema_definition(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.build_schema_definition(*args, **kwargs)

    def build_evidence_bundle(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.build_evidence_bundle(*args, **kwargs)
