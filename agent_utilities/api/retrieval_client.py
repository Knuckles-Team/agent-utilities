"""The single typed composition of the generated EG client for AU-SEMANTIC-R025.

``AU-SEMANTIC-R025`` moves AU's retrieval and neural-search modules (the
hybrid, semantic, reranking, hierarchical-document, memory, code-context,
lineage and autocut retrievers under ``kg/retrieval/**``, ``kg/neural/**``,
and the OpenSearch change-data-capture indexer in ``kg/search/**``) to EG;
AU retains only context compilation and the capability index. This module is
the ``.1`` slice: the typed composition AU's retained context-compilation
call sites will route retrieval through, plus the refusal behavior when the
installed EG client does not serve the needed retrieval surface. The
call-site migrations themselves (``.2``+) land separately.

Nothing here reconstructs hybrid/semantic/reranking retrieval locally; the
protocol below only names the generated-client surface a caller is allowed
to depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class RetrievalClientUnavailable(RuntimeError):
    """The connected/installed EG client does not serve retrieval ops.

    Raised instead of falling back to any local reconstruction of the moved
    hybrid/semantic/reranking/memory/code-context/lineage/autocut retrievers.
    """


@runtime_checkable
class GeneratedRetrievalSurface(Protocol):
    """The generated-EG-client methods AU's context compilation needs.

    Each attribute name is the EG-side method the caller invokes; this is a
    structural check against the installed client, never a grep over AU's
    own internal names (AU-SEMANTIC-R027).
    """

    def hybrid_search(self, *args: Any, **kwargs: Any) -> Any: ...

    def semantic_search(self, *args: Any, **kwargs: Any) -> Any: ...

    def rerank(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class RetrievalClient:
    """AU's one typed composition of the generated EG client for this surface.

    Construct only via :meth:`for_client`, which validates the installed
    client actually implements :class:`GeneratedRetrievalSurface` before
    handing back a usable composition. AU keeps no local retrieval authority
    beyond context compilation; everything here delegates.
    """

    client: GeneratedRetrievalSurface

    @classmethod
    def for_client(cls, eg_client: Any) -> RetrievalClient:
        if eg_client is None:
            raise RetrievalClientUnavailable(
                "an epistemic-graph client is required for retrieval ops"
            )
        if not isinstance(eg_client, GeneratedRetrievalSurface):
            missing = [
                name
                for name in ("hybrid_search", "semantic_search", "rerank")
                if not callable(getattr(eg_client, name, None))
            ]
            raise RetrievalClientUnavailable(
                "the installed EG client does not serve the retrieval surface "
                f"(missing: {', '.join(missing) or 'unknown'})"
            )
        return cls(client=eg_client)

    def hybrid_search(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.hybrid_search(*args, **kwargs)

    def semantic_search(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.semantic_search(*args, **kwargs)

    def rerank(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.rerank(*args, **kwargs)
