#!/usr/bin/python
"""SPARQL-capable Graph Backends.

Provides remote (Fuseki, Stardog) backends that implement the full ``GraphBackend``
ABC with native SPARQL support.

Use ``create_backend("jena_fuseki")`` / ``create_backend("stardog")`` to instantiate
via the standard factory.
"""

from .jena_fuseki_backend import JenaFusekiBackend
from .stardog_backend import StardogSparqlBackend

__all__ = [
    "JenaFusekiBackend",
    "StardogSparqlBackend",
]
