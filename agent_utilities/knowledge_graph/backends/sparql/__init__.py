#!/usr/bin/python
"""SPARQL-capable Graph Backends.

Provides remote (Fuseki) backends that implement
the full ``GraphBackend`` ABC with native SPARQL support.

Use ``create_backend("jena_fuseki")`` to
instantiate via the standard factory.
"""

from .jena_fuseki_backend import JenaFusekiBackend

__all__ = [
    "JenaFusekiBackend",
]
