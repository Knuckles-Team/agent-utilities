"""Built-in document-source connectors (CONCEPT:AU-ECO.connector.document-source-framework).

Each module in this package registers its connector via the ``@register_source``
decorator (CONCEPT:AU-ECO.connector.factory-ingestion-adaptor). They are imported by
:func:`agent_utilities.protocols.source_connectors.registry.discover`, which walks
this package so the decorators run on the live ingestion path. Importing this
package directly does **not** import the submodules (discovery is explicit) — this
keeps connector heavy-deps lazy.

Reference connectors:
  * :mod:`web` — recursive same-domain crawler (``web``).
  * :mod:`reader` — single-URL readability reader → clean markdown (``reader``,
    CONCEPT:AU-KG.enrichment.multimodal-readers).
  * :mod:`filesystem` — directory walk (``filesystem``).
  * :mod:`rest` — paginated JSON endpoint (``rest``).
  * :mod:`mcp_package` — adapter over the agent-package fleet (``mcp:<pkg>``).
  * :mod:`mcp_tool` — any MCP server's record-listing tool as a paginated,
    checkpointed ingestion source (``mcp_tool``, CONCEPT:AU-KG.ingest.mcp-tool-connector).
"""
