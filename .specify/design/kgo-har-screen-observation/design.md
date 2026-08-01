# Design Document: A captured desktop frame materializes as durable graph entities in ONE round-trip, not a raw screenshot blob

CONCEPT:AU-KG.ontology.owl-screen-bridge

> `agent_utilities/knowledge_graph/core/graph_compute.py`
> (`GraphComputeEngine.observe_screen`), promotable-type additions in
> `agent_utilities/knowledge_graph/core/owl_bridge.py:145`.

## Decision — computer-use (GUI) events are typed, promotable graph entities (`ComputerUseSession` / `ScreenObservation` / `UIElement`), not opaque screenshot storage

`observe_screen` (`graph_compute.py:3139-3166`) turns a captured desktop frame
into structured, durable graph state in a single RPC round-trip: the PNG
itself is not stored — "only its dimensions + content hash persist" — while
the AT-SPI accessibility `elements` (`role, name, x, y, w, h` per element)
become one `UIElement` node per accessible element, linked under a
`ComputerUseSession` + `ScreenObservation` frame node. `owl_bridge.py:145-150`
registers `computerusesession`, `screenobservation`, `uielement`, `guiaction`
as OWL-promotable node types, so a computer-use trajectory becomes reasoned-
over graph structure rather than an opaque blob a human would have to visually
re-inspect.

**The rejected alternative is treating a screenshot as a storage artifact** —
save the PNG (or a video of frames) and let a human or a downstream vision
model re-derive structure from pixels every time it's needed. Structuring the
frame at capture time means "what UI elements were on screen, in what layout,
in what session, at what sequence position" is a graph query, not a
re-render-and-re-parse operation, and it composes directly with the rest of
the ontology (a `GuiAction` can link to the `UIElement` it targeted). The
content-hash-only persistence of the PNG (`prev_hash` param, `graph_compute.py:
3146`) is itself a deliberate half-measure: enough to detect whether the frame
changed from the previous one (dedup a static screen), without paying to store
every frame's actual pixels durably.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/graph_compute.py` (`observe_screen`),
  `knowledge_graph/core/owl_bridge.py` promotable-type table, any computer-use
  agent trajectory recording.
- **Backward Compatible**: Yes — additive entity types; no existing node type
  is affected.
- **Known weak point**: because the PNG's pixels are never durably stored
  (only dimensions + hash), there is no way to visually re-inspect exactly
  what a past `ScreenObservation` looked like after the fact — only the
  structured `UIElement` extraction survives, so a bug in the AT-SPI
  extraction itself is unrecoverable from the persisted record.
