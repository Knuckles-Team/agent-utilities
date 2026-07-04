---
name: kg-modality-blob
description: >-
  Store, fetch, and stream content-addressed binary objects (media, files,
  large payloads) in the epistemic-graph engine's blob store. Use when you need
  a binary object store keyed by content hash — uploading/downloading media or
  large attachments, deduplicating blobs, or attaching binary payloads to graph
  nodes ("store this file", "get blob by hash", "stream this media").
license: MIT
tags: [graph-os, engine, modality, blob, storage, media]
tier: modality
wraps: [engine_blob]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Blob (content-addressed object store)

Fronts the epistemic-graph engine's **`blob`** domain: a streamed,
content-addressed store for binary media blobs. Objects are keyed by their
content hash (dedupe-by-content), so the same bytes stored twice cost one copy,
and a blob can be referenced from graph nodes/edges as a first-class attachment.

This is the **modality** tier — a thin wrapper over the low-level engine surface
(`engine_blob`), which is action-routed 1:1 over the `epistemic_graph` client's
`BlobClient`. It is distinct from the high-level document/media ingestion path;
use this when you want direct byte-level put/get/stream against the store.

## What the wrapped verb does

`engine_blob` is one action-routed tool over the engine's blob sub-client. Call
it with an `action` (a `BlobClient` method) and `params_json` (its JSON kwargs).
Call with an **empty `action`** to list the exact actions the live engine
exposes (put/store, get/fetch, stat, delete, and streamed variants) — the action
set is discovered from the client, so it never drifts.

## How to reach it

**Via the multiplexer (recommended on demand):**
1. `load_tools(tools=["engine_blob"])` — mounts the tool live.
2. `engine_blob(action="", params_json="{}")` — list available actions.
3. `engine_blob(action="<method>", params_json="{...}", graph="")` — invoke it
   (empty `graph` ⇒ the deployment default graph).
4. `unload_tools(...)` when done, to reclaim context.

**Direct MCP on graph-os:** the `engine_blob` tool is registered on the graph-os
server. Per-method verbose tools (`engine_blob_<method>`) appear only under
`MCP_TOOL_MODE=verbose|both`.

**REST twin:** `POST /engine/blob` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// list what the live blob domain supports
engine_blob(action="", params_json="{}")

// fetch a blob by its content hash (exact arg names come from the action list)
engine_blob(action="get", params_json="{\"hash\": \"sha256:…\"}")
```

Binary results are returned base64-wrapped as `{"__bytes_b64__": "…"}`.
