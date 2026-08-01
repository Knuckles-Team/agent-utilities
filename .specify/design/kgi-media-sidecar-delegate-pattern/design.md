# Design Document: Heavy media decode is delegated to governed fleet sidecar agents, never done in-engine, through ONE reusable delegate loop

CONCEPT:AU-KG.ingest.media-sidecar-delegation

> `agent_utilities/media/sidecar_contract.py` (the typed contract),
> `agent_utilities/media/sidecar_delegate.py` (the reusable delegate loop),
> `agent_utilities/media/image_sidecar.py`, `agent_utilities/media/pdf_sidecar.py`
> (the two shipped modalities), `agent_utilities/knowledge_graph/etl/lineage.py:37,305,369`
> (PROV-O activity/claim recording), `agent_utilities/mcp/tools/media_sidecar_tools.py`.
> ADR: `reports/wave4/ADR-media-sidecar.md` (HG-7, accepted 2026-07-23).

## Decision — the engine stays pure-Rust; OCR/JPEG/PDF/audio/video decode runs in governed fleet delegate agents reached through ONE reusable delegate loop, generalized from an existing pattern rather than built fresh per modality

`sidecar_contract.py:1-6` states the architectural line directly, citing its
own accepted ADR: **"the engine stays pure-Rust; all heavy media decode
(OCR, JPEG, MP3/AAC, H.264/VP9, Whisper) runs in governed fleet delegate
agents, never in-engine."** This module is "the typed contract that makes
that delegation standardized rather than ad hoc."

**The rejected alternative is in-engine media decode** — linking OCR/codec/
ASR libraries into the Rust epistemic-graph engine itself. It is named as
the alternative the ADR explicitly rejected: keeping the engine pure-Rust
means it never carries the dependency surface, memory-safety risk, or build
complexity of C/C++ media-decode libraries (libjpeg, ffmpeg, Whisper's
model runtime); the cost is a network hop to a fleet agent for every
extraction instead of an in-process call.

**Three sub-decisions make the delegation standardized rather than
ad hoc** (`sidecar_contract.py:8-33`):

1. **Input identity is a CAS blob ref, never a raw engine path.** A sidecar
   receives `SidecarBlobRef` (digest + media type) plus the bytes themselves,
   base64-encoded over the MCP JSON wire — "the only thing a JSON-RPC
   transport can actually carry across a process/host boundary." The
   rejected alternative — handing a sidecar a path into the engine's own
   blob store — fails because that path is private storage layout, "often
   unreachable cross-host."
2. **A fail-closed capability manifest** (`SIDECAR_CAPABILITIES`) declares
   which sidecar/tool may produce which evidence-locus kind — "same style as
   the connector-manifest gate... in spirit — fail-closed, additive,
   never-guessed" but deliberately NOT that gate's Ed25519-signed OWL-compile
   pipeline, because a sidecar here is an already-trusted in-fleet MCP agent,
   not an external, independently-onboarded data source. `assert_capable`
   is the fail-closed guard every per-locus write-back calls before writing.
3. **Every delegation call is governed** — recorded as one PROV-O
   `:PROVENANCE_ACTIVITY` node and, when it produced locus writes, one
   directly-verified `:Claim` that every write-back's `claim_id` links to via
   the existing `SUPPORTS` edge convention (`etl/lineage.py`'s
   `record_media_sidecar_activity`/`record_media_sidecar_claim`) — "what
   makes a sidecar's result challengeable through the standard why/why-not
   machinery... with no new resolver."

**The delegate loop itself is a generalization, not a new pattern**
(`sidecar_delegate.py:1-18`): `delegate_extract` generalizes `graph_mine_deep`'s
existing delegate shape (`AU-KG.mining.dsm-forecast-delegation`) for media —
same `call_tool_once`/`McpToolSourceConnector` transport, same PROV-O
activity recording, same defensive str-vs-dict result decoding the transport's
BUG-7 fix established. **The rejected alternative was building a bespoke
delegation mechanism per media modality** (one for PDF, a different one for
images, a third for audio) — the ADR's own framing (W4.6, "standardize the
sidecar delegate pattern") names this as the thing being fixed: `delegate_extract`
is the ONE reusable component both shipped modalities (`pdf_sidecar.py`,
`image_sidecar.py`) call, and it never raises — an unknown modality, an
unreachable sidecar, or a malformed response all degrade to
`SidecarDelegationResult(available=False, error=...)`, matching every other
delegate tool in the codebase.

## Risk Assessment

- **Blast Radius**: `media/sidecar_contract.py`, `media/sidecar_delegate.py`,
  `media/image_sidecar.py`, `media/pdf_sidecar.py`,
  `knowledge_graph/etl/lineage.py`, `mcp/tools/media_sidecar_tools.py`.
- **Backward Compatible**: Yes — new modalities register into the same
  capability manifest and delegate loop.
- **Breaking Changes**: None.
- **Known weak point**: the capability manifest's trust model is
  deliberately weaker than the connector-manifest gate's signed pipeline
  ("an already-trusted in-fleet MCP agent... not an external,
  independently-onboarded data source") — a compromised or misbehaving
  in-fleet sidecar is not caught by the same cryptographic verification an
  external source would be subject to.
