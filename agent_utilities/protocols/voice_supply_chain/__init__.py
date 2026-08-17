"""Voice-model supply-chain acquisition (GOC-36 — voice model supply chain qualification).

The AU-side governed acquisition adapter for Piper voice models: pinned Hugging Face
fetch, SHA-256 verification, quarantine, manifest schemas, and license-decision
recording. See :mod:`.manifest` and :mod:`.acquisition` module docstrings for the
authority model (this package has no promotion authority — see those docstrings)
and DEF-017 scope guard (Piper-only, not a generic Hugging Face loader).

CONCEPT:AU-KG.ingest.voice-model-acquisition — Governed, digest-verified, pinned-revision
acquisition of Piper voice-model assets into a local quarantine, with license/consent
decision recording gating any later promotion handoff.
"""

from __future__ import annotations
