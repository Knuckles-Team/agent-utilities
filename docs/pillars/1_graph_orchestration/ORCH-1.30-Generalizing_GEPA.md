# Held-out generalization for native program optimization

Generalization is enforced by the native program contract rather than a Python
prompt optimizer. Training examples carry explicit train/validation/test splits,
opaque evidence loci, modality labels, and bounded scores. Candidate evaluations
remain separate from the training corpus and promotion can require every observed
modality to meet a non-regression threshold.

The Agent Utilities boundary keeps raw examples ephemeral and sends only opaque
references and numeric evaluation coordinates. The Rust compiler validates split,
policy, evidence, modality, budget, and promotion invariants before producing a
candidate or plan. This makes held-out selection consistent across text, document,
image, audio, video, graph, table, time-series, vector, spatial, tensor, code, trace,
and binary evidence.

See `docs/architecture/evolvable_surface.md` and
`epistemic-graph/crates/eg-program/src/optimizer.rs` for the current contract.
