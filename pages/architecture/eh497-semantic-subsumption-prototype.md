# EH-497 semantic subsumption prototype cut

`SemanticSubsumptionEngine` aligns one detached node embedding against an
ordered mapping of class prototype embeddings. That calculation is cosine
argmax, with the first class winning ties and scores at or below zero yielding
no match. It is distinct from EG `OwlReason`, which derives class facts from
RDF axioms and graph evidence. Replacing this operation with `OwlReason` would
change its input and output meaning.

EG `eg-numeric::prototype::best_cosine_prototype` owns the cosine scan. The
PyO3 boundary takes one bounded list of prototypes, including empty or ragged
zero vectors, and returns an index and score. AU converts prototype values at
the boundary, maps the index back to the first class key, applies the existing
threshold, and constructs the existing `SubsumptionAlignmentNode` and primary
parent lineage. The public AU object and its local PyGraph callers retain the
same shape; the class ranking loop is no longer in AU.

Rust unit coverage pins first tie, negative/zero, and ragged zero behavior.
AU focused coverage pins threshold inclusion, lineage, and DTO fields. The
new extension must be built and its direct AU tests run before composing this
cut; a source-level shim check alone is not the native gate.
