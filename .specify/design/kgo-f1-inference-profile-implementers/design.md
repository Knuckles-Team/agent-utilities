# Design Document: A Model's sampling profile is an ontology-reasoned link, not a config lookup — bounded by the same value types the profile-tuning loop writes through

CONCEPT:AU-KG.ontology.inference-profile-implementers

> `agent_utilities/knowledge_graph/ontology/interfaces.py:1043-1089`
> (`InferenceProfile`, `SamplingConfigurable`),
> `agent_utilities/models/knowledge_graph.py:500`,
> `agent_utilities/models/model_registry.py:740`.

## Decision — `InferenceProfile` is a shape (not a concrete node type), and `Model` implements `SamplingConfigurable` by declaring a `HAS_PROFILE` link — so reasoning extrapolates tuned profiles across models/roles

`interfaces.py:1043-1050` states the design: "An `InferenceProfile` is the
shape of a SamplingProfile (ORCH-1.58): the sampling knobs as shared
properties. The bounds are enforced by the matching value types (KG-2.94,
Temperature/TopP/...) at the SHACL write gate; the interface declares the
shape so Functions/queries can target 'any inference profile'. A Model
implements `SamplingConfigurable`: it must declare a `HAS_PROFILE` link to
its tuned profile, letting the reasoner extrapolate profiles across
models/roles." Every knob property (`temperature`, `top_p`, `top_k`,
`max_tokens`, …) is declared with its bounding value type reference (e.g.
`temperature` → the `Temperature` value type, `[0,2]`) rather than a bare
`double` — the interface's shape and the write-time SHACL bounds are the SAME
declared contract, not two independently-maintained ones.

**The rejected alternative is treating a model's sampling configuration as
plain config** — a dict/JSON blob a caller reads by convention, with no
shared shape and no way for a Function/query to ask "which models are tuned
for task-class X" without knowing each model's config format ahead of time.
Making `InferenceProfile` an ontology Interface — and `Model` a concrete type
that *implements* `SamplingConfigurable` by declaring the `HAS_PROFILE` link
— means that question becomes ordinary interface-conformance reasoning: any
object implementing `SamplingConfigurable` is discoverable via
`find_implementers`, and its tuned profile is one link-traversal away,
regardless of which concrete model type it is.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/interfaces.py`
  (`InferenceProfile`, `SamplingConfigurable`), `models/knowledge_graph.py`
  (`RegistryNodeType.MODEL`/`INFERENCE_PROFILE`), `models/model_registry.py`.
- **Backward Compatible**: Yes — additive interface + link constraint; a
  model with no declared profile simply doesn't implement
  `SamplingConfigurable`.
- **Known weak point**: `HAS_PROFILE` is declared but not shown here as a
  hard `min_count` requirement across every registered Model — a model that
  never declares the link is invisible to profile-extrapolation reasoning
  with no structural signal distinguishing "deliberately untuned" from
  "forgot to link."
