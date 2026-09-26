# Object & Link Types — Type Reference

> Source: <https://www.palantir.com/docs/foundry/object-link-types/type-reference/>
> Captured during the ontology-parity effort. Concrete data-model reference only.

## Ontology types (schema definitions)

- **Object type** — schema for a real-world entity/event; comprises individual objects with shared characteristics (e.g. `Airport` type → `JFK`, `LHR` objects).
- **Property** — a characteristic of an entity/event; object-level attribute with a typed value (e.g. Airport `name`, `country`).
- **Shared property** — a property usable on multiple object types; consistent modeling + centralized metadata.
- **Link type** — schema for a relationship between two object types; an individual link is one relationship instance.
- **Action type** — schema for a set of changes/edits to objects, property values, and links, including side-effect behaviors.
- **Object type groups** — classification primitive for ontology search/exploration.
- **Interfaces** — describe the shape of an object type and its capabilities; provide polymorphism; enforce consistent modeling.

## Data type categories

### Value types

- Semantic wrappers around field types with metadata/constraints.
- Customizable within a space (not globally static).
- Examples: emails, URLs, UUIDs, enumerations.
- Enable domain-specific validation and expressiveness.

### Type-level vs instance-level

| Aspect | Type definition | Instance |
|---|---|---|
| Scope | Metadata structure | Actual data values |
| Content | Display names, property types, descriptions | Primary keys, property values |
| Nature | Static schema | Dynamic data |
