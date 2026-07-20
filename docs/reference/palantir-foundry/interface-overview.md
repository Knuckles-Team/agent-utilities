# Interfaces Overview

> Source: <https://www.palantir.com/docs/foundry/interfaces/interface-overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## Definition

An interface is "an Ontology type that describes the shape of an object type and its capabilities," enabling consistent modeling across implementing object types without knowing type-specific details.

## Core properties

- **Local vs shared properties** — interface properties may be defined locally on the interface (recommended) or via shared properties.
- **Link-type constraints** — govern which link types may connect to implementers; packagable with the interface.

## Implementation & polymorphism

- **Multiple implementation** — many object types can implement the same interface, enabling polymorphic workflows.
- **Interface usage** — workflows interact with several object types in aggregate or independently without knowing specifics.

## Inheritance & extension

- **Extension** — a child interface inherits the parent's properties, then adds more specific ones.
- **Multi-level inheritance** — interfaces can extend multiple interfaces, including ones that themselves extend others; properties inherit through layers.

## Programmatic targeting

- **TypeScript v2 functions** — full, type-safe, polymorphic targeting.
- **Object Set Service** — partial: search/sort by interface; aggregation in development.
- **Ontology SDK** — TypeScript available; Java/Python in development.

## Platform support status

| Feature | Status |
|---|---|
| Ontology Manager | Full |
| Marketplace packaging | Full |
| TypeScript v2 functions | Full |
| Actions | Partial (no direct interface link-type reference) |
| Workshop | Not supported |
| TypeScript v1 / Python functions | Not supported |
