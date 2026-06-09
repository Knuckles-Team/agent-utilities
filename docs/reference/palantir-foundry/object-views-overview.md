# Object Views Overview

> Source: <https://www.palantir.com/docs/foundry/object-views/overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## View types

### Standard object views
- Out-of-the-box representations that automatically reflect an object type's configuration.
- Available for all object types; require no configuration; remain a fallback even when configured views exist; users can toggle back at any time.

### Configured object views
- Fully customizable, built in Workshop.
- Become the default once created; support contextualized workflows; coexist with standard views; user-toggleable.

## Form factors

Both standard and configured views support two presentation modes:

- **Full object views** — comprehensive, in-depth overview (e.g. complete patient history: demographics, vitals, procedures, prescriptions, diagnoses, trends).
- **Panel object views** — integration-focused, critical data only, optimized for embedding in other applications (e.g. demographics + vitals summary).

## Data components

Object views present:
- **Property data** — core attributes/metadata.
- **Object links** — relationships to other ontology objects.
- **Related applications** — integrated app references/contexts.
- **Actions** — available operations on the object.
