# Object Edits Overview

> Source: <https://www.palantir.com/docs/foundry/object-edits/overview/>
> Captured during the ontology-parity effort.
>
> NOTE: the dedicated `object-edits/overview` page is a navigation/landing page with
> no substantive feature body. Fetched twice (retried once) — both times it yielded
> only the landing facts below; the detailed edit/undo model is documented inline
> within the Action Types and Object Backend pages. The "Edit model" section that
> follows is therefore reconstructed from the adjacent canonical Foundry pages
> captured in this same directory — flagged so it is not mistaken for a verbatim
> capture of this URL.

## Landing-page facts (verbatim from this URL)

- Actions are "a single transaction that changes the properties of one or more objects, based on user-defined logic."
- Users can "edit property values, add and remove links, and create and delete objects by applying Actions."
- Constraint: "Actions are not yet supported on object types with Foundry stream datasources."
- The page cross-references Actions, Workshop, Object Views, and the APIs sections for detail.

## Edit model (reconstructed from adjacent canonical pages)

- An **edit** is the unit of mutation an Action applies: a property value set, a link add/remove, or an object create/delete, committed as part of a single Action transaction.
- Edits are **tracked per object** by the object backend (Object Storage tracks user-generated edits), forming an action/edit history.
- **Undo / revert** and **user edit-history tracking** are listed as first-class Action observability/governance capabilities (see `action-types-overview.md`).
- Edits drive **writeback** to the source datasource and are processed by the **Object Data Funnel** to keep the index synchronized (see `object-backend-overview.md`).

See also: `action-types-overview.md`, `object-backend-overview.md`.
