# Ontology School

A short, structured curriculum for learning this platform's Knowledge Graph —
its ontology model, and how to query it. This closes Ontology-Playground
coverage row #16 ("Ontology School": structured courses, Markdown lessons,
presentation mode, quizzes) as a real, extensible **framework** with two
starter lessons authored from existing platform material, not a placeholder.

**Honest scope note:** Ontology-Playground's own Ontology School ships nine
full courses with a review workflow. This ships the course/lesson **structure**
— the manifest schema, the renderer contract (Markdown body +
presentation-mode slide-split at `##` + an optional quiz), and two complete,
accurate lessons — so the feature genuinely exists and is trivial to extend
with more lessons over time. Growing it to nine courses is ongoing content
work, deliberately out of scope for this change.

## How this is structured

[`manifest.yaml`](manifest.yaml) is the single source of truth: a list of
courses, each with one or more lessons (a title, a Markdown body path, and an
optional multiple-choice quiz). Nothing here is code — adding a lesson is
adding a Markdown file plus a manifest entry, no new machinery required.

The same manifest (mirrored into `agent-webui`'s own `src/content/learn/` so
the in-app **Learn** view works standalone, with no backend round-trip — see
that repo's `LearnView`) drives:

- a course list,
- a lesson reader (plain Markdown, rendered with the app's existing renderer),
- a presentation mode that slide-splits a lesson at every `##` heading,
- and, where a lesson defines one, a simple multiple-choice quiz.

## Courses

### [Intro to the Ontology Model](lessons/ontology-model-101/01-interfaces-object-types-and-links.md)

What an interface, an object type, and a link type are, and how they compose
into a governed schema — with a short quiz at the end.

### [Querying with UQL](lessons/querying-with-uql/01-your-first-uql-pipeline.md)

The engine's native cross-modal query language: one pipelined text query
composing graph traversal, vector rank, lexical search, time travel, and
epistemic reasoning over a single snapshot — with a short quiz at the end.

## Extending this

To add a lesson: write a Markdown file under `docs/learn/lessons/<course-id>/`,
then add a `lessons:` entry (`id`, `title`, `body`, optional `quiz`) to
[`manifest.yaml`](manifest.yaml) — and mirror both into `agent-webui`'s
`src/content/learn/` so the in-app view picks it up. To add a course, add a
new top-level entry to the same manifest. No code changes are required for
either.
