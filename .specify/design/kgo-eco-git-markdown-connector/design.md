# Design Document: Git gets its own native connector, not a config knob on FilesystemConnector

CONCEPT:AU-ECO.connector.git-markdown-revision-connector

> `agent_utilities/protocols/source_connectors/connectors/git_markdown.py:1-40`
> (module docstring), `:118` (domain-pack presets), `:387`, `:644`
> (diff-driven incremental batch).

## Decision — a new, small, zero-infra native connector, reusing `FilesystemConnector`'s conventions rather than its code

AGENTS.md's standing rule is that a new *external* source is a declarative
`mcp_tool` preset, never a new native connector module — but it explicitly
carves out an exception for **zero-infra defaults** (filesystem, sqlite:
"things that must work with nothing deployed"), because those substrates
have no server/protocol/auth for a preset to describe. A local git working
tree is exactly that shape: no server, no credentials, no network, just a
directory on disk — so the real question this module's docstring poses is
narrower than "preset or connector": **does git need its own connector, or
is a directory-with-a-`.git`-folder already covered by `FilesystemConnector`?**

It needs its own connector because git has real revision semantics a plain
filesystem walk cannot express, and revision is specifically what the
adopting use case asked for (the git commit SHA as `source_version`):

* `FilesystemConnector.poll` waters on `st_mtime_ns` (or a caller-supplied
  content-hash snapshot) — neither is a *revision*: mtime resets on a fresh
  checkout, and there is no source-native "as of commit X" concept to cite.
* Content is read via `git show <sha>:<path>` — bound to the exact git
  object at that revision — never the live, possibly-dirty working tree
  `FilesystemConnector` reads, which is what makes `source_version` an
  independently-verifiable fact instead of an opaque watermark.
* A git diff between two revisions is a natural incremental change feed
  (`git diff --name-status <old>..<new>`: add/modify/copy/rename map onto
  `upsert`, delete maps onto a tombstone `ChangeEnvelope`) — a materially
  different poll algorithm from mtime-comparison, not a config knob on top
  of one.

**The rejected alternative is named directly in the docstring**: extend
`FilesystemConnector` with a git-aware mode instead of writing a second
module. Rejected because the walk algorithm itself differs enough
(`git ls-tree`/`git show`/`git diff` at a fixed revision vs. a live directory
walk) that branching one connector's `poll`/read path on "is this a git
repo?" would be more confusing than two small, independent implementations
that merely share conventions (namespace hashing, fail-closed ACL defaults,
governed-envelope emission after `graphql_document`'s precedent) rather than
code.

## Risk Assessment

- **Blast Radius**:
  `agent_utilities/protocols/source_connectors/connectors/git_markdown.py`
  only — a new connector registration, not a change to
  `FilesystemConnector` or the source-connector protocol.
- **Backward Compatible**: Yes — purely additive; existing filesystem-backed
  sources are unaffected.
- **Known weak point**: explicitly scoped to revision-tracked content only —
  the module docstring states it does not implement the domain-pack
  framework itself, only the connector each domain pack's preset dispatches
  through.
