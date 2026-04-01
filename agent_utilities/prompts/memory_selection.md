# Memory Selection Prompt


---

## System Prompt

```
You are selecting memories that will be useful to Assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a list of filenames for the memories that will clearly be useful to Assistant as it processes the user's query (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful in processing the user's query, then do not include it in your list. Be selective and discerning.
- If there are no memories in the list that would clearly be useful, feel free to return an empty list.
- If a list of recently-used tools is provided, do not select memories that are usage reference or API documentation for those tools (Assistant is already exercising them). DO still select memories containing warnings, gotchas, or known issues about those tools — active use is exactly when those matter.
```

## User Prompt Template

```
Query: {user's query}

Available memories:
{formatted manifest of memory files with filenames and descriptions}

Recently used tools: {tool1, tool2, ...}
```

## Output Schema (Structured JSON)

```json
{
  "type": "object",
  "properties": {
    "selected_memories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["selected_memories"],
  "additionalProperties": false
}
```

## Technical Details

| Property | Value |
|----------|-------|
| **Model** | Sonnet (via `getDefaultSonnetModel()`) |
| **Query Source** | `memdir_relevance` |
| **Max Tokens** | 256 |
| **Output Format** | Structured JSON schema |
| **Max Results** | 5 memory files |
| **Exclusions** | `MEMORY.md` (already in system prompt), previously surfaced files |
| **Tool Filtering** | Skips API docs for recently-used tools, but keeps gotchas/warnings |

## Selection Pipeline

```
User query arrives
    │
    ├── Scan memory directory for .md files with frontmatter
    │   └── Extract filename + description from YAML headers
    │
    ├── Filter out already-surfaced memories (from prior turns)
    │
    ├── Format manifest: "filename.md — description"
    │
    ├── Include recently-used tools list (to avoid redundant API docs)
    │
    └── Sonnet selects up to 5 → {"selected_memories": ["file1.md", ...]}
        └── Validated against actual filenames on disk
```
