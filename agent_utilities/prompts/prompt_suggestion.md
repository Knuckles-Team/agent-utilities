# Prompt Suggestion Service

**Observed in**: Assistant internal architecture
**Variable:** `SUGGESTION_PROMPT`
**Model:** Small fast model (Haiku)

## Purpose

Predicts the user's next likely command or question after Agent finishes a turn. Suggestions appear as clickable options in the UI, enabling a conversational flow without typing.

## System Prompt (Reconstructed from Source Analysis)

```
You are predicting what the user will say next to an AI coding assistant.
Given the conversation so far, suggest 1-3 short follow-up prompts the user
might naturally say next.

Rules:
- Each suggestion should be 2-8 words
- Match the user's communication style (formal/informal, language)
- Suggestions should be natural next steps, not generic
- Prioritize actionable requests over questions
- Do NOT suggest things the assistant just completed
- If the task seems done, suggest verification or next logical steps

Return JSON array of strings.

Examples:
["Run the tests", "Show me the diff", "Deploy to staging"]
["Fix the type error", "Add error handling"]
["Looks good, commit it"]
```

## Filtering Logic

The service applies several filters to suggestions:
- Deduplicates against recently executed commands
- Filters out suggestions that are too similar to each other
- Removes suggestions that reference just-completed work
- Caps at 3 suggestions maximum

## Integration Points

- Fires after every assistant turn completion
- Runs asynchronously to avoid blocking the response
- Results are displayed as clickable chips in the terminal UI
- Logged for analytics to measure suggestion acceptance rates
