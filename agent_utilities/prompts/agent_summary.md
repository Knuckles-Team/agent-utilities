# Agent Summary (Background Progress)

**Observed in**: Assistant internal architecture
**Model:** Small fast model (Haiku)

## Purpose

Generates periodic background progress updates for sub-agents running in coordinator mode. Provides the parent agent with real-time awareness of what each worker is doing.

## Prompt (Reconstructed from Source Analysis)

```
Summarize what the agent is currently doing in 1 short sentence. Use present tense.
Focus on the specific action, not the overall task.

Good: "Reading the authentication middleware to understand token validation"
Good: "Running pytest on the user service after fixing the import error"
Bad: "Working on the task" (too vague)
Bad: "The agent is currently in the process of examining..." (too verbose)
```

## Design Constraints

- **Single sentence**: Maximum one sentence, present tense
- **Action-specific**: Describe the current action, not the overall goal
- **No meta-commentary**: Don't describe the agent itself, describe what it's doing
- **Present tense**: Always use present-tense verbs ("Reading", "Running", "Fixing")

## Integration Points

- Runs on a periodic timer during coordinator mode
- Results are displayed in the coordinator's status view
- Helps the lead agent decide when to check in on workers
- Fires only when sub-agents are actively executing tool calls
