# YOLO/Auto-Mode Classifier System


---

## Architecture Overview

The YOLO classifier is Assistant's **most security-critical component**. When the user enables "auto mode" (previously called "YOLO mode"), this classifier evaluates every tool invocation before it runs.

### How It Works

1. **Base Prompt**: Loaded from `yolo-classifier-prompts/auto_mode_system_prompt.txt` (bundled at build time via `bun:bundle` feature flag `TRANSCRIPT_CLASSIFIER`)
2. **Permissions Template**: Swapped at runtime depending on user type:
   - **External users**: `permissions_external.txt`
   - **System employees** (`USER_TYPE=ant`): `permissions_System.txt`
3. **User Rules**: Three customizable sections are injected from `settings.autoMode`:
   - `allow` — actions the user explicitly permits
   - `soft_deny` — actions that should be blocked (but user can override)
   - `environment` — context about the user's environment

### Classifier Tool Schema

The classifier uses a forced tool call (`classify_result`) for structured output:

```json
{
  "name": "classify_result",
  "description": "Report the security classification result for the agent action",
  "input_schema": {
    "type": "object",
    "properties": {
      "thinking": {
        "type": "string",
        "description": "Brief step-by-step reasoning."
      },
      "shouldBlock": {
        "type": "boolean",
        "description": "Whether the action should be blocked (true) or allowed (false)"
      },
      "reason": {
        "type": "string",
        "description": "Brief explanation of the classification decision"
      }
    },
    "required": ["thinking", "shouldBlock", "reason"]
  }
}
```

### System Prompt Assembly

```
buildYoloSystemPrompt(context):
  1. Load BASE_PROMPT from auto_mode_system_prompt.txt
  2. Replace <permissions_template> with external OR System template
  3. Replace <user_allow_rules_to_replace> with user's allow rules
  4. Replace <user_deny_rules_to_replace> with user's deny rules
  5. Replace <user_environment_to_replace> with user's environment info
```

### Agent.md Integration

The classifier also receives the user's Agent.md content as a prefixed user message:

```
The following is the user's Agent.md configuration. These are
instructions the user provided to the agent and should be treated
as part of the user's intent when evaluating actions.

<user_Agent_md>
{content}
</user_Agent_md>
```

### Safe Allowlisted Tools

These tools are pre-approved and skip the classifier entirely:

- Read-only tools (file reading, glob, grep)
- The classifier tool itself (`classify_result`)
- Various safe utilities

### Conversation Transcript

The classifier receives a **compact transcript** of the conversation:
- User messages: text only
- Assistant messages: **tool_use blocks only** (text is excluded to prevent the model from crafting text that influences the classifier)

### PowerShell Deny Guidance

On Windows, additional deny rules are injected:
- PowerShell commands that modify system state
- Registry modifications
- Service management commands

## Key Design Decisions

1. **Text exclusion**: Assistant text blocks are deliberately excluded from the transcript to prevent prompt injection through model-authored text
2. **Cache-friendly**: Agent.md prefix uses `cache_control` for prompt caching across classifier calls
3. **Feature-gated**: The entire system is behind the `TRANSCRIPT_CLASSIFIER` build flag
4. **Dual templates**: System employees get different permission defaults than external users
5. **User customization**: The three-section allow/deny/environment system lets users tune the classifier's behavior
