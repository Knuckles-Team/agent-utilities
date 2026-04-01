# Tool-Specific Prompts


---

## Bash Tool (`Bash`)

> Observed in: Assistant tool system

```
Executes a given bash command and returns its output.

The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).

IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool:

- File search: Use Glob (NOT find or ls)
- Content search: Use Grep (NOT grep or rg)
- Read files: Use Read (NOT cat/head/tail)
- Edit files: Use Edit (NOT sed/awk)
- Write files: Use Write (NOT echo >/cat <<EOF)
- Communication: Output text directly (NOT echo/printf)

# Instructions
- If your command will create new directories or files, first use this tool to run `ls` to verify the parent directory exists and is the correct location.
- Always quote file paths that contain spaces with double quotes.
- Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`.
- You may specify an optional timeout in milliseconds (up to 600000ms / 10 minutes). By default, your command will timeout after 120000ms (2 minutes).
- You can use the `run_in_background` parameter to run the command in the background.
- When issuing multiple commands:
  - If commands are independent, make multiple Bash tool calls in parallel.
  - If commands depend on each other, use '&&' to chain them.
  - DO NOT use newlines to separate commands.
- For git commands:
  - Prefer to create a new commit rather than amending an existing commit.
  - Before running destructive operations, consider safer alternatives.
  - Never skip hooks (--no-verify) unless explicitly asked.
- Avoid unnecessary `sleep` commands.
```

### Git Operations (External Users)

The Bash tool includes extensive git commit and PR instructions:
- **Git Safety Protocol**: Never update git config, never run destructive commands unless explicitly asked
- **Commit workflow**: `git status` → `git diff` → `git log` → draft message → `git add` → `git commit`
- **PR workflow**: Uses `gh pr create` with structured body format
- **HEREDOC format**: Always uses `git commit -m "$(cat <<'EOF' ... EOF)"` for proper formatting

### Sandbox System

When sandboxing is enabled, the Bash tool includes filesystem and network restriction configs:
- Read deny-only lists
- Write allow-only lists
- Network allowed/denied hosts
- Temporary files must use `$TMPDIR`, not `/tmp`

---

## File Edit Tool (`Edit`)

> Observed in: Assistant tool system

```
Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces).
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file.
```

---

## Agent Tool

> Observed in: Assistant tool system

```
Launch a new agent to handle complex, multi-step tasks autonomously.

The Agent tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do
- Launch multiple agents concurrently whenever possible, to maximize performance
- When the agent is done, it will return a single message back to you. The result is not visible to the user — send a text summary.
- You can optionally run agents in the background using the run_in_background parameter
- To continue a previously spawned agent, use SendMessage with the agent's ID or name
- The agent's outputs should generally be trusted
- Clearly tell the agent whether you expect it to write code or just to do research
- You can optionally set `isolation: "worktree"` to run the agent in a temporary git worktree
```

### Fork Subagent Feature

When fork subagent is enabled, additional guidance is included:

```
## When to fork

Fork yourself (omit `subagent_type`) when the intermediate tool output isn't worth keeping in your context.
- Research: fork open-ended questions. Launch parallel forks in one message.
- Implementation: prefer to fork implementation work that requires more than a couple of edits.

Forks are cheap because they share your prompt cache. Don't set `model` on a fork.

**Don't peek.** Do not Read or tail the output_file unless the user explicitly asks.
**Don't race.** Never fabricate or predict fork results.
```

---

## Other Tool Prompts

| Tool | File | Purpose |
|------|------|---------|
| `Read` | `FileReadTool/prompt.ts` | Read file contents with line numbers |
| `Write` | `FileWriteTool/prompt.ts` | Create or overwrite files |
| `Glob` | `GlobTool/prompt.ts` | Find files by pattern matching |
| `Grep` | `GrepTool/prompt.ts` | Search file contents with regex |
| `WebFetch` | `WebFetchTool/prompt.ts` | Fetch URL content |
| `WebSearch` | `WebSearchTool/prompt.ts` | Search the web |
| `NotebookEdit` | `NotebookEditTool/prompt.ts` | Edit Jupyter notebooks |
| `Config` | `ConfigTool/prompt.ts` | Manage Assistant configuration |
| `EnterPlanMode` | `EnterPlanModeTool/prompt.ts` | Enter planning mode |
| `ExitPlanMode` | `ExitPlanModeTool/prompt.ts` | Exit planning mode |
| `EnterWorktree` | `EnterWorktreeTool/prompt.ts` | Enter git worktree |
| `ExitWorktree` | `ExitWorktreeTool/prompt.ts` | Exit git worktree |
| `SendMessage` | `SendMessageTool/prompt.ts` | Send messages to agents/teammates |
| `TaskCreate` | `TaskCreateTool/prompt.ts` | Create background tasks |
| `TaskGet` | `TaskGetTool/prompt.ts` | Get task status |
| `TaskList` | `TaskListTool/prompt.ts` | List tasks |
| `TaskStop` | `TaskStopTool/prompt.ts` | Stop a running task |
| `TaskUpdate` | `TaskUpdateTool/prompt.ts` | Update task properties |
| `TodoWrite` | `TodoWriteTool/prompt.ts` | Write TODO items |
| `TeamCreate` | `TeamCreateTool/prompt.ts` | Create team agents |
| `ScheduleCron` | `ScheduleCronTool/prompt.ts` | Schedule cron jobs |
| `RemoteTrigger` | `RemoteTriggerTool/prompt.ts` | Trigger remote execution |
| `Sleep` | `SleepTool/prompt.ts` | Sleep/wait |
| `Skill` | `SkillTool/prompt.ts` | Invoke project skill files |
| `LSP` | `LSPTool/prompt.ts` | Language Server Protocol |
| `MCP` | `MCPTool/prompt.ts` | Model Context Protocol tools |
| `PowerShell` | `PowerShellTool/prompt.ts` | Windows PowerShell execution |
| `Brief` | `BriefTool/prompt.ts` | Set brief output mode |
| `AskUserQuestion` | `AskUserQuestionTool/prompt.ts` | Ask the user a question |
| `ReadMcpResource` | `ReadMcpResourceTool/prompt.ts` | Read MCP server resources |
| `ListMcpResources` | `ListMcpResourcesTool/prompt.ts` | List MCP resources |
| `ToolSearch` | `ToolSearchTool/prompt.ts` | Search available tools |
