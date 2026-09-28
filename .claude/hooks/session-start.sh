#!/bin/bash
# Claude Code on the web: provision a fresh container so every gate can run.
set -euo pipefail
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
cd "$CLAUDE_PROJECT_DIR"
scripts/bootstrap.sh
# Put the upgraded uv and the repository environment first on PATH for the
# rest of the session.
{
  echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'
  echo "export VIRTUAL_ENV=\"$CLAUDE_PROJECT_DIR/.venv\""
  echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\""
} >> "${CLAUDE_ENV_FILE:-/dev/null}"
