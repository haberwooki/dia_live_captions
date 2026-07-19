#!/usr/bin/env bash
# SessionStart hook: inject .claude/HANDOFF.md into context at session start,
# resume, /clear, and post-compact — so Claude always knows where work stood.
#
# Stdout from a SessionStart command hook is added to Claude's context.
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
HANDOFF="$PROJECT_DIR/.claude/HANDOFF.md"

if [ -f "$HANDOFF" ]; then
  echo "=== SESSION HANDOFF (from .claude/HANDOFF.md) ==="
  echo "Read this before doing anything else. 'Next steps' and 'Dead ends' are authoritative."
  echo
  cat "$HANDOFF"
  echo
  echo "=== END HANDOFF ==="
fi

exit 0
