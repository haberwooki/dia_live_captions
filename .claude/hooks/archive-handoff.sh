#!/usr/bin/env bash
# SessionEnd hook: archive a timestamped copy of HANDOFF.md when a session
# ends, and keep only the 20 most recent copies. Cheap insurance: even if a
# later session mangles the handoff file, you can recover the earlier state.
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
HANDOFF="$PROJECT_DIR/.claude/HANDOFF.md"
ARCHIVE_DIR="$PROJECT_DIR/.claude/handoff-archive"

if [ -f "$HANDOFF" ]; then
  mkdir -p "$ARCHIVE_DIR"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  cp "$HANDOFF" "$ARCHIVE_DIR/HANDOFF-$STAMP.md"

  # Prune: keep the 20 newest archives
  ls -1t "$ARCHIVE_DIR"/HANDOFF-*.md 2>/dev/null | tail -n +21 | while IFS= read -r old; do
    rm -f "$old"
  done
fi

exit 0
