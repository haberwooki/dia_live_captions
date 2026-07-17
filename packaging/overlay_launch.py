"""PyInstaller entry point for the windowed EXE (livecaptions-overlay.exe).

Same main() as the CLI, but defaults to the overlay so double-clicking the
Start Menu shortcut Just Works. Built console=False, so configure_runtime()
redirects the (absent) stdout/stderr to a log file. Any explicit mode flag the
user passes (e.g. --diarize-live, --wav) still wins; we only add --overlay when
no mode was requested.
"""
import sys

from livecaptions.__main__ import main

_MODE_FLAGS = ("--overlay", "--demo", "--screenshot", "--list-devices",
               "--wav", "--diarize", "--download-models", "--sessions",
               "--search", "--export", "--rename-speaker", "--name-speakers")

if __name__ == "__main__":
    if not any(a.split("=", 1)[0] in _MODE_FLAGS for a in sys.argv[1:]):
        sys.argv.append("--overlay")
    main()
