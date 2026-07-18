"""Self-update: check GitHub Releases for a newer installer and run it.

Upgrading only replaces the app binaries in %LOCALAPPDATA%\\Programs\\LiveCaptions.
The models and transcripts live in %LOCALAPPDATA%\\live-captions, a separate tree
the installer never touches — so an upgrade keeps them automatically (no ~1.5 GB
re-download). Pure logic; the GUI (settings window) drives it with a progress bar.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from typing import Callable, Optional, Tuple

from . import __version__

REPO = "haberwooki/dia_live_captions"
_API = f"https://api.github.com/repos/{REPO}/releases/latest"


def current_version() -> str:
    return __version__


def _tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.strip().lstrip("vV").split(".") if x.isdigit())


def latest_release(timeout: float = 10.0) -> Tuple[str, Optional[str]]:
    """(tag, installer_url) of the latest GitHub release. Raises on network error."""
    req = urllib.request.Request(_API, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "livecaptions-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    tag = data.get("tag_name", "")
    url = next((a["browser_download_url"] for a in data.get("assets", [])
                if a.get("name", "").lower().endswith(".exe")), None)
    return tag, url


def is_newer(tag: str) -> bool:
    """True if `tag` is a newer version than what's running."""
    try:
        return _tuple(tag) > _tuple(__version__)
    except Exception:
        return False


def download(url: str, on_progress: Optional[Callable[[float], None]] = None,
             timeout: float = 30.0) -> str:
    """Download the installer to a temp file, reporting fraction (0..1) via
    on_progress. Returns the path. on_progress may raise to cancel."""
    dest = os.path.join(tempfile.gettempdir(), "LiveCaptions-Setup-update.exe")
    req = urllib.request.Request(url, headers={"User-Agent": "livecaptions-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        read = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if on_progress and total:
                on_progress(read / total)
    return dest


def run_installer(path: str) -> None:
    """Launch the downloaded installer silently and return immediately. It upgrades
    in place (CloseApplications in the .iss lets it close+replace our running exe);
    the caller should quit the app right after. Models/transcripts are untouched."""
    subprocess.Popen([path, "/SILENT", "/NOCANCEL"], close_fds=True)
