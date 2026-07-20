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
# The LIST, not /releases/latest. GitHub's "latest" pointer follows PUBLISH time,
# not version number: when a service outage let v0.4.0 finish building before
# v0.3.4, v0.3.4 published later and grabbed the "latest" pointer, so the update
# button offered an OLDER version and never saw v0.4.0. We sort by version
# ourselves so out-of-order publishing can never mislead the updater again.
_API = f"https://api.github.com/repos/{REPO}/releases?per_page=30"


def current_version() -> str:
    return __version__


def _tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.strip().lstrip("vV").split(".") if x.isdigit())


def _installer_url(release: dict) -> Optional[str]:
    return next((a["browser_download_url"] for a in release.get("assets", [])
                 if a.get("name", "").lower().endswith(".exe")), None)


def latest_release(timeout: float = 10.0) -> Tuple[str, Optional[str]]:
    """(tag, installer_url) of the HIGHEST-VERSION release with an installer.

    Highest by version number, not by publish time — see _API. Drafts and
    prereleases are skipped, and a release without an .exe asset (a build that
    failed to upload) is passed over so the updater never points at a dead link.
    """
    req = urllib.request.Request(_API, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "livecaptions-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        releases = json.load(r)

    best_tag, best_url, best_ver = "", None, ()
    for rel in releases:
        if rel.get("draft") or rel.get("prerelease"):
            continue
        url = _installer_url(rel)
        if not url:
            continue
        ver = _tuple(rel.get("tag_name", ""))
        if ver > best_ver:
            best_ver, best_tag, best_url = ver, rel.get("tag_name", ""), url
    return best_tag, best_url


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
    the caller should quit the app right after. Models/transcripts are untouched.

    /RELAUNCH=1 is our own parameter: a silent install normally launches nothing
    (correct for winget), so this is what starts the new version back up."""
    subprocess.Popen([path, "/SILENT", "/NOCANCEL", "/RELAUNCH=1"], close_fds=True)
