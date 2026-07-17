"""Frozen-build runtime setup: app-owned model cache + a diagnostic log.

Called once at the very top of `main()`, before anything that imports
huggingface_hub — so the cache-location env vars are in place before the first
download. A no-op in the important respects when running from source, so dev
setups (and an existing ~/.cache/huggingface) are left untouched.

Two problems this solves, both specific to the packaged app:

1. **Clean uninstall.** By default every weight (Whisper ~1.5 GB, Sortformer,
   pyannote) lands in the shared ``~/.cache/huggingface`` alongside other apps'
   models, which an uninstaller can't safely delete. When frozen we redirect the
   HF cache into ``%LOCALAPPDATA%\\live-captions\\hf`` so it's ours to remove.
2. **No diagnostics in a windowed build.** The overlay EXE is built
   ``console=False``, so ``sys.stdout``/``stderr`` are None and every ``print``
   the app already makes vanishes. We point them at a log file instead, keeping
   the existing print-based diagnostics without rewriting them as logging.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import platformdirs

APP_NAME = "live-captions"


def data_dir() -> Path:
    """%LOCALAPPDATA%\\live-captions on Windows — same root as models/ and the DB."""
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _redirect_hf_cache() -> None:
    """Point the Hugging Face cache at our app dir — packaged build only.

    Respects any HF_* the user set explicitly (don't override a deliberate
    choice). Moves HF_HOME too, so ``huggingface-cli`` token lookup and the
    weight cache stay together under one removable directory. Packaged pyannote
    users supply a token via LC_HF_TOKEN / HF_TOKEN, which resolve_token() reads
    before the on-disk token file, so moving HF_HOME doesn't strand them.
    """
    hf = data_dir() / "hf"
    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE"):
        os.environ.setdefault(var, str(hf))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _capture_stdio_to_log() -> None:
    """When frozen with no console (windowed EXE), send stdout/stderr to a file.

    Preserves the app's existing print() diagnostics. Best-effort: if the log
    can't be opened we leave the (None) streams alone rather than crash.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        log_dir = data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Line-buffered so a crash still leaves the last lines on disk.
        f = open(log_dir / "live-captions.log", "a", encoding="utf-8",
                 buffering=1, errors="replace")
    except OSError:
        return
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def configure_runtime() -> None:
    """Idempotent; safe to call from a source run (only stdio capture and the
    cache redirect are gated on `frozen`)."""
    if is_frozen():
        _redirect_hf_cache()
        _capture_stdio_to_log()
