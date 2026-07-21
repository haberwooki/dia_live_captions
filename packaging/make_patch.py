"""Build-time helper for differential updates.

Splits the frozen bundle into what changes every release (the two exes) and what
rarely does (`_internal`: torch, NeMo, cuBLAS — ~800 MB). Emits:

  <bundle>/internal.sha256    a hash of the _internal tree, shipped INSIDE both the
                              full and patch installers so an installed app can later
                              tell whether its heavy libraries match a new release.
  <staging>/                  the patch payload: the two exes + internal.sha256, from
                              which the patch installer is built.
  Output/manifest.json        {version, internal_sha256, ...} uploaded to the release;
                              the updater reads it to decide patch vs full download.

Run by packaging/build.ps1 after PyInstaller and before the installers compile.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

EXES = ("livecaptions.exe", "livecaptions-overlay.exe")
INTERNAL = "_internal"
HASH_FILE = "internal.sha256"


def internal_hash(bundle_dir: str) -> str:
    """A deterministic SHA-256 over the _internal tree.

    Order-independent of the filesystem: paths are sorted and each file contributes
    its relative path plus its content hash, so the same bundle always yields the
    same digest and any changed/added/removed library flips it. Uses forward-slash
    relative paths so a hash computed on the build machine matches regardless of OS
    path separators.
    """
    root = os.path.join(bundle_dir, INTERNAL)
    files = []
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            files.append((rel, full))
    files.sort()

    digest = hashlib.sha256()
    for rel, full in files:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def build(bundle_dir: str, staging_dir: str, output_dir: str, version: str) -> dict:
    """Write internal.sha256, stage the patch payload, and write manifest.json."""
    h = internal_hash(bundle_dir)

    with open(os.path.join(bundle_dir, HASH_FILE), "w", encoding="utf-8") as f:
        f.write(h)

    # Stage the patch payload: exes + the hash file, nothing from _internal.
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)
    for name in (*EXES, HASH_FILE):
        src = os.path.join(bundle_dir, name)
        if not os.path.exists(src):
            raise SystemExit(f"expected {name} in the bundle; is this a real build?")
        shutil.copy2(src, os.path.join(staging_dir, name))

    manifest = {
        "version": version,
        "internal_sha256": h,
        "full": f"LiveCaptions-Setup-{version}.exe",
        "patch": f"LiveCaptions-Patch-{version}.exe",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    bundle, staging, output, version = sys.argv[1:5]
    m = build(bundle, staging, output, version)
    print(f"internal.sha256 = {m['internal_sha256']}")
    print(f"manifest -> {os.path.join(output, 'manifest.json')}")
