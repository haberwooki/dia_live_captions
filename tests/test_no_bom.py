"""Text config files must not start with a UTF-8 BOM.

A BOM in pyproject.toml is not cosmetic: tomllib rejects it with "Invalid
statement (at line 1, column 1)", which failed an entire release build. PowerShell's
`Set-Content -Encoding utf8` writes a BOM, and an editor can too, so this guards the
files most likely to be hand-bumped.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Files that are (a) parsed by something BOM-intolerant, or (b) routinely edited by
# the release version bump, which is where the BOM crept in.
GUARDED = [
    "pyproject.toml",
    "src/livecaptions/__init__.py",
    "packaging/livecaptions.iss",
]


@pytest.mark.parametrize("rel", GUARDED)
def test_file_has_no_utf8_bom(rel):
    data = (ROOT / rel).read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), (
        f"{rel} starts with a UTF-8 BOM — tomllib and some tools choke on it. "
        f"Write it as UTF-8 without a BOM (avoid PowerShell Set-Content -Encoding utf8).")


def test_pyproject_actually_parses():
    """The concrete failure that took down a build: prove tomllib can read it."""
    import tomllib
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"]
