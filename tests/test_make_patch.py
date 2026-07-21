"""The differential build must produce a hash that identifies the native libraries.

The updater trusts internal.sha256 to decide "are the heavy libraries unchanged, so
a small patch is safe?". If the hash missed a changed library, the updater would ship
new exes against old libraries — a broken install. So: same tree -> same hash, any
change -> different hash, and the staged patch payload must NOT contain _internal.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "make_patch",
    os.path.join(os.path.dirname(__file__), "..", "packaging", "make_patch.py"))
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def _bundle(root, internal_files, exes=("livecaptions.exe", "livecaptions-overlay.exe")):
    os.makedirs(os.path.join(root, "_internal"), exist_ok=True)
    for rel, content in internal_files.items():
        p = os.path.join(root, "_internal", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(content)
    for e in exes:
        with open(os.path.join(root, e), "wb") as f:
            f.write(b"exe")
    return root


def test_same_tree_same_hash(tmp_path):
    a = _bundle(str(tmp_path / "a"), {"torch.dll": b"1", "sub/nemo.pyd": b"2"})
    b = _bundle(str(tmp_path / "b"), {"torch.dll": b"1", "sub/nemo.pyd": b"2"})
    assert mp.internal_hash(a) == mp.internal_hash(b)


def test_changed_library_flips_the_hash(tmp_path):
    a = _bundle(str(tmp_path / "a"), {"cublas.dll": b"v12"})
    b = _bundle(str(tmp_path / "b"), {"cublas.dll": b"v13"})   # a dependency bumped
    assert mp.internal_hash(a) != mp.internal_hash(b)


def test_added_or_removed_library_flips_the_hash(tmp_path):
    a = _bundle(str(tmp_path / "a"), {"torch.dll": b"1"})
    b = _bundle(str(tmp_path / "b"), {"torch.dll": b"1", "extra.dll": b"9"})
    assert mp.internal_hash(a) != mp.internal_hash(b)


def test_a_renamed_file_flips_the_hash(tmp_path):
    """Same bytes, different path — the path is part of the identity."""
    a = _bundle(str(tmp_path / "a"), {"old_name.dll": b"same"})
    b = _bundle(str(tmp_path / "b"), {"new_name.dll": b"same"})
    assert mp.internal_hash(a) != mp.internal_hash(b)


def test_exe_changes_do_not_affect_the_internal_hash(tmp_path):
    """The exes are the VOLATILE layer; the hash tracks only _internal, so a code
    change (new exes, same libraries) leaves the hash — and thus 'patch is safe'."""
    root = _bundle(str(tmp_path), {"torch.dll": b"1"})
    before = mp.internal_hash(root)
    with open(os.path.join(root, "livecaptions-overlay.exe"), "wb") as f:
        f.write(b"NEW CODE")
    assert mp.internal_hash(root) == before


class TestBuild:
    def test_writes_hash_manifest_and_stages_only_the_exes(self, tmp_path):
        bundle = _bundle(str(tmp_path / "bundle"),
                         {"torch.dll": b"big", "cublas.dll": b"huge"})
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "out")
        manifest = mp.build(bundle, staging, output, "0.5.0")

        # the hash file ships inside the bundle
        with open(os.path.join(bundle, "internal.sha256")) as f:
            assert f.read().strip() == manifest["internal_sha256"]

        # the manifest matches
        import json
        with open(os.path.join(output, "manifest.json")) as f:
            assert json.load(f)["internal_sha256"] == manifest["internal_sha256"]

        # the patch payload has the exes + hash, and CRUCIALLY not _internal
        staged = set(os.listdir(staging))
        assert staged == {"livecaptions.exe", "livecaptions-overlay.exe",
                          "internal.sha256"}
        assert not os.path.exists(os.path.join(staging, "_internal")), \
            "the patch must NOT carry the 800 MB it exists to avoid"

    def test_a_bundle_missing_an_exe_fails_loudly(self, tmp_path):
        bundle = _bundle(str(tmp_path / "bundle"), {"x.dll": b"1"},
                         exes=("livecaptions.exe",))   # overlay exe missing
        with pytest.raises(SystemExit):
            mp.build(bundle, str(tmp_path / "s"), str(tmp_path / "o"), "0.5.0")
