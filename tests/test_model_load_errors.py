"""A failed model load must say WHY, and must survive a half-built cache.

Real report from v0.2.0: the overlay showed "couldn't start: 1". That is
`str(SystemExit(1))` — load_model raised a bare exit code, so the only thing that
reached the user was the number 1. The underlying cause was a Hugging Face cache
whose blob had downloaded but whose snapshot link was not created yet, so
CTranslate2 opened a directory containing no model.bin. It repaired itself minutes
later, which made the app look randomly broken.
"""
import pytest

from livecaptions.asr import whisper as W


class Boom(Exception):
    pass


def _settings(**kw):
    from livecaptions.config import Settings
    return Settings(**kw)


def test_missing_weights_is_recognised():
    assert W._missing_weights(Exception(
        "Unable to open file 'model.bin' in model 'C:\\...\\snapshots\\abc'")) is True
    assert W._missing_weights(Exception("No such file or directory")) is True
    # An unrelated failure must NOT be treated as a cache problem, or we would
    # re-download the weights every time cuBLAS is missing.
    assert W._missing_weights(Exception(
        "Library cublas64_12.dll is not found or cannot be loaded")) is False


def test_failure_reports_the_reason_not_an_exit_code(monkeypatch):
    monkeypatch.setattr(W, "_import_whisper_model", lambda: object())
    monkeypatch.setattr(W, "_cuda_device_present", lambda: False)

    def explode(*a, **k):
        raise Boom("Unable to open file 'model.bin'")
    monkeypatch.setattr(W, "_new_model", explode)

    with pytest.raises(SystemExit) as ei:
        W.load_model(_settings(device="cpu"))

    text = str(ei.value)
    assert text != "1", "a bare exit code reached the user again"
    assert "Could not load" in text and "model.bin" in text


def test_incomplete_cache_is_repaired_not_surrendered_to(monkeypatch):
    """local_files_only fails, the online path fails the same way -> re-fetch."""
    calls = []

    def fake_model(name, device=None, compute_type=None, local_files_only=False):
        calls.append(("model", name, local_files_only))
        if name == "repaired/path":
            return "MODEL"
        raise Boom("Unable to open file 'model.bin' in model 'snapshots/abc'")

    import sys
    import types
    mod = types.ModuleType("faster_whisper.utils")
    mod.download_model = lambda name, local_files_only=False: calls.append(
        ("download", name)) or "repaired/path"
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", mod)

    got = W._new_model(fake_model, "tiny.en", "cpu", "int8")

    assert got == "MODEL"
    assert ("download", "tiny.en") in calls, "never attempted to repair the cache"
    assert calls[0] == ("model", "tiny.en", True), "should try the cache first"


def test_unrelated_errors_do_not_trigger_a_redownload(monkeypatch):
    """A missing CUDA library must fall through to the CPU candidate, not spend
    minutes re-downloading weights that are perfectly fine."""
    def fake_model(name, device=None, compute_type=None, local_files_only=False):
        raise Boom("Library cublas64_12.dll is not found or cannot be loaded")

    import sys
    import types
    mod = types.ModuleType("faster_whisper.utils")
    mod.download_model = lambda *a, **k: pytest.fail("re-downloaded for a DLL error")
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", mod)

    with pytest.raises(Boom):
        W._new_model(fake_model, "tiny.en", "cuda", "float16")
