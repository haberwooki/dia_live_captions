"""Model loading + the off-thread inference worker.

The worker runs faster-whisper on a single background thread draining a bounded,
drop-oldest queue. One GPU worker only: model.transcribe is not safely
concurrent and 8 GB VRAM holds one medium model. This is what moves inference
(and resampling) OFF the audio/segmenter thread.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import soxr

from ..events import TranscriptEvent
from ..util import drop_oldest_put
from .segmenter import Utterance

WHISPER_SR = 16000


class _HideModules:
    """A meta_path finder that hides top-level modules from the import system."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ModuleNotFoundError(f"{fullname} hidden during ctranslate2 import")
        return None


def _import_whisper_model():
    """Import faster-whisper with torch/transformers hidden.

    CTranslate2's __init__ imports its model-CONVERTER subpackages, which pull in
    transformers and torch — ~3 s each, and inference never uses either. ct2 guards
    both with try/except ImportError, so hiding them is safe and cuts
    `import ctranslate2` from 6-13 s to ~0.5 s (measured). The block is removed
    immediately, so live diarization (which imports torch lazily) still works.
    """
    blocker = _HideModules({"torch", "transformers"})
    sys.meta_path.insert(0, blocker)
    try:
        from faster_whisper import WhisperModel
    finally:
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass
    return WhisperModel


def _missing_weights(err: Exception) -> bool:
    """True for the 'cache is half-built' failure: the blob downloaded but the
    snapshot link wasn't created yet, so CTranslate2 opens a directory with no
    model.bin in it. Seen in the wild — four consecutive launch failures while the
    cache finished linking itself, then it worked."""
    s = str(err).lower()
    return "unable to open file" in s or "no such file" in s


def _new_model(WhisperModel, name: str, device: str, compute: str):
    """Prefer the already-cached model: faster-whisper otherwise makes a Hugging
    Face round-trip on every launch to revalidate a snapshot we already have
    (1.8-4.5 s). Falls back to the normal online path when it isn't cached yet."""
    try:
        return WhisperModel(name, device=device, compute_type=compute, local_files_only=True)
    except Exception as local_err:
        try:
            return WhisperModel(name, device=device, compute_type=compute)
        except Exception as online_err:
            if not (_missing_weights(local_err) or _missing_weights(online_err)):
                raise
            # Repair an incomplete cache rather than telling the user to reinstall.
            print("  model cache looks incomplete; re-fetching the weights...")
            from faster_whisper.utils import download_model
            path = download_model(name, local_files_only=False)
            return WhisperModel(path, device=device, compute_type=compute)


def _cuda_device_present() -> bool:
    """True if an NVIDIA GPU is visible to CTranslate2. Queries the driver
    (nvcuda.dll), not cuBLAS — so it says 'there is a GPU', while the self-test
    below is what proves the CUDA *libraries* actually load. Never raises."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def load_model(settings):
    """Load Whisper, preferring GPU and falling back to CPU. GPU (cuBLAS)
    problems often surface only at the FIRST inference, so each candidate is
    validated with a tiny self-test before we commit to it."""
    WhisperModel = _import_whisper_model()
    frozen = getattr(sys, "frozen", False)
    if settings.device == "cpu":
        candidates = [("cpu", settings.cpu_compute)]
    elif settings.device == "cuda":
        candidates = [("cuda", settings.gpu_compute)]
    elif _cuda_device_present():
        candidates = [("cuda", settings.gpu_compute), ("cpu", settings.cpu_compute)]
    else:
        # No NVIDIA GPU — skip the cuda attempt so there's no scary exception,
        # just a clean, fast CPU start.
        print("No NVIDIA GPU detected; using CPU.")
        candidates = [("cpu", settings.cpu_compute)]

    if candidates[0][0] == "cpu" and settings.model_name in ("medium", "large", "large-v2", "large-v3"):
        print(f"  Note: '{settings.model_name}' on CPU is slow; --streaming likely won't keep up. "
              f"Consider --model small.en (or tiny.en for streaming).")

    last_err = None
    for device, compute in candidates:
        try:
            print(f"Loading faster-whisper '{settings.model_name}' on {device} ({compute})...")
            t0 = time.time()
            model = _new_model(WhisperModel, settings.model_name, device, compute)
            list(model.transcribe(np.zeros(WHISPER_SR, dtype=np.float32),
                                  beam_size=1, vad_filter=False)[0])
            print(f"Model ready on {device} in {time.time() - t0:.1f}s.")
            return model
        except Exception as e:
            last_err = e
            print(f"  {device} path unavailable: {e}")
            if device == "cuda":
                # A GPU is present but the CUDA libraries didn't load. In a
                # packaged build that's a bundling problem, not something the
                # user can pip-install away.
                if "cublas" in str(e).lower() or "cudnn" in str(e).lower() or "library" in str(e).lower():
                    print("  GPU libraries failed to load"
                          + ("" if frozen else " (install: pip install nvidia-cublas-cu12 "
                             "nvidia-cuda-runtime-cu12)") + "; falling back to CPU.")
                else:
                    print("  Falling back to CPU (slower).")
    # Carry the REASON in the exception, not just an exit code: this propagates to
    # the overlay, and `str(SystemExit(1))` renders as the useless text "1".
    msg = f"Could not load the speech model on any device. Last error: {last_err}"
    print(f"\n{msg}")
    raise SystemExit(msg)


class WhisperWorker:
    def __init__(self, model: WhisperModel, *, language, beam_size, source_id, maxsize=4):
        self._model = model
        self._language = language
        self._beam = beam_size
        self._source_id = source_id
        self._q: "queue.Queue[Optional[Utterance]]" = queue.Queue(maxsize=maxsize)
        self._on_event = None
        self._thread = None
        self.dropped = 0                 # utterances dropped because inference fell behind
        self.done = threading.Event()

    def start(self, on_event) -> None:
        self._on_event = on_event
        self.done.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, utt: Utterance) -> None:
        if drop_oldest_put(self._q, utt):
            self.dropped += 1

    def close(self) -> None:
        """Signal a clean drain-then-stop (blocks briefly if full to keep order)."""
        self._q.put(None)

    def _run(self) -> None:
        while True:
            utt = self._q.get()
            if utt is None:
                break
            try:
                self._transcribe(utt)
            except Exception as e:
                print(f"(transcription error: {e})")
        self.done.set()

    def _transcribe(self, utt: Utterance) -> None:
        if utt.rate != WHISPER_SR:
            audio = soxr.resample(utt.samples, utt.rate, WHISPER_SR).astype(np.float32)
        else:
            audio = utt.samples.astype(np.float32)
        t0 = time.time()
        segments, _ = self._model.transcribe(
            audio, language=self._language, beam_size=self._beam, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text and self._on_event is not None:
            self._on_event(TranscriptEvent(
                text=text, source=self._source_id,
                t_start=utt.t_start, t_end=utt.t_end, infer_lag=time.time() - t0))

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
