"""Voice-activity detection for streaming — Silero VAD v6 via faster-whisper's
bundled ONNX model. Runs on onnxruntime's CPU execution provider (the only one
installed), so it uses ZERO VRAM — the 8 GB stays reserved for the Whisper
worker. No torch.

Used for two things in the streaming pipeline: gating (don't transcribe pure
silence/music → fewer hallucinations) and endpointing (finalize a line when
speech pauses).
"""
from __future__ import annotations

from typing import List

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps


class SpeechGate:
    def __init__(self, sampling_rate: int = 16000, threshold: float = 0.5,
                 min_silence_ms: int = 150):
        self._sr = sampling_rate
        self._opts = VadOptions(threshold=threshold, min_silence_duration_ms=min_silence_ms)

    def segments(self, audio: np.ndarray) -> List[dict]:
        """Speech regions as [{'start': sample, 'end': sample}, ...] (CPU VAD)."""
        if audio.size == 0:
            return []
        return get_speech_timestamps(audio, self._opts, self._sr)

    def has_speech(self, audio: np.ndarray) -> bool:
        return bool(self.segments(audio))

    def trailing_silence_sec(self, audio: np.ndarray) -> float:
        """Seconds of silence at the END of the buffer (0 if it ends in speech)."""
        segs = self.segments(audio)
        if not segs:
            return audio.size / self._sr if self._sr else 0.0
        return max(0.0, (audio.size - segs[-1]["end"]) / self._sr)
