"""The narrow-waist data types every layer shares.

Pipeline:
    AudioSource -> AudioBlock -> Segmenter -> Utterance -> WhisperWorker
                -> TranscriptEvent -> Sink

Two seams:
  * AudioBlock     — raw mono PCM produced by a capture backend (WASAPI, WAV).
  * TranscriptEvent — one caption unit produced by a TranscriptionSource
    (local Whisper now; cloud / Discord later).

Multi-source merge decision (see docs/architecture.md): sources are INDEPENDENT
and every event is tagged with `source`. A future MicCaptureSource emits
`source="mic"`; a sink merges sources into one transcript by `t_start` — no
schema change. `speaker` (M4/M5 diarization) and `is_final` (M3 streaming
partials) are reserved: M1 always emits `speaker=None`, `is_final=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AudioBlock:
    """A block of mono PCM audio. `samples` is float32 in [-1, 1] at `rate` Hz."""

    samples: np.ndarray
    rate: int
    t: float = 0.0          # source-relative start time (seconds), monotonic

    @property
    def duration(self) -> float:
        return len(self.samples) / self.rate if self.rate else 0.0


@dataclass(frozen=True)
class TranscriptEvent:
    """One caption unit — the seam every transcription backend emits."""

    text: str
    source: str                        # which stream: "loopback", "mic", "discord:alice"
    t_start: float                     # source-relative seconds
    t_end: float
    is_final: bool = True              # reserved for M3 streaming partials (always True for now)
    speaker: Optional[str] = None      # reserved for M4/M5 diarization
    confidence: Optional[float] = None
    infer_lag: Optional[float] = None  # inference wall-time in seconds (latency rig)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start
