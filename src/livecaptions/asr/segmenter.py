"""Utterance segmentation — a pure RMS state machine (no I/O, fully testable).

Consumes a stream of AudioBlocks and emits an Utterance whenever it detects the
end of speech (a trailing-silence gap) or hits a max-length cap. This is the
keystone the characterization tests target; they assert on utterance boundaries,
speech duration, and timing from a synthetic RMS envelope — never on decoded
Whisper text, which is non-deterministic across model/precision/beam.

Durations are tracked in SAMPLES (integers), not accumulated float seconds, so
the boundaries are exact and don't drift over long utterances. RMS is compared
in int16 units so the SILENCE_RMS tuning carried over from M0 keeps its meaning
even though samples are float32 in [-1, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..events import AudioBlock

INT16_FULL_SCALE = 32768.0


@dataclass
class Utterance:
    """A segmented span of audio ready to transcribe (native-rate float32 mono)."""

    samples: np.ndarray
    rate: int
    t_start: float
    t_end: float
    speech_sec: float   # seconds of actual speech (excludes trailing/most silence)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.rate if self.rate else 0.0


@dataclass
class Segmenter:
    """RMS voice-activity segmenter.

    silence_rms:     int16 RMS at/below which a block is "silence"
    end_silence_sec: trailing silence that ends an utterance
    min_utt_sec:     ignore utterances with less than this much *speech*
                     (gates on speech only, NOT speech+trailing-silence — the M0
                     bug the review caught was counting the silence tail here)
    max_utt_sec:     force-flush a still-running utterance after this long
    """

    silence_rms: float = 350.0
    end_silence_sec: float = 0.6
    min_utt_sec: float = 0.4
    max_utt_sec: float = 12.0

    _rate: Optional[int] = field(default=None, init=False)
    _buf: List[np.ndarray] = field(default_factory=list, init=False)
    _utt_samples: int = field(default=0, init=False)     # speech + buffered silence
    _speech_samples: int = field(default=0, init=False)  # speech only
    _silence_samples: int = field(default=0, init=False) # current trailing-silence run
    _speaking: bool = field(default=False, init=False)
    _t_start: float = field(default=0.0, init=False)
    _t_end: float = field(default=0.0, init=False)

    @staticmethod
    def block_rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) * INT16_FULL_SCALE

    def push(self, block: AudioBlock) -> List[Utterance]:
        """Feed one block; return any utterances completed by it (usually 0 or 1)."""
        out: List[Utterance] = []
        if self._rate is None:
            self._rate = block.rate
        n = len(block.samples)
        rms = self.block_rms(block.samples)
        t_block_end = block.t + (n / self._rate if self._rate else 0.0)

        if rms > self.silence_rms:
            if not self._speaking:
                self._t_start = block.t     # first speech block of this utterance
            self._speaking = True
            self._silence_samples = 0
            self._buf.append(block.samples)
            self._utt_samples += n
            self._speech_samples += n
            self._t_end = t_block_end
        elif self._speaking:
            # trailing silence: keep a little tail, then decide to flush
            self._buf.append(block.samples)
            self._utt_samples += n
            self._silence_samples += n
            self._t_end = t_block_end
            if self._silence_samples >= self.end_silence_sec * self._rate:
                u = self._emit()
                if u is not None:
                    out.append(u)

        if self._speaking and self._utt_samples >= self.max_utt_sec * self._rate:
            u = self._emit()
            if u is not None:
                out.append(u)
        return out

    def flush(self) -> Optional[Utterance]:
        """End-of-stream: emit whatever speech is still buffered (or None)."""
        return self._emit()

    def _emit(self) -> Optional[Utterance]:
        utt: Optional[Utterance] = None
        rate = self._rate or 0
        if rate and self._buf and self._speech_samples >= self.min_utt_sec * rate:
            utt = Utterance(
                samples=np.concatenate(self._buf).astype(np.float32),
                rate=rate,
                t_start=self._t_start,
                t_end=self._t_end,
                speech_sec=self._speech_samples / rate,
            )
        self._buf = []
        self._utt_samples = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._speaking = False
        return utt
