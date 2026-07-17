"""Online (streaming) Whisper: a rolling audio buffer transcribed repeatedly,
with LocalAgreement-2 committing stable words. Inspired by ufal/whisper_streaming
(MIT), adapted to faster-whisper word timestamps.

Not thread-managed here — the StreamingTranscriptionSource drives it from a
worker thread. `insert_audio` appends 16 kHz mono audio; `process` runs one
transcription pass and returns (newly committed words, unconfirmed tail).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .hypothesis import HypothesisBuffer, Word

WHISPER_SR = 16000


class OnlineASR:
    def __init__(self, model, *, language, beam_size, max_buffer_sec: float = 15.0):
        self._model = model
        self._language = language
        self._beam = beam_size
        self._max_buffer = max_buffer_sec
        self.buffer = np.zeros(0, dtype=np.float32)   # 16 kHz mono
        self.offset = 0.0                              # global time of buffer[0], seconds
        self.hyp = HypothesisBuffer()

    def insert_audio(self, audio16k: np.ndarray) -> None:
        self.buffer = np.append(self.buffer, audio16k.astype(np.float32))

    def buffer_sec(self) -> float:
        return len(self.buffer) / WHISPER_SR

    def _transcribe_words(self) -> List[Word]:
        segments, _ = self._model.transcribe(
            self.buffer, language=self._language, beam_size=self._beam,
            word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
        words: List[Word] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append((float(w.start), float(w.end), w.word))
        return words

    def process(self) -> Tuple[List[Word], List[Word]]:
        """One streaming step. Returns (newly_committed, unconfirmed_tail)."""
        words = self._transcribe_words()
        self.hyp.insert(words, self.offset)
        committed = self.hyp.flush()
        self._maybe_trim()
        return committed, list(self.hyp.buffer)

    def _maybe_trim(self) -> None:
        """Once the buffer is too long, drop audio up to the last committed word."""
        if self.buffer_sec() > self._max_buffer:
            self.trim_to_committed()

    def trim_to_committed(self) -> None:
        """Drop audio up to the last committed word (its text is final, so we never
        re-decode it); keep the unconfirmed tail audio."""
        cut_time = self.hyp.last_committed_time
        rel = cut_time - self.offset
        n = int(rel * WHISPER_SR)
        if 0 < n < len(self.buffer):
            self.buffer = self.buffer[n:]
            self.offset = cut_time
            self.hyp.pop_committed(cut_time - 0.1)

    def reset(self, offset: float = 0.0) -> None:
        """Start a fresh segment (after a pause). `offset` is the global stream time
        of the next audio, so word timestamps stay on one monotonic clock — the live
        diarizer's speaker timeline is aligned against them."""
        self.buffer = np.zeros(0, dtype=np.float32)
        self.offset = offset
        self.hyp = HypothesisBuffer()
