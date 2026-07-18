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
    # Cut only once the buffer is half again over the cap — see _clamp_buffer.
    HARD_CAP_FACTOR = 1.5

    def __init__(self, model, *, language, beam_size, max_buffer_sec: float = 15.0):
        self._model = model
        self._language = language
        self._beam = beam_size
        self._max_buffer = max_buffer_sec
        self.buffer = np.zeros(0, dtype=np.float32)   # 16 kHz mono
        self.offset = 0.0                              # global time of buffer[0], seconds
        self.hyp = HypothesisBuffer()
        self.dropped_sec = 0.0                         # audio discarded by the hard cap

    def insert_audio(self, audio16k: np.ndarray) -> None:
        self.buffer = np.append(self.buffer, audio16k.astype(np.float32))

    def buffer_sec(self) -> float:
        return len(self.buffer) / WHISPER_SR

    def _transcribe_words(self) -> List[Word]:
        # temperature=0.0 disables faster-whisper's fallback loop, which re-decodes a
        # cut up to 6 times when it looks "bad". Measured: it fired on ~10.7% of cuts,
        # turning a 0.79 s pass into 7-9.5 s of frozen overlay, and every time it fired
        # the extra passes produced the same text. The fallback decodes are also
        # sampled (nondeterministic), which works against LocalAgreement-2 — it commits
        # only what two consecutive passes agree on. The batch worker keeps the default.
        segments, _ = self._model.transcribe(
            self.buffer, language=self._language, beam_size=self._beam, temperature=0.0,
            word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
        words: List[Word] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append((float(w.start), float(w.end), w.word))
        return words

    def process(self) -> Tuple[List[Word], List[Word]]:
        """One streaming step. Returns (newly_committed, unconfirmed_tail)."""
        self._clamp_buffer()          # BEFORE the decode: clamping after it would leave
        words = self._transcribe_words()   # a steady state of cap + one pass of audio
        self.hyp.insert(words, self.offset)
        committed = self.hyp.flush()
        self._maybe_trim()
        return committed, list(self.hyp.buffer)

    def _maybe_trim(self) -> None:
        """Once the buffer is too long, drop audio up to the last committed word."""
        if self.buffer_sec() > self._max_buffer:
            self.trim_to_committed()

    def _clamp_buffer(self) -> None:
        """Enforce max_buffer as a HARD cap, independent of what has committed.

        trim_to_committed() cannot be relied on to do this. It sets
        `offset = cut_time`, so on the very next call `rel == 0`, `n == 0`, and the
        `0 < n` guard is dead until a NEW word commits. When nothing commits — exactly
        what happens once decoding falls behind — the buffer grows without bound and
        every pass re-decodes more audio, feeding back into longer passes and
        multi-second frozen captions. Measured buffers of 46/71/100 s against a 15 s
        cap before this existed.

        The hysteresis factor keeps this off the healthy path: a healthy run sits just
        over the cap between trims (measured 15.5 s against 15 s), and cutting there
        would eat live speech mid-utterance. This only fires in the runaway state — a
        state that is ALREADY dropping audio upstream via the drop-oldest block queue.
        """
        limit = self._max_buffer * self.HARD_CAP_FACTOR
        if self.buffer_sec() <= limit:
            return
        n = len(self.buffer) - int(self._max_buffer * WHISPER_SR)
        if n <= 0:
            return
        self.buffer = self.buffer[n:]
        # Arithmetic, NOT `= some hypothesis time`: the live diarizer aligns speaker
        # turns against these word timestamps, so the clock must stay exact.
        self.offset += n / WHISPER_SR
        self.dropped_sec += n / WHISPER_SR
        self.hyp.pop_committed(self.offset - 0.1)
        # Unconfirmed words now older than the audio can never be re-agreed with, and
        # would otherwise block the next LocalAgreement round on a guaranteed mismatch.
        self.hyp.buffer = [w for w in self.hyp.buffer if w[1] > self.offset]

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
        self.hyp = HypothesisBuffer()   # dropped_sec is cumulative: not reset here
