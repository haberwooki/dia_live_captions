"""LocalAgreement-2 hypothesis buffer for streaming ASR.

Re-implementation of the LocalAgreement-2 policy from ufal/whisper_streaming
(Dominik Macháček et al., MIT license): commit the longest common prefix that
two consecutive Whisper hypotheses agree on, so committed text never rewrites,
while the disagreeing tail stays "unconfirmed" (a live partial). This is a
clean reimplementation of that algorithm, adapted to feed word timestamps from
faster-whisper.

A "word" here is a (start, end, text) tuple with times in GLOBAL stream seconds.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

Word = Tuple[float, float, str]


class HypothesisBuffer:
    def __init__(self):
        self.committed_in_buffer: List[Word] = []   # everything committed so far
        self.buffer: List[Word] = []                # previous hypothesis' unconfirmed tail
        self.new: List[Word] = []                   # incoming hypothesis (post-filter)
        self.last_committed_time: float = 0.0
        self.last_committed_word: Optional[str] = None

    def insert(self, words: List[Word], offset: float) -> None:
        """Offer a fresh hypothesis (word times relative to the current buffer;
        `offset` converts them to global time). Drops words at or before what's
        already committed, and de-duplicates a repeated n-gram at the boundary."""
        shifted = [(a + offset, b + offset, t) for a, b, t in words]
        self.new = [(a, b, t) for a, b, t in shifted if a > self.last_committed_time - 0.1]

        if not self.new:
            return
        a0 = self.new[0][0]
        if abs(a0 - self.last_committed_time) < 1 and self.committed_in_buffer:
            cn = len(self.committed_in_buffer)
            nn = len(self.new)
            for i in range(1, min(cn, nn, 5) + 1):   # up to a 5-gram overlap
                tail_committed = " ".join(self.committed_in_buffer[-i:][j][2] for j in range(i))
                head_new = " ".join(self.new[j][2] for j in range(i))
                if tail_committed == head_new:
                    del self.new[:i]
                    break

    def flush(self) -> List[Word]:
        """Commit the longest common prefix of the new hypothesis and the
        previous tail; return the newly committed words (may be empty)."""
        committed: List[Word] = []
        while self.new and self.buffer:
            if self.new[0][2] == self.buffer[0][2]:
                w = self.new.pop(0)
                committed.append(w)
                self.last_committed_word = w[2]
                self.last_committed_time = w[1]
                self.buffer.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, upto_time: float) -> None:
        """Forget committed words that end before `upto_time` (bounds memory
        after the audio buffer is trimmed)."""
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= upto_time:
            self.committed_in_buffer.pop(0)
