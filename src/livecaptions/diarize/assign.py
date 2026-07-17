"""Attach speakers to transcribed words, and group them into caption lines.

This is the WhisperX-style pattern: the diarizer yields speaker turns, the ASR
yields word timestamps, and each word is assigned to the speaker whose turn
overlaps it most. Pure functions — no I/O, no models — so they're unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .base import SpeakerTurn

Word = Tuple[float, float, str]          # (start, end, text) — as emitted by our ASR


@dataclass
class LabeledWord:
    start: float
    end: float
    text: str
    speaker: Optional[str]


@dataclass
class SpeakerSegment:
    """Consecutive words from one speaker — one caption line."""

    start: float
    end: float
    speaker: Optional[str]
    text: str


def best_speaker(start: float, end: float, turns: Sequence[SpeakerTurn]) -> Optional[str]:
    """The speaker whose turn overlaps [start, end) most. Falls back to the
    nearest turn when a word lands in a diarization gap."""
    best: Optional[str] = None
    best_overlap = 0.0
    for t in turns:
        overlap = min(end, t.end) - max(start, t.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = t.speaker
    if best is not None or not turns:
        return best
    # no overlap at all — snap to the temporally nearest turn
    mid = (start + end) / 2.0
    nearest = min(turns, key=lambda t: 0.0 if t.start <= mid <= t.end
                  else min(abs(mid - t.start), abs(mid - t.end)))
    return nearest.speaker


def assign_speakers(words: Sequence[Word], turns: Sequence[SpeakerTurn]) -> List[LabeledWord]:
    return [LabeledWord(s, e, t, best_speaker(s, e, turns)) for (s, e, t) in words]


def group_into_segments(labeled: Sequence[LabeledWord],
                        max_gap: float = 1.0) -> List[SpeakerSegment]:
    """Merge consecutive same-speaker words into caption lines. A new line also
    starts on a long pause, so one speaker's monologue doesn't become one blob."""
    segments: List[SpeakerSegment] = []
    for w in labeled:
        if (segments
                and segments[-1].speaker == w.speaker
                and w.start - segments[-1].end <= max_gap):
            prev = segments[-1]
            segments[-1] = SpeakerSegment(prev.start, w.end, prev.speaker,
                                          f"{prev.text} {w.text.strip()}".strip())
        else:
            segments.append(SpeakerSegment(w.start, w.end, w.speaker, w.text.strip()))
    return segments


def format_transcript(segments: Sequence[SpeakerSegment]) -> str:
    """Human-readable speaker-labeled transcript."""
    lines = []
    for s in segments:
        stamp = f"{int(s.start // 60):02d}:{s.start % 60:05.2f}"
        who = s.speaker or "UNKNOWN"
        lines.append(f"[{stamp}] {who}: {s.text}")
    return "\n".join(lines)
