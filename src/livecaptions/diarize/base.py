"""The Diarizer seam — anything that turns audio into speaker turns."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class SpeakerTurn:
    """A span of audio attributed to one speaker (times in seconds)."""

    start: float
    end: float
    speaker: str        # e.g. "SPEAKER_00" — a label, not a name

    @property
    def duration(self) -> float:
        return self.end - self.start


class Diarizer(abc.ABC):
    name: str = "unknown"

    @abc.abstractmethod
    def diarize(self, audio16k: np.ndarray) -> List[SpeakerTurn]:
        """Return speaker turns for float32 mono 16 kHz audio, sorted by start."""
