"""A source that emits growing PARTIALS then a FINAL per utterance — used to
validate the overlay's in-place overwrite / reflow / scroll before real
streaming (M3) exists. Runs on a background thread like any source, so it also
exercises the thread -> GUI-thread marshaling."""
from __future__ import annotations

import threading
from typing import List, Optional

from ..events import TranscriptEvent
from .base import EventCallback, TranscriptionSource

_DEMO_UTTERANCES = [
    "the overlay renders partial captions in place",
    "each word arrives dimmed until the line is finalized",
    "then it commits solid and the next line begins",
    "long lines wrap and older lines scroll off the top of the caption bar",
]


class DemoTranscriptionSource(TranscriptionSource):
    source_id = "demo"

    def __init__(self, utterances: Optional[List[str]] = None,
                 word_delay: float = 0.16, gap: float = 0.9, loop: bool = False):
        self._utts = list(utterances) if utterances is not None else list(_DEMO_UTTERANCES)
        self._word_delay = word_delay
        self._gap = gap
        self._loop = loop
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.finished = threading.Event()

    def start(self, on_event: EventCallback, monitor=None) -> None:
        self._thread = threading.Thread(target=self._run, args=(on_event,), daemon=True)
        self._thread.start()

    def _emit_utterance(self, on_event: EventCallback, text: str, t: float) -> bool:
        words = text.split()
        for i in range(1, len(words) + 1):
            if self._stop.wait(self._word_delay):
                return False
            on_event(TranscriptEvent(
                text=" ".join(words[:i]), source=self.source_id,
                t_start=t, t_end=t + i * self._word_delay, is_final=False))
        on_event(TranscriptEvent(
            text=text, source=self.source_id,
            t_start=t, t_end=t + len(words) * self._word_delay, is_final=True))
        return not self._stop.wait(self._gap)

    def _run(self, on_event: EventCallback) -> None:
        t = 0.0
        while True:
            for utt in self._utts:
                if not self._emit_utterance(on_event, utt, t):
                    self.finished.set()
                    return
                t += len(utt.split()) * self._word_delay + self._gap
            if not self._loop:
                break
        self.finished.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
