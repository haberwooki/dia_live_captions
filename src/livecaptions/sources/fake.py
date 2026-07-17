"""A source that emits canned events on a timer — for wiring/UI tests with no
audio device and no GPU. Proves the seam with a second real implementation."""
from __future__ import annotations

import threading
from typing import List, Optional

from ..events import TranscriptEvent
from .base import EventCallback, TranscriptionSource

_DEFAULT_LINES = [
    "hello from the fake source",
    "the seam works without any audio or gpu",
    "this is the third and final canned caption",
]


class FakeTranscriptionSource(TranscriptionSource):
    source_id = "fake"

    def __init__(self, lines: Optional[List[str]] = None, interval: float = 0.5):
        self._lines = list(lines) if lines is not None else list(_DEFAULT_LINES)
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.finished = threading.Event()

    def start(self, on_event: EventCallback, monitor=None) -> None:
        self._thread = threading.Thread(target=self._run, args=(on_event,), daemon=True)
        self._thread.start()

    def _run(self, on_event: EventCallback) -> None:
        t = 0.0
        for line in self._lines:
            if self._stop.wait(self._interval):   # respects stop() promptly
                break
            on_event(TranscriptEvent(text=line, source=self.source_id,
                                     t_start=t, t_end=t + self._interval))
            t += self._interval
        self.finished.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
