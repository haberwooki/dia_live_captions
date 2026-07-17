"""The TranscriptionSource seam.

Every backend that produces captions implements this: local Whisper now, cloud
(M4) and Discord (M5) later. `start(on_event)` begins emitting on a background
thread and invokes `on_event` from that thread; sinks must be thread-safe or
marshal to their own thread (the Qt overlay will). `stop()` tears down cleanly
and must not call `on_event` afterward.
"""
from __future__ import annotations

import abc
from typing import Callable

from ..events import TranscriptEvent

EventCallback = Callable[[TranscriptEvent], None]


class TranscriptionSource(abc.ABC):
    #: identifies this stream on every event it emits ("loopback", "mic", ...)
    source_id: str = "unknown"

    @abc.abstractmethod
    def start(self, on_event: EventCallback) -> None:
        """Begin emitting events to `on_event` (called from a background thread)."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop and release all resources; no events after this returns."""
