"""The AudioSource seam — a producer of AudioBlocks."""
from __future__ import annotations

import abc
from typing import Callable, Optional

from ..events import AudioBlock

BlockCallback = Callable[[AudioBlock], None]
#: called once when the stream ends: None on a clean end-of-stream (WAV EOF),
#: or the exception that stopped a live capture.
EndCallback = Callable[[Optional[BaseException]], None]


class AudioSource(abc.ABC):
    is_live: bool = True     # False for finite sources (WAV) that end on their own
    name: str = "unknown"
    rate: int = 16000

    @abc.abstractmethod
    def start(self, on_block: BlockCallback, on_end: EndCallback) -> None:
        """Begin producing AudioBlocks (from a background/callback thread)."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop producing and release the device/thread cleanly."""
