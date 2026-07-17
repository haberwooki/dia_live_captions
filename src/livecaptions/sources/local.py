"""Local transcription source: AudioSource -> Segmenter -> WhisperWorker.

Threads:
  * the AudioSource's capture/callback thread just enqueues blocks (minimal work);
  * a segmenter thread drains blocks, runs the RMS segmenter and the monitor
    callback, and submits utterances;
  * the WhisperWorker thread runs inference (off the audio thread).
`finished` is set once the audio stream has ended AND the worker has drained.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from ..asr.segmenter import Segmenter
from ..asr.whisper import WhisperWorker
from ..capture.base import AudioSource
from ..events import AudioBlock
from ..util import drop_oldest_put
from .base import EventCallback, TranscriptionSource

MonitorCallback = Callable[[float], None]   # receives per-block RMS (int16 units)


class LocalTranscriptionSource(TranscriptionSource):
    def __init__(self, audio: AudioSource, model, settings, *, source_id: str = "loopback"):
        self.source_id = source_id
        self._audio = audio
        self._seg = Segmenter(
            silence_rms=settings.silence_rms,
            end_silence_sec=settings.end_silence_sec,
            min_utt_sec=settings.min_utt_sec,
            max_utt_sec=settings.max_utt_sec,
        )
        self._worker = WhisperWorker(
            model, language=settings.language, beam_size=settings.beam_size, source_id=source_id)
        self._block_q: "queue.Queue[Optional[AudioBlock]]" = queue.Queue(
            maxsize=max(1, int(20 / settings.block_sec)))
        self._monitor: Optional[MonitorCallback] = None
        self._seg_thread = None
        self._audio_error: Optional[BaseException] = None
        self.dropped_blocks = 0
        self.finished = threading.Event()

    def start(self, on_event: EventCallback, monitor: Optional[MonitorCallback] = None) -> None:
        self._monitor = monitor
        self._worker.start(on_event)
        self._seg_thread = threading.Thread(target=self._segment_loop, daemon=True)
        self._seg_thread.start()
        self._audio.start(self._enqueue_block, self._on_audio_end)

    def _enqueue_block(self, block: AudioBlock) -> None:
        if drop_oldest_put(self._block_q, block):
            self.dropped_blocks += 1

    def _on_audio_end(self, error: Optional[BaseException]) -> None:
        self._audio_error = error
        self._block_q.put(None)   # tell the segmenter loop the stream ended

    def _segment_loop(self) -> None:
        while True:
            block = self._block_q.get()
            if block is None:
                break
            rms = Segmenter.block_rms(block.samples)
            if self._monitor is not None:
                self._monitor(rms)
            for utt in self._seg.push(block):
                self._worker.submit(utt)
        # stream ended: transcribe any trailing speech, then drain + stop worker
        tail = self._seg.flush()
        if tail is not None:
            self._worker.submit(tail)
        self._worker.close()
        self._worker.join()
        if self._audio_error is not None:
            print(f"Audio capture failed: {self._audio_error}")
        self.finished.set()

    def stop(self) -> None:
        self._audio.stop()          # stop producing blocks
        self._block_q.put(None)     # ensure the segmenter loop terminates
