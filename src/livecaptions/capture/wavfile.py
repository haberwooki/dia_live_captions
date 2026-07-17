"""Replay a 16-bit PCM WAV as an AudioSource (the deterministic test rig)."""
from __future__ import annotations

import os
import threading
import time
import wave

import numpy as np

from ..events import AudioBlock
from .base import AudioSource, BlockCallback, EndCallback


class WavFileSource(AudioSource):
    is_live = False

    def __init__(self, path: str, block_sec: float = 0.1, paced: bool = True):
        try:
            with wave.open(path, "rb") as w:
                self.rate = w.getframerate()
                self._channels = w.getnchannels()
                sampwidth = w.getsampwidth()
                self._raw = w.readframes(w.getnframes())
        except (OSError, wave.Error, EOFError) as e:
            raise SystemExit(f"{path}: cannot read as a WAV file ({e})")
        if sampwidth != 2:
            raise SystemExit(f"{path}: need a 16-bit PCM WAV (sample width {sampwidth} bytes)")
        self.name = f"WAV:{os.path.basename(path)}"
        self._block_frames = max(1, int(self.rate * block_sec))
        self._block_sec = block_sec
        self._paced = paced
        self._stop = threading.Event()
        self._thread = None

    def start(self, on_block: BlockCallback, on_end: EndCallback) -> None:
        self._thread = threading.Thread(target=self._run, args=(on_block, on_end), daemon=True)
        self._thread.start()

    def _run(self, on_block: BlockCallback, on_end: EndCallback) -> None:
        err = None
        try:
            data = np.frombuffer(self._raw, dtype=np.int16)
            if self._channels > 1:
                usable = (len(data) // self._channels) * self._channels
                data = data[:usable].reshape(-1, self._channels).mean(axis=1)
            data = data.astype(np.float32) / 32768.0
            n = self._block_frames
            t = 0.0
            for i in range(0, len(data), n):
                if self._stop.is_set():
                    break
                chunk = data[i:i + n]
                on_block(AudioBlock(samples=chunk, rate=self.rate, t=t))
                t += len(chunk) / self.rate
                if self._paced:
                    time.sleep(self._block_sec)
        except Exception as e:
            err = e
        finally:
            on_end(err)   # always release the consumer (clean EOF or error)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
