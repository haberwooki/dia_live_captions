"""WASAPI loopback capture as an AudioSource (callback mode + blocking fallback)."""
from __future__ import annotations

import threading

import numpy as np
import pyaudiowpatch as pyaudio

from ..events import AudioBlock
from .base import AudioSource, BlockCallback, EndCallback


class WasapiLoopbackSource(AudioSource):
    """Callback (non-blocking) loopback capture. Primary path: a callback that
    stops firing is detectable by the consumer's watchdog and tears down cleanly."""

    is_live = True

    def __init__(self, dev: dict, block_sec: float = 0.1):
        self.name = dev["name"]
        self.index = dev["index"]
        self.rate = int(dev["defaultSampleRate"])
        self.channels = int(dev["maxInputChannels"])
        self._block_frames = int(self.rate * block_sec)
        self._pa = None
        self._stream = None
        self._t = 0.0

    def start(self, on_block: BlockCallback, on_end: EndCallback) -> None:
        self._pa = pyaudio.PyAudio()
        ch = self.channels

        def _cb(in_data, frame_count, time_info, status):
            try:
                s = np.frombuffer(in_data, dtype=np.int16)
                if ch > 1:
                    s = s.reshape(-1, ch).mean(axis=1)
                samples = s.astype(np.float32) / 32768.0
                t = self._t
                self._t += len(samples) / self.rate
                on_block(AudioBlock(samples=samples, rate=self.rate, t=t))
            except Exception as e:  # never let the audio callback die silently
                on_end(e)
                return (None, pyaudio.paComplete)
            return (None, pyaudio.paContinue)

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16, channels=ch, rate=self.rate,
                frames_per_buffer=self._block_frames, input=True,
                input_device_index=self.index, stream_callback=_cb,
            )
            self._stream.start_stream()
        except Exception as e:
            on_end(e)

    def stop(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        finally:
            if self._pa is not None:
                self._pa.terminate()


class BlockingWasapiSource(AudioSource):
    """Blocking-read fallback (--blocking). Documented interim: if the endpoint
    delivers nothing, stream.read() blocks in C and this thread can't be torn
    down cleanly (the watchdog still reports it, but the thread leaks to exit)."""

    is_live = True

    def __init__(self, dev: dict, block_sec: float = 0.1):
        self.name = dev["name"]
        self.index = dev["index"]
        self.rate = int(dev["defaultSampleRate"])
        self.channels = int(dev["maxInputChannels"])
        self._block_frames = int(self.rate * block_sec)
        self._stop = threading.Event()
        self._thread = None
        self._t = 0.0

    def start(self, on_block: BlockCallback, on_end: EndCallback) -> None:
        self._thread = threading.Thread(target=self._run, args=(on_block, on_end), daemon=True)
        self._thread.start()

    def _run(self, on_block: BlockCallback, on_end: EndCallback) -> None:
        pa = pyaudio.PyAudio()
        ch = self.channels
        frames = self._block_frames
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=self.rate,
                             frames_per_buffer=frames, input=True,
                             input_device_index=self.index)
        except Exception as e:
            on_end(e)
            pa.terminate()
            return
        err = None
        try:
            while not self._stop.is_set():
                data = stream.read(frames, exception_on_overflow=False)
                s = np.frombuffer(data, dtype=np.int16)
                if ch > 1:
                    s = s.reshape(-1, ch).mean(axis=1)
                samples = s.astype(np.float32) / 32768.0
                t = self._t
                self._t += len(samples) / self.rate
                on_block(AudioBlock(samples=samples, rate=self.rate, t=t))
        except Exception as e:
            err = e
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            on_end(err)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
