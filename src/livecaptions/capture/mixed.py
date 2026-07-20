"""Capture the microphone alongside system audio, so your own words get captioned.

WASAPI loopback carries only what Windows PLAYS. In a call that is everyone except
you — your voice goes straight out of the microphone and is never rendered locally.
This mixes the two capture devices into one stream so a single transcription pass
covers the whole conversation.

Mixing rather than transcribing twice is deliberate: one Whisper pass costs what it
already costs, and attribution does not need a second one. We know which DEVICE each
sound arrived on, so a line where the microphone was the loud one is yours with
certainty — better than the diarizer can do for anyone else, since it is measurement
rather than inference. Per-block levels ride along on AudioBlock.levels.

Clock drift: two devices free-run on their own clocks and slowly diverge. This tracks
the system stream as the clock and takes whatever microphone audio has arrived,
padding with silence or dropping the oldest surplus. Over an hour that is a fraction
of a second of alignment error inside a block — invisible to a recogniser working on
whole utterances, and far cheaper than resampling one stream onto the other's clock.
"""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from ..events import AudioBlock
from .base import AudioSource, BlockCallback, EndCallback

#: Cap on buffered microphone audio. If the mic runs ahead (or the consumer stalls)
#: we drop the OLDEST audio: stale mic samples would be mixed against the wrong
#: moment of system audio and mis-attribute a line.
_MAX_MIC_BACKLOG_SEC = 2.0


def default_input_device(p) -> Optional[dict]:
    """The microphone Windows is currently set to use, or None."""
    import pyaudiowpatch as pyaudio
    try:
        info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        dev = p.get_device_info_by_index(info["defaultInputDevice"])
        return dev if dev.get("maxInputChannels", 0) > 0 else None
    except Exception:
        return None


def input_devices(p) -> list:
    """Microphones, excluding loopback endpoints (which are outputs in disguise)."""
    import pyaudiowpatch as pyaudio
    out = []
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except Exception:
        return out
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if (d.get("hostApi") == wasapi["index"] and d.get("maxInputChannels", 0) > 0
                and "[Loopback]" not in d.get("name", "")):
            out.append(d)
    return out


def _to_mono(raw: bytes, channels: int) -> np.ndarray:
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


class MixedSource(AudioSource):
    """System audio + microphone, mixed, with per-device levels on every block.

    Presents the same interface as WasapiLoopbackSource, so the rest of the
    pipeline neither knows nor cares that two devices are involved.
    """

    def __init__(self, sys_dev: dict, mic_dev: dict, block_sec: float = 0.1,
                 mic_gain: float = 1.0):
        self.name = f"{sys_dev['name']} + {mic_dev['name']}"
        self.rate = int(sys_dev["defaultSampleRate"])
        self.mic_name = mic_dev["name"]
        self._sys_dev, self._mic_dev = sys_dev, mic_dev
        self._mic_gain = float(mic_gain)
        self._block_frames = int(self.rate * block_sec)
        self._mic_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._mic_buf = np.zeros(0, dtype=np.float32)
        self._t = 0.0
        self._pa = None
        self._sys_stream = None
        self._mic_stream = None
        self._lock = threading.Lock()

    # ---- microphone side: buffer whatever arrives, mixed against system blocks ----
    def _mic_cb(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio
        ch = min(2, int(self._mic_dev["maxInputChannels"]))
        self._mic_q.put(_to_mono(in_data, ch))
        return (None, pyaudio.paContinue)

    def _take_mic(self, n: int) -> np.ndarray:
        """n frames of microphone audio aligned with the current system block."""
        while not self._mic_q.empty():
            try:
                self._mic_buf = np.concatenate([self._mic_buf, self._mic_q.get_nowait()])
            except queue.Empty:
                break
        cap = int(_MAX_MIC_BACKLOG_SEC * self.rate)
        if self._mic_buf.size > cap:
            self._mic_buf = self._mic_buf[-cap:]        # keep the most recent audio
        if self._mic_buf.size >= n:
            out, self._mic_buf = self._mic_buf[:n], self._mic_buf[n:]
            return out
        out = np.zeros(n, dtype=np.float32)
        out[: self._mic_buf.size] = self._mic_buf       # short: pad, never stall
        self._mic_buf = np.zeros(0, dtype=np.float32)
        return out

    def start(self, on_block: BlockCallback, on_end: EndCallback) -> None:
        import pyaudiowpatch as pyaudio
        self._pa = pyaudio.PyAudio()

        sys_ch = min(2, int(self._sys_dev["maxInputChannels"]))

        def sys_cb(in_data, frame_count, time_info, status):
            try:
                system = _to_mono(in_data, sys_ch)
                mic = self._take_mic(system.size) * self._mic_gain
                # Average rather than sum: two loud sources would otherwise clip,
                # and clipping is far worse for recognition than being 6 dB quieter.
                mixed = (system + mic) * 0.5
                with self._lock:
                    t = self._t
                    self._t += system.size / self.rate
                on_block(AudioBlock(samples=mixed, rate=self.rate, t=t,
                                    levels={"system": _rms(system), "mic": _rms(mic)}))
            except Exception as e:
                on_end(e)
                return (None, pyaudio.paComplete)
            return (None, pyaudio.paContinue)

        # The microphone opens FIRST: if it fails (in use, revoked permission) the
        # caller can fall back to system-only rather than losing captions entirely.
        self._mic_stream = self._pa.open(
            format=pyaudio.paInt16, channels=min(2, int(self._mic_dev["maxInputChannels"])),
            rate=self.rate, input=True, input_device_index=self._mic_dev["index"],
            frames_per_buffer=self._block_frames, stream_callback=self._mic_cb)
        self._mic_stream.start_stream()

        self._sys_stream = self._pa.open(
            format=pyaudio.paInt16, channels=sys_ch, rate=self.rate, input=True,
            input_device_index=self._sys_dev["index"],
            frames_per_buffer=self._block_frames, stream_callback=sys_cb)
        self._sys_stream.start_stream()

    def stop(self) -> None:
        for stream in (self._sys_stream, self._mic_stream):
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
        self._sys_stream = self._mic_stream = None
        try:
            if self._pa is not None:
                self._pa.terminate()
        finally:
            self._pa = None
