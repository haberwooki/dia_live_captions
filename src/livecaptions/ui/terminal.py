"""Terminal sink: prints captions, a live VU meter, and audio-health warnings.

`on_event(TranscriptEvent)` is the sink; `on_block(rms)` is the per-block monitor
feed; a background watchdog thread prints the two-signal audio-health warnings
and dropped-block notices. The live VU line is suppressed on a non-tty
(redirected output) so logs stay clean.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from typing import Callable, Optional

from ..events import TranscriptEvent


class StatusLine:
    def __init__(self):
        self._tty = sys.stdout.isatty()
        self._len = 0

    def status(self, text: str) -> None:
        if not self._tty:
            return
        pad = max(0, self._len - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self._len = len(text)

    def message(self, text: str) -> None:
        if self._tty and self._len:
            sys.stdout.write("\r" + " " * self._len + "\r")
            self._len = 0
        print(text, flush=True)


def _vu_meter(rms: float, name: str) -> str:
    dbfs = 20 * math.log10(max(rms, 1e-6) / 32768.0)
    width = 28
    level = int(max(0, min(width, (dbfs + 60) / 60 * width)))
    bar = "#" * level + "-" * (width - level)
    return f"  [{bar}] {dbfs:6.1f} dBFS  listening: {name[:34]}"


class TerminalUI:
    def __init__(self, *, source_name: str, is_live: bool, silence_rms_floor: float,
                 no_blocks_warn_sec: float, silence_warn_sec: float,
                 get_dropped: Optional[Callable[[], int]] = None):
        self._status = StatusLine()
        self._name = source_name
        self._is_live = is_live
        self._floor = silence_rms_floor
        self._no_blocks = no_blocks_warn_sec
        self._silence_warn = silence_warn_sec
        self._get_dropped = get_dropped
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wd_thread = None
        self._last_block = 0.0
        self._last_sound = 0.0
        self._warned_idle = False
        self._warned_silent = False
        self._reported_drops = 0

    def start(self) -> None:
        if self._is_live:
            now = time.monotonic()
            self._last_block = now
            self._last_sound = now
            self._wd_thread = threading.Thread(target=self._watchdog, daemon=True)
            self._wd_thread.start()

    def on_event(self, event: TranscriptEvent) -> None:
        speaker = f"{event.speaker}: " if event.speaker else ""
        if not event.is_final:
            # streaming partial: show in place on the status line
            self._status.status(f"  … {speaker}{event.text}"[-100:])
            return
        lag = f", {event.infer_lag:.2f}s to transcribe" if event.infer_lag else ""
        stamp = time.strftime("%H:%M:%S")
        self._status.message(f"[{stamp}] {speaker}{event.text}    ({event.duration:.1f}s audio{lag})")

    def on_block(self, rms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_block = now
            self._warned_idle = False
            if rms > self._floor:
                self._last_sound = now
                self._warned_silent = False
        if self._is_live:
            self._status.status(_vu_meter(rms, self._name))

    def message(self, text: str) -> None:
        self._status.message(text)

    def _watchdog(self) -> None:
        while not self._stop.wait(0.25):
            now = time.monotonic()
            with self._lock:
                last_block, last_sound = self._last_block, self._last_sound
                warned_idle, warned_silent = self._warned_idle, self._warned_silent
            if now - last_block > self._no_blocks:
                if not warned_idle:
                    self._status.message(
                        f"(no audio from '{self._name}' in {self._no_blocks:.0f}s - "
                        f"idle/phantom endpoint? play some audio, or pick another with "
                        f"--list-devices / --device)")
                    with self._lock:
                        self._warned_idle = True
            elif now - last_sound > self._silence_warn and not warned_silent:
                self._status.message(
                    f"(audio from '{self._name}' but near-silent for {self._silence_warn:.0f}s - "
                    f"wrong or muted output device?)")
                with self._lock:
                    self._warned_silent = True
            if self._get_dropped is not None:
                d = self._get_dropped()
                if d - self._reported_drops >= 50:
                    self._reported_drops = d
                    self._status.message(f"([behind] dropped {d} audio blocks - "
                                         f"inference slower than realtime)")

    def stop(self) -> None:
        self._stop.set()
        if self._wd_thread is not None:
            self._wd_thread.join(timeout=1)
