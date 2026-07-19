"""Optional audio saving — keep a session's audio so it can be re-diarized later.

Live diarization is best-effort (Streaming Sortformer, one mixed stream, decided
in real time); an offline pass over the same audio is much better. That is only
possible if the audio still exists, so this writes the 16 kHz mono stream the ASR
already produced straight to a WAV beside the transcript store, named by session
id so a transcript row can find it.

OFF by default (``save_audio``). Keeping audio is a privacy decision the user has
not already made by keeping transcripts — the words were being stored, the voices
were not — and a disk decision at ~115 MB/hour, hence the ``audio_max_mb`` cap.

Crash safety — the header is kept CURRENT rather than repaired on open. stdlib
``Wave_write.writeframes`` re-patches the RIFF/data sizes after every write, and
the seek it does to patch them flushes the preceding frames out of Python's
buffer first. So the bytes a size field describes are always already on disk: a
killed process leaves a header that at worst UNDER-counts the frames present,
which every reader tolerates, and can never over-count and cause a truncated
read. Repair-on-open was the alternative and was rejected because it would
oblige every reader — ffmpeg, Audacity, a future offline pass — to know about
the repair before the file is usable.
"""
from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

#: Everything upstream is resampled to 16 kHz mono float32 before it reaches the
#: ASR (sources/streaming_local.py); recording that same buffer means the saved
#: audio is exactly what was transcribed, and needs no resampling here.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2                 # 16-bit PCM: half the size of float32, and Whisper
                                 # itself only ever saw 16-bit-grade audio anyway
HEADER_BYTES = 44                # canonical PCM RIFF header, what `wave` writes
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH
BYTES_PER_HOUR = BYTES_PER_SEC * 3600        # ~110 MiB; quote this in the UI
AUDIO_DIRNAME = "audio"

#: A RIFF size field is 32-bit, so a WAV cannot exceed 4 GiB (~37 h here) however
#: generous the user's cap is. Stop cleanly at that wall instead of letting struct
#: raise mid-session.
_RIFF_LIMIT_BYTES = 0xFFFFFFFF - 36


def audio_dir(base: Path | str | None = None) -> Path:
    """Where session audio lives: beside the transcript store, since the two are
    only useful together. DB_PATH is read at CALL time, not bound at import (same
    reason as store.db.connect) so tests and a future 'store location' setting
    redirect the audio with the transcripts instead of writing to the real one."""
    if base is not None:
        return Path(base)
    from ..store.db import DB_PATH
    return Path(DB_PATH).parent / AUDIO_DIRNAME


def session_audio_path(session_id: int, base: Path | str | None = None) -> Path:
    return audio_dir(base) / f"session_{int(session_id)}.wav"


def find_session_audio(session_id: int, base: Path | str | None = None) -> Optional[Path]:
    """The audio for a session, or None if it was never saved or has been deleted."""
    path = session_audio_path(session_id, base)
    return path if path.is_file() else None


def delete_session_audio(session_id: int, base: Path | str | None = None) -> bool:
    """Delete one session's audio. True if a file was actually removed."""
    path = session_audio_path(session_id, base)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        print(f"(couldn't delete {path.name}: {e})")
        return False


def audio_files(base: Path | str | None = None) -> List[Path]:
    try:
        return sorted(p for p in audio_dir(base).glob("session_*.wav") if p.is_file())
    except OSError:
        return []


def total_audio_bytes(base: Path | str | None = None) -> int:
    """Disk used by saved audio — what the user needs to see before deciding."""
    total = 0
    for path in audio_files(base):
        try:
            total += path.stat().st_size
        except OSError:
            pass                 # deleted between listing and stat; just don't count it
    return total


def delete_all_audio(base: Path | str | None = None) -> Tuple[int, int]:
    """Delete every saved recording. Returns (files removed, bytes freed)."""
    files = bytes_freed = 0
    for path in audio_files(base):
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        files += 1
        bytes_freed += size
    return files, bytes_freed


def format_bytes(n: int) -> str:
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def _to_pcm16(samples) -> bytes:
    a = np.asarray(samples)
    if a.dtype == np.int16:
        return a.tobytes()
    # Clip before scaling: an over-driven block (auto-gain can push past 1.0) would
    # otherwise wrap from full-scale positive to full-scale negative and record as
    # a loud click in audio the user may later listen to.
    return (np.clip(a.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class SessionRecorder:
    """Streams 16 kHz mono PCM to a WAV as chunks arrive — nothing is buffered in
    RAM beyond the current chunk, because an hour of session is ~115 MB.

    It never raises at the caller. A full disk or a reached cap stops the
    recording and reports why: losing the optional audio must not take the
    captions down with it.
    """

    def __init__(self, path: Path | str, *, session_id: Optional[int] = None,
                 rate: int = SAMPLE_RATE, max_mb: float = 0.0):
        self.path = Path(path)
        self.session_id = session_id
        self.rate = int(rate)
        self.max_mb = float(max_mb or 0.0)
        cap = int(self.max_mb * 1024 * 1024)
        if 0 < cap <= _RIFF_LIMIT_BYTES:
            self._max_bytes, self._cap_desc = cap, f"{self.max_mb:g} MB"
        else:
            self._max_bytes, self._cap_desc = _RIFF_LIMIT_BYTES, "4 GB (WAV format)"
        self._lock = threading.Lock()
        self._frames = 0
        self._fh = None
        self._wav: Optional[wave.Wave_write] = None
        self.stopped = False
        self.stop_reason = ""
        self._open()

    @classmethod
    def from_settings(cls, settings, session_id: Optional[int],
                      *, base: Path | str | None = None) -> Optional["SessionRecorder"]:
        """The recorder for this session, or None when audio saving is off — which
        is the default. getattr keeps this working before the settings fields land."""
        if not getattr(settings, "save_audio", False):
            return None
        if session_id is None:
            return None          # no session row to attach it to, so nothing to name it
        return cls(session_audio_path(session_id, base), session_id=session_id,
                   max_mb=float(getattr(settings, "audio_max_mb", 2048) or 0))

    # ---- writing ----
    def write(self, samples) -> bool:
        """Append a chunk of 16 kHz mono audio (float32 in [-1, 1], or int16).

        Returns False once recording has stopped — the caller carries on
        transcribing either way, and can read `stop_reason` to tell the user.
        """
        with self._lock:
            if self.stopped or self._wav is None:
                return False
            data = _to_pcm16(samples)
            if not data:
                return True
            if self.size_bytes + len(data) > self._max_bytes:
                self._shut(f"Audio saving stopped at the {self._cap_desc} limit "
                           f"({format_bytes(self.size_bytes)} saved). Captions continue.")
                return False
            try:
                self._wav.writeframes(data)     # also re-patches the header (see module docstring)
                self._fh.flush()                # ...and get both out of Python's buffer
            except Exception as e:
                # Anything at all — full disk, removed drive, RIFF overflow. Audio is
                # the optional half of this app; it stops quietly, captures keep going.
                self._shut(f"Audio saving stopped — write failed: {e}")
                return False
            self._frames += len(data) // SAMPLE_WIDTH
            return True

    # ---- state ----
    @property
    def size_bytes(self) -> int:
        """Bytes this recording occupies on disk, header included."""
        return HEADER_BYTES + self._frames * SAMPLE_WIDTH

    @property
    def duration_sec(self) -> float:
        return self._frames / self.rate if self.rate else 0.0

    @property
    def size_label(self) -> str:
        return format_bytes(self.size_bytes)

    # ---- lifecycle ----
    def stop(self, reason: str = "") -> None:
        """Finish the file. Safe to call repeatedly, from any thread, and after a
        cap or an error already stopped it."""
        with self._lock:
            self._shut(reason or "Recording finished.")

    def _shut(self, reason: str) -> None:
        if self.stopped:
            return               # keep the FIRST reason: the cap/error, not "finished"
        self.stopped = True
        self.stop_reason = reason
        try:
            if self._wav is not None:
                self._wav.close()          # final header patch
        except Exception:
            pass                 # an unpatchable header is still a readable file
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        self._wav = self._fh = None

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # The file handle is ours, not wave's, so closing is explicit and ordered.
            self._fh = open(self.path, "wb")
            self._wav = wave.open(self._fh, "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(SAMPLE_WIDTH)
            self._wav.setframerate(self.rate)
            # Commit an empty-but-valid header now. `wave` otherwise writes the header
            # on the first writeframes SIZED FOR THAT CHUNK, which is the one moment it
            # could claim more data than is on disk; forcing it out at length zero means
            # every later size is written only after the frames it counts (module docstring).
            self._wav.writeframes(b"")
            self._fh.flush()
        except Exception as e:
            self._shut(f"Couldn't record to {self.path.name}: {e}")

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def __del__(self):
        # A recorder dropped without stop() is otherwise finalised in whatever order
        # the collector picks — `wave` flushing a handle that was already closed,
        # printing an ignored exception. Closing here keeps the WAV tidy regardless.
        try:
            self._shut("Recording dropped without stop().")
        except Exception:
            pass
