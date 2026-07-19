"""Optional audio saving — keep a session's audio so it can be re-diarized later.

Live diarization is best-effort (Streaming Sortformer, one mixed stream, decided
in real time); an offline pass over the same audio is much better. That is only
possible if the audio still exists, so this writes the 16 kHz mono stream the ASR
already produced straight to a WAV beside the transcript store, named by session
id so a transcript row can find it.

OFF by default (``save_audio``). Keeping audio is a privacy decision the user has
not already made by keeping transcripts — the words were being stored, the voices
were not — and a disk decision at ~115 MB/hour, hence the ``audio_max_mb`` cap.

Crash safety — the header is kept CURRENT while recording. stdlib
``Wave_write.writeframes`` re-patches the RIFF/data sizes after every write, and
the seek it does to patch them flushes the preceding frames out of Python's
buffer first. So the bytes a size field describes are always already on disk: a
killed process leaves a header that at worst UNDER-counts the frames present,
which every reader tolerates, and can never over-count and cause a truncated
read.

Under-counting is still lost audio, though — a header stale by one patch reports
a shorter file than exists, and a header killed before its first patch reports
zero. So ``find_session_audio`` repairs a stale header IN PLACE before handing
the path out. That is deliberately not "repair on open": the repair happens once,
at the single lookup every consumer goes through, and what they then open is an
ordinary WAV. ffmpeg, Audacity and a future offline pass still need to know
nothing about it. The repair only ever grows the counts to match the bytes
actually on disk, so it cannot manufacture a truncated read.

ORPHAN AUDIO — deleting a transcript does NOT delete its recording. Nothing in
this module is wired to transcript deletion; ``delete_session_audio(session_id)``
exists for that and currently has no callers, so a user who deletes a private
conversation from the Transcripts tab still has its raw voices on disk. Until the
Transcripts tab calls it, ``delete_all_audio`` in Settings is the only way out.
"""
from __future__ import annotations

import struct
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

#: How long stop() will wait for an in-flight write before leaving the close to
#: the writer thread. Long enough for any healthy disk, short enough that a hung
#: one does not freeze the GUI thread that called stop().
_STOP_IO_WAIT_SEC = 0.25


def audio_dir(base: Path | str | None = None, settings=None) -> Path:
    """Where session audio lives: beside the transcript store, since the two are
    only useful together. DB_PATH is read at CALL time, not bound at import (same
    reason as store.db.connect) so tests and a future 'store location' setting
    redirect the audio with the transcripts instead of writing to the real one.

    `settings.audio_save_dir` overrides it. This must stay in agreement with the
    Speakers tab, which locates a session's WAV the same way — when the two
    disagreed, re-diarization simply reported "no saved audio" forever.
    """
    if base is not None:
        return Path(base)
    if settings is not None:
        configured = str(getattr(settings, "audio_save_dir", "") or "").strip()
        if configured:
            return Path(configured)
    from ..store.db import DB_PATH
    return Path(DB_PATH).parent / AUDIO_DIRNAME


def session_audio_path(session_id: int, base: Path | str | None = None,
                       settings=None) -> Path:
    """The one place a session's audio filename is spelled. Both the recorder and
    the Speakers tab go through here — when each had its own idea of the name, the
    tab reported "no saved audio" for files sitting on disk."""
    return audio_dir(base, settings) / f"session_{int(session_id)}.wav"


def repair_wav_header(path: Path | str) -> int:
    """Grow a stale RIFF/data size to cover the frames actually on disk.

    A process killed between a write and its header patch leaves a WAV that opens
    fine but reports too few frames — the audio is there, no reader can see it.
    Returns the number of frames recovered (0 when the header was already right,
    or when the file is not one of ours to touch).

    Conservative on purpose: it only accepts the canonical 44-byte PCM layout this
    module writes, and only ever raises the counts, never lowers them, so it can
    never make a header claim more audio than the file holds.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
        if size <= HEADER_BYTES:
            return 0
        with open(path, "r+b") as fh:
            head = fh.read(HEADER_BYTES)
            if len(head) < HEADER_BYTES:
                return 0
            if head[0:4] != b"RIFF" or head[8:12] != b"WAVE":
                return 0
            # fmt chunk of 16 bytes immediately followed by data is what `wave`
            # emits for PCM; anything else may have chunks after `data`, and then
            # the trailing bytes are not audio.
            if head[12:16] != b"fmt " or head[16:20] != struct.pack("<I", 16):
                return 0
            if head[36:40] != b"data":
                return 0
            block_align = struct.unpack_from("<H", head, 32)[0]
            declared = struct.unpack_from("<I", head, 40)[0]
            if block_align <= 0:
                return 0
            available = size - HEADER_BYTES
            actual = available - (available % block_align)   # ignore a torn frame
            if actual <= declared:
                return 0
            fh.seek(4)
            fh.write(struct.pack("<I", HEADER_BYTES - 8 + actual))
            fh.seek(40)
            fh.write(struct.pack("<I", actual))
            fh.flush()
        return (actual - declared) // block_align
    except (OSError, struct.error):
        # A locked, read-only or half-written file is still readable at its stale
        # length; failing the lookup over it would be worse than the short read.
        return 0


def find_session_audio(session_id: int, base: Path | str | None = None,
                       settings=None) -> Optional[Path]:
    """The audio for a session, or None if it was never saved or has been deleted.

    Repairs a stale header on the way out (module docstring) so every consumer
    gets the full recording without knowing a kill happened.
    """
    path = session_audio_path(session_id, base, settings)
    if not path.is_file():
        return None
    repair_wav_header(path)
    return path


def delete_session_audio(session_id: int, base: Path | str | None = None,
                         settings=None) -> Tuple[bool, str]:
    """Delete one session's audio. Returns (removed, error).

    ``error`` is empty when the file was removed or was already gone, and carries
    the reason when deletion actually failed — a file locked by a player, a
    read-only drive. Reported rather than printed because print() goes nowhere in
    the windowed PyInstaller build, which would make "Delete" look like it worked.
    """
    path = session_audio_path(session_id, base, settings)
    try:
        path.unlink()
        return True, ""
    except FileNotFoundError:
        return False, ""
    except OSError as e:
        return False, f"Couldn't delete {path.name}: {e}"


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
    """16 kHz MONO samples to PCM16 bytes. Raises on anything else — write()
    catches it (see SessionRecorder.write)."""
    a = np.asarray(samples)
    if a.ndim == 2 and 1 in a.shape:
        a = a.reshape(-1)                # sounddevice hands mono back as (N, 1)
    if a.ndim != 1:
        # Interleaved multi-channel written as mono plays back at N times speed and
        # is not what the ASR transcribed, so it is refused rather than downmixed.
        raise ValueError(f"expected mono audio, got shape {np.shape(samples)}")
    if a.dtype == np.int16:
        return a.tobytes()
    # Clip before scaling: an over-driven block (auto-gain can push past 1.0) would
    # otherwise wrap from full-scale positive to full-scale negative and record as
    # a loud click in audio the user may later listen to.
    return (np.clip(a.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class SessionRecorder:
    """Streams 16 kHz mono PCM to a WAV as chunks arrive — nothing is buffered in
    RAM beyond the current chunk, because an hour of session is ~115 MB.

    It never raises at the caller — not on a full disk, not on a reached cap, not
    on a chunk it cannot convert. Any of those stops the recording and reports
    why: losing the optional audio must not take the captions down with it. The
    caller is the AUDIO THREAD, which has no exception handler of its own.
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
        # Two locks, because they are held for very different lengths of time.
        # _state guards the flags and the frame count and is never held across I/O,
        # so stop() from the GUI thread can always take it. _io guards the file
        # handle and IS held across a write that a stalled disk can hang for
        # seconds; stop() only ever waits on it with a timeout.
        self._state = threading.Lock()
        self._io = threading.Lock()
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
        with self._state:
            if self.stopped:
                return False
            size = self.size_bytes

        try:
            data = _to_pcm16(samples)
        except Exception as e:
            # Conversion is inside the guarantee too: a wrong-shaped or non-numeric
            # chunk must not raise into the audio thread just because it failed
            # before reaching the disk.
            self._finish(f"Audio saving stopped — unusable audio chunk: {e}")
            return False
        if not data:
            return True
        if size + len(data) > self._max_bytes:
            self._finish(f"Audio saving stopped at the {self._cap_desc} limit "
                         f"({format_bytes(size)} saved). Captions continue.")
            return False

        with self._io:
            if self._wav is None or self._fh is None:
                return False
            try:
                self._wav.writeframes(data)     # also re-patches the header (see module docstring)
                self._fh.flush()                # ...and get both out of Python's buffer
            except Exception as e:
                # Anything at all — full disk, removed drive, RIFF overflow. Audio is
                # the optional half of this app; it stops quietly, captures keep going.
                self._mark_stopped(f"Audio saving stopped — write failed: {e}")
                self._close_locked()
                return False
            with self._state:
                self._frames += len(data) // SAMPLE_WIDTH
                handover = self.stopped
            if handover:
                # stop() gave up waiting on this write and left the file to us.
                self._close_locked()
                return False
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
        cap or an error already stopped it.

        Called from the GUI thread, so it never blocks behind an in-flight write:
        the recording is marked stopped immediately, and if the disk is hung the
        close is handed to the writer, which does it as soon as its write returns.
        A file left open that way is still valid on disk — the header is current
        (module docstring) — and find_session_audio repairs it if it is not.
        """
        self._mark_stopped(reason or "Recording finished.")
        self._close_files(timeout=_STOP_IO_WAIT_SEC)

    def _mark_stopped(self, reason: str) -> None:
        with self._state:
            if self.stopped:
                return           # keep the FIRST reason: the cap/error, not "finished"
            self.stopped = True
            self.stop_reason = reason

    def _finish(self, reason: str) -> None:
        self._mark_stopped(reason)
        self._close_files()

    def _close_files(self, timeout: Optional[float] = None) -> bool:
        """Close the handles. False if the writer held _io longer than `timeout`."""
        if not self._io.acquire(timeout=timeout if timeout is not None else -1):
            return False
        try:
            self._close_locked()
        finally:
            self._io.release()
        return True

    def _close_locked(self) -> None:
        wav, fh, self._wav, self._fh = self._wav, self._fh, None, None
        try:
            if wav is not None:
                wav.close()                # final header patch
        except Exception:
            pass                 # an unpatchable header is still a readable file
        try:
            if fh is not None:
                fh.close()
        except Exception:
            pass

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
            self._finish(f"Couldn't record to {self.path.name}: {e}")

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def __del__(self):
        # A recorder dropped without stop() is otherwise finalised in whatever order
        # the collector picks — `wave` flushing a handle that was already closed,
        # printing an ignored exception. Closing here keeps the WAV tidy regardless.
        try:
            self._mark_stopped("Recording dropped without stop().")
            self._close_files(timeout=_STOP_IO_WAIT_SEC)
        except Exception:
            pass
