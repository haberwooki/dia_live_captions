"""TranscriptWriter — persists finalized captions off the hot path.

It's a sink: hand it `TranscriptEvent`s and it batches them onto a dedicated
writer thread. Nothing here ever runs on the audio callback, the segmenter, or
the Qt GUI thread — a slow disk must never stall capture or the overlay.

Only FINAL events are stored; partials are transient by definition.
"""
from __future__ import annotations

import queue
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..events import TranscriptEvent
from .db import DB_PATH, connect

_FLUSH_SEC = 1.0          # write at least this often
_FLUSH_BATCH = 25         # ...or as soon as this many are queued


class TranscriptWriter:
    def __init__(self, path: Path | str = DB_PATH, *, source: str = "loopback",
                 title: Optional[str] = None):
        self._path = path
        self._source = source
        self._title = title
        self._q: "queue.Queue[Optional[TranscriptEvent]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._conn: Optional[sqlite3.Connection] = None
        self.session_id: Optional[int] = None
        self.count = 0
        self._ready = threading.Event()

    # ---- lifecycle ----
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def on_event(self, event: TranscriptEvent) -> None:
        """Sink entry point — cheap, never blocks the caller."""
        if event.is_final and event.text.strip():
            self._q.put(event)

    def stop(self, timeout: float = 5.0) -> None:
        self._q.put(None)                     # drain-then-exit sentinel
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ---- writer thread ----
    def _run(self) -> None:
        try:
            self._conn = connect(self._path)
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at, source, title) VALUES (?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), self._source, self._title))
            self.session_id = cur.lastrowid
            self._conn.commit()
        except Exception as e:
            print(f"(transcript store unavailable: {e})")
            self._ready.set()
            return
        self._ready.set()

        pending: List[TranscriptEvent] = []
        closing = False
        while not closing:
            try:
                item = self._q.get(timeout=_FLUSH_SEC)
                if item is None:
                    closing = True
                else:
                    pending.append(item)
                    if len(pending) < _FLUSH_BATCH:
                        continue
            except queue.Empty:
                pass
            if pending:
                self._flush(pending)
                pending = []

        self._close_session()

    def _flush(self, events: List[TranscriptEvent]) -> None:
        rows: List[Tuple] = [
            (self.session_id, e.t_start, e.t_end,
             datetime.now().isoformat(timespec="seconds"), e.speaker, e.text.strip())
            for e in events
        ]
        try:
            self._conn.executemany(
                "INSERT INTO utterances (session_id, t_start, t_end, wall_clock, speaker, text)"
                " VALUES (?, ?, ?, ?, ?, ?)", rows)
            self._conn.commit()
            self.count += len(rows)
        except Exception as e:
            print(f"(transcript write failed: {e})")

    def _close_session(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute("UPDATE sessions SET ended_at=? WHERE id=?",
                               (datetime.now().isoformat(timespec="seconds"), self.session_id))
            self._conn.commit()
        finally:
            self._conn.close()
