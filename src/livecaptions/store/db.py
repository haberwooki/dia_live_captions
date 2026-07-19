"""Schema and connections for the transcript store.

Design notes:
  * WAL mode so the writer thread never blocks readers (search while capturing).
  * FTS5 as an EXTERNAL-CONTENT table over `utterances`: the text is stored once,
    and renaming a speaker touches only `utterances.speaker` with zero reindex.
  * Triggers keep the index in sync with inserts/updates/deletes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import platformdirs

DB_PATH = Path(platformdirs.user_data_dir("live-captions", appauthor=False)) / "transcripts.db"

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,          -- ISO-8601 local time
    ended_at    TEXT,
    source      TEXT,                   -- e.g. 'loopback', 'WAV:clip.wav'
    title       TEXT
);

CREATE TABLE IF NOT EXISTS utterances (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_start     REAL NOT NULL,          -- seconds within the session
    t_end       REAL NOT NULL,
    wall_clock  TEXT NOT NULL,          -- ISO-8601
    speaker     TEXT,                   -- diarization label, or a real name once assigned
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_session ON utterances(session_id);
CREATE INDEX IF NOT EXISTS idx_utt_speaker ON utterances(speaker);
"""

# Only created when the SQLite build actually has FTS5 (probed at runtime).
# Search degrades to LIKE without it.
_FTS_SCHEMA = """
-- external-content FTS: text lives in `utterances`, this is just the index
CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
    text,
    content='utterances',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS utt_ai AFTER INSERT ON utterances BEGIN
    INSERT INTO utterances_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS utt_ad AFTER DELETE ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS utt_au AFTER UPDATE OF text ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO utterances_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def connect(path: Path | str | None = None, *, create: bool = True) -> sqlite3.Connection:
    """Open the transcript store. `path` resolves to DB_PATH at CALL time, not at
    import time: a default argument would bind the module value once, so anything
    redirecting DB_PATH (a test, a future 'store location' setting) would be
    silently ignored and write to the user's real transcripts instead."""
    path = Path(path if path is not None else DB_PATH)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")     # readers never block the writer
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if create:
        init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_BASE_SCHEMA)
    if fts5_available(conn):
        conn.executescript(_FTS_SCHEMA)
    conn.commit()


def has_fts(conn: sqlite3.Connection) -> bool:
    """True when the full-text index exists (else search falls back to LIKE)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='utterances_fts'").fetchone()
    return row is not None
