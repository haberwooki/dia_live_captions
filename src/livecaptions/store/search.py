"""Full-text search over saved transcripts.

FTS5 with BM25 ranking and snippet() highlighting when available; a LIKE scan
otherwise (some SQLite builds ship without FTS5).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from .db import has_fts


@dataclass
class Hit:
    utterance_id: int
    session_id: int
    wall_clock: str
    t_start: float
    speaker: Optional[str]
    text: str
    snippet: str
    score: float          # lower is better with bm25(); 0.0 in the LIKE fallback


def search(conn: sqlite3.Connection, query: str, *, speaker: Optional[str] = None,
           since: Optional[str] = None, limit: int = 20) -> List[Hit]:
    """`query` is an FTS5 MATCH expression (supports "quoted phrases", AND/OR/NOT)."""
    where, params = [], []
    if speaker:
        where.append("u.speaker = ?")
        params.append(speaker)
    if since:
        where.append("u.wall_clock >= ?")
        params.append(since)

    if has_fts(conn):
        sql = """
            SELECT u.id, u.session_id, u.wall_clock, u.t_start, u.speaker, u.text,
                   snippet(utterances_fts, 0, '[', ']', '...', 12) AS snip,
                   bm25(utterances_fts) AS score
            FROM utterances_fts f
            JOIN utterances u ON u.id = f.rowid
            WHERE utterances_fts MATCH ?
        """
        args = [query] + params
        if where:
            sql += " AND " + " AND ".join(where)
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
    else:
        sql = """
            SELECT u.id, u.session_id, u.wall_clock, u.t_start, u.speaker, u.text,
                   u.text AS snip, 0.0 AS score
            FROM utterances u
            WHERE u.text LIKE ?
        """
        args = [f"%{query}%"] + params
        if where:
            sql += " AND " + " AND ".join(where)
        sql += " ORDER BY u.id DESC LIMIT ?"
        args.append(limit)

    rows = conn.execute(sql, args).fetchall()
    return [Hit(r["id"], r["session_id"], r["wall_clock"], r["t_start"],
                r["speaker"], r["text"], r["snip"], float(r["score"])) for r in rows]


def rename_speaker(conn: sqlite3.Connection, old: str, new: str,
                   session_id: Optional[int] = None) -> int:
    """Rename a speaker label (e.g. SPEAKER_00 -> 'Sarah').

    Only touches `utterances.speaker` — because FTS5 is external-content over the
    text column, this costs no reindex. Returns rows changed. Reversible: just
    rename back.
    """
    sql = "UPDATE utterances SET speaker=? WHERE speaker=?"
    params = [new, old]
    if session_id is not None:
        sql += " AND session_id=?"
        params.append(session_id)
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.rowcount


def recent_sessions(conn: sqlite3.Connection, limit: int = 10) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT s.*, COUNT(u.id) AS utterances,
                  (SELECT COUNT(DISTINCT speaker) FROM utterances
                    WHERE session_id = s.id AND speaker IS NOT NULL) AS speakers
           FROM sessions s LEFT JOIN utterances u ON u.session_id = s.id
           GROUP BY s.id ORDER BY s.id DESC LIMIT ?""", (limit,)).fetchall()


def delete_session(conn: sqlite3.Connection, session_id: int) -> int:
    """Delete a session and its lines. Returns the number of lines removed.

    Deletes the utterances explicitly rather than leaning on ON DELETE CASCADE:
    the full-text index is maintained by AFTER DELETE triggers on `utterances`,
    and whether those fire for cascade-deleted rows depends on pragmas that are
    easy to change by accident. Doing it directly means a deleted session cannot
    keep turning up in search.
    """
    n = conn.execute("DELETE FROM utterances WHERE session_id=?", (session_id,)).rowcount
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    return int(n or 0)
