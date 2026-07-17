"""Export a saved session. Exports are derived views — the DB is the storage."""
from __future__ import annotations

import json
import sqlite3
from typing import List


def _rows(conn: sqlite3.Connection, session_id: int) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM utterances WHERE session_id=? ORDER BY t_start", (session_id,)).fetchall()


def _ts(seconds: float, sep: str = ",") -> str:
    h, rem = divmod(max(0.0, seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def to_srt(conn: sqlite3.Connection, session_id: int) -> str:
    out = []
    for i, r in enumerate(_rows(conn, session_id), 1):
        who = f"{r['speaker']}: " if r["speaker"] else ""
        out.append(f"{i}\n{_ts(r['t_start'])} --> {_ts(r['t_end'])}\n{who}{r['text']}\n")
    return "\n".join(out)


def to_vtt(conn: sqlite3.Connection, session_id: int) -> str:
    out = ["WEBVTT\n"]
    for r in _rows(conn, session_id):
        who = f"<v {r['speaker']}>" if r["speaker"] else ""
        out.append(f"{_ts(r['t_start'], '.')} --> {_ts(r['t_end'], '.')}\n{who}{r['text']}\n")
    return "\n".join(out)


def to_jsonl(conn: sqlite3.Connection, session_id: int) -> str:
    return "\n".join(json.dumps({
        "t_start": r["t_start"], "t_end": r["t_end"], "wall_clock": r["wall_clock"],
        "speaker": r["speaker"], "text": r["text"],
    }, ensure_ascii=False) for r in _rows(conn, session_id))


def to_markdown(conn: sqlite3.Connection, session_id: int) -> str:
    sess = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    out = [f"# Transcript — session {session_id}", ""]
    if sess:
        out += [f"*{sess['started_at']} · source: {sess['source'] or 'unknown'}*", ""]
    last = None
    for r in _rows(conn, session_id):
        who = r["speaker"] or "Unknown"
        if who != last:                      # group consecutive turns under one heading
            out.append(f"\n**{who}**")
            last = who
        out.append(f"- `{_ts(r['t_start'], '.')[:8]}` {r['text']}")
    return "\n".join(out) + "\n"


FORMATS = {"srt": to_srt, "vtt": to_vtt, "jsonl": to_jsonl, "md": to_markdown}


def export(conn: sqlite3.Connection, session_id: int, fmt: str) -> str:
    if fmt not in FORMATS:
        raise SystemExit(f"unknown export format {fmt!r} (use: {', '.join(FORMATS)})")
    return FORMATS[fmt](conn, session_id)
