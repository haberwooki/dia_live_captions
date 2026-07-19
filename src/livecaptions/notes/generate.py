"""Ask a model to summarise a saved session, and keep the answer.

The hard part is not the summary, it is the to-dos: a model asked for action
items will happily produce them for a meeting that had none. So the schema makes
every to-do carry a verbatim quote, the prompt says an empty list is a correct
answer, and an unclear owner must be null rather than the nearest name.

Nothing is sent without `consented=True`. That gate is a parameter rather than a
convention so a future GUI cannot skip it by forgetting to ask.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from ..store.naming import build_transcript, session_labels

#: Summaries need the whole arc of a session, so this is larger than naming's
#: budget — but still a cap: the user is shown the exact size before sending.
MAX_TRANSCRIPT_CHARS = 120_000

_SYSTEM = """\
You are writing notes for someone who missed a meeting, from an automatic \
speech-recognition transcript. Lines are labelled with the speaker \
(SPEAKER_00, or a real name if one has been assigned).

Summarise what was actually said. Do not infer the meeting's purpose, the \
company, the product, or anyone's role from context and then write notes about \
that -- if the transcript only covers ten minutes of one topic, the notes cover \
ten minutes of one topic. ASR output contains mishearings; where a passage is \
garbled, leave it out rather than guessing at it.

A to-do belongs in the list only if someone committed to doing something: "I'll \
send the draft on Friday", "can you chase legal?" followed by agreement. \
Suggestions nobody took up, hypotheticals ("we could rewrite it"), and topics \
merely discussed are NOT to-dos. Each to-do must carry `evidence`: the verbatim \
quote it comes from, copied from the transcript, not paraphrased. If nothing \
was committed to, return an empty list -- that is a correct answer, and far \
better than a plausible invention.

Set `owner` to the speaker label or name of the person who will do the work, \
exactly as it appears in the transcript. If the transcript does not make the \
owner clear, set owner to null. Null is a correct answer; the nearest name is \
not. Note that the person who asks ("Sarah, can you take this?") is usually not \
the owner -- the person who agrees is.

`decisions` are things settled in the session ("we're going with option B"), not \
open questions. `key_points` are the substance someone would need to follow what \
happened, in the order they came up. Keep every field in the transcript's own \
language."""


class ToDo(BaseModel):
    text: str = Field(description="The action, phrased as a task: 'Send the draft to legal'")
    owner: Optional[str] = Field(
        description="Speaker label or name of whoever committed to it, exactly as it "
        "appears in the transcript, or null if the transcript does not make it clear"
    )
    evidence: str = Field(
        description="Verbatim quote from the transcript in which this was committed to"
    )


class SessionNotes(BaseModel):
    """What the model returns. This is the schema sent to the provider, so it
    holds nothing we know locally (see StoredNotes for that).

    The three lists have no defaults on purpose. With `default_factory=list` they
    drop out of the schema's `required`, and on the plain json_object fallback
    path (which local models take whenever they reject strict json_schema) a
    reply of just {"summary": "..."} would validate — silently giving the user
    notes with no to-dos that look exactly like a session where nothing was
    committed to. Required means the model has to say [] explicitly, so an empty
    list is an answer rather than an omission."""

    summary: str = Field(description="A few sentences covering what happened, in prose")
    key_points: List[str] = Field(
        description="The substance of the session, in the order it came up; [] if none")
    todos: List[ToDo] = Field(
        description="Only things someone committed to doing; [] if nobody did")
    decisions: List[str] = Field(
        description="Things settled in the session; [] if nothing was settled")


class StoredNotes(SessionNotes):
    """Notes plus the provenance the UI needs: when, by which model, and whether
    the model saw the whole session. Subclassing keeps these out of the schema
    the model is asked to fill in — it cannot claim a session wasn't truncated."""

    session_id: int
    generated_at: str = ""
    model_label: str = ""
    truncated: bool = False


class ConsentRequired(RuntimeError):
    """Raised instead of sending a transcript the user never agreed to send."""


class NotesNotStored(RuntimeError):
    """The model answered but the answer could not be written to the store.

    Carries the notes in `.notes` so the caller can still show, copy or export a
    result that may have cost money — the alternative is throwing away a paid
    call because of a database problem."""

    def __init__(self, message: str, notes: "StoredNotes"):
        super().__init__(message)
        self.notes = notes


class NotesUnreadable(RuntimeError):
    """A stored row exists but cannot be decoded. Deliberately not reported as
    'no notes': that would hide the corruption behind an offer to generate."""


_TABLE = "session_notes"

#: One statement per execute(). NOT executescript(): that issues an implicit
#: COMMIT first, and store.db hands out ONE connection shared with the live
#: TranscriptWriter — so merely reading notes during a recording would commit
#: the writer's in-flight transaction.
_SCHEMA_STATEMENTS = (
    f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
    session_id   INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    generated_at TEXT NOT NULL,
    model_label  TEXT NOT NULL DEFAULT '',
    truncated    INTEGER NOT NULL DEFAULT 0,
    notes_json   TEXT NOT NULL
)""",
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Own table, created on demand: this feature is optional and must not make
    store.db's schema depend on it.

    Guarded on a catalogue lookup so the common case is a pure read, and never
    commits: the caller's transaction is the caller's to end."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)
    ).fetchone()
    if exists:
        return
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)


def save_notes(conn: sqlite3.Connection, notes: StoredNotes) -> None:
    """Write (or replace) the notes for a session. Regenerating overwrites, so a
    session never accumulates stale competing versions."""
    ensure_schema(conn)
    payload = SessionNotes.model_validate(notes.model_dump()).model_dump_json()
    try:
        conn.execute(
            "INSERT INTO session_notes (session_id, generated_at, model_label, truncated,"
            " notes_json) VALUES (?,?,?,?,?)"
            " ON CONFLICT(session_id) DO UPDATE SET generated_at=excluded.generated_at,"
            " model_label=excluded.model_label, truncated=excluded.truncated,"
            " notes_json=excluded.notes_json",
            (notes.session_id, notes.generated_at, notes.model_label,
             int(notes.truncated), payload),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        # Almost always the foreign key: the session was deleted while the model
        # was thinking. The notes themselves are fine, so hand them back.
        raise NotesNotStored(
            f"The notes were generated but could not be saved: session "
            f"{notes.session_id} is no longer in the transcript store ({e}). "
            f"Copy or export them now — they will not be there after a restart.",
            notes,
        ) from e


def load_notes(conn: sqlite3.Connection, session_id: int) -> Optional[StoredNotes]:
    ensure_schema(conn)
    row = conn.execute("SELECT * FROM session_notes WHERE session_id=?",
                       (session_id,)).fetchone()
    if row is None:
        return None
    try:
        stored = json.loads(row["notes_json"])
    except (ValueError, TypeError) as e:
        raise NotesUnreadable(
            f"The saved notes for session {session_id} are damaged and cannot be "
            f"read ({e}). Generating them again will replace them.") from e
    try:
        return StoredNotes(
            session_id=session_id,
            generated_at=row["generated_at"],
            model_label=row["model_label"],
            truncated=bool(row["truncated"]),
            **stored,
        )
    except ValidationError as e:
        raise NotesUnreadable(
            f"The saved notes for session {session_id} are not in a shape this "
            f"version understands ({e.error_count()} problem(s)). Generating them "
            f"again will replace them.") from e


def delete_notes(conn: sqlite3.Connection, session_id: int) -> bool:
    ensure_schema(conn)
    cur = conn.execute("DELETE FROM session_notes WHERE session_id=?", (session_id,))
    conn.commit()
    return cur.rowcount > 0


def _build_request(conn: sqlite3.Connection, session_id: int,
                   max_chars: int) -> tuple[str, str, bool]:
    """The exact two messages that go to the provider, so the number the user
    consents to and the number actually sent come from the same code."""
    transcript, truncated = build_transcript(conn, session_id, max_chars=max_chars)
    speakers = session_labels(conn, session_id)
    who = ", ".join(speakers) if speakers else "(the transcript is not split by speaker)"
    user = (f"Speakers in this transcript: {who}\n"
            + ("\nNote: this is only the earlier part of a longer session.\n"
               if truncated else "")
            + f"\nTranscript:\n{transcript}")
    return transcript, user, truncated


def payload_chars(conn: sqlite3.Connection, session_id: int,
                  max_chars: int = MAX_TRANSCRIPT_CHARS) -> int:
    """Everything that will be sent, not just the transcript: the system prompt
    is ~1.6 KB and the speaker header grows with the session. A consent dialog
    that understates the payload is asking for consent to something else."""
    _, user, _ = _build_request(conn, session_id, max_chars)
    return len(_SYSTEM) + len(user)


def privacy_notice(config, conn: sqlite3.Connection, session_id: int,
                   max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """The sentence to show before asking for consent: how much goes where, and
    whether the model will see the whole session."""
    from ..llm.providers import describe_privacy

    _, user, truncated = _build_request(conn, session_id, max_chars)
    notice = describe_privacy(config, len(_SYSTEM) + len(user))
    if truncated:
        notice += ("\n\n(This session is longer than the limit, so only its earlier "
                   "part is sent — later to-dos may be missed.)")
    return notice


#: Fields a reply must actually contain. Validation already enforces this for
#: providers that validate, but a provider handing back a model_construct()ed or
#: hand-built object would sail past it, and "the model said nothing" must never
#: reach the user looking like "there were no to-dos".
_REQUIRED_OF_MODEL = ("summary", "key_points", "todos", "decisions")


def _reject_partial_reply(result) -> None:
    from ..llm.providers import LLMError

    supplied = getattr(result, "model_fields_set", None)
    if supplied is None:
        return
    missing = [f for f in _REQUIRED_OF_MODEL if f not in supplied]
    if missing:
        raise LLMError(
            "The model's reply left out " + ", ".join(missing) + ". No notes were "
            "saved: an answer with missing sections cannot be told apart from a "
            "session where there genuinely was nothing to record.")


def generate_notes(conn: sqlite3.Connection, session_id: int, provider, *,
                   consented: bool = False,
                   max_chars: int = MAX_TRANSCRIPT_CHARS,
                   store: bool = True) -> StoredNotes:
    """Summarise a session with `provider` and (by default) save the result.

    `consented` must be exactly True: the caller has told the user how much text
    is going where and the user agreed. Anything else raises before a single
    character is read, let alone sent. Raises LLMError from the provider if the
    model fails or answers in the wrong shape.
    """
    if consented is not True:
        raise ConsentRequired(
            "Notes were not generated: this sends transcript text to the AI "
            "provider, which needs your explicit agreement first.")

    transcript, user, truncated = _build_request(conn, session_id, max_chars)
    if not transcript.strip():
        raise ValueError(f"Session {session_id} has no transcript to summarise.")

    result = provider.complete(_SYSTEM, user, SessionNotes)
    _reject_partial_reply(result)

    notes = StoredNotes(
        session_id=session_id,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        model_label=getattr(provider, "label", ""),
        truncated=truncated,
        **SessionNotes.model_validate(result.model_dump()).model_dump(),
    )
    if store:
        save_notes(conn, notes)
    return notes


def to_markdown(notes: StoredNotes, title: str = "") -> str:
    """Render notes for export or the clipboard. Sections with nothing in them
    say so rather than vanishing — an empty to-do list is a real result here, and
    silently omitting it reads like the feature failed."""
    head = title or f"Session {notes.session_id}"
    out = [f"# Notes — {head}", ""]

    provenance = [p for p in (notes.generated_at, notes.model_label) if p]
    if provenance:
        out += ["*Generated " + " · ".join(provenance) + "*", ""]
    if notes.truncated:
        out += ["> Only the earlier part of this session was sent to the model.", ""]

    out += ["## Summary", "", notes.summary.strip() or "_No summary._", ""]

    out += ["## Key points", ""]
    out += [f"- {p}" for p in notes.key_points] or ["_None recorded._"]
    out.append("")

    out += ["## To-dos", ""]
    if notes.todos:
        for t in notes.todos:
            out.append(f"- [ ] {t.text} — **{t.owner or 'Unassigned'}**")
            if t.evidence:
                out.append(f"  > {t.evidence}")
    else:
        out.append("_No one committed to anything in this session._")
    out.append("")

    out += ["## Decisions", ""]
    out += [f"- {d}" for d in notes.decisions] or ["_None recorded._"]

    return "\n".join(out) + "\n"
