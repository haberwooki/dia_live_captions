"""Session notes: a summary, key points, decisions and to-dos for a saved session.

Like speaker naming, this is one of the few features that can send transcript
text off the machine, so it carries the same rules: nothing is sent without the
caller passing explicit consent, and the model is told to ground every to-do in
a verbatim quote rather than invent plausible-sounding action items.

Notes are stored in their own table (see generate.ensure_schema) keyed by
session, so they survive a restart and can be regenerated at any time.
"""
from .generate import (
    ConsentRequired,
    SessionNotes,
    StoredNotes,
    ToDo,
    delete_notes,
    ensure_schema,
    generate_notes,
    load_notes,
    privacy_notice,
    save_notes,
    to_markdown,
)

__all__ = [
    "ConsentRequired",
    "SessionNotes",
    "StoredNotes",
    "ToDo",
    "delete_notes",
    "ensure_schema",
    "generate_notes",
    "load_notes",
    "privacy_notice",
    "save_notes",
    "to_markdown",
]
