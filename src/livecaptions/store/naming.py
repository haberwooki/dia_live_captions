"""Propose real names for diarization labels (SPEAKER_00 -> "Sarah") via Claude.

This is the one part of live-captions that is NOT local: it sends transcript text
to the Anthropic API. So it is opt-in per invocation, gated on explicit consent,
and every proposal must be confirmed by a human before anything is written. The
rename itself goes through store.search.rename_speaker, which is reversible.

Credentials resolve the normal SDK way -- ANTHROPIC_API_KEY, or an `ant auth
login` profile. We never read a key from config.toml: the repo should not be a
place secrets can land.
"""
from __future__ import annotations

import sqlite3
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

#: Names are usually established in the first few minutes ("Hi, this is Mike"),
#: so when a session is too big to send we keep the head rather than sampling.
MAX_TRANSCRIPT_CHARS = 60_000

_SYSTEM = """\
You are labelling speakers in a meeting transcript produced by automatic speech \
recognition and diarization. The diarizer assigned anonymous labels (SPEAKER_00, \
SPEAKER_01, ...). Your job is to work out each label's real name.

Only assign a name you can support with a verbatim quote from the transcript -- \
someone introducing themselves ("this is Mike"), being addressed ("Sarah, can you \
take this one?"), or being referred to unambiguously. Do not infer names from \
topic, role, accent, speaking style, or how much someone talks.

If there is no supporting quote for a label, return name=null for it. A missing \
name is a correct answer; a guessed one is not. The transcript is ASR output, so \
names may be misspelled -- use the spelling that appears most often.

Watch for the off-by-one trap: a name spoken BY a label usually belongs to a \
DIFFERENT label (the person being addressed). The speaker of "Thanks, Sarah" is \
generally not Sarah."""


class SpeakerName(BaseModel):
    label: str = Field(description="The diarization label, e.g. SPEAKER_00")
    name: Optional[str] = Field(
        description="Real name, or null if the transcript does not support one"
    )
    confidence: Literal["high", "medium", "low"]
    evidence: str = Field(
        description="Verbatim quote from the transcript supporting the name, "
        "or an explanation of why no name could be determined"
    )


class SpeakerNames(BaseModel):
    speakers: List[SpeakerName]


def session_labels(conn: sqlite3.Connection, session_id: int) -> List[str]:
    rows = conn.execute(
        "SELECT DISTINCT speaker FROM utterances "
        "WHERE session_id=? AND speaker IS NOT NULL ORDER BY speaker",
        (session_id,),
    ).fetchall()
    return [r["speaker"] for r in rows]


def build_transcript(conn: sqlite3.Connection, session_id: int,
                     max_chars: int = MAX_TRANSCRIPT_CHARS) -> tuple[str, bool]:
    """Render the session as 'SPEAKER_00: text' lines. Returns (text, truncated)."""
    rows = conn.execute(
        "SELECT speaker, text FROM utterances WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    lines, total, truncated = [], 0, False
    for r in rows:
        line = f"{r['speaker'] or 'UNKNOWN'}: {r['text']}"
        if total + len(line) + 1 > max_chars:
            truncated = True
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines), truncated


def propose_names(transcript: str, labels: List[str], *,
                  model: str = "claude-opus-4-8", provider=None) -> List[SpeakerName]:
    """Ask a model to name each label. Caller must already have consent.

    `provider` comes from llm.providers, so this works with Claude, any
    OpenAI-compatible API, or a local model. Defaults to Claude to keep the CLI
    behaving as it did.
    """
    if provider is None:
        from ..llm.providers import AnthropicProvider, ProviderConfig
        provider = AnthropicProvider(ProviderConfig(kind="anthropic", model=model))

    result = provider.complete(
        _SYSTEM,
        (f"Labels to identify: {', '.join(labels)}\n\n"
         f"Transcript:\n{transcript}\n\n"
         f"Return one entry per label listed above."),
        SpeakerNames,
    )
    proposed = {s.label: s for s in result.speakers}
    # Trust our own label list over the model's -- never drop or invent a label.
    return [
        proposed.get(label, SpeakerName(label=label, name=None, confidence="low",
                                        evidence="(model returned no entry for this label)"))
        for label in labels
    ]
