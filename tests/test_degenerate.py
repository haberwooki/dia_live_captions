"""The repetition guard must catch stuck-decoder loops WITHOUT eating real speech.

JUNK_FINALS only ever matched 21 exact strings. Whisper's real failure mode on
music/noise is a repetition loop, which faster-whisper used to catch internally via
compression_ratio_threshold during temperature fallback — a fallback v0.1.7 disabled
because it froze the overlay for seconds. is_degenerate() is the replacement.

The false-positive tests matter more than the true-positive ones: dropping a caption
the user actually said is worse than letting one bad line through.
"""
import pytest

from livecaptions.sources.streaming_local import is_degenerate

DEGENERATE = [
    "Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay.",
    "the the the the the the the the the the the the",
    "Thank you. Thank you. Thank you. Thank you. Thank you. Thank you.",
    "I'm going to go. I'm going to go. I'm going to go. I'm going to go.",
    "la la la la la la la la la la la la la la la la",
    "Subscribe to my channel. Subscribe to my channel. Subscribe to my channel. "
    "Subscribe to my channel.",
]

REAL_SPEECH = [
    "",
    "Okay.",
    "yeah yeah yeah",                       # genuine emphasis, short
    "no no no, that's not what I meant",
    "very very good",
    "So the thing about the buffer is that it never actually gets trimmed.",
    "I think we should ship it, but only after the tests pass on Windows.",
    "one two three four five six seven eight nine ten",   # distinct tokens
    "That's what she said. That's actually pretty funny.",  # partial echo, not a loop
    "We need to talk about the the duplicated word there.",  # stutter, not a loop
    "It costs about fifty dollars, maybe fifty five, I can't remember exactly.",
]


@pytest.mark.parametrize("text", DEGENERATE)
def test_catches_repetition_loops(text):
    assert is_degenerate(text) is True, f"should have been caught: {text!r}"


@pytest.mark.parametrize("text", REAL_SPEECH)
def test_does_not_eat_real_speech(text):
    assert is_degenerate(text) is False, f"real speech was dropped: {text!r}"


def test_short_lines_are_always_allowed():
    """A brief line can't be confidently called degenerate, so it must pass."""
    for text in ("hi", "okay okay", "no no no", "what?"):
        assert is_degenerate(text) is False


def test_guard_is_applied_only_to_finals():
    """Partials update ~2x/second and a half-formed partial can look repetitive;
    only finals are filtered, so a partial is never suppressed."""
    import inspect

    from livecaptions.sources import streaming_local
    src = inspect.getsource(streaming_local.StreamingTranscriptionSource._emit)
    assert "is_final and" in src and "is_degenerate(text)" in src
