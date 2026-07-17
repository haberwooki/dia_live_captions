"""Tests for word->speaker assignment and caption grouping (pure logic)."""
from livecaptions.diarize.assign import (
    assign_speakers,
    best_speaker,
    format_transcript,
    group_into_segments,
)
from livecaptions.diarize.base import SpeakerTurn

TURNS = [
    SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
    SpeakerTurn(2.0, 4.0, "SPEAKER_01"),
]


def test_best_speaker_by_max_overlap():
    assert best_speaker(0.1, 0.9, TURNS) == "SPEAKER_00"
    assert best_speaker(2.5, 3.0, TURNS) == "SPEAKER_01"


def test_word_straddling_boundary_goes_to_larger_overlap():
    # 1.8-2.6 overlaps SPEAKER_00 by 0.2 and SPEAKER_01 by 0.6 -> SPEAKER_01
    assert best_speaker(1.8, 2.6, TURNS) == "SPEAKER_01"


def test_word_in_gap_snaps_to_nearest_turn():
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(5.0, 6.0, "SPEAKER_01")]
    assert best_speaker(1.2, 1.4, turns) == "SPEAKER_00"   # nearest is the first turn
    assert best_speaker(4.6, 4.8, turns) == "SPEAKER_01"


def test_no_turns_yields_none():
    assert best_speaker(0.0, 1.0, []) is None


def test_assign_and_group_merges_same_speaker():
    words = [(0.0, 0.4, "hello"), (0.5, 0.9, "there"), (2.1, 2.5, "hi"), (2.6, 3.0, "back")]
    labeled = assign_speakers(words, TURNS)
    assert [w.speaker for w in labeled] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    segs = group_into_segments(labeled)
    assert len(segs) == 2
    assert segs[0].speaker == "SPEAKER_00" and segs[0].text == "hello there"
    assert segs[1].speaker == "SPEAKER_01" and segs[1].text == "hi back"


def test_group_splits_on_long_pause_same_speaker():
    words = [(0.0, 0.4, "one"), (9.0, 9.4, "two")]   # same speaker, 8.6s gap
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00")]
    segs = group_into_segments(assign_speakers(words, turns), max_gap=1.0)
    assert len(segs) == 2


def test_format_transcript():
    segs = group_into_segments(assign_speakers([(0.0, 0.4, "hello")], TURNS))
    out = format_transcript(segs)
    assert "SPEAKER_00: hello" in out
