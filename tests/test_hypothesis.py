"""Characterization tests for the LocalAgreement-2 HypothesisBuffer.

Deterministic: feed sequences of (start, end, word) hypotheses and assert on
what commits. The core guarantee is that committed words never rewrite and only
the agreed-across-two-hypotheses prefix commits.
"""
from livecaptions.asr.hypothesis import HypothesisBuffer


def words(*ws, dt=0.3, start=0.0):
    """Build (start, end, word) tuples with sequential timing."""
    out = []
    t = start
    for w in ws:
        out.append((t, t + dt, w))
        t += dt
    return out


def test_first_hypothesis_commits_nothing():
    h = HypothesisBuffer()
    h.insert(words("the", "cat"), 0.0)
    assert h.flush() == []            # nothing to agree with yet
    assert [w[2] for w in h.buffer] == ["the", "cat"]


def test_agreement_commits_common_prefix():
    h = HypothesisBuffer()
    h.insert(words("the", "cat"), 0.0)
    h.flush()
    h.insert(words("the", "cat", "sat"), 0.0)
    committed = h.flush()
    assert [w[2] for w in committed] == ["the", "cat"]   # agreed prefix commits
    assert [w[2] for w in h.buffer] == ["sat"]           # unconfirmed tail remains


def test_disagreement_does_not_commit():
    h = HypothesisBuffer()
    h.insert(words("the", "dog"), 0.0)
    h.flush()
    h.insert(words("the", "cat"), 0.0)   # disagrees at word 2
    committed = h.flush()
    assert [w[2] for w in committed] == ["the"]          # only the agreed "the" commits


def test_streaming_extends_commit_over_time():
    h = HypothesisBuffer()
    seq = [
        ["the"],
        ["the", "quick"],
        ["the", "quick", "brown"],
        ["the", "quick", "brown", "fox"],
    ]
    committed_all = []
    for s in seq:
        h.insert(words(*s), 0.0)
        committed_all += [w[2] for w in h.flush()]
    # each word commits once agreed by the next hypothesis; "fox" stays unconfirmed
    assert committed_all == ["the", "quick", "brown"]
    assert [w[2] for w in h.buffer] == ["fox"]


def test_committed_never_rewrites():
    h = HypothesisBuffer()
    h.insert(words("hello", "world"), 0.0)
    h.flush()
    h.insert(words("hello", "world", "again"), 0.0)
    h.flush()
    # a later hypothesis that disagrees on an already-committed word cannot undo it
    before = list(h.committed_in_buffer)
    h.insert(words("hello", "there", "again"), 0.0)
    h.flush()
    assert h.committed_in_buffer[:len(before)] == before
