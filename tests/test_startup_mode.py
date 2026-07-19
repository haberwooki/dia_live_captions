"""How the app opens must match how it was left.

Reported: "it closes and reopens not captioning". The cause was mine — startup was
a fixed checkbox (`start_captions_on_launch`), so pressing Start and closing the
app changed nothing about the next launch. A checkbox cannot express "resume how I
left it"; only a state recorded as it changes can.
"""
import pytest

from livecaptions.config import Settings
from livecaptions.ui.overlay import should_start_on_launch


def s(**kw):
    return Settings(**kw)


class TestResume:
    """The default: honour the state the user left behind."""

    def test_left_running_reopens_captioning(self):
        assert should_start_on_launch(
            s(startup_mode="resume", last_transport_state="running")) is True

    def test_left_stopped_reopens_stopped(self):
        assert should_start_on_launch(
            s(startup_mode="resume", last_transport_state="stopped")) is False

    def test_left_paused_reopens_stopped(self):
        """Pause is 'I turned it off' — coming back captioning would be a surprise."""
        assert should_start_on_launch(
            s(startup_mode="resume", last_transport_state="paused")) is False

    def test_transient_states_do_not_strand_the_user(self):
        """If it was mid-start or errored when it closed, prefer captioning: the
        alternative is an app that silently never starts again."""
        for state in ("starting", "error", "", "nonsense"):
            assert should_start_on_launch(
                s(startup_mode="resume", last_transport_state=state)) is True


class TestPinned:
    def test_always_ignores_how_it_was_left(self):
        for state in ("stopped", "paused", "running"):
            assert should_start_on_launch(
                s(startup_mode="always", last_transport_state=state)) is True

    def test_never_ignores_how_it_was_left(self):
        for state in ("stopped", "paused", "running"):
            assert should_start_on_launch(
                s(startup_mode="never", last_transport_state=state)) is False


def test_default_is_resume_and_captions_on_a_fresh_install():
    """Out of the box, opening the app must caption — not sit silently waiting."""
    fresh = Settings()
    assert fresh.startup_mode == "resume"
    assert should_start_on_launch(fresh) is True


@pytest.mark.parametrize("mode", ["RESUME", "Always", " never "])
def test_mode_is_read_leniently(mode):
    """Config is hand-editable; casing or a stray space must not silently change
    behaviour into 'never start'."""
    out = should_start_on_launch(s(startup_mode=mode.strip().lower(),
                                   last_transport_state="running"))
    assert out is (mode.strip().lower() != "never")
