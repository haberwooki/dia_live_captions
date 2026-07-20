"""The updater must offer the newest VERSION, whatever order releases published in.

Real incident (2026-07-20): a GitHub Actions outage queued two builds. When runners
returned, v0.4.0 finished and published FIRST, then v0.3.4 published second — so
v0.3.4 was the most recently published release. GitHub's /releases/latest follows
publish time, not version, so it named v0.3.4 "latest", the update button offered an
OLDER version, and a user on 0.3.2 could not reach 0.4.0. The fix is to sort the
release LIST by version ourselves; these pin that it holds.
"""
import json

import pytest

from livecaptions import updater as U


def _release(tag, *, exe=True, draft=False, prerelease=False):
    assets = [{"name": f"LiveCaptions-Setup-{tag.lstrip('v')}.exe",
               "browser_download_url": f"https://example/{tag}/setup.exe"}] if exe else []
    return {"tag_name": tag, "draft": draft, "prerelease": prerelease, "assets": assets}


def _fake_api(monkeypatch, releases):
    """Feed a canned /releases list, in whatever order GitHub would return it."""
    import io

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    payload = json.dumps(releases).encode()
    monkeypatch.setattr(U.urllib.request, "urlopen", lambda req, timeout=10: R(payload))


def test_picks_highest_version_even_when_published_out_of_order(monkeypatch):
    """The exact incident: v0.3.4 appears first (most recently published), v0.4.0
    second, and v0.4.0 must still win."""
    _fake_api(monkeypatch, [_release("v0.3.4"), _release("v0.4.0"), _release("v0.3.3")])
    tag, url = U.latest_release()
    assert tag == "v0.4.0"
    assert "v0.4.0" in url


def test_two_digit_and_multi_component_versions_order_correctly(monkeypatch):
    """String sorting would put v0.10.0 below v0.9.0; tuple sorting must not."""
    _fake_api(monkeypatch, [_release("v0.9.0"), _release("v0.10.0"), _release("v0.10.1")])
    assert U.latest_release()[0] == "v0.10.1"


def test_skips_a_release_with_no_installer(monkeypatch):
    """A build that failed to upload its .exe must be passed over, or the button
    points at a dead link."""
    _fake_api(monkeypatch, [_release("v0.5.0", exe=False), _release("v0.4.0")])
    assert U.latest_release()[0] == "v0.4.0"


def test_skips_drafts_and_prereleases(monkeypatch):
    _fake_api(monkeypatch, [
        _release("v0.6.0", draft=True), _release("v0.5.9", prerelease=True),
        _release("v0.4.0")])
    assert U.latest_release()[0] == "v0.4.0"


def test_no_usable_release_returns_empty(monkeypatch):
    _fake_api(monkeypatch, [_release("v0.6.0", exe=False), _release("v0.5.0", draft=True)])
    tag, url = U.latest_release()
    assert tag == "" and url is None


class TestIsNewer:
    def test_a_higher_version_is_newer(self, monkeypatch):
        monkeypatch.setattr(U, "__version__", "0.3.2")
        assert U.is_newer("v0.4.0") is True

    def test_the_same_version_is_not(self, monkeypatch):
        monkeypatch.setattr(U, "__version__", "0.4.0")
        assert U.is_newer("v0.4.0") is False

    def test_an_older_tag_is_not_newer(self, monkeypatch):
        """After the fix, the button must never offer a downgrade even if the API
        somehow surfaces one."""
        monkeypatch.setattr(U, "__version__", "0.4.0")
        assert U.is_newer("v0.3.4") is False

    def test_v_prefix_is_optional(self, monkeypatch):
        monkeypatch.setattr(U, "__version__", "0.3.2")
        assert U.is_newer("0.4.0") is True
