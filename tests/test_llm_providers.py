"""The AI layer is the only thing that can send data off the machine.

So these tests are mostly about boundaries: keys never touching config.toml, the
user being told what leaves and where it goes, and failures being explained rather
than swallowed. The protocol details (schema retry, fenced JSON) matter because
local models are sloppier than hosted ones and the fallback path is what makes
"run it locally" actually work.
"""
import json

import pytest
from pydantic import BaseModel

from livecaptions.config import Settings
from livecaptions.llm import providers as P


class Answer(BaseModel):
    name: str
    count: int


def test_config_has_no_place_to_put_a_key():
    """A key in config.toml would land in backups, screenshots and bug reports."""
    fields = set(Settings.model_fields)
    for suspicious in ("llm_api_key", "api_key", "anthropic_api_key", "openai_api_key"):
        assert suspicious not in fields, f"{suspicious} must not be a config field"


def test_unconfigured_provider_says_so_clearly():
    with pytest.raises(P.LLMError) as e:
        P.build(P.ProviderConfig(kind="none"))
    assert "Settings" in str(e.value)


class TestPrivacyDisclosure:
    def test_remote_says_it_leaves_the_machine(self):
        cfg = P.ProviderConfig(kind="anthropic", model="claude-opus-4-8")
        msg = P.describe_privacy(cfg, 12_345)
        assert "12,345" in msg, "must state how much is being sent"
        assert "leaves this machine" in msg

    def test_local_says_it_stays(self):
        cfg = P.ProviderConfig(kind="local", model="llama3.1",
                               base_url="http://localhost:11434/v1")
        msg = P.describe_privacy(cfg, 500)
        assert "does not leave this machine" in msg
        assert "localhost" in msg

    def test_kinds_are_classified_correctly(self):
        assert P.ProviderConfig(kind="local").sends_data_off_machine is False
        assert P.ProviderConfig(kind="none").sends_data_off_machine is False
        assert P.ProviderConfig(kind="anthropic").sends_data_off_machine is True
        assert P.ProviderConfig(kind="openai").sends_data_off_machine is True


class TestOpenAICompatible:
    def _provider(self, monkeypatch, responses):
        """responses: list of (status_or_None, body) consumed in order."""
        calls = []
        cfg = P.ProviderConfig(kind="local", model="llama3.1",
                               base_url="http://localhost:11434/v1")
        prov = P.OpenAICompatProvider(cfg)

        def fake_post(payload, timeout):
            calls.append(payload)
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return {"choices": [{"message": {"content": item}}]}
        monkeypatch.setattr(prov, "_post", fake_post)
        return prov, calls

    def test_happy_path(self, monkeypatch):
        prov, calls = self._provider(monkeypatch, [json.dumps({"name": "x", "count": 2})])
        out = prov.complete("sys", "user", Answer)
        assert (out.name, out.count) == ("x", 2)
        assert calls[0]["response_format"]["type"] == "json_schema", "should try strict first"

    def test_falls_back_when_strict_schema_is_unsupported(self, monkeypatch):
        """Older Ollama and many proxies reject json_schema — without this fallback,
        local models would simply not work."""
        prov, calls = self._provider(monkeypatch, [
            P.LLMError("Server returned 400. unknown response_format"),
            json.dumps({"name": "y", "count": 1})])
        out = prov.complete("sys", "user", Answer)
        assert out.name == "y"
        assert len(calls) == 2
        assert calls[1]["response_format"]["type"] == "json_object"

    def test_strips_a_markdown_fence(self, monkeypatch):
        """Small models wrap JSON in ``` even when told not to."""
        prov, _ = self._provider(
            monkeypatch, ['```json\n{"name": "z", "count": 7}\n```'])
        assert prov.complete("s", "u", Answer).count == 7

    def test_bad_key_is_not_retried(self, monkeypatch):
        prov, calls = self._provider(monkeypatch, [
            P.LLMError("The server rejected the API key (401). nope"),
            json.dumps({"name": "n", "count": 0})])
        with pytest.raises(P.LLMError) as e:
            prov.complete("s", "u", Answer)
        assert "rejected the API key" in str(e.value)
        assert len(calls) == 1, "retrying a rejected key just wastes time"

    def test_server_down_is_explained_not_swallowed(self, monkeypatch):
        prov, calls = self._provider(monkeypatch, [
            P.LLMError("Couldn't reach http://localhost:11434/v1 — is the server running?")])
        with pytest.raises(P.LLMError) as e:
            prov.complete("s", "u", Answer)
        assert "is the server running" in str(e.value)
        assert len(calls) == 1

    def test_unparseable_reply_reports_the_format_problem(self, monkeypatch):
        prov, _ = self._provider(monkeypatch, ["not json at all", "still not json"])
        with pytest.raises(P.LLMError) as e:
            prov.complete("s", "u", Answer)
        assert "didn't match the expected format" in str(e.value)

    def test_missing_model_or_url_is_caught_before_any_request(self):
        with pytest.raises(P.LLMError):
            P.OpenAICompatProvider(P.ProviderConfig(kind="local", base_url="http://x/v1")
                                   ).complete("s", "u", Answer)
        with pytest.raises(P.LLMError):
            P.OpenAICompatProvider(P.ProviderConfig(kind="local", model="m")
                                   ).complete("s", "u", Answer)


def test_from_settings_reads_the_configured_provider():
    s = Settings(llm_provider="local", llm_model="llama3.1",
                 llm_base_url="http://localhost:11434/v1")
    prov = P.from_settings(s)
    assert isinstance(prov, P.OpenAICompatProvider)
    assert "llama3.1" in prov.label and "11434" in prov.label


def test_local_presets_cover_the_common_servers():
    assert "11434" in P.LOCAL_PRESETS["Ollama"]
    assert "1234" in P.LOCAL_PRESETS["LM Studio"]
