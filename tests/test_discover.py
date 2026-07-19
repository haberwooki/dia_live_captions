"""Notes should work from one click, without anyone configuring a provider.

Asking someone to pick a provider, know that Ollama listens on 11434, and type an
exact model name is three chances to give up before getting a summary. If a local
server is running, all of that is discoverable — these pin that it is discovered
correctly, and that a server which cannot actually answer is not selected.
"""
import json

import pytest

from livecaptions.llm import discover as D


def _fake_models(monkeypatch, by_url):
    """by_url: {base_url: [model ids]} — anything else refuses the connection."""
    def fake_get_json(url, timeout):
        for base, models in by_url.items():
            if url.startswith(base.rstrip("/")):
                return {"data": [{"id": m} for m in models]}
        raise OSError("connection refused")
    monkeypatch.setattr(D, "_get_json", fake_get_json)


class TestPickModel:
    def test_skips_models_that_cannot_write_notes(self):
        """An Ollama install is usually full of embedding models; picking one would
        fail at the first request with a baffling error."""
        assert D.pick_model(["nomic-embed-text", "llama3.1:8b"]) == "llama3.1:8b"
        assert D.pick_model(["bge-large", "mxbai-embed-large"]) is None
        assert D.pick_model(["whisper-large"]) is None

    def test_prefers_a_general_instruction_model(self):
        got = D.pick_model(["codellama:7b", "qwen2.5:14b"])
        assert got == "qwen2.5:14b"

    def test_falls_back_to_whatever_is_there(self):
        assert D.pick_model(["some-unknown-model"]) == "some-unknown-model"

    def test_no_models_is_not_a_crash(self):
        assert D.pick_model([]) is None


class TestDiscovery:
    def test_finds_a_running_ollama(self, monkeypatch):
        _fake_models(monkeypatch, {"http://localhost:11434/v1": ["llama3.1:8b"]})
        servers = D.find_local_servers()
        assert [s.name for s in servers] == ["Ollama"]
        assert servers[0].best_model == "llama3.1:8b"

    def test_finds_lm_studio_too(self, monkeypatch):
        _fake_models(monkeypatch, {"http://localhost:1234/v1": ["mistral-7b-instruct"]})
        assert [s.name for s in D.find_local_servers()] == ["LM Studio"]

    def test_nothing_running_is_reported_as_nothing(self, monkeypatch):
        _fake_models(monkeypatch, {})
        assert D.find_local_servers() == []
        assert D.autoconfigure() is None

    def test_a_server_with_no_usable_model_is_not_selected(self, monkeypatch):
        """Reachable but useless is worse than absent: auto-selecting it would fail
        at the first request instead of saying 'nothing found'."""
        _fake_models(monkeypatch, {"http://localhost:11434/v1": ["nomic-embed-text"]})
        assert D.find_local_servers() == []
        assert D.autoconfigure() is None


class TestAutoconfigure:
    def test_returns_settings_ready_to_persist(self, monkeypatch):
        _fake_models(monkeypatch, {"http://localhost:11434/v1": ["gemma2:9b"]})
        cfg = D.autoconfigure()
        assert cfg["llm_provider"] == "local"
        assert cfg["llm_base_url"] == "http://localhost:11434/v1"
        assert cfg["llm_model"] == "gemma2:9b"
        assert cfg["_server"] == "Ollama"

    def test_the_result_actually_builds_a_working_provider(self, monkeypatch):
        """The point of autoconfigure is that its output needs no further editing."""
        from livecaptions.config import Settings
        from livecaptions.llm import providers as P
        _fake_models(monkeypatch, {"http://localhost:11434/v1": ["llama3.1:8b"]})
        cfg = D.autoconfigure()
        cfg.pop("_server")
        provider = P.from_settings(Settings(**cfg))
        assert isinstance(provider, P.OpenAICompatProvider)
        assert "llama3.1:8b" in provider.label


def test_a_broken_server_reply_does_not_raise(monkeypatch):
    """A server answering with something other than the OpenAI shape must read as
    'nothing found', not crash the button."""
    monkeypatch.setattr(D, "_get_json", lambda url, timeout: {"unexpected": True})
    assert D.list_models("http://localhost:11434/v1") == []

    monkeypatch.setattr(D, "_get_json",
                        lambda url, timeout: (_ for _ in ()).throw(ValueError("not json")))
    assert D.list_models("http://localhost:11434/v1") == []


def test_discovery_only_talks_to_localhost():
    """This runs without consent, so it must never reach off the machine."""
    from livecaptions.llm.providers import LOCAL_PRESETS
    for url in LOCAL_PRESETS.values():
        assert url.startswith("http://localhost:") or url.startswith("http://127.0.0.1:")
