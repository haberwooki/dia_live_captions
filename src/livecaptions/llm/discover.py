"""Find a local model server so notes work without anyone configuring anything.

Asking someone to pick a provider, know that Ollama listens on 11434, and type an
exact model name before they can get a summary is three chances to give up. If a
local server is running, everything needed is discoverable: the port is one of two,
and the server lists its own models.

Nothing here sends data anywhere — it only asks local servers what they have.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

from .providers import LOCAL_PRESETS

#: Model names that cannot write notes, however happily the server lists them.
#: Embedding and speech models are the common clutter in an Ollama install.
_NOT_CHAT = ("embed", "bge-", "gte-", "nomic", "minilm", "e5-", "clip",
             "whisper", "rerank", "moondream", "llava-phi")

#: Rough preference when several will do. Instruction-tuned general models write
#: better structured notes than base or code models; ordering is a nudge, not a rule.
_PREFERRED = ("llama3.3", "llama3.2", "llama3.1", "llama3", "qwen2.5", "qwen3",
              "mistral", "mixtral", "gemma3", "gemma2", "phi4", "phi3", "command-r")


@dataclass
class LocalServer:
    name: str
    base_url: str
    models: List[str] = field(default_factory=list)

    @property
    def best_model(self) -> Optional[str]:
        return pick_model(self.models)


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def list_models(base_url: str, timeout: float = 2.0) -> List[str]:
    """Model ids a server offers, via the OpenAI-compatible /models endpoint."""
    url = base_url.rstrip("/") + "/models"
    try:
        data = _get_json(url, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        name = (item or {}).get("id") if isinstance(item, dict) else None
        if isinstance(name, str) and name:
            out.append(name)
    return out


def pick_model(models: List[str]) -> Optional[str]:
    """The most plausible note-writing model from what a server offers."""
    usable = [m for m in models if not any(bad in m.lower() for bad in _NOT_CHAT)]
    if not usable:
        return None
    for wanted in _PREFERRED:
        for m in usable:
            if wanted in m.lower():
                return m
    return usable[0]


def find_local_servers(timeout: float = 2.0) -> List[LocalServer]:
    """Local servers that are actually running, with the models they offer.

    Only servers that answered AND listed at least one usable model are returned:
    a reachable server with nothing loaded would otherwise be auto-selected and
    then fail at the first request, which is worse than reporting nothing found.
    """
    found = []
    for name, url in LOCAL_PRESETS.items():
        models = list_models(url, timeout)
        if models and pick_model(models):
            found.append(LocalServer(name=name, base_url=url, models=models))
    return found


def autoconfigure(timeout: float = 2.0) -> Optional[dict]:
    """Settings for the first usable local server, or None if none is running.

    Returns a dict ready to persist: {llm_provider, llm_base_url, llm_model}.
    """
    for server in find_local_servers(timeout):
        model = server.best_model
        if model:
            return {"llm_provider": "local", "llm_base_url": server.base_url,
                    "llm_model": model, "_server": server.name}
    return None
