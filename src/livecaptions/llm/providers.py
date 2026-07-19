"""Talk to whichever model the user chose: Claude, any OpenAI-compatible API, or a
local server (Ollama / LM Studio).

Everything else in this app is local, so this module is the boundary where data
can leave the machine. Two rules follow from that and are enforced by callers:
nothing is sent without explicit consent, and the user is told how much is going.
A local provider keeps even that on the machine, which is why it is a first-class
option rather than an afterthought.

HTTP is stdlib urllib, not httpx/requests: this has to keep working inside the
frozen PyInstaller build, and stdlib cannot fail to be collected.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Type

from pydantic import BaseModel

#: Ready-made endpoints for the two common local servers, so "run it locally"
#: doesn't require knowing a port.
LOCAL_PRESETS = {
    "Ollama": "http://localhost:11434/v1",
    "LM Studio": "http://localhost:1234/v1",
}

PROVIDER_KINDS = ("none", "anthropic", "openai", "local")


class LLMError(RuntimeError):
    """Anything that stopped us getting a usable answer. Message is user-facing."""


@dataclass
class ProviderConfig:
    kind: str = "none"
    model: str = ""
    base_url: str = ""
    api_key: Optional[str] = None
    #: Seconds to wait for a reply. The default suits a connectivity check; long
    #: jobs raise it. Measured: a local reasoning model (qwen3.5-9b via LM Studio)
    #: took 133 s on an EIGHT-LINE transcript, emitting 13,776 characters of
    #: reasoning before the answer — a real session needs far more headroom.
    timeout: float = 120.0

    @property
    def is_local(self) -> bool:
        return self.kind == "local"

    @property
    def sends_data_off_machine(self) -> bool:
        return self.kind in ("anthropic", "openai")


class Provider:
    """Returns a validated pydantic object, or raises LLMError."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def label(self) -> str:
        c = self.config
        where = c.base_url or {"anthropic": "api.anthropic.com"}.get(c.kind, "")
        return f"{c.model or '(no model set)'} @ {where or c.kind}"

    def complete(self, system: str, user: str, schema: Type[BaseModel]) -> BaseModel:
        raise NotImplementedError

    def test(self) -> str:
        """Cheap round-trip so the user can confirm setup before trusting it with a
        transcript. Returns a short human-readable result."""
        class Ping(BaseModel):
            ok: bool
            reply: str

        out = self.complete(
            "You are a connectivity check. Answer briefly.",
            'Reply with ok=true and reply="pong".', Ping)
        return f"Connected — {self.label} replied “{out.reply}”."


class AnthropicProvider(Provider):
    def complete(self, system: str, user: str, schema: Type[BaseModel]) -> BaseModel:
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("The anthropic package isn't available in this build.") from e
        # No api_key passed: the SDK resolves ANTHROPIC_API_KEY or an `ant auth
        # login` profile. A key typed into the GUI is put in the environment by the
        # caller, never written to config.
        kwargs = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        try:
            client = anthropic.Anthropic(**kwargs)
            response = client.messages.parse(
                model=self.config.model or "claude-opus-4-8",
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("The model declined to answer this request.")
        return response.parsed_output


class OpenAICompatProvider(Provider):
    """Anything speaking the OpenAI chat-completions protocol: OpenAI itself,
    OpenRouter, Groq, Together, Ollama and LM Studio."""

    def _post(self, payload: dict, timeout: float) -> dict:
        base = (self.config.base_url or "").rstrip("/")
        if not base:
            raise LLMError("No server URL set for this provider.")
        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if e.code in (401, 403):
                raise LLMError(f"The server rejected the API key ({e.code}). {detail}") from e
            raise LLMError(f"Server returned {e.code}. {detail}") from e
        except urllib.error.URLError as e:
            # By far the most common local failure: the server simply isn't running.
            raise LLMError(f"Couldn't reach {base} — is the server running? ({e.reason})") from e
        except TimeoutError as e:
            raise LLMError(f"{base} didn't respond in time.") from e

    def complete(self, system: str, user: str, schema: Type[BaseModel],
                 timeout: Optional[float] = None) -> BaseModel:
        timeout = self.config.timeout if timeout is None else timeout
        if not self.config.model:
            raise LLMError("No model name set for this provider.")
        js = schema.model_json_schema()
        base_payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
        }
        # Three rungs, because local servers disagree about all of this. Measured
        # against LM Studio
        # (qwen3.5-9b): strict json_schema returns HTTP 200 with EMPTY content, and
        # json_object is rejected outright with "must be 'json_schema' or 'text'".
        # Plain text with the schema in the prompt is the only rung that worked
        # there, and it works everywhere else too — so it must exist as the floor.
        schema_prompt = [
            {"role": "system",
             "content": f"{system}\n\nReply with JSON matching this schema, and "
                        f"nothing else:\n{json.dumps(js)}"},
            {"role": "user", "content": user},
        ]
        attempts = [
            dict(base_payload, response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": js}}),
            dict(base_payload, response_format={"type": "json_object"},
                 messages=schema_prompt),
            dict(base_payload, response_format={"type": "text"}, messages=schema_prompt),
        ]
        last: Optional[Exception] = None
        for payload in attempts:
            try:
                data = self._post(payload, timeout)
                content = data["choices"][0]["message"]["content"]
                if not (content or "").strip():
                    # A 200 with no content is a rung that silently doesn't work
                    # (LM Studio does this for json_schema). Treat it as a failure
                    # so the next rung is tried, rather than reporting a confusing
                    # "reply didn't match the format" for an empty reply.
                    raise LLMError("the server returned an empty reply")
                return schema.model_validate_json(_strip_fence(content))
            except LLMError as e:
                last = e
                if "rejected the API key" in str(e) or "Couldn't reach" in str(e):
                    raise                      # retrying won't help
            except Exception as e:
                last = e
        raise LLMError(f"The model's reply didn't match the expected format. {last}")


def _strip_fence(text: str) -> str:
    """Small models like wrapping JSON in ```json fences even when told not to."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
        t = t.strip()
        if t.startswith("json"):
            t = t[4:].strip()
    return t


def config_from_settings(settings, api_key: Optional[str] = None) -> ProviderConfig:
    kind = str(getattr(settings, "llm_provider", "none") or "none").lower()
    return ProviderConfig(
        kind=kind if kind in PROVIDER_KINDS else "none",
        model=str(getattr(settings, "llm_model", "") or ""),
        base_url=str(getattr(settings, "llm_base_url", "") or ""),
        api_key=api_key,
    )


def build(config: ProviderConfig) -> Provider:
    if config.kind == "anthropic":
        return AnthropicProvider(config)
    if config.kind in ("openai", "local"):
        return OpenAICompatProvider(config)
    raise LLMError("No AI provider is configured. Choose one in Settings → AI.")


def from_settings(settings, api_key: Optional[str] = None) -> Provider:
    return build(config_from_settings(settings, api_key))


def resolve_api_key(kind: str) -> Optional[str]:
    """Key for a provider kind: environment first, then Credential Manager.
    Local servers usually need none."""
    from .credentials import resolve_key
    env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(kind)
    return resolve_key(kind, env)


def describe_privacy(config: ProviderConfig, payload_chars: int) -> str:
    """Exactly what is about to happen, in one sentence — shown before sending."""
    if config.is_local:
        return (f"This sends {payload_chars:,} characters of transcript to your local "
                f"server at {config.base_url}. It does not leave this machine.")
    where = config.base_url or "the Anthropic API"
    return (f"This sends {payload_chars:,} characters of transcript text to {where} "
            f"({config.model}). That leaves this machine.")


def available_labels() -> List[str]:
    return ["Not configured", "Claude (Anthropic)", "OpenAI-compatible API",
            "Local model (Ollama / LM Studio)"]
