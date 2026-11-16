"""Language-model backends. Each one answers a prompt with a parsed JSON object.

* ``lmstudio[:model]``: local model through LM Studio's OpenAI-compatible server (default
  host ``http://localhost:1234``; the first loaded model is used when none is given);
* ``ollama:model``: local model through Ollama's OpenAI-compatible endpoint;
* ``openai:model``: OpenAI API (``OPENAI_API_KEY``) or any compatible server
  (``OPENAI_BASE_URL``);
* ``claude:model``: Anthropic API (``pip install derivcheck[claude]``, ``ANTHROPIC_API_KEY``);
* ``offline``: no model; LaTeX is segmented and parsed by rules.

The default comes from the ``DERIVCHECK_LLM`` environment variable, else ``offline``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    """The model could not be reached or did not return usable JSON."""


def parse_json(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model answer (drops <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("the model's answer contains no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"the model's answer is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise LLMError("the model's answer is not a JSON object")
    return data


class Backend:
    """Base class; ``calls`` counts requests made to the model."""

    name = "offline"
    is_llm = False

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Ask for JSON; if the answer does not parse, ask once more with the error."""
        answer = self.complete(system, user)
        try:
            return parse_json(answer)
        except LLMError as exc:
            retry = (
                f"{user}\n\nYour previous answer could not be used ({exc}). "
                "Answer again with one JSON object and nothing else."
            )
            return parse_json(self.complete(system, retry))


class OfflineBackend(Backend):
    name = "offline"


class OpenAICompatibleBackend(Backend):
    """Chat-completions endpoint of LM Studio, Ollama, OpenAI or any compatible server."""

    is_llm = True

    def __init__(
        self,
        label: str,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        super().__init__()
        self.label = label
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.label}:{self.model}" if self.model else self.label

    def _request(self, path: str, payload: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method="POST" if payload is not None else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"{self.label} at {self.base_url} did not answer: {exc}") from None

    def _resolve_model(self) -> str:
        if not self.model:
            models = self._request("/models").get("data") or []
            if not models:
                raise LLMError(f"{self.label} has no model loaded")
            self.model = models[0]["id"]
        return self.model

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self._resolve_model(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }
        self.calls += 1
        data = self._request("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected answer from {self.label}: {str(data)[:200]}") from None


class ClaudeBackend(Backend):
    is_llm = True

    def __init__(self, model: str) -> None:
        super().__init__()
        try:
            import anthropic
        except ImportError:
            raise LLMError("the Claude backend needs: pip install 'derivcheck[claude]'") from None
        self.model = model
        self.client = anthropic.Anthropic()

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"claude:{self.model}"

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Claude did not answer: {exc}") from None
        return "".join(getattr(block, "text", "") for block in message.content)


def get_backend(spec: str | None = None) -> Backend:
    """Build a backend from ``kind[:model]``; see the module docstring."""
    spec = (spec or os.environ.get("DERIVCHECK_LLM") or "offline").strip()
    kind, _, model = spec.partition(":")
    kind = kind.lower()
    if kind == "offline":
        return OfflineBackend()
    if kind == "lmstudio":
        host = os.environ.get("LMSTUDIO_HOST", "http://localhost:1234")
        return OpenAICompatibleBackend(
            "lmstudio",
            host + "/v1",
            model or os.environ.get("LMSTUDIO_MODEL"),
            os.environ.get("LMSTUDIO_API_KEY"),
        )
    if kind == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = model or os.environ.get("OLLAMA_MODEL")
        if not model:
            raise ValueError("give the Ollama model, e.g. --llm ollama:qwen3:32b")
        return OpenAICompatibleBackend("ollama", host + "/v1", model)
    if kind == "openai":
        model = model or os.environ.get("OPENAI_MODEL")
        if not model:
            raise ValueError("give the OpenAI model, e.g. --llm openai:<model>")
        return OpenAICompatibleBackend(
            "openai",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model,
            os.environ.get("OPENAI_API_KEY"),
        )
    if kind in ("claude", "anthropic"):
        model = model or os.environ.get("ANTHROPIC_MODEL")
        if not model:
            raise ValueError("give the Claude model, e.g. --llm claude:<model>")
        return ClaudeBackend(model)
    raise ValueError(f"unknown backend {kind!r}; use offline, lmstudio, ollama, openai or claude")
