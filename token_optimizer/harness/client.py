"""LLM clients — one tiny interface, several backends.

All a client must do is map a prompt string to (text, prompt_tokens,
completion_tokens). Real backends use only the prompt; the offline MockClient
also peeks at `meta` so the whole harness can run with zero network and zero
money to prove the plumbing works.

Pick a backend with env vars (see build_client_from_env):
    LLM_PROVIDER = mock | ollama | groq | openai | openai_compatible
    LLM_MODEL    = model name for that provider
    LLM_API_KEY  = key (for groq/openai/openai_compatible)
    LLM_BASE_URL = override base url (optional)

Free paths: `mock` (offline), `ollama` (local models), `groq` (free API tier).
"""

import os
import random
import time
from typing import Optional, Dict

import requests

from .tokens import count_tokens


class GenerationResult:
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int,
                 cost: float = 0.0, info: Optional[Dict] = None):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost = cost                 # USD, when a price is known (Phase 4)
        self.info = info or {}           # e.g. {"cache_hit": True} / {"route": "cheap"}

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    name = "base"

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        raise NotImplementedError


class MockClient(LLMClient):
    """Offline, zero-cost, no key. NOT a real model.

    It's a crude lexical-overlap baseline: it returns the context sentence with
    the most word overlap with the question. Its accuracy is low by design — its
    only job is to prove the harness runs end to end (dataset -> prompt -> answer
    -> score -> table). Swap in `ollama` or `groq` for real quality numbers.
    """

    name = "mock"

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        meta = meta or {}
        question = meta.get("question", "")
        context = meta.get("context", prompt)

        from .scoring import normalize_answer
        q_tokens = set(normalize_answer(question).split())

        # Split context into rough sentences.
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        if not sentences:
            sentences = [context]

        def overlap(sent: str) -> int:
            s_tokens = set(normalize_answer(sent).split())
            return len(q_tokens & s_tokens)

        best = max(sentences, key=overlap)
        return GenerationResult(
            text=best,
            prompt_tokens=count_tokens(prompt),
            completion_tokens=count_tokens(best),
        )


class OllamaClient(LLMClient):
    """Local models via Ollama (https://ollama.com). Zero cost, no key.

    Run `ollama pull llama3.2` then `ollama serve`. Token counts come straight
    from Ollama's prompt_eval_count / eval_count.
    """

    name = "ollama"

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        pt = data.get("prompt_eval_count") or count_tokens(prompt)
        ct = data.get("eval_count") or count_tokens(text)
        return GenerationResult(text, pt, ct)


class OpenAICompatibleClient(LLMClient):
    """Any OpenAI-compatible chat endpoint: Groq, OpenAI, Together, etc.

    Groq has a free tier and is OpenAI-compatible, which makes it the best
    cheap "real model" for this harness. Token counts come from the response's
    usage field when present.
    """

    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str, model: str,
                 max_retries: int = 6, request_delay: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.request_delay = request_delay  # proactive sleep before each call (sec)

    def generate(self, prompt: str, meta: Optional[Dict] = None) -> GenerationResult:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            if self.request_delay:
                time.sleep(self.request_delay)
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
            except requests.exceptions.RequestException as e:
                last_err = e
                self._sleep_backoff(attempt, None)
                continue

            # Rate limited or transient server error -> wait and retry.
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = requests.exceptions.HTTPError(
                    f"{resp.status_code} from API", response=resp
                )
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt, resp)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            pt = usage.get("prompt_tokens") or count_tokens(prompt)
            ct = usage.get("completion_tokens") or count_tokens(text)
            return GenerationResult(text, pt, ct)

        # Retries exhausted.
        raise last_err if last_err else RuntimeError("request failed")

    def _sleep_backoff(self, attempt: int, resp) -> None:
        """Honor Retry-After if present, else exponential backoff with jitter."""
        wait = None
        if resp is not None:
            ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    wait = None
        if wait is None:
            wait = min(60.0, (2 ** attempt)) + random.uniform(0, 1.0)
        time.sleep(wait)


def build_client_from_env() -> LLMClient:
    """Construct a client from environment variables. Defaults to offline mock."""
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()

    if provider == "mock":
        return MockClient()

    if provider == "ollama":
        return OllamaClient(model=os.environ.get("LLM_MODEL", "llama3.2"))

    if provider in ("openai", "groq", "openai_compatible"):
        base = os.environ.get("LLM_BASE_URL")
        if not base and provider == "groq":
            base = "https://api.groq.com/openai/v1"
        if not base and provider == "openai":
            base = "https://api.openai.com/v1"
        if not base:
            raise ValueError("LLM_BASE_URL is required for openai_compatible provider")
        key = os.environ.get("LLM_API_KEY", "")
        default_model = "llama-3.1-8b-instant" if provider == "groq" else "gpt-4o-mini"
        model = os.environ.get("LLM_MODEL", default_model)
        request_delay = float(os.environ.get("REQUEST_DELAY", "0.0"))
        max_retries = int(os.environ.get("MAX_RETRIES", "6"))
        return OpenAICompatibleClient(base, key, model,
                                      max_retries=max_retries,
                                      request_delay=request_delay)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")