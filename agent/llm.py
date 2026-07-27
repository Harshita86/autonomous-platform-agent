"""LLM port + provider adapters.

The planner depends on `LLM.complete()`, never on a vendor SDK — so swapping
providers is a config change, not a code change. Providers are tried in the
order given by LLM_PROVIDER (or a default priority), and the first one that
actually answers wins. That makes 'which of my keys works?' an observable
property of the system instead of a guess.

Each adapter imports its SDK lazily, so you only need the package for the
provider you actually use.
"""
from __future__ import annotations

import os

DEFAULT_ORDER = ["anthropic", "openai", "gemini"]

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}

ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class LLMUnavailable(RuntimeError):
    """No configured provider could answer."""


# --- provider adapters: each returns plain text ---

def _call_anthropic(system: str, user: str, model: str) -> str:
    from anthropic import Anthropic

    msg = Anthropic().messages.create(
        model=model,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _call_openai(system: str, user: str, model: str) -> str:
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_gemini(system: str, user: str, model: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # System prompt is prepended rather than passed as a config object: fewer
    # SDK-surface assumptions, identical effect for this extraction task.
    resp = client.models.generate_content(model=model, contents=f"{system}\n\n{user}")
    return resp.text or ""


ADAPTERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
}


class LLM:
    """Tries each configured provider in order; returns the first success."""

    def __init__(self, order: list[str] | None = None):
        env_order = os.environ.get("LLM_PROVIDER", "").strip()
        self.order = order or ([p.strip() for p in env_order.split(",") if p.strip()] or DEFAULT_ORDER)
        self.last_provider: str | None = None
        self.errors: list[tuple[str, str]] = []
        # Counted so 'how much reasoning did this run need?' is measurable. Plan
        # reuse shows up here as calls dropping to zero.
        self.calls: int = 0

    def _model_for(self, provider: str) -> str:
        return os.environ.get(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]

    def configured(self, provider: str) -> bool:
        return bool(os.environ.get(ENV_KEYS[provider]))

    def complete(self, system: str, user: str) -> str:
        self.errors = []
        for provider in self.order:
            if provider not in ADAPTERS:
                self.errors.append((provider, "unknown provider name"))
                continue
            if not self.configured(provider):
                self.errors.append((provider, f"{ENV_KEYS[provider]} not set"))
                continue
            try:
                self.calls += 1
                text = ADAPTERS[provider](system, user, self._model_for(provider))
                self.last_provider = provider
                return text
            except Exception as exc:  # noqa: BLE001 — try the next provider
                self.errors.append((provider, f"{type(exc).__name__}: {exc}"))

        detail = "; ".join(f"{p}: {e}" for p, e in self.errors) or "no providers configured"
        raise LLMUnavailable(detail)
