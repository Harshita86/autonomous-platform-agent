"""The ONLY component that knows Linear exists. Everything platform-specific is
here; the rest of the agent talks to `execute()`. Swapping platforms = new adapter."""
from __future__ import annotations

import httpx

LINEAR_URL = "https://api.linear.app/graphql"


class LinearError(RuntimeError):
    pass


class LinearAdapter:
    def __init__(self, api_key: str | None, timeout: float = 20.0):
        if not api_key:
            raise LinearError("LINEAR_API_KEY is missing — set it in .env")
        self._client = httpx.Client(
            base_url=LINEAR_URL,
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )
        # Every real API round-trip increments this. The executor reads deltas
        # to attribute api_calls per step — this is the source of the metric.
        self.call_count = 0

    def execute(self, query: str, variables: dict | None = None) -> dict:
        self.call_count += 1
        resp = self._client.post("", json={"query": query, "variables": variables or {}})
        if resp.status_code >= 400:
            raise LinearError(f"HTTP {resp.status_code}: {resp.text}")
        payload = resp.json()
        if payload.get("errors"):
            raise LinearError(str(payload["errors"]))
        return payload["data"]

    def close(self) -> None:
        self._client.close()
