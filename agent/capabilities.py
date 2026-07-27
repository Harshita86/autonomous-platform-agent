"""Capability registry — the procedural memory in front of the store.

Capabilities are data. The registry is the single authority on 'can this system
do X?'. The planner is free to *name* any capability it wants; the registry
decides whether it exists. That is what turns an impossible instruction into an
honest BLOCKED report instead of a false success.
"""
from __future__ import annotations

import json

from .memory.store import MemoryStore

# The irreducible built-ins, kept deliberately small: anything beyond these has to
# be synthesized at runtime, otherwise "synthesis" is just a lookup table of
# pre-written operations.
PRIMITIVES = [
    {
        "name": "resolve_team",
        "description": "Resolve a team name to its Linear team id (cached in semantic memory).",
        "handler": "builtin:resolve_team",
        "params": ["name"],
    },
    {
        "name": "resolve_label",
        "description": "Resolve an issue-label name to its Linear label id (cached in semantic memory).",
        "handler": "builtin:resolve_label",
        "params": ["name"],
    },
    {
        "name": "create_issue",
        "description": "Create an issue in a team, with optional description and labels.",
        "handler": "builtin:create_issue",
        "params": ["title", "description", "team", "labels"],
    },
]


class CapabilityRegistry:
    def __init__(self, memory: MemoryStore):
        self._m = memory
        self._seed()

    def _seed(self) -> None:
        for p in PRIMITIVES:
            if self._m.get_capability(p["name"]) is None:
                self._m.put_capability(
                    name=p["name"],
                    kind="primitive",
                    description=p["description"],
                    handler=p["handler"],
                    params=p["params"],
                    provenance="seeded at first run",
                )

    def get(self, name: str):
        return self._m.get_capability(name)

    def has(self, name: str) -> bool:
        return self._m.get_capability(name) is not None

    # A capability that has been tried enough times and still fails more often than
    # not is worse than no capability: it burns API calls to produce failures.
    MIN_SAMPLES = 3
    UNRELIABLE_BELOW = 0.5

    @classmethod
    def reliability(cls, row) -> float | None:
        return round(row["successes"] / row["invocations"], 2) if row["invocations"] else None

    @classmethod
    def confidence(cls, row) -> tuple[float, str | None]:
        """How far this capability has earned trust, and why not more.

        Grounded in its own record rather than in the model's self-assessment: a
        capability that has run once and worked is not the same claim as one that
        has worked forty times, and the difference is worth saying out loud.
        """
        inv, suc = row["invocations"], row["successes"]
        if inv == 0:
            return 0.5, f"'{row['name']}' has never run before"
        score = suc / inv
        if inv < cls.MIN_SAMPLES:
            return round(min(0.6, 0.4 + 0.1 * inv), 2), (
                f"'{row['name']}' is unproven — only {inv} prior "
                f"invocation{'s' if inv != 1 else ''}"
            )
        if score < cls.UNRELIABLE_BELOW:
            return round(score, 2), (
                f"'{row['name']}' is unreliable — succeeded {suc} of {inv} times"
            )
        if score < 0.8:
            return round(score, 2), (
                f"'{row['name']}' is inconsistent — succeeded {suc} of {inv} times"
            )
        return round(min(1.0, score), 2), None

    @classmethod
    def is_unreliable(cls, row) -> bool:
        score = cls.reliability(row)
        return (
            row["invocations"] >= cls.MIN_SAMPLES
            and score is not None
            and score < cls.UNRELIABLE_BELOW
        )

    def catalogue(self) -> list[dict]:
        """What the planner is told it can use, best-first.

        Reliability is not decoration: the list is ordered by it, and a capability
        that has proven unreliable is labelled so the planner stops selecting it.
        Enforcement also happens at execution time — see Executor._ensure_reliable.
        """
        out = []
        for row in self._m.list_capabilities():
            score = self.reliability(row)
            out.append(
                {
                    "name": row["name"],
                    "description": row["description"],
                    "params": json.loads(row["params_json"]),
                    "kind": row["kind"],
                    "reliability": score,
                    "status": (
                        "unreliable — avoid unless nothing else fits"
                        if self.is_unreliable(row)
                        else ("proven" if (score or 0) >= 0.8 and row["invocations"] >= self.MIN_SAMPLES
                              else "unproven")
                    ),
                }
            )
        out.sort(key=lambda c: (-(c["reliability"] if c["reliability"] is not None else 0.5)))
        return out

    def register_synthesized(
        self, name: str, description: str, graphql: str, params: list[str],
        provenance: str, tests: str,
    ) -> int:
        return self._m.put_capability(
            name=name, kind="synthesized", description=description,
            handler="graphql", params=params, graphql=graphql,
            provenance=provenance, tests=tests,
        )

    def record(self, name: str, success: bool) -> None:
        self._m.record_invocation(name, success)
