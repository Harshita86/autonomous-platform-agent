"""Typed contracts. Every LLM output is validated into one of these before it
runs or enters memory — the 'LLM proposes, the system disposes' boundary."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class StepStatus(str, Enum):
    ok = "ok"
    failed = "failed"
    skipped = "skipped"
    blocked = "blocked"  # no capability exists for this step


def normalize_token(text: str) -> str:
    """Canonical form of one signature component: lowercase, and every run of
    non-alphanumerics collapsed to a single hyphen. Without this the LLM's free
    choice of wording splits one intent across several keys ('high priority' vs
    'high-priority'), which silently fragments the learning metrics."""
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def singularize(word: str) -> str:
    """Conservative singularisation so 'issues' and 'issue' are one intent."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "as")):
        return word[:-1]
    return word


# Wordings that mean the same thing to the platform. Kept explicit and small —
# a synonym table that guesses too much would merge genuinely different intents.
_MODIFIER_ALIASES = {
    "urgent": "urgent-priority",
    "high": "high-priority",
    "highest": "urgent-priority",
    "critical": "urgent-priority",
    "bugs": "bug",
}


class IntentSignature(BaseModel):
    """Canonical identity of a task. This is the memory key — not an embedding —
    so 'the same task' is identifiable and its metrics are aggregable.

    Normalisation is enforced here, at the contract boundary, so no caller can
    write an unnormalised key into memory.
    """

    action: str
    entity: str
    modifiers: list[str] = Field(default_factory=list)

    @field_validator("action", "entity")
    @classmethod
    def _norm_component(cls, value: str) -> str:
        return singularize(normalize_token(value)) or "unknown"

    @field_validator("modifiers")
    @classmethod
    def _norm_modifiers(cls, values: list[str]) -> list[str]:
        out = set()
        for raw in values:
            token = normalize_token(str(raw))
            if token:
                out.add(_MODIFIER_ALIASES.get(token, token))
        return sorted(out)

    def key(self) -> str:
        return f"{self.action}:{self.entity}:{','.join(self.modifiers)}"


def _as_text(value: Any) -> str:
    """Flatten a described field to text.

    Models describe an input either as a name or as a small object
    ({"name": "team_id", "type": "string", "description": ...}). Both are
    reasonable answers to 'what does this capability need', so both are accepted
    and normalised rather than rejected — this is documentation for the
    synthesizer, and being strict about its shape only loses information.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name") or value.get("field") or ""
        detail = value.get("description") or value.get("type") or ""
        return f"{name}: {detail}".strip(": ").strip() or str(value)
    if isinstance(value, list):
        return ", ".join(_as_text(v) for v in value)
    return str(value)


class CapabilitySpec(BaseModel):
    """What a *missing* capability would need to do. Emitted by the planner when
    it names a capability the registry doesn't have — the input to synthesis."""

    purpose: str = ""
    inputs: list[str] = Field(default_factory=list)
    output: str = ""

    @field_validator("purpose", "output", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return _as_text(value) if value is not None else ""

    @field_validator("inputs", mode="before")
    @classmethod
    def _coerce_inputs(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [_as_text(v) for v in value]


class PlanStep(BaseModel):
    id: str
    capability: str  # a name; the registry decides whether it exists
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    spec: CapabilitySpec | None = None  # populated for candidate-new capabilities
    # Name of an earlier step whose results this step should be applied to, once
    # per item. Lets 'find X and update all of them' be one plan instead of an
    # unbounded number of hardcoded steps.
    for_each: str | None = None


class Plan(BaseModel):
    instruction: str
    intent: IntentSignature
    steps: list[PlanStep]


class StepResult(BaseModel):
    step_id: str
    capability: str
    status: StepStatus
    api_calls: int = 0
    latency_ms: int = 0
    error: str | None = None
    note: str | None = None
    # How much the agent trusts the capability this step used, from its own track
    # record. Reported rather than kept internal: acting on a capability that has
    # only ever run once is a different claim from acting on a proven one.
    confidence: float | None = None
    caveat: str | None = None


class ExecutionReport(BaseModel):
    """Structured report returned after every run. No silent half-completions and
    no false successes: a step with no capability is reported as BLOCKED."""

    instruction: str
    outcome: Literal["success", "partial", "failed", "blocked", "rolled_back"]
    steps: list[StepResult]
    total_api_calls: int
    total_latency_ms: int
    summary: str
    gaps: list[str] = Field(default_factory=list)  # capability names that were missing
    confidence: float = 1.0          # the weakest link in the run
    caveats: list[str] = Field(default_factory=list)
