"""Metrics Harness — the before/after learning proof, computed from the episodic
store (the source of truth). No LLM call, so it is reproducible offline.

Two rules keep the number honest:
  1. Only SUCCESSFUL runs count. A blocked run makes 0 API calls; counting it as
     an improvement would be measuring failure as progress.
  2. Only runs with the SAME plan shape are compared. Once the agent learned to
     also set priority, the task genuinely costs more calls — comparing it against
     the earlier, less complete plan would be comparing two different tasks.
"""
from __future__ import annotations

import json

from .memory.store import MemoryStore


def _shape(plan_json: str) -> str:
    try:
        steps = json.loads(plan_json).get("steps", [])
        return " → ".join(s.get("capability", "?") for s in steps)
    except Exception:  # noqa: BLE001 — a malformed plan simply has no shape
        return "(unparseable)"


def before_after(memory: MemoryStore, signature: str) -> str:
    rows = memory.episodes_for_signature(signature)
    summary = memory.summary_for(signature)
    if not rows and not summary:
        return f"No runs recorded yet for signature '{signature}'."

    total_runs = len(rows) + (summary["runs"] if summary else 0)
    lines = [
        f"Signature : {signature}",
        f"Runs      : {total_runs}",
    ]
    if summary:
        # Compaction removed the raw rows but kept the baseline, so the earliest
        # cost this signature ever had is still reportable.
        lines.append(
            f"History   : {summary['runs']} earlier run(s) compacted — "
            f"first cost {summary['first_api_calls']} API calls / "
            f"{summary['first_latency_ms']}ms, best since {summary['min_api_calls']}"
        )
    lines += [
        "",
        f"{'run':>4}  {'api':>4}  {'llm':>4}  {'ms':>6}  {'outcome':<8}  plan shape",
        f"{'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*40}",
    ]
    groups: dict[str, list] = {}
    for i, r in enumerate(rows, 1):
        shape = _shape(r["plan_json"])
        llm = r["llm_calls"] if "llm_calls" in r.keys() else 0
        lines.append(
            f"{i:>4}  {r['api_calls']:>4}  {llm:>4}  {r['latency_ms']:>6}  "
            f"{r['outcome']:<8}  {shape}"
        )
        if r["outcome"] == "success":
            groups.setdefault(shape, []).append(r)

    lines.append("")
    # Compare like with like: the most complete successful decomposition.
    best = max(groups.items(), key=lambda kv: (len(kv[1]), len(kv[0])), default=None)
    if not best or len(best[1]) < 2:
        lines.append(
            "→ Not enough comparable successful runs yet. Run the same instruction "
            "again to see the cache take effect (only successful runs with an "
            "identical plan shape are compared)."
        )
        return "\n".join(lines)

    shape, runs = best
    first, last = runs[0], runs[-1]
    lines.append(f"Comparing {len(runs)} successful runs of the plan: {shape}")
    lines.append(
        f"  API calls  : {first['api_calls']} → {last['api_calls']}"
        f"   (best {min(r['api_calls'] for r in runs)})"
    )
    lines.append(
        f"  Latency ms : {first['latency_ms']} → {last['latency_ms']}"
        f"   (best {min(r['latency_ms'] for r in runs)})"
    )

    def llm_of(row) -> int:
        return row["llm_calls"] if "llm_calls" in row.keys() else 0

    lines.append(f"  LLM calls  : {llm_of(first)} → {llm_of(last)}")
    if llm_of(last) < llm_of(first):
        lines.append(
            "→ Learned: the decomposition itself is remembered — a proven plan is "
            "reused for this instruction shape, so no LLM reasoning is needed."
        )
    if last["api_calls"] < first["api_calls"]:
        saved = first["api_calls"] - last["api_calls"]
        lines.append(
            f"→ Learned: {saved} fewer API call(s) per run — name→id resolutions were "
            f"cached in semantic memory, so lookups are no longer repeated."
        )
    else:
        lines.append(
            "→ No reduction on this plan shape yet (the first run already ran fully "
            "warm, or the cache was cleared)."
        )
    return "\n".join(lines)
