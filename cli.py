"""Single-command entrypoint.

    python cli.py run "create a bug report for the login timeout"
    python cli.py metrics "create a bug report for the login timeout"
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from agent.adapters.linear import LinearAdapter
from agent.capabilities import CapabilityRegistry
from agent.executor import Executor
from agent.memory.store import MemoryStore
from agent.metrics import before_after
from agent.contracts import PlanStep
from agent.planner import Planner, PlanningUnavailable
from agent.synthesizer import Synthesizer


def _build():
    load_dotenv()
    adapter = LinearAdapter(os.environ.get("LINEAR_API_KEY"))
    memory = MemoryStore(os.environ.get("MEMORY_DB", "memory.db"))
    registry = CapabilityRegistry(memory)
    planner = Planner(registry, memory, default_team=os.environ.get("LINEAR_TEAM"))
    synthesizer = Synthesizer(adapter, memory, registry)
    executor = Executor(adapter, memory, registry, synthesizer)
    return adapter, memory, registry, planner, executor


def cmd_run(args: argparse.Namespace) -> None:
    adapter, memory, registry, planner, executor = _build()
    try:
        try:
            plan = planner.plan(args.instruction)
        except PlanningUnavailable as exc:
            print("BLOCKED — 0 steps completed, 0 API calls")
            print(f"  ⚠ planning: {exc}")
            print("\n  The instruction was NOT attempted. No changes were made.")
            raise SystemExit(2)
        if args.dry_run:
            print(f"DRY RUN — planned by {planner.planner_used}, nothing will be executed\n")
            print(f"  instruction : {plan.instruction}")
            print(f"  signature   : {plan.intent.key()}")
            if planner.reuse_note:
                print(f"  memory      : {planner.reuse_note}")
            print("\n  steps:")
            for i, step in enumerate(plan.steps, 1):
                known = registry.has(step.capability)
                mark = "have" if known else "MISSING → would synthesize"
                print(f"    {i}. {step.capability:<26} [{mark}]")
                if step.params:
                    print(f"       params: {step.params}")
                if step.for_each:
                    print(f"       for each result of: {step.for_each}")
                if not known and step.spec:
                    print(f"       purpose: {step.spec.purpose}")
            missing = [s.capability for s in plan.steps if not registry.has(s.capability)]
            print(
                f"\n  {len(plan.steps)} step(s); "
                + (f"{len(missing)} would require synthesis: {', '.join(missing)}"
                   if missing else "all capabilities already available")
            )
            return

        report = executor.run(
            plan,
            llm_calls=planner.llm_calls,
            allow_duplicate=args.allow_duplicate,
            rollback=not args.no_rollback,
            allow_destructive=args.allow_destructive,
        )
        if planner.reuse_note:
            print(f"MEMORY   {planner.reuse_note}\n")
        if executor.synthesis_log:
            print("SYNTHESIS")
            for entry in executor.synthesis_log:
                print(f"  {entry}")
            print()
        print(report.summary)
        print(f"\nplanned by       : {planner.planner_used}")
        print(f"intent signature : {plan.intent.key()}")
        print(f"total API calls  : {report.total_api_calls}")
        print(f"LLM calls        : {planner.llm_calls}")
        print(f"confidence       : {report.confidence:.2f}")
        if report.gaps:
            raise SystemExit(2)  # non-zero exit: the instruction was not fulfilled
    finally:
        adapter.close()
        memory.close()


def cmd_metrics(args: argparse.Namespace) -> None:
    """Reads the episodic store directly — no LLM call, so the learning proof is
    computable offline and independent of any provider."""
    adapter, memory, _r, _p, _e = _build()
    try:
        if args.signature:
            print(before_after(memory, args.signature))
            return
        rows = memory._db.execute(
            "SELECT signature, COUNT(*) n FROM episodes GROUP BY signature ORDER BY n DESC"
        ).fetchall()
        print("Known intent signatures (pass one to see its before/after table):\n")
        for r in rows:
            print(f"  runs={r['n']:<3} {r['signature']}")
    finally:
        adapter.close()
        memory.close()


def cmd_memory(_: argparse.Namespace) -> None:
    """Show current memory state across all three layers — for the before/after demo."""
    adapter, memory, _r, _p, _e = _build()
    try:
        caps = memory.list_capabilities()
        print("PROCEDURAL — capabilities")
        for c in caps:
            rel = f"{c['successes']}/{c['invocations']}" if c["invocations"] else "unused"
            print(f"  {c['name']:<22} v{c['version']}  {c['kind']:<12} {rel}")

        print("\nSEMANTIC — resolutions (name -> id cache)")
        rows = memory._db.execute("SELECT kind, name, value FROM resolutions ORDER BY kind, name")
        for r in rows:
            print(f"  {r['kind']:<8} {r['name']:<16} -> {r['value']}")

        cons = memory.list_constraints()
        print("\nSEMANTIC — constraints discovered at runtime")
        for c in cons:
            print(f"  {c['key']:<30} [{c['kind']}] {c['value'][:60]}")
        if not cons:
            print("  (none yet)")

        print("\nEPISODIC — runs per intent signature")
        rows = memory._db.execute(
            "SELECT signature, COUNT(*) n, MIN(api_calls) lo, MAX(api_calls) hi "
            "FROM episodes GROUP BY signature ORDER BY n DESC"
        )
        for r in rows:
            print(f"  {r['signature']:<34} runs={r['n']}  api_calls {r['hi']} -> {r['lo']}")
    finally:
        adapter.close()
        memory.close()


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Remove issues this agent created during testing.

    Dry-run by default: deletion is irreversible, so it must be asked for
    explicitly. The synthesis sandbox is never deleted — memory references it.
    """
    adapter, memory, registry, _p, executor = _build()
    try:
        sandbox = memory.get_constraint("linear.sandbox.issue_id")
        rows = [
            r for r in memory._db.execute(
                "SELECT * FROM created_entities WHERE kind='issue' ORDER BY ts"
            ).fetchall()
            if r["entity_id"] != sandbox
        ]
        if not rows:
            print("Nothing to clean up: no agent-created issues are recorded in memory.")
            return

        print(f"{len(rows)} issue(s) created by this agent:\n")
        for r in rows:
            print(f"  {r['identifier'] or r['entity_id']:<10} {r['fingerprint'][:60]}")
        if sandbox:
            print(f"\n  (synthesis sandbox {sandbox[:8]}… is protected and will be kept)")

        if not args.confirm:
            print("\nDry run — nothing deleted. Re-run with --confirm to delete these.")
            return

        cap = registry.get("delete_issue")
        if cap is None:
            print("\nNo delete_issue capability yet — run an instruction that triggers "
                  "rollback first, or delete these manually in Linear.")
            raise SystemExit(1)

        deleted = 0
        for r in rows:
            step = PlanStep(id="cleanup", capability="delete_issue",
                            params={"id": r["entity_id"]})
            try:
                executor._graphql(cap, step, {"id": r["entity_id"],
                                              "issue_id": r["entity_id"]})
                memory.forget_created("issue", r["entity_id"])
                deleted += 1
                print(f"  deleted {r['identifier'] or r['entity_id']}")
            except Exception as exc:  # noqa: BLE001 — report, continue
                print(f"  FAILED {r['identifier'] or r['entity_id']}: {exc}")
        print(f"\n{deleted}/{len(rows)} deleted.")
    finally:
        adapter.close()
        memory.close()


def cmd_compact(args: argparse.Namespace) -> None:
    """Fold old episodes into aggregates. Dry run unless --confirm."""
    adapter, memory, _r, _p, _e = _build()
    try:
        report = memory.compact(keep_recent=args.keep, apply=args.confirm)
        if not report:
            print(f"Nothing to compact: every signature has {args.keep} or fewer runs "
                  f"beyond what must be preserved.")
            return
        total = sum(r["folded"] for r in report)
        print(f"{'signature':<40} {'fold':>5} {'keep':>5} {'patterns':>9}")
        print("-" * 62)
        for r in report:
            print(f"{r['signature']:<40} {r['folded']:>5} {r['kept']:>5} "
                  f"{r['patterns_preserved']:>9}")
        print(f"\n{total} episode(s) would fold into per-signature aggregates.")
        print("Preserved: the most recent runs, the cheapest successful run of each "
              "learned pattern, and each signature's first-run baseline.")
        if not args.confirm:
            print("\nDry run — nothing changed. Re-run with --confirm to apply.")
        else:
            print("\nCompacted.")
    finally:
        adapter.close()
        memory.close()


def cmd_doctor(_: argparse.Namespace) -> None:
    """Report which LLM providers are configured and which actually answer.
    Answers 'which of my keys works?' by asking them, not by guessing."""
    load_dotenv()
    from agent.llm import ADAPTERS, ENV_KEYS, LLM

    llm = LLM()
    print(f"provider order: {', '.join(llm.order)}\n")
    for provider in llm.order:
        if provider not in ADAPTERS:
            print(f"  ?  {provider:<10} unknown provider name")
            continue
        if not llm.configured(provider):
            print(f"  –  {provider:<10} no key ({ENV_KEYS[provider]} not set in .env)")
            continue
        model = llm._model_for(provider)
        try:
            ADAPTERS[provider]("Reply with the single word: ok", "ping", model)
            print(f"  ✓  {provider:<10} WORKS   (model: {model})")
        except Exception as exc:  # noqa: BLE001 — reporting, not handling
            print(f"  ✗  {provider:<10} failed  ({model}) — {type(exc).__name__}: {exc}")

    print(f"\nLINEAR_API_KEY   : {'set' if os.environ.get('LINEAR_API_KEY') else 'MISSING'}")
    print(f"LINEAR_TEAM      : {os.environ.get('LINEAR_TEAM') or 'not set'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(required=True)

    p_doc = sub.add_parser("doctor", help="check which LLM providers and keys work")
    p_doc.set_defaults(fn=cmd_doctor)

    p_mem = sub.add_parser("memory", help="show memory state across all three layers")
    p_mem.set_defaults(fn=cmd_memory)

    p_run = sub.add_parser("run", help="execute an instruction on Linear")
    p_run.add_argument("instruction")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="plan only: show the steps and any capability gaps, execute nothing",
    )
    p_run.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="create an entity even if memory says an identical one already exists",
    )
    p_run.add_argument(
        "--allow-destructive",
        action="store_true",
        help="permit an irreversible operation to be applied across every match",
    )
    p_run.add_argument(
        "--no-rollback",
        action="store_true",
        help="leave partial work in place instead of compensating it on failure",
    )
    p_run.set_defaults(fn=cmd_run)

    p_clean = sub.add_parser("cleanup", help="delete issues this agent created while testing")
    p_clean.add_argument("--confirm", action="store_true",
                         help="actually delete (default is a dry run)")
    p_clean.set_defaults(fn=cmd_cleanup)

    p_comp = sub.add_parser("compact", help="fold old episodes into aggregates")
    p_comp.add_argument("--keep", type=int, default=5,
                        help="most recent runs to keep per signature (default 5)")
    p_comp.add_argument("--confirm", action="store_true",
                        help="apply (default is a dry run)")
    p_comp.set_defaults(fn=cmd_compact)

    p_met = sub.add_parser("metrics", help="show run-1 vs run-N for an intent signature")
    p_met.add_argument("signature", nargs="?", help="omit to list all known signatures")
    p_met.set_defaults(fn=cmd_metrics)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
