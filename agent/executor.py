"""Executor: resolves each step against the capability registry, runs it, isolates
partial failures, and writes the episode to memory.

Two invariants:
  1. A step whose capability is not registered is BLOCKED, never improvised. This
     is what prevents a false success on an instruction the system cannot perform.
  2. Nothing after a blocked/failed prerequisite is executed — no half-completions.
"""
from __future__ import annotations

import json
import re
import time

from .adapters.linear import LinearAdapter, LinearError
from .capabilities import CapabilityRegistry
from .contracts import (
    CapabilitySpec,
    ExecutionReport,
    Plan,
    PlanStep,
    StepResult,
    StepStatus,
)
from .memory.store import MemoryStore
from .planner import generalize
from .transforms import run_pipeline

TEAMS_Q = "query { teams { nodes { id name } } }"
LABELS_Q = "query { issueLabels { nodes { id name } } }"
CREATE_M = """
mutation Create($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id identifier title url }
  }
}
"""


class Executor:
    def __init__(
        self,
        adapter: LinearAdapter,
        memory: MemoryStore,
        registry: CapabilityRegistry,
        synthesizer=None,
    ):
        self._a = adapter
        self._m = memory
        self._r = registry
        self._s = synthesizer  # optional: enables runtime capability synthesis
        self.synthesis_log: list[str] = []
        self.rollback_log: list[str] = []
        self.prevented = 0  # calls avoided by a learned constraint
        self._allow_duplicate = False
        self._rollback = True
        self._undo: list[dict] = []
        self.synthesized_this_run: set[str] = set()
        self._allow_destructive = False

    def run(
        self,
        plan: Plan,
        llm_calls: int = 0,
        allow_duplicate: bool = False,
        rollback: bool = True,
        allow_destructive: bool = False,
    ) -> ExecutionReport:
        results: list[StepResult] = []
        gaps: list[str] = []
        refused: list[str] = []
        ctx: dict[str, str] = {}
        t0 = time.time()
        halted = False
        self._allow_duplicate = allow_duplicate
        self._rollback = rollback
        self._allow_destructive = allow_destructive
        self._undo = []
        self.rollback_log = []
        self.prevented = 0
        self.synthesized_this_run = set()

        for step in plan.steps:
            if halted:
                results.append(
                    StepResult(
                        step_id=step.id, capability=step.capability,
                        status=StepStatus.skipped,
                        note="not attempted — an earlier step did not complete",
                    )
                )
                continue

            cap = self._r.get(step.capability)
            if cap is None and self._s is not None:
                # Capability gap: try to build the missing capability at runtime.
                try:
                    if self._s.synthesize(step, ctx):
                        cap = self._r.get(step.capability)
                        self.synthesized_this_run.add(step.capability)
                except Exception as exc:  # noqa: BLE001 — synthesis is best-effort
                    self._s.log.append(f"  synthesis aborted: {exc}")
                self.synthesis_log.extend(self._s.log)
                self._s.log = []

            if cap is not None:
                cap = self._ensure_reliable(cap, step, ctx)

            if cap is not None and self._blocks_as_destructive(step):
                # Composition makes the agent more capable than any single
                # capability: once delete_issue exists, 'delete everything' becomes
                # one find plus one loop. Irreversible bulk work therefore needs an
                # explicit go-ahead rather than inheriting the run's authority.
                results.append(
                    StepResult(
                        step_id=step.id, capability=step.capability,
                        status=StepStatus.blocked,
                        error="destructive bulk operation requires explicit confirmation",
                        note=f"'{step.capability}' would be applied to every result of "
                             f"'{step.for_each}'; re-run with --allow-destructive to permit it",
                    )
                )
                # Refused, not missing: the capability exists and was withheld.
                refused.append(step.capability)
                halted = True
                continue

            if cap is None:
                # The honest path: we cannot do this, and we say so.
                gaps.append(step.capability)
                results.append(
                    StepResult(
                        step_id=step.id, capability=step.capability,
                        status=StepStatus.blocked,
                        error=f"no capability registered for '{step.capability}'",
                        note=step.spec.purpose if step.spec else step.description,
                    )
                )
                halted = True
                continue

            confidence, caveat = CapabilityRegistry.confidence(cap)
            calls_before = self._a.call_count
            s0 = time.time()
            try:
                note = self._dispatch(cap, step, ctx)
                status, err = StepStatus.ok, None
            except Exception as exc:  # partial-failure isolation
                status, err, note = StepStatus.failed, str(exc), None
                halted = True

            # Confidence is read BEFORE recording this run's outcome, so it
            # reflects what was known at the moment the decision was made.
            self._r.record(step.capability, status is StepStatus.ok)
            results.append(
                StepResult(
                    step_id=step.id, capability=step.capability, status=status,
                    api_calls=self._a.call_count - calls_before,
                    latency_ms=int((time.time() - s0) * 1000),
                    error=err, note=note,
                    confidence=confidence, caveat=caveat,
                )
            )

        outcome = self._outcome(results, gaps + refused)
        if outcome != "success" and self._rollback and self._undo:
            self._compensate()
            outcome = "rolled_back"

        total_calls = sum(r.api_calls for r in results)
        total_ms = int((time.time() - t0) * 1000)

        self._m.add_episode(
            instruction=plan.instruction,
            signature=plan.intent.key(),
            plan_json=plan.model_dump_json(),
            results_json=json.dumps([r.model_dump() for r in results]),
            outcome=outcome,
            api_calls=total_calls,
            latency_ms=total_ms,
            llm_calls=llm_calls,
            gen_template=generalize(plan.instruction, plan) if outcome == "success" else None,
        )

        scored = [r for r in results if r.confidence is not None]
        confidence = min((r.confidence for r in scored), default=1.0)
        caveats = [r.caveat for r in scored if r.caveat]
        if any(r.capability in self.synthesized_this_run for r in results):
            confidence = min(confidence, 0.5)
            caveats.append(
                "capabilities built during this run have been validated once, not proven"
            )

        # Confidence answers 'should you trust that this run did what was asked?'.
        # A run that did not fulfil the instruction cannot answer that with a high
        # number, however reliable the individual capabilities were — a blocked run
        # reporting 1.0 would be the same false claim the rest of the design exists
        # to prevent.
        if outcome != "success":
            ceiling = {"blocked": 0.0, "failed": 0.1, "partial": 0.2, "rolled_back": 0.2}
            confidence = min(confidence, ceiling.get(outcome, 0.2))
            caveats.insert(0, f"the instruction was not fulfilled (outcome: {outcome})")

        return ExecutionReport(
            instruction=plan.instruction, outcome=outcome, steps=results,
            total_api_calls=total_calls, total_latency_ms=total_ms, gaps=gaps,
            confidence=round(confidence, 2), caveats=caveats,
            summary=self._summary(results, outcome, total_calls, gaps,
                                  round(confidence, 2), caveats, refused),
        )

    # Verbs whose effect cannot be undone by this agent.
    DESTRUCTIVE = ("delete", "remove", "destroy", "purge", "wipe", "archive")

    def _blocks_as_destructive(self, step: PlanStep) -> bool:
        """Irreversible work applied across a whole result set needs consent."""
        if self._allow_destructive or not step.for_each:
            return False
        return step.capability.lower().startswith(self.DESTRUCTIVE)

    def _ensure_reliable(self, cap, step: PlanStep, ctx: dict):
        """Act on the reliability score instead of merely recording it.

        A synthesized capability that has failed more often than it has worked is
        rebuilt before use. Because put_capability versions rather than overwrites,
        the previous version is retained — a worse replacement can never destroy a
        known-good one. Primitives are exempt: they are the floor of the system.
        """
        if cap["kind"] == "primitive" or self._s is None:
            return cap
        if not CapabilityRegistry.is_unreliable(cap):
            return cap

        score = CapabilityRegistry.reliability(cap)
        self.synthesis_log.append(
            f"'{cap['name']}' is unreliable ({cap['successes']}/{cap['invocations']} "
            f"= {score}) — rebuilding before use"
        )
        try:
            if self._s.synthesize(step, ctx):
                self.synthesis_log.extend(self._s.log)
                self._s.log = []
                return self._r.get(step.capability)
        except Exception as exc:  # noqa: BLE001 — keep the old version on failure
            self.synthesis_log.append(f"  rebuild failed ({exc}); using the existing version")
        self.synthesis_log.extend(self._s.log)
        self._s.log = []
        return self._r.get(step.capability)

    # --- dispatch ---
    def _dispatch(self, cap, step: PlanStep, ctx: dict[str, str]) -> str:
        handler = cap["handler"]
        if handler == "builtin:resolve_team":
            return self._resolve("team", step.params.get("name"), TEAMS_Q, "teams", ctx, "team_id")
        if handler == "builtin:resolve_label":
            name = step.params.get("name") or ""
            # Key is case-normalised: the planner may spell the same label
            # differently across steps, and a case mismatch here would silently
            # drop the label from the created issue.
            return self._resolve(
                "label", name, LABELS_Q, "issueLabels", ctx, f"label_id::{name.lower()}"
            )
        if handler == "builtin:create_issue":
            return self._create(step, ctx)
        if handler == "transform":
            return self._transform(cap, step, ctx)
        if handler == "graphql":
            if step.for_each:
                return self._for_each(cap, step, ctx)
            return self._graphql(cap, step, ctx)
        raise LinearError(f"capability '{cap['name']}' has unknown handler '{handler}'")

    def _resolve(self, kind, name, query, root, ctx, ctx_key) -> str:
        if not name:
            raise LinearError(f"{kind} name was not provided")
        cached = self._m.get_resolution(kind, name)
        if cached:
            self._remember(ctx, kind, ctx_key, cached)
            return f"cache hit: {kind} '{name}' -> {cached}  (0 API calls)"

        # Constraint pre-validation: a previous run discovered the valid values for
        # this field, so a name that is not among them is known to fail. Rejecting
        # it here costs 0 API calls, where the first encounter cost 1 wasted call.
        known = self._m.get_constraint(f"linear.{kind}.valid_names")
        if known:
            valid = [v.strip() for v in known.split(",") if v.strip()]
            if valid and name.lower() not in {v.lower() for v in valid}:
                self.prevented += 1
                raise LinearError(
                    f"{kind} '{name}' is known-invalid from a previously learned "
                    f"constraint (valid: {', '.join(valid)}) — skipped the API call"
                )

        data = self._a.execute(query)  # 1 API call, only on a cold cache
        match = next((n for n in data[root]["nodes"] if n["name"].lower() == name.lower()), None)
        if not match:
            available = ", ".join(sorted(n["name"] for n in data[root]["nodes"]))
            # A discovered constraint: this name is not a valid value for this field.
            self._m.put_constraint(
                key=f"linear.{kind}.valid_names", kind="validation", value=available
            )
            raise LinearError(f"{kind} '{name}' not found. Valid {kind}s: {available}")
        self._m.put_resolution(kind, name, match["id"])
        self._remember(ctx, kind, ctx_key, match["id"])
        return f"resolved {kind} '{name}' -> {match['id']}  (cached for next run)"

    @staticmethod
    def _remember(ctx: dict, kind: str, ctx_key: str, value: str) -> None:
        """Expose resolved ids under stable names so synthesized capabilities can
        bind to them ({{team_id}}, {{label_ids}}) without knowing our internals."""
        ctx[ctx_key] = value
        if kind == "label":
            ctx.setdefault("label_ids", []).append(value)

    def _create(self, step: PlanStep, ctx: dict[str, str]) -> str:
        team_id = ctx.get("team_id")
        if not team_id:
            raise LinearError("team was not resolved; cannot create issue")

        # Models return `labels` as either a list or a bare string. Iterating a
        # string yields its characters, which produced a request for eleven
        # single-letter labels — so the shape is normalised before use.
        raw_labels = step.params.get("labels") or []
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels] if raw_labels.strip() else []
        resolved = ctx.get("label_ids", [])
        label_ids: list[str] = []
        missing: list[str] = []
        for raw in raw_labels:
            name = str(raw)
            key = f"label_id::{name.lower()}"
            if key in ctx:
                label_ids.append(ctx[key])
            elif re.fullmatch(r"\{\{?\s*[\w.:]+\s*\}?\}", name) and resolved:
                # The planner sometimes writes a reference to the resolve step
                # instead of the literal name; the ids it means are already in
                # context, so use those rather than failing on the placeholder.
                label_ids.extend(i for i in resolved if i not in label_ids)
            else:
                missing.append(name)
        if missing:
            # Never silently drop part of the request.
            raise LinearError(f"labels requested but not resolved: {missing}")
        title = self._fill(step.params.get("title") or "(untitled)", ctx)

        # Idempotency: this agent already created this issue in an earlier run, so
        # creating it again would duplicate real work rather than fulfil anything.
        if not self._allow_duplicate:
            prior = self._m.find_created("issue", team_id, title)
            if prior:
                ctx["issue_id"] = prior["entity_id"]
                return (
                    f"already exists as {prior['identifier']} — skipped creation "
                    f"(idempotent; use --allow-duplicate to force): {prior['url']}"
                )

        payload = {"teamId": team_id, "title": title}
        description = self._fill(step.params.get("description") or "", ctx)
        if description:
            payload["description"] = description
        if label_ids:
            payload["labelIds"] = label_ids

        data = self._a.execute(CREATE_M, {"input": payload})
        result = data["issueCreate"]
        if not result.get("success"):
            raise LinearError("issueCreate returned success=false")
        issue = result["issue"]
        ctx["issue_id"] = issue["id"]
        self._m.record_created(
            "issue", team_id, title, issue["id"], issue["identifier"], issue["url"]
        )
        # Register for compensation: if a later step fails, this must be undone.
        self._undo.append(
            {"kind": "issue", "entity_id": issue["id"], "label": issue["identifier"]}
        )
        return f"created {issue['identifier']}: {issue['url']}"

    @staticmethod
    def latest_results(ctx: dict, source: str | None = None) -> list[dict]:
        """The rows a transform operates on: an explicitly named source, else the
        most recent result set produced in this run."""
        if source:
            key = source if source.endswith("::results") else f"{source}::results"
            if key in ctx:
                return ctx[key]
        found = [v for k, v in ctx.items() if k.endswith("::results") and isinstance(v, list)]
        if not found:
            raise LinearError(
                "transform has no input: no earlier step produced a result set"
            )
        return found[-1]

    def _transform(self, cap, step: PlanStep, ctx: dict) -> str:
        """Run a synthesized transform pipeline over data retrieved earlier in the run."""
        artifact = json.loads(cap["graphql"])  # same column stores the artifact
        rows = self.latest_results(ctx, artifact.get("source") or step.params.get("source"))
        output = run_pipeline(rows, artifact["pipeline"])

        ctx[f"{cap['name']}::output"] = output
        if isinstance(output, list):
            ctx[f"{cap['name']}::results"] = output
        preview = output if isinstance(output, str) else json.dumps(output)
        return (
            f"{cap['name']} ok (transform over {len(rows)} row(s)) → "
            f"{len(preview)} chars: {preview[:80]}..."
        )

    @staticmethod
    def _fill(value, ctx: dict):
        """Resolve {{name}} references in a step parameter from run context, so a
        transform's output can become an issue description."""
        if not isinstance(value, str):
            return value

        def sub(match):
            key = match.group(1).strip()
            for candidate in (key, f"{key}::output", f"{key}::results"):
                if candidate in ctx:
                    got = ctx[candidate]
                    return got if isinstance(got, str) else json.dumps(got, indent=2)
            return match.group(0)  # unresolved: leave the text untouched

        # Single braces are accepted as well as double: models emit either, and a
        # placeholder that silently survives into an issue body is a visible defect.
        # Substitution only happens when the key actually resolves, so ordinary text
        # containing braces is left alone.
        return re.sub(r"\{\{?\s*([\w.:]+)\s*\}?\}", sub, value)

    def _for_each(self, cap, step: PlanStep, ctx: dict) -> str:
        """Apply this capability once per item produced by an earlier step.

        Iteration lives here rather than in the plan so the number of entities found
        at runtime cannot change the plan's shape — the decomposition stays stable
        and comparable across runs, which the learning metric depends on.
        """
        items = ctx.get(f"{step.for_each}::results")
        if items is None:
            raise LinearError(
                f"step declares for_each='{step.for_each}' but that step produced no "
                f"result set to iterate"
            )
        if not items:
            return f"no items returned by '{step.for_each}' — nothing to apply"

        done, failures = 0, []
        for item in items:
            item_ctx = {**ctx, "issue_id": item.get("id"), "id": item.get("id")}
            try:
                self._graphql(cap, step, item_ctx)
                done += 1
            except Exception as exc:  # noqa: BLE001 — one bad item must not hide the rest
                failures.append(f"{item.get('identifier') or item.get('id')}: {exc}")
        note = f"applied to {done}/{len(items)} item(s) from '{step.for_each}'"
        if failures:
            # A partial across items is reported, never rounded up to success.
            raise LinearError(f"{note}; failures — {'; '.join(failures[:3])}")
        return note

    @staticmethod
    def _extract_nodes(data) -> list[dict] | None:
        """Find the result set in a query response: the first list of objects that
        carry an id. Generic, so it works for any synthesized retrieval."""
        if isinstance(data, dict):
            for value in data.values():
                found = Executor._extract_nodes(value)
                if found is not None:
                    return found
        elif isinstance(data, list):
            if all(isinstance(v, dict) for v in data) and any("id" in v for v in data):
                return data
        return None

    def _graphql(self, cap, step: PlanStep, ctx: dict) -> str:
        """Generic path for synthesized capabilities. The stored artifact is data —
        an operation plus a variables template — so execution is deterministic
        substitution, identical to how it was validated at synthesis time."""
        from .synthesizer import build_variables

        artifact = json.loads(cap["graphql"])
        variables = build_variables(
            artifact["variables_template"],
            artifact.get("value_maps") or {},
            step.params,
            ctx,
        )
        data = self._a.execute(artifact["graphql"], variables)

        # A retrieval's results become available to later steps by name.
        nodes = self._extract_nodes(data)
        if nodes is not None:
            ctx[f"{cap['name']}::results"] = nodes
            preview = ", ".join(
                str(n.get("identifier") or n.get("name") or n.get("id"))[:20] for n in nodes[:5]
            )
            more = f" (+{len(nodes) - 5} more)" if len(nodes) > 5 else ""
            return f"found {len(nodes)} result(s): {preview}{more}" if nodes else "found 0 results"
        return f"{cap['name']} ok (synthesized): {json.dumps(data)[:160]}"

    # --- compensation ---
    def _compensate(self) -> None:
        """Undo completed work in reverse order when the run did not fulfil the
        instruction. Without this, a partial leaves a half-configured entity behind —
        the worst outcome, because it looks like success in the UI.

        The inverse is a capability like any other: if `delete_issue` is not
        registered it is synthesized on demand, then recorded as the inverse of
        `create_issue` so the next rollback needs no reasoning at all.
        """
        self.rollback_log.append(
            f"run did not fulfil the instruction — compensating {len(self._undo)} action(s) in reverse"
        )
        for action in reversed(self._undo):
            name = f"delete_{action['kind']}"
            cap = self._r.get(name)
            if cap is None and self._s is not None:
                need = PlanStep(
                    id="undo",
                    capability=name,
                    description=f"Delete a Linear {action['kind']} by id, to undo a failed run",
                    params={"id": action["entity_id"]},
                    spec=CapabilitySpec(
                        purpose=f"delete an existing {action['kind']} given its id",
                        inputs=["id"],
                        output="success flag",
                    ),
                )
                try:
                    if self._s.synthesize(need, {"issue_id": action["entity_id"]}):
                        cap = self._r.get(name)
                except Exception as exc:  # noqa: BLE001
                    self._s.log.append(f"  inverse synthesis aborted: {exc}")
                self.synthesis_log.extend(self._s.log)
                self._s.log = []

            if cap is None:
                self.rollback_log.append(
                    f"  ✗ {action['label']}: no inverse capability available — LEFT IN PLACE"
                )
                continue
            try:
                self._graphql(cap, PlanStep(id="undo", capability=name,
                                            params={"id": action["entity_id"]}),
                              {"issue_id": action["entity_id"], "id": action["entity_id"]})
                self._m.forget_created(action["kind"], action["entity_id"])
                self._r.record(name, True)
                self._link_inverse(action["kind"], name)
                self.rollback_log.append(f"  ✓ {action['label']}: undone via {name}")
            except Exception as exc:  # noqa: BLE001 — report, never mask
                self._r.record(name, False)
                self.rollback_log.append(
                    f"  ✗ {action['label']}: rollback FAILED ({exc}) — LEFT IN PLACE"
                )

    def _link_inverse(self, kind: str, inverse_name: str) -> None:
        """Persist the create→delete pairing so later rollbacks skip discovery."""
        forward = f"create_{kind}"
        row = self._r.get(inverse_name)
        if self._r.get(forward) is not None and row is not None:
            self._m._db.execute(
                "UPDATE capabilities SET inverse_capability_id=? "
                "WHERE name=? AND superseded=0",
                (inverse_name, forward),
            )
            self._m._db.commit()

    # --- reporting ---
    @staticmethod
    def _outcome(results: list[StepResult], gaps: list[str]) -> str:
        if gaps:
            # Some steps may have succeeded, but the instruction was NOT fulfilled.
            return "blocked"
        if results and all(r.status is StepStatus.ok for r in results):
            return "success"
        if results and all(r.status is StepStatus.failed for r in results):
            return "failed"
        return "partial"

    def _summary(self, results, outcome: str, total_calls: int, gaps: list[str],
                 confidence: float = 1.0, caveats: list[str] | None = None,
                 refused: list[str] | None = None) -> str:
        glyph = {"ok": "✓", "failed": "✗", "skipped": "–", "blocked": "⚠"}
        done = sum(1 for r in results if r.status is StepStatus.ok)
        lines = [f"{outcome.upper()} — {done}/{len(results)} steps completed, {total_calls} API calls"]
        for r in results:
            lines.append(f"  {glyph[r.status.value]} {r.capability}: {r.note or r.error or ''}")
        if confidence < 0.8:
            band = "LOW" if confidence < 0.5 else "MODERATE"
            lines.append("")
            lines.append(f"  CONFIDENCE {band} ({confidence:.2f}) — the agent is unsure about:")
            for c in dict.fromkeys(caveats or []):
                lines.append(f"    · {c}")
        if self.prevented:
            lines.append("")
            lines.append(
                f"  PREVENTED — {self.prevented} API call(s) skipped by a learned constraint"
            )
        if self.rollback_log:
            lines.append("")
            lines.append("  ROLLBACK")
            for entry in self.rollback_log:
                lines.append(f"  {entry}")
        if refused:
            lines.append("")
            lines.append(f"  REFUSED — {', '.join(refused)} withheld for safety")
            lines.append("  The capability exists but the operation is irreversible and")
            lines.append("  would apply to every match. Re-run with --allow-destructive.")
        if gaps:
            lines.append("")
            lines.append(f"  CAPABILITY GAP — missing: {', '.join(gaps)}")
            lines.append("  The instruction was NOT completed. Synthesis is required.")
        return "\n".join(lines)
