"""Executor behaviour against a fake platform.

A fake adapter is used deliberately: partial failure and rollback are the paths
that must never regress, and they are painful to trigger reliably against a live
API. The end-to-end tests against real Linear live in test_integration.py — this
file is about the failure paths.
"""
from __future__ import annotations

import json

import pytest

from agent.capabilities import CapabilityRegistry
from agent.contracts import IntentSignature, Plan, PlanStep, StepStatus
from agent.executor import Executor
from agent.memory.store import MemoryStore
from agent.metrics import before_after


class FakeAdapter:
    """Speaks enough GraphQL to exercise the executor. `fail_on` makes a chosen
    operation raise, so failure paths are deterministic."""

    def __init__(self, fail_on: str | None = None):
        self.call_count = 0
        self.fail_on = fail_on
        self.operations: list[str] = []
        self._next_issue = 1

    def execute(self, query: str, variables: dict | None = None) -> dict:
        self.call_count += 1
        self.operations.append(query.strip().split("\n")[0][:40])

        if self.fail_on and self.fail_on in query:
            raise RuntimeError(f"simulated platform failure in {self.fail_on}")

        if "teams" in query:
            return {"teams": {"nodes": [{"id": "team-1", "name": "Engineering"}]}}
        if "issueLabels" in query:
            return {"issueLabels": {"nodes": [{"id": "label-1", "name": "Bug"}]}}
        if "issueCreate" in query:
            n = self._next_issue
            self._next_issue += 1
            return {"issueCreate": {"success": True, "issue": {
                "id": f"issue-{n}", "identifier": f"FAKE-{n}",
                "title": (variables or {}).get("input", {}).get("title", ""),
                "url": f"https://example.invalid/FAKE-{n}"}}}
        if "issueDelete" in query:
            return {"issueDelete": {"success": True}}
        return {"ok": True}

    def close(self) -> None:
        pass


@pytest.fixture
def wiring(tmp_path):
    memory = MemoryStore(str(tmp_path / "test.db"))
    registry = CapabilityRegistry(memory)
    return memory, registry


def _plan(steps: list[PlanStep], instruction: str = "test") -> Plan:
    return Plan(
        instruction=instruction,
        intent=IntentSignature(action="create", entity="issue"),
        steps=steps,
    )


def test_successful_run_reports_every_step(wiring):
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="create_issue", params={"title": "disk full"}),
    ]))
    assert report.outcome == "success"
    assert [s.status for s in report.steps] == [StepStatus.ok, StepStatus.ok]


def test_failure_halts_and_marks_later_steps_skipped(wiring):
    """The half-completion guarantee: nothing after a failure is attempted, and
    every step is still accounted for in the report."""
    memory, registry = wiring
    adapter = FakeAdapter(fail_on="teams")
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="create_issue", params={"title": "never created"}),
    ]))
    assert report.outcome != "success"
    assert report.steps[0].status is StepStatus.failed
    assert report.steps[1].status is StepStatus.skipped
    assert not any("issueCreate" in op for op in adapter.operations)


def test_missing_capability_blocks_and_does_not_improvise(wiring):
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="teleport_issue"),
    ]))
    assert report.outcome == "blocked"
    assert report.gaps == ["teleport_issue"]
    assert adapter.call_count == 0  # nothing was attempted


def test_rollback_compensates_multiple_actions_in_reverse(wiring):
    """Item under test: a run that created several entities before failing must
    undo all of them, newest first."""
    memory, registry = wiring
    adapter = FakeAdapter()
    # Register the inverse up front so the test needs no LLM.
    registry.register_synthesized(
        name="delete_issue",
        description="delete an issue by id",
        graphql=json.dumps({
            "graphql": "mutation D($id: String!) { issueDelete(id: $id) { success } }",
            "variables_template": {"id": "{{id}}"},
            "value_maps": {},
        }),
        params=["id"],
        provenance="registered by test",
        tests="{}",
    )

    executor = Executor(adapter, memory, registry)
    report = executor.run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="create_issue", params={"title": "first"}),
        PlanStep(id="2", capability="create_issue", params={"title": "second"},
                 for_each=None),
        PlanStep(id="3", capability="teleport_issue"),  # forces the run to fail
    ]))

    assert report.outcome == "rolled_back"
    undone = [line for line in executor.rollback_log if "undone" in line]
    assert len(undone) == 2, executor.rollback_log
    # Reverse order: the most recently created entity is undone first.
    assert "FAKE-2" in undone[0] and "FAKE-1" in undone[1]
    assert memory.find_created("issue", "team-1", "first") is None
    assert memory.find_created("issue", "team-1", "second") is None


def test_rollback_can_be_disabled(wiring):
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(
        _plan([
            PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
            PlanStep(id="1", capability="create_issue", params={"title": "kept"}),
            PlanStep(id="2", capability="teleport_issue"),
        ]),
        rollback=False,
    )
    assert report.outcome == "blocked"
    assert memory.find_created("issue", "team-1", "kept") is not None


def test_idempotency_skips_a_duplicate_create(wiring):
    memory, registry = wiring
    adapter = FakeAdapter()
    executor = Executor(adapter, memory, registry)
    steps = [
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="create_issue", params={"title": "same title"}),
    ]
    executor.run(_plan(steps))
    calls_after_first = adapter.call_count
    second = executor.run(_plan(steps))

    assert second.outcome == "success"
    assert adapter.call_count == calls_after_first  # no new platform call
    assert "already exists" in (second.steps[1].note or "")


def test_learned_constraint_prevents_a_doomed_call(wiring):
    memory, registry = wiring
    memory.put_constraint("linear.label.valid_names", "validation", "Bug, Feature")
    adapter = FakeAdapter()
    executor = Executor(adapter, memory, registry)
    report = executor.run(_plan([
        PlanStep(id="0", capability="resolve_label", params={"name": "Blocker"}),
    ]))
    assert report.outcome != "success"
    assert executor.prevented == 1
    assert adapter.call_count == 0  # the call was never made


def test_labels_are_never_silently_dropped(wiring):
    """A case mismatch between steps once dropped the label without any error."""
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="resolve_label", params={"name": "bug"}),
        PlanStep(id="2", capability="create_issue",
                 params={"title": "x", "labels": ["Bug"]}),  # different case
    ]))
    assert report.outcome == "success"
    assert report.steps[2].status is StepStatus.ok


def test_labels_given_as_a_bare_string_are_not_split_into_characters(wiring):
    """A model returned labels="Improvement"; iterating the string asked Linear
    for eleven single-letter labels."""
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="resolve_label", params={"name": "Bug"}),
        PlanStep(id="2", capability="create_issue",
                 params={"title": "x", "labels": "Bug"}),  # a string, not a list
    ]))
    assert report.outcome == "success", report.summary


def test_label_placeholder_falls_back_to_resolved_ids(wiring):
    """Planners sometimes write "{resolve_label}" instead of the literal name;
    the ids it refers to are already in context."""
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="resolve_label", params={"name": "Bug"}),
        PlanStep(id="2", capability="create_issue",
                 params={"title": "x", "labels": ["{resolve_label}"]}),
    ]))
    assert report.outcome == "success", report.summary


def test_unresolvable_label_still_fails_loudly(wiring):
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="create_issue",
                 params={"title": "x", "labels": ["NeverResolved"]}),
    ]))
    assert report.outcome != "success"


def test_report_states_low_confidence_for_an_unproven_capability(wiring):
    """Knowing is not enough — the run has to say what it is unsure about."""
    memory, registry = wiring
    registry.register_synthesized(
        name="set_thing", description="set a thing",
        graphql=json.dumps({"graphql": "mutation { ok }",
                            "variables_template": {}, "value_maps": {}}),
        params=[], provenance="test", tests="{}",
    )
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="set_thing"),
    ]))
    assert report.confidence < 0.8
    assert any("set_thing" in c for c in report.caveats), report.caveats
    assert "CONFIDENCE" in report.summary


def test_proven_primitive_reports_full_confidence(wiring):
    memory, registry = wiring
    adapter = FakeAdapter()
    executor = Executor(adapter, memory, registry)
    steps = [PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"})]
    for _ in range(6):
        executor.run(_plan(steps))
    report = executor.run(_plan(steps))
    assert report.confidence >= 0.8
    assert report.caveats == []
    assert "CONFIDENCE" not in report.summary


def test_compaction_preserves_baseline_patterns_and_recent_runs(tmp_path):
    """Compaction is the one exception to append-only history, so it must not
    cost the agent anything it actually uses."""
    memory = MemoryStore(str(tmp_path / "compact.db"))
    pattern = '{"pattern": "create a bug for the \\u00000\\u0000", "slots": []}'
    for i in range(12):
        memory.add_episode(
            instruction=f"create a bug for the thing {i}",
            signature="create:issue:bug",
            plan_json='{"instruction":"x","intent":{"action":"create","entity":"issue"},"steps":[]}',
            results_json="[]",
            outcome="success",
            api_calls=9 - min(i, 8),      # gets cheaper over time
            latency_ms=100 + i,
            llm_calls=1 if i == 0 else 0,
            gen_template=pattern,
        )
    before = memory.episodes_for_signature("create:issue:bug")
    assert len(before) == 12

    memory.compact(keep_recent=3, apply=True)
    after = memory.episodes_for_signature("create:issue:bug")

    # Recent runs survive, and so does the cheapest run of the learned pattern.
    assert 3 <= len(after) < 12
    assert min(r["api_calls"] for r in after) == 1
    # Patterns still available for reuse.
    assert memory.successful_patterns(), "compaction destroyed the learned pattern"

    # The baseline needed for the before/after claim is preserved in the aggregate.
    summary = memory.summary_for("create:issue:bug")
    assert summary["first_api_calls"] == 9
    assert summary["runs"] + len(after) == 12
    assert "9" in before_after(memory, "create:issue:bug")


def test_unfulfilled_runs_cannot_claim_high_confidence(wiring):
    """A blocked run executed nothing; reporting full confidence in it would be
    the same false claim the rest of the design prevents."""
    memory, registry = wiring
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="nonexistent_thing"),
    ]))
    assert report.outcome == "blocked"
    assert report.confidence == 0.0
    assert any("not fulfilled" in c for c in report.caveats), report.caveats


def test_rolled_back_run_reports_low_confidence(wiring):
    memory, registry = wiring
    adapter = FakeAdapter()
    registry.register_synthesized(
        name="delete_issue", description="delete an issue by id",
        graphql=json.dumps({
            "graphql": "mutation D($id: String!) { issueDelete(id: $id) { success } }",
            "variables_template": {"id": "{{id}}"}, "value_maps": {}}),
        params=["id"], provenance="test", tests="{}",
    )
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="resolve_team", params={"name": "Engineering"}),
        PlanStep(id="1", capability="create_issue", params={"title": "x"}),
        PlanStep(id="2", capability="nonexistent_thing"),
    ]))
    assert report.outcome == "rolled_back"
    assert report.confidence <= 0.2


def test_destructive_bulk_operation_is_refused_without_consent(wiring):
    """Once delete_issue exists, 'delete everything' is one find plus one loop.
    Irreversible bulk work must not inherit the run's authority."""
    memory, registry = wiring
    registry.register_synthesized(
        name="delete_issue", description="delete an issue by id",
        graphql=json.dumps({
            "graphql": "mutation D($id: String!) { issueDelete(id: $id) { success } }",
            "variables_template": {"id": "{{id}}"}, "value_maps": {}}),
        params=["id"], provenance="test", tests="{}",
    )
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="delete_issue", for_each="find_everything"),
    ]))
    assert report.outcome == "blocked"
    assert "REFUSED" in report.summary
    assert not any("issueDelete" in op for op in adapter.operations)


def test_destructive_bulk_runs_when_explicitly_allowed(wiring):
    memory, registry = wiring
    registry.register_synthesized(
        name="delete_issue", description="delete an issue by id",
        graphql=json.dumps({
            "graphql": "mutation D($id: String!) { issueDelete(id: $id) { success } }",
            "variables_template": {"id": "{{id}}"}, "value_maps": {}}),
        params=["id"], provenance="test", tests="{}",
    )
    adapter = FakeAdapter()
    executor = Executor(adapter, memory, registry)
    report = executor.run(
        _plan([PlanStep(id="0", capability="delete_issue", for_each="finder")]),
        allow_destructive=True,
    )
    # No result set exists, so it fails honestly rather than being refused.
    assert "REFUSED" not in report.summary


def test_single_delete_is_not_treated_as_bulk(wiring):
    """The guard is about breadth, not about the verb."""
    memory, registry = wiring
    registry.register_synthesized(
        name="delete_issue", description="delete an issue by id",
        graphql=json.dumps({
            "graphql": "mutation D($id: String!) { issueDelete(id: $id) { success } }",
            "variables_template": {"id": "{{id}}"}, "value_maps": {}}),
        params=["id"], provenance="test", tests="{}",
    )
    adapter = FakeAdapter()
    report = Executor(adapter, memory, registry).run(_plan([
        PlanStep(id="0", capability="delete_issue", params={"id": "issue-1"}),
    ]))
    assert report.outcome == "success", report.summary
