"""Fast tests — no network, no LLM, no API keys.

These cover the deterministic gates, which is where the value is: every one of
them exists because a run did something wrong, so a regression here is a
regression in correctness, not style.

    pytest tests/ -q
"""
from __future__ import annotations

import json

import pytest

from agent.contracts import IntentSignature, Plan, PlanStep
from agent.planner import (
    PlanningUnavailable,
    Planner,
    _assert_no_prompt_leak,
    _coverage_gaps,
    _infer_for_each,
    _order_steps,
    _rebind,
    generalize,
    match_generalized,
)
from agent.transforms import TransformError, run_pipeline

ROWS = [
    {"id": "1", "identifier": "ENG-1", "title": "login broken",
     "priorityLabel": "High", "assignee": None},
    {"id": "2", "identifier": "ENG-2", "title": "slow page",
     "priorityLabel": "Low", "assignee": None},
    {"id": "3", "identifier": "ENG-3", "title": "crash",
     "priorityLabel": "High", "assignee": {"name": "Harshita"}},
]


# --- intent signatures: metric integrity -------------------------------------

def test_equivalent_intents_collapse_to_one_key():
    a = IntentSignature(action="Create", entity="Issues", modifiers=["High Priority", "bug"])
    b = IntentSignature(action="create", entity="issue", modifiers=["high-priority", "Bug"])
    assert a.key() == b.key() == "create:issue:bug,high-priority"


def test_different_intents_stay_distinct():
    a = IntentSignature(action="create", entity="issue", modifiers=[])
    b = IntentSignature(action="delete", entity="issue", modifiers=[])
    assert a.key() != b.key()


# --- transforms ---------------------------------------------------------------

def test_pipeline_groups_and_renders():
    out = run_pipeline(ROWS, [
        {"op": "filter", "field": "assignee", "is_null": True},
        {"op": "group_by", "field": "priorityLabel"},
        {"op": "render_markdown", "heading": "Triage"},
    ])
    assert "# Triage" in out
    assert "## High (1)" in out and "## Low (1)" in out
    assert "crash" not in out  # assigned issue was filtered out


def test_pipeline_reads_nested_fields():
    out = run_pipeline(ROWS, [{"op": "filter", "field": "assignee.name",
                               "equals": "Harshita"}])
    assert [r["identifier"] for r in out] == ["ENG-3"]


def test_unknown_operation_is_refused_not_skipped():
    # Silently ignoring an op would drop part of the instruction.
    with pytest.raises(TransformError):
        run_pipeline(ROWS, [{"op": "hallucinated_op"}])


# --- plan reuse: the wrong-parameters bug -------------------------------------

def _plan(instruction: str, title: str) -> Plan:
    return Plan(
        instruction=instruction,
        intent=IntentSignature(action="create", entity="issue"),
        steps=[PlanStep(id="s0", capability="create_issue", params={"title": title})],
    )


def test_learns_pattern_from_unquoted_instruction():
    instruction = "create a bug report for the login timeout issue"
    gen = generalize(instruction, _plan(instruction, "login timeout issue"))
    assert gen is not None
    bound = match_generalized(gen, "create a bug report for the signup crash issue")
    assert bound is not None
    reused = _rebind(json.dumps(json.loads(_plan(instruction, "login timeout issue")
                                           .model_dump_json())),
                     bound, "create a bug report for the signup crash issue")
    assert reused.steps[0].params["title"] == "signup crash issue"


def test_reuse_never_carries_old_parameters():
    """'escalate issue 5' and 'escalate issue 7' once shared a template, so the
    second run reused the first one's number — silent wrong data."""
    old = "escalate issue 5 to high priority"
    gen = generalize(old, _plan(old, "issue 5"))
    bound = match_generalized(gen, "escalate issue 7 to high priority")
    if bound is not None:
        reused = _rebind(_plan(old, "issue 5").model_dump_json(), bound,
                         "escalate issue 7 to high priority")
        assert reused is None or "5" not in reused.steps[0].params["title"]


def test_unrelated_instruction_does_not_match_pattern():
    instruction = "create a bug report for the login timeout issue"
    gen = generalize(instruction, _plan(instruction, "login timeout issue"))
    assert match_generalized(gen, "delete every issue in the workspace") is None


# --- ordering and iteration ----------------------------------------------------

def test_retrieval_then_shaping_then_create():
    steps = [
        PlanStep(id="0", capability="create_weekly_triage_summary_page"),
        PlanStep(id="1", capability="group_issues_by_priority"),
        PlanStep(id="2", capability="find_open_unassigned_issues"),
    ]
    order = [s.capability for s in _order_steps(_infer_for_each(steps))]
    assert order.index("find_open_unassigned_issues") == 0
    assert order.index("group_issues_by_priority") < order.index(
        "create_weekly_triage_summary_page"
    )


def test_shaping_step_never_iterates_per_item():
    steps = [
        PlanStep(id="0", capability="find_bugs"),
        PlanStep(id="1", capability="group_by_priority", for_each="find_bugs"),
    ]
    out = _infer_for_each(steps)
    assert out[1].for_each is None  # a transform consumes the whole set


def test_mutation_is_bound_to_the_retrieval():
    steps = [
        PlanStep(id="0", capability="find_unassigned_bugs"),
        PlanStep(id="1", capability="set_issue_priority", params={"priority": "high"}),
    ]
    out = _infer_for_each(steps)
    assert out[1].for_each == "find_unassigned_bugs"


# --- the gates -----------------------------------------------------------------

def test_coverage_gate_catches_dropped_priority():
    gaps = _coverage_gaps("create a high priority bug", [{"capability": "create_issue"}])
    assert gaps, "a plan with no priority step must be rejected"


def test_coverage_gate_passes_when_addressed():
    gaps = _coverage_gaps(
        "create a high priority bug",
        [{"capability": "create_issue"}, {"capability": "set_issue_priority"}],
    )
    assert gaps == []


def test_prompt_leak_gate_blocks_exfiltration():
    system = "You are the planner for an agent that operates on Linear an issue tracker"
    steps = [{"params": {"title": "You are the planner for an agent that operates on "
                                  "Linear an issue tracker"}}]
    with pytest.raises(PlanningUnavailable):
        _assert_no_prompt_leak(system, steps)


def test_prompt_leak_gate_allows_normal_titles():
    system = "You are the planner for an agent that operates on Linear an issue tracker"
    _assert_no_prompt_leak(system, [{"params": {"title": "checkout fails on mobile safari"}}])


# --- fallback refuses rather than guessing --------------------------------------

@pytest.mark.parametrize(
    "instruction",
    [
        "delete all issues in the Engineering team",   # destructive verb
        "change the title of ENG-1 to 'triaged'",      # not a create
        "asdkjh qwe zxcv",                             # no intent
        "create two issues: 'one' and 'two'",          # would half-complete
    ],
)
def test_fallback_refuses_what_it_cannot_plan(instruction):
    with pytest.raises(PlanningUnavailable):
        Planner._extract_fallback(instruction)


def test_fallback_handles_a_simple_create():
    raw = Planner._extract_fallback("create an issue titled 'disk full' with label Bug")
    assert raw["steps"][-1]["capability"] == "create_issue"
    assert raw["steps"][-1]["params"]["title"] == "disk full"


def test_reuse_rebinds_values_inside_list_parameters():
    """A learned pattern that masked only scalars once produced a plan that
    resolved one label and then asked to attach a different one."""
    instruction = "create an issue titled 'disk full' with label Bug"
    plan = Plan(
        instruction=instruction,
        intent=IntentSignature(action="create", entity="issue"),
        steps=[
            PlanStep(id="s0", capability="resolve_label", params={"name": "Bug"}),
            PlanStep(id="s1", capability="create_issue",
                     params={"title": "disk full", "labels": ["Bug"]}),
        ],
    )
    gen = generalize(instruction, plan)
    bound = match_generalized(gen, "create an issue titled 'cold probe' with label Improvement")
    assert bound is not None
    reused = _rebind(plan.model_dump_json(), bound,
                     "create an issue titled 'cold probe' with label Improvement",
                     instruction)
    assert reused is not None
    assert reused.steps[0].params["name"] == "Improvement"
    assert reused.steps[1].params["labels"] == ["Improvement"]
    assert reused.steps[1].params["title"] == "cold probe"


def test_for_each_pointing_at_a_single_create_is_dropped():
    """A model told to pair find with for_each sometimes points it at create_issue,
    which produces one entity and no set — the run then rolled back needlessly."""
    from agent.planner import _valid_for_each
    steps = [
        PlanStep(id="0", capability="create_issue", params={"title": "x"}),
        PlanStep(id="1", capability="set_issue_priority", for_each="create_issue"),
    ]
    out = _valid_for_each(steps)
    assert out[1].for_each is None


def test_for_each_pointing_at_a_retrieval_is_kept():
    from agent.planner import _valid_for_each
    steps = [
        PlanStep(id="0", capability="find_unassigned_bugs"),
        PlanStep(id="1", capability="set_issue_priority", for_each="find_unassigned_bugs"),
    ]
    assert _valid_for_each(steps)[1].for_each == "find_unassigned_bugs"


def test_value_map_tolerates_the_phrasing_a_planner_uses():
    """'high priority' reached the API as a string where an Int was expected,
    because the map is keyed on the bare word."""
    from agent.synthesizer import _map_value
    m = {"urgent": 1, "high": 2, "medium": 3, "low": 4}
    assert _map_value("high priority", m) == 2
    assert _map_value("High Priority", m) == 2
    assert _map_value("Urgent!", m) == 1
    assert _map_value("bogus", m) is None


def test_unmapped_value_fails_with_a_readable_message():
    from agent.synthesizer import build_variables
    from agent.adapters.linear import LinearError
    with pytest.raises(LinearError) as exc:
        build_variables({"p": "{{priority}}"}, {"priority": {"high": 2}},
                        {"priority": "bogus"}, {})
    assert "known values" in str(exc.value)


def test_step_name_placeholders_resolve_to_the_value_they_produced():
    """A synthesized template referencing {resolve_team} reached the API as
    literal text and failed as 'eq must be a UUID'."""
    from agent.synthesizer import build_variables
    out = build_variables(
        {"filter": {"team": {"id": {"eq": "{resolve_team}"}}}},
        {}, {}, {"team_id": "a186-uuid"},
    )
    assert out["filter"]["team"]["id"]["eq"] == "a186-uuid"


def test_single_and_double_brace_placeholders_both_bind():
    from agent.synthesizer import build_variables
    ctx = {"issue_id": "abc"}
    assert build_variables({"id": "{{issue_id}}"}, {}, {}, ctx)["id"] == "abc"
    assert build_variables({"id": "{issue_id}"}, {}, {}, ctx)["id"] == "abc"


def test_unresolvable_placeholder_names_what_was_available():
    from agent.synthesizer import build_variables
    from agent.adapters.linear import LinearError
    with pytest.raises(LinearError) as exc:
        build_variables({"id": "{{nope}}"}, {}, {}, {"team_id": "x"})
    assert "available" in str(exc.value)


def test_param_value_that_is_itself_a_step_placeholder_resolves_via_context():
    """The planner wrote params={'issue_id': '{create_issue}', ...} instead of a
    plain identifier. Trusting that literal sent the text '{create_issue}' to
    Linear as an id, which failed as 'Entity not found: Issue' and triggered an
    unnecessary rollback of a real issue."""
    from agent.synthesizer import build_variables
    out = build_variables(
        {"id": "{{issue_id}}", "input": {"priority": "{{priority}}"}},
        {"priority": {"high": 2}},
        {"issue_id": "{create_issue}", "priority": "High"},
        {"issue_id": "real-uuid-here"},
    )
    assert out["id"] == "real-uuid-here"
    assert out["input"]["priority"] == 2


def test_ordinary_literal_param_values_still_pass_through_untouched():
    from agent.synthesizer import build_variables
    out = build_variables({"title": "{{title}}"}, {}, {"title": "a plain title"}, {})
    assert out["title"] == "a plain title"


def test_a_description_restating_the_instruction_does_not_block_the_title_slot():
    """The description param often nearly repeats the whole instruction. Treated
    as the longest slot it swallowed the entire sentence, so the title's shorter
    span could no longer be found and generalize() silently returned None,
    meaning nothing was learned from that run at all."""
    instruction = "log a defect about the notification delay problem"
    plan = Plan(
        instruction=instruction,
        intent=IntentSignature(action="create", entity="issue"),
        steps=[PlanStep(id="s0", capability="create_issue", params={
            "title": "Notification delay problem",
            "description": "Log a defect about the notification delay problem",
        })],
    )
    gen = generalize(instruction, plan)
    assert gen is not None
    bound = match_generalized(gen, "log a defect about the search index corruption problem")
    assert bound is not None


def test_restatement_field_is_refreshed_not_treated_as_a_leak():
    """A description restating almost the whole old instruction is correctly left
    unmasked (masking it would swallow the pattern), but that made it stale by
    construction and the safety net blocked every reuse of this plan shape. It
    should be refreshed to the new instruction instead of blocking reuse."""
    old = "log a defect about the cart total mismatch problem"
    new = "log a defect about the shipping calculator crash problem"
    plan = Plan(
        instruction=old,
        intent=IntentSignature(action="create", entity="issue"),
        steps=[PlanStep(id="s0", capability="create_issue", params={
            "title": "Cart total mismatch problem",
            "description": "Log a defect about the cart total mismatch problem",
        })],
    )
    gen = generalize(old, plan)
    bound = match_generalized(gen, new)
    assert bound is not None
    reused = _rebind(plan.model_dump_json(), bound, new, old)
    assert reused is not None
    assert reused.steps[0].params["title"] == "shipping calculator crash problem"
    assert "cart total mismatch" not in reused.steps[0].params["description"].lower()


def test_a_genuine_short_leak_still_blocks_reuse():
    """The floor only forgives near-full-sentence restatements. A short stale
    fragment (the original masking-bug scenario) must still refuse reuse."""
    reused = _rebind(
        Plan(instruction="x", intent=IntentSignature(action="create", entity="issue"),
             steps=[PlanStep(id="s0", capability="create_issue",
                             params={"labels": ["Bug"]})]).model_dump_json(),
        {"values": {}, "slots": []},
        "create an issue titled 'y'",
        "create a bug titled 'x' with label Bug",
    )
    assert reused is None  # a short leaked fragment still refuses reuse
