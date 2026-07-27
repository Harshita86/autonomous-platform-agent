"""End-to-end tests against the real Linear API.

Skipped automatically unless LINEAR_API_KEY and an LLM key are present, so
`pytest tests/` stays runnable with no credentials.

    pytest tests/test_integration.py -q -s

These create real issues in the configured team. Use a sandbox workspace, and
`python cli.py cleanup --confirm` afterwards.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest
from dotenv import load_dotenv

from agent.adapters.linear import LinearAdapter
from agent.capabilities import CapabilityRegistry
from agent.contracts import StepStatus
from agent.executor import Executor
from agent.memory.store import MemoryStore
from agent.planner import Planner
from agent.synthesizer import Synthesizer

load_dotenv()

LLM_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
pytestmark = pytest.mark.skipif(
    not os.environ.get("LINEAR_API_KEY") or not any(os.environ.get(k) for k in LLM_KEYS),
    reason="needs LINEAR_API_KEY and at least one LLM provider key",
)


@pytest.fixture(scope="module")
def agent():
    """Shares the real memory file on purpose: the learning behaviour is only
    meaningful against accumulated history, so these tests must not wipe it."""
    adapter = LinearAdapter(os.environ["LINEAR_API_KEY"])
    memory = MemoryStore(os.environ.get("MEMORY_DB", "memory.db"))
    registry = CapabilityRegistry(memory)
    planner = Planner(registry, memory, default_team=os.environ.get("LINEAR_TEAM"))
    executor = Executor(adapter, memory, registry, Synthesizer(adapter, memory, registry))
    yield adapter, memory, registry, planner, executor
    adapter.close()
    memory.close()


def _run(agent, instruction: str):
    _adapter, _memory, _registry, planner, executor = agent
    plan = planner.plan(instruction)
    report = executor.run(plan, llm_calls=planner.llm_calls)
    return plan, report


def test_instruction_creates_a_real_issue(agent):
    adapter, *_ = agent
    marker = f"integration probe {uuid.uuid4().hex[:8]}"
    _plan, report = _run(agent, f"create a bug titled '{marker}' with label Bug")

    assert report.outcome == "success", report.summary
    # Confirm on the platform itself rather than trusting our own report.
    found = adapter.execute(
        "query($q:String!){ issues(filter:{title:{eq:$q}}) "
        "{ nodes { identifier title labels { nodes { name } } } } }",
        {"q": marker},
    )["issues"]["nodes"]
    assert found, "the agent reported success but Linear has no such issue"
    assert "Bug" in [l["name"] for l in found[0]["labels"]["nodes"]]


def test_memory_makes_the_same_task_cheaper(agent):
    """Cold vs warm: the id cache must change behaviour, not just be written."""
    _adapter, memory, *_ = agent
    memory._db.execute("DELETE FROM resolutions WHERE kind='label' AND name='improvement'")
    memory._db.commit()

    _p1, cold = _run(agent, f"create an issue titled 'cold {uuid.uuid4().hex[:6]}' with label Improvement")
    _p2, warm = _run(agent, f"create an issue titled 'warm {uuid.uuid4().hex[:6]}' with label Improvement")

    assert cold.outcome == "success" and warm.outcome == "success"
    assert warm.total_api_calls < cold.total_api_calls
    assert any("cache hit" in (s.note or "") for s in warm.steps)


def test_learned_pattern_removes_the_llm_call(agent):
    """Plan reuse on unquoted natural language, with parameters rebound."""
    _adapter, _memory, _registry, planner, _executor = agent
    subject = uuid.uuid4().hex[:8]
    _run(agent, f"create a bug report for the {subject} defect")
    plan, report = _run(agent, f"create a bug report for the {subject}b defect")

    assert report.outcome == "success"
    assert planner.planner_used == "memory:reused-plan"
    assert planner.llm_calls == 0
    # The reused plan must carry the NEW subject, never the previous one.
    titles = [str(s.params.get("title", "")) for s in plan.steps]
    assert any(f"{subject}b" in t for t in titles), titles


def test_synthesized_capabilities_are_validated_and_reused(agent):
    _adapter, memory, *_ = agent
    synthesized = [c for c in memory.list_capabilities() if c["kind"] == "synthesized"]
    assert synthesized, "no capability has been synthesized yet — run the demo first"

    for cap in synthesized:
        assert cap["tests_json"], f"{cap['name']} was registered without a test record"
        artifact = json.loads(cap["graphql"])
        # Every artifact is data: either a platform operation or a transform pipeline.
        assert ("graphql" in artifact) or ("pipeline" in artifact), cap["name"]
        assert cap["provenance"]


def test_partial_failure_creates_nothing(agent):
    _plan, report = _run(
        agent, f"create an issue titled 'never {uuid.uuid4().hex[:6]}' with label DoesNotExist"
    )
    assert report.outcome != "success"
    created = [s for s in report.steps
               if s.capability == "create_issue" and s.status is StepStatus.ok]
    assert not created, "a prerequisite failed but an issue was still created"
