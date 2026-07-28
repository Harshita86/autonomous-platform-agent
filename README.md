# Autonomous Platform Agent — Linear

An agent that takes an instruction in plain English and carries it out on a real Linear workspace
through the GraphQL API. It keeps structured memory across sessions, uses that memory to make
different decisions on later runs, and when it meets something it can't do it builds that
capability at runtime — reasoning about what's needed, testing it against the live API, and keeping
it only if the test passes.

No agent framework: about 3,300 lines of Python for the system, plus 800 of tests. Three runtime
dependencies and one LLM SDK.

- [ARCHITECTURE.md](ARCHITECTURE.md) — what memory stores, how synthesis works, what the learning signal is
- [DEMO.md](DEMO.md) — the three instructions for the walkthrough

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill it in
python cli.py doctor        # checks which LLM provider answers
```

`.env` needs `LINEAR_API_KEY`, `LINEAR_TEAM` (any team in your workspace, used only when an
instruction doesn't name one), and at least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or
`GEMINI_API_KEY`. Providers are tried in order and the first that answers wins, so an exhausted
quota is a config change rather than a code change.

`.env` and `memory.db` are gitignored.

## Running it

```bash
python cli.py run "create a bug report for the login timeout issue"
python cli.py run "<instruction>" --dry-run     # show the plan, execute nothing
python cli.py memory                            # all three memory layers
python cli.py metrics create:issue:bug          # before/after learning numbers
python cli.py compact                           # fold old episodes into aggregates
python cli.py cleanup                           # remove test issues (--confirm to delete)
```

A run exits with code 2 when the instruction wasn't fulfilled, so failure is scriptable.

## How a run works

```
instruction
   │
   ├─ does a learned sentence pattern match a past success?
   │     yes → reuse that plan, rebind its values, no model call
   │     no  → ask the model, then check what it produced:
   │              does the plan echo my own prompt back at me?
   │              does it cover everything the instruction asked for?
   │              are the steps in an order the data allows?
   │
   └─ run each step
        ├─ capability missing → introspect the schema, generate, TEST, register
        ├─ ids and constraints answered from memory at zero API cost
        └─ anything left half-done is compensated in reverse
```

Every check in the middle of that exists because a run did something wrong. One turned "delete all
issues" into a created issue and called it a success. Another quietly dropped "high priority" from
the plan and still reported success — the issue in Linear had no priority set. A third wrote my own
system prompt into an issue title when asked to.

Capabilities are stored as data rather than code: a parametrized GraphQL operation with a variables
template, or a list of named transform steps. Nothing generated is executed as code, so there's
nothing to sandbox, and you can read every capability the agent has taught itself straight out of
the database.

## Dependencies

`pydantic` for the typed contracts LLM output has to pass through, `httpx` for the one place that
speaks HTTP, and `python-dotenv` for config. Memory is stdlib `sqlite3`. The three LLM SDKs in
`requirements.txt` are alternatives, not additions — each adapter imports lazily, so you only need
the one whose key you set.

I didn't use an agent framework. The three things this assignment is actually about — the memory
architecture, synthesis, and the learning loop — are the three a framework would own and hide.
LangGraph would fit the plan/execute/reflect loop and I'd reach for it in a production service, but
it wouldn't have changed a single design decision here.

## Layout

```
cli.py                    run · dry-run · memory · metrics · compact · cleanup · doctor
agent/
  contracts.py            typed models; intent-signature normalisation
  planner.py              instruction → plan, and every check applied to it
  executor.py             runs the plan, iterates, rolls back, records the episode
  synthesizer.py          introspect → generate → test → register
  transforms.py           declarative pipeline: filter, sort, group, render
  capabilities.py         the registry — the authority on "can I do this?"
  metrics.py              before/after numbers, read from the episodic store
  memory/store.py         SQLite: episodes, capabilities, constraints, resolutions
  adapters/linear.py      the only file that knows the platform is Linear
tests/
```

## Tests

```bash
pip install pytest && pytest tests/ -q
```

49 tests. The unit and executor tests need no network or credentials; the executor ones run against
a fake platform because the paths that must not regress — partial failure, multi-step rollback,
duplicate suppression, refusing destructive bulk work — are hard to trigger reliably against a live
API. The integration tests use the real one and skip themselves when there are no keys.

## Memory

| Layer | Table | Holds |
|---|---|---|
| Episodic | `episodes`, `created_entities` | every run, its plan and cost, and what it created |
| Procedural | `capabilities` | each capability, versioned, with success rate and inverse |
| Semantic | `constraints`, `resolutions` | name→id, valid values, permission boundaries, rate limits |

Delete `memory.db` and the agent still works. It's just naive again, which is the point: memory is
supposed to change behaviour, not sit in a table being written to and never read.
