# Autonomous Platform Agent, for Linear

Give it an instruction in plain English and it works out how to do that on a real Linear workspace
through the GraphQL API. It keeps structured memory across sessions and uses that memory to make
different decisions on later runs. When it meets something it can't do, it builds the capability at
runtime: reasons about what's needed, tests it against the live API, and keeps it only if the test
passes.

No agent framework. About 3,300 lines of Python for the system and 800 for tests, on three runtime
dependencies plus one LLM SDK.

- [ARCHITECTURE.md](ARCHITECTURE.md) covers what memory stores, how synthesis works, and what the learning signal is.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill it in
python cli.py doctor        # checks which LLM provider answers
```

`.env` wants `LINEAR_API_KEY`, `LINEAR_TEAM` (any team in your workspace, only used when an
instruction doesn't name one), and at least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or
`GEMINI_API_KEY`. Providers get tried in order and the first that answers wins, so an exhausted
quota is a config change rather than a code change. I found that out the hard way.

`.env` and `memory.db` are gitignored.

## Running it

```bash
python cli.py run "create a bug report for the login timeout issue"
python cli.py run "<instruction>" --dry-run     # show the plan, execute nothing
python cli.py memory                            # all three memory layers
python cli.py metrics create:issue:bug          # before and after numbers
python cli.py compact                           # fold old episodes into aggregates
python cli.py cleanup                           # remove test issues (--confirm to delete)
```

A run exits with code 2 when the instruction wasn't fulfilled, so failure is scriptable.

## How a run works

```
instruction
   |
   +- does a learned sentence pattern match a past success?
   |     yes -> reuse that plan, rebind its values, no model call
   |     no  -> ask the model, then check what came back:
   |              is it echoing my own prompt at me?
   |              does it cover everything the instruction asked for?
   |              are the steps in an order the data allows?
   |
   +- run each step
        +- capability missing -> introspect schema, generate, TEST, register
        +- ids and constraints answered from memory at zero API cost
        +- anything left half done gets compensated in reverse
```

Every check in the middle of that exists because a run did something wrong. One turned "delete all
issues" into a created issue and called it success. Another quietly dropped "high priority" from the
plan and still reported success, and when I opened the issue in Linear the priority wasn't set. A
third wrote my own system prompt into an issue title when I asked it to.

Capabilities are stored as data rather than code: a parametrized GraphQL operation with a variables
template, or a list of named transform steps. Nothing generated gets executed as code, so there's
nothing to sandbox, and you can read every capability the agent taught itself straight out of the
database.

## Dependencies

`pydantic` for the typed contracts LLM output has to pass through, `httpx` for the one place that
speaks HTTP, `python-dotenv` for config. Memory is stdlib `sqlite3`. The three LLM SDKs in
`requirements.txt` are alternatives rather than additions; each adapter imports lazily, so you only
need the one whose key you set.

I didn't reach for an agent framework. The three things this assignment is actually about are the
memory architecture, synthesis, and the learning loop, and those are the three a framework would own
and hide. LangGraph fits this loop and I'd use it in a production service. It wouldn't have changed
a design decision here.

## Layout

```
cli.py                    run, dry-run, memory, metrics, compact, cleanup, doctor
agent/
  contracts.py            typed models, intent signature normalisation
  planner.py              instruction to plan, and every check applied to it
  executor.py             runs the plan, iterates, rolls back, records the episode
  synthesizer.py          introspect, generate, test, register
  transforms.py           declarative pipeline: filter, sort, group, render
  capabilities.py         the registry, and the authority on "can I do this?"
  metrics.py              before and after numbers, read from the episodic store
  memory/store.py         SQLite: episodes, capabilities, constraints, resolutions
  adapters/linear.py      the only file that knows the platform is Linear
tests/
```

## Tests

```bash
pip install pytest && pytest tests/ -q
```

49 of them. The unit and executor tests need no network and no credentials. The executor ones run
against a fake platform on purpose, because the paths that must not regress (partial failure,
multi-step rollback, duplicate suppression, refusing destructive bulk work) are painful to trigger
reliably against a live API. The integration tests use the real one and skip themselves when there
are no keys.

## Memory

| Layer | Table | Holds |
|---|---|---|
| Episodic | `episodes`, `created_entities` | every run, its plan and cost, and what it created |
| Procedural | `capabilities` | each capability, versioned, with success rate and inverse |
| Semantic | `constraints`, `resolutions` | name to id, valid values, permission boundaries, rate limits |

Delete `memory.db` and the agent still works. It's just naive again, which is rather the point.
Memory is supposed to change what the thing does, not sit in a table getting written to and never
read.
