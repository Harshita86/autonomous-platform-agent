# Architecture

Platform: Linear, because the GraphQL schema is introspectable. The agent can ask the platform what
operations exist instead of reading a list I wrote down, and that's what separates synthesis from a
lookup table.

Two ideas run through the rest. The model proposes and the code disposes: anything the LLM returns
clears a deterministic check before it touches state. And capabilities are stored as data, not code.
Neither was drawn up front. My first version took "delete all issues in the Engineering team",
created a new issue titled "delete all issues in the Engineering team", and reported SUCCESS. It did
the wrong thing and told me it went fine. Everything defensive here traces back to a run like that.

One boundary worth knowing: 38 lines in `adapters/linear.py` know which platform this is.

## 1. What memory stores, and why it's shaped that way

One SQLite file, three tables, split by what the knowledge is for.

`episodes` holds each run: instruction, plan, per-step results, cost. It also holds the sentence
pattern that run answers, with the wording-specific values masked out. That mask is what lets a
proven decomposition get reused later.

`capabilities` holds every capability, versioned, with its success rate, the test it passed, and its
inverse. A synthesized one is stored and called exactly like a built-in.

`constraints` and `resolutions` hold what it learned about Linear. Name to id mappings. Which label
names are valid, learned by getting one wrong. A permission boundary it hit when my plan wouldn't
let it create a team. Linear's 10,000 query complexity limit, found because introspecting every
mutation at once got rejected.

No vector store, deliberately. I don't want roughly similar past prompts, I want to know whether
I've done this exact task and what it cost. That's a keyed lookup, and it's what makes the metric
aggregable.

One writer owns all three tables, so three invariants hold. Episodes are append only, because the
before and after claim depends on it. Capabilities are versioned rather than overwritten, so a bad
re-synthesis can't destroy a working one. Constraints only accumulate. Compaction is the one
deliberate exception, and it keeps the first run baseline and every learned pattern.

## 2. How capability synthesis works

Three capabilities are built in. Everything else gets built at runtime or honestly refused.

When the planner names one that doesn't exist, the synthesizer introspects the live schema and
shortlists real operations, ranked on Linear's `<entity><Verb>` naming. Reads and writes sit in
separate pools, because a retrieval answered by a mutation would be a bad day. It asks the model for
a parametrized operation, then runs that against the real API before registering it. Mutations get
redirected at a sandbox issue, so a test can't damage anything real. A failed attempt feeds its API
error into the next one, and a permission error ends the loop, since no rewording fixes an account
limit.

Not every gap is an API call. Grouping and rendering happen in memory, so those get synthesized as
declarative pipelines and tested against the rows they'll actually process.

I considered generating Python and running it. More flexible, but then I need a sandbox and I've
moved the problem rather than solved it. A capability here is a stored operation plus a variables
template, or a list of named pipeline steps. Nothing is `exec`'d, and you can read every capability
the agent taught itself out of the database.

## 3. The learning signal, run 1 vs run N

| | run 1 | run N |
|---|---|---|
| API calls | 2 | 0 |
| LLM calls | 1 | 0 |

Run 1 resolves the team and label ids and asks the model to decompose the sentence. By run N it
serves both from memory and recognises the sentence pattern, so it reuses the plan and never calls
the model. Measured across 62 comparable runs.

Two things keep that honest: only successful runs count, and only runs with the same plan shape get
compared. A blocked run makes zero API calls, so my first version of this metric scored failure as
the best result I'd had. I fixed the measurement before trusting it.

## What I'd build next

Ordering by data dependency rather than by name. "Create a team and add an issue to it" schedules
the lookup before the create, because the rule reads step names. Fixing that, plus harvesting single
object results into context, also makes "move these issues into that project" work.

Semantic plan reuse. Patterns match lexically, so a paraphrase misses. Embedding the masked pattern
would generalise it.

Multi-agent decomposition, and I'd rather be straight here. The synthesizer is already a specialist
the planner hands work to, with its own prompt, retry loop and test oracle. I could have wrapped it
in agent framing and claimed the box. It wouldn't have changed one decision the system makes, so I
left it and wrote this instead.
