# Architecture

Platform: **Linear**, chosen because its GraphQL schema is introspectable. The agent asks the
platform what operations exist rather than reading a list I wrote, which is the difference between
synthesis and a lookup table.

Two decisions shape the rest. **The model proposes, the code disposes**: every LLM output must
clear a deterministic gate before it touches state. And **capabilities are data, not code**. Both
were forced on me by failures; the first version turned "delete all issues" into a newly created
issue titled *"delete all issues"* and reported SUCCESS. One boundary: 38 lines in
`adapters/linear.py` know which platform this is. Nothing else does.

## 1. What memory stores, and why it's shaped that way

One SQLite file, three layers, split by what the knowledge is *for*:

- **Episodic** (`episodes`) — each run's instruction, plan, per-step results and cost, plus the
  *sentence pattern* it answers with wording-derived values masked out. That mask is what makes a
  proven decomposition reusable.
- **Procedural** (`capabilities`) — every capability, versioned, with success rate, validation
  record and inverse. A synthesized capability is stored and invoked identically to a built-in.
- **Semantic** (`constraints`, `resolutions`) — what it learned about the platform: name→id
  mappings, valid field values, permission boundaries, and Linear's 10,000 query-complexity limit,
  which I found only because introspecting every mutation at once was rejected.

**Rejected:** a vector store of past prompts. I don't want *roughly similar*, I want *have I done
this exact task and what did it cost*. That's a keyed lookup, and it makes the metric aggregable.

A single writer owns all three tables so three invariants hold: episodes **append-only** (the
before/after claim depends on it), capabilities **versioned** rather than overwritten (a bad
re-synthesis can't destroy a working one), constraints **monotonic**. Compaction is the one
explicit exception, and it preserves the first-run baseline and each learned pattern.

## 2. How capability synthesis works

Three capabilities are built in. Everything else is built at runtime or honestly refused.

When the planner names a capability that doesn't exist, the synthesizer introspects the live schema,
shortlists real operations (ranked on Linear's `<entity><Verb>` convention, reads and writes kept
separate so a retrieval can't be answered by a mutation), asks the model for a parametrized
operation, and **executes it against the real API before registering it**. Mutations are redirected
at a sandbox issue. Each failure feeds the real API error into the next attempt; a permission error
ends the loop, since no rewording fixes an account limit.

Not every gap is an API call. Grouping and rendering are in-memory work, synthesized as declarative
pipelines and tested against the rows they will actually process.

**Rejected:** generating Python and running it. That buys flexibility and costs a sandbox. Instead a
capability is a stored operation plus a variables template, or a list of named pipeline steps —
substitution is deterministic, nothing is `exec`'d, and every capability is readable in the database.

## 3. The learning signal, run 1 vs run N

| | run 1 | run N |
|---|---|---|
| API calls | 2 | **0** |
| LLM calls | 1 | **0** |

Run 1 resolves the team and label ids and asks the model to decompose the sentence. By run N it
serves both ids from semantic memory and recognises the sentence pattern, so it reuses the proven
plan and never calls the model. Measured across 62 comparable runs.

Two properties keep that honest: only **successful** runs count, and only runs with an **identical
plan shape** are compared. A blocked run makes zero API calls, and scoring that as an improvement
would be measuring failure as progress.

## What I'd build next

**Producer/consumer ordering.** Steps are ordered by name, so "create a team and add an issue to it"
schedules the lookup before the create. Ordering on data dependency, plus harvesting single-object
results into context, fixes that and "move these issues into that project" together.

**Semantic plan reuse.** Patterns match lexically, so a paraphrase misses; embedding the masked
pattern would generalise it.

**Multi-agent decomposition.** The synthesizer is already a specialist the planner delegates to,
with its own prompt and test oracle. Formalising that wouldn't change a decision the system makes,
so I left it and said why.
