"""Capability synthesis at runtime.

Reason -> generate -> TEST against the real API -> register. The test is the part
that matters: anything can describe an endpoint to an LLM, but only a validated
artifact may enter capability memory.

A synthesized capability is DATA, never generated code:
    graphql            a parametrized GraphQL operation
    variables_template a JSON template with {{placeholder}} slots
    value_maps         string -> API value translations (e.g. high -> 2)
Substitution is deterministic, so there is nothing to sandbox and the artifact is
inspectable in the database. This is the 'LLM proposes, the system disposes' rule
applied to code generation itself.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .adapters.linear import LinearAdapter, LinearError
from .capabilities import CapabilityRegistry
from .contracts import PlanStep
from .llm import LLM
from .memory.store import MemoryStore
from .transforms import TransformError, run_pipeline

SANDBOX_TITLE = "[agent-synthesis-sandbox] safe to delete"
SANDBOX_KEY = "linear.sandbox.issue_id"
# Versioned: the cached summary's shape changed when queries were added, and a
# stale entry would silently hide half the schema from candidate selection.
SCHEMA_KEY = "linear.schema.operations.v2"

# Introspection is split into cheap phases: Linear enforces a query-complexity
# limit (discovered at runtime — asking for every mutation's args at once is
# rejected at complexity 16384 > 10000), so we fetch names first and only then
# the input type of the operations we actually shortlist.
_MUTATIONS_Q = "query { __schema { mutationType { fields { name } } } }"
_QUERIES_Q = "query { __schema { queryType { fields { name } } } }"
_INPUT_TYPE_Q = """
query T($name: String!) { __type(name: $name) {
  inputFields { name type { kind name ofType { kind name } } } } }
"""
_CREATE_SANDBOX = """
mutation C($input: IssueCreateInput!) {
  issueCreate(input: $input) { success issue { id identifier } } }
"""

_SYSTEM = """You synthesise a single Linear GraphQL capability.

You are given: the capability name and purpose, the parameter names the caller will
supply, the runtime context keys available, and the REAL schema signatures for
candidate operations (from live introspection). Use only fields that appear there.

Return ONLY JSON:
{
  "operation_name": "<the GraphQL field you chose, e.g. issueUpdate>",
  "graphql": "<a complete parametrized mutation/query using GraphQL variables>",
  "variables_template": { ... },
  "value_maps": { "<param>": { "<word>": <api value> } },
  "description": "<one line>"
}

RULES
- variables_template mirrors the operation's variables. Use "{{name}}" where a value
  must be filled from a caller parameter or a runtime context key. A string that is
  exactly "{{name}}" is replaced by the typed value, so integers stay integers.
- Put word->number or word->enum translations in value_maps (Linear priority is an
  Int: urgent=1, high=2, medium=3, low=4; 0 means none).
- Request a "success" field in the selection set when the operation offers one, so
  the result can be validated.
- For a QUERY (retrieval): Linear list queries are connections. Select a RICH node
  set, because later steps group, sort and summarise these rows and can only use
  fields you return. For issues always select at least:
  `{ nodes { id identifier title url priority priorityLabel createdAt
             state { name type } assignee { name } labels { nodes { name } } } }`
  Express conditions through the filter input (e.g. `{ assignee: { null: true } }`,
  `{ labels: { name: { eq: "Bug" } } }`, `{ state: { type: { neq: "completed" } } }`).
  Pass the whole filter object as one variable and cap results with `first: 50`.
- Identifiers must come from runtime context keys, never from caller parameters.
  A parameter like team or label holds a human-readable NAME; the matching id is
  already resolved and available as a context key such as team_id, label_ids or
  issue_id. Any field the schema types as an ID or UUID must be filled from those.
- Do not invent fields. If the candidates cannot express the purpose, return
  {"error": "<why>"}."""


_TRANSFORM_SYSTEM = """You synthesise a DATA TRANSFORM capability — work the platform
API cannot do, such as grouping, sorting, filtering in memory, counting, or rendering
a summary from rows already retrieved.

You are given the purpose and a sample of the ACTUAL rows the transform will receive.
Use only field names that appear in that sample.

Supported operations (use no others):
  {"op":"filter","field":"<f>","is_null":true|"not_null":true|"equals":<v>|"contains":<v>}
  {"op":"sort_by","field":"<f>","desc":true|false}
  {"op":"group_by","field":"<f>"}
  {"op":"limit","n":<int>}
  {"op":"count"}
  {"op":"render_markdown","heading":"<title>","label_field":"<f>","text_field":"<f>"}

Return ONLY JSON:
{"pipeline":[ ...operations in order... ],"description":"<one line>"}

Notes
- If the purpose is to produce readable text (a summary, a report, a page body),
  end the pipeline with render_markdown.
- Nested fields use dots, e.g. "assignee.name". A missing assignee is null.
- If the purpose cannot be expressed with these operations, return
  {"error":"<why>"}."""

# Purposes that are in-memory data work rather than platform operations.
_TRANSFORM_WORDS = re.compile(
    r"\b(group|groups|grouping|sort|order|rank|summar|report|format|render|compile|"
    r"aggregate|count|tally|organi[sz]e|breakdown|digest|bucket)\w*\b",
    re.I,
)


class SynthesisFailed(RuntimeError):
    pass


def _substitute(node: Any, resolve) -> Any:
    """Fill {{placeholders}}. An exact '{{x}}' string yields the typed value;
    inline placeholders yield string interpolation."""
    if isinstance(node, dict):
        return {k: _substitute(v, resolve) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, resolve) for v in node]
    if isinstance(node, str):
        exact = re.fullmatch(r"\{\{\s*([\w.]+)\s*\}\}", node)
        if exact:
            return resolve(exact.group(1))
        return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", lambda m: str(resolve(m.group(1))), node)
    return node


def build_variables(
    template: dict, value_maps: dict, params: dict, ctx: dict
) -> dict:
    """Deterministic binding of a synthesized capability's variables."""
    def resolve(name: str):
        if name in params and params[name] is not None:
            value = params[name]
        elif name in ctx:
            value = ctx[name]
        else:
            raise LinearError(f"missing value for '{name}'")
        mapping = value_maps.get(name)
        if mapping and isinstance(value, str):
            key = value.strip().lower()
            if key in mapping:
                return mapping[key]
        return value

    return _substitute(template, resolve)


class Synthesizer:
    def __init__(
        self,
        adapter: LinearAdapter,
        memory: MemoryStore,
        registry: CapabilityRegistry,
        llm: LLM | None = None,
        attempts: int = 3,
    ):
        self._a = adapter
        self._m = memory
        self._r = registry
        self._llm = llm or LLM()
        self._attempts = attempts
        self.log: list[str] = []

    # --- reason: what does the platform actually offer? ---
    def _schema_summary(self) -> list[dict]:
        cached = self._m.get_constraint(SCHEMA_KEY)
        if cached:
            return json.loads(cached)
        data = self._a.execute(_MUTATIONS_Q)
        fields = [
            {"field": f["name"], "op": "mutation"}
            for f in data["__schema"]["mutationType"]["fields"]
        ]
        # Read side too: retrieval capabilities ('find all unassigned bugs') are
        # queries, and without these the agent can only ever write.
        qdata = self._a.execute(_QUERIES_Q)
        fields += [
            {"field": f["name"], "op": "query"}
            for f in qdata["__schema"]["queryType"]["fields"]
        ]
        # Discovered once, reused forever: a runtime-learned fact about the platform.
        self._m.put_constraint(SCHEMA_KEY, "schema", json.dumps(fields))
        self._m.put_constraint(
            "linear.graphql.max_complexity",
            "rate_limit",
            "10000 — introspect incrementally, never the whole mutation set with args",
        )
        return fields

    # Intent words -> the CRUD verb Linear actually names its mutations with.
    _ACTIONS = {
        "update": ("set", "assign", "unassign", "change", "update", "edit", "modify",
                   "prioritise", "prioritize", "rename", "move", "close", "reopen"),
        "create": ("create", "add", "new", "file", "log", "open", "raise", "report"),
        "delete": ("delete", "remove", "destroy"),
        "archive": ("archive",),
    }
    _READ_WORDS = ("find", "search", "list", "get", "fetch", "retrieve", "query", "all",
                   "every", "lookup", "collect")
    _ENTITIES = ("issue", "project", "comment", "label", "team", "cycle", "attachment",
                 "milestone", "user", "document", "roadmap")

    def _candidates(self, name: str, purpose: str, limit: int = 5) -> list[dict]:
        """Rank real schema operations against the need.

        Naive substring overlap ranks 'issueImportCreateLinearV2' above 'issueUpdate';
        Linear's naming is <entity><Verb>, so score that structure explicitly and
        prefer the shortest (most canonical) field on ties.
        """
        text = f"{name} {purpose}".lower()
        words = set(re.findall(r"[a-z]+", text))
        entity = next((e for e in self._ENTITIES if e in words or f"{e}s" in words), None)
        verbs = {v for v, syns in self._ACTIONS.items() if words & set(syns)}

        # The capability NAME decides read vs write, not loose word matching over the
        # purpose: 'find open unassigned issues' contains 'open', which is a creation
        # synonym, and that alone was enough to send a retrieval need to the mutations.
        lname = name.lower()
        if lname.startswith(("find", "search", "list", "get", "fetch", "retrieve", "query")):
            wants_read, verbs = True, set()
        elif lname.startswith(("create", "add", "new", "make", "update", "set", "delete",
                               "remove", "assign", "archive", "close", "move", "rename")):
            wants_read = False
        else:
            wants_read = bool(words & set(self._READ_WORDS)) and not verbs

        scored = []
        for f in self._schema_summary():
            fl = f["field"].lower()
            is_query = f.get("op") == "query"
            # A retrieval need must not be answered with a mutation, and vice versa.
            if wants_read != is_query:
                continue
            score = 0
            if entity and fl.startswith(entity):
                score += 3
            elif entity and entity in fl:
                score += 1
            if verbs and any(v in fl for v in verbs):
                score += 3
            score += sum(1 for w in words if len(w) > 4 and w in fl)
            if score:
                # Shorter field names are the canonical operations.
                scored.append((score - len(f["field"]) / 100.0, f))
        scored.sort(key=lambda s: -s[0])
        out = []
        for _score, f in scored[:limit]:
            enriched = dict(f)
            field = f["field"]
            if f.get("op") == "query":
                # Linear's convention: the `issues` query filters with `IssueFilter`.
                guess = field[0].upper() + field[1:]
                guess = (guess[:-1] if guess.endswith("s") else guess) + "Filter"
                enriched["returns"] = "connection: { nodes { ... } }"
            else:
                # ...and mutation `issueUpdate` takes `IssueUpdateInput`.
                guess = field[0].upper() + field[1:] + "Input"
            fields = self._input_fields(guess)
            if fields:
                enriched["input_type"] = guess
                enriched["input_fields"] = fields[:40]
            out.append(enriched)
        return out

    def _input_fields(self, type_name: str) -> list[dict]:
        cache_key = f"linear.schema.input.{type_name}"
        cached = self._m.get_constraint(cache_key)
        if cached:
            return json.loads(cached)
        try:
            d = self._a.execute(_INPUT_TYPE_Q, {"name": type_name})
        except LinearError:
            return []
        t = d.get("__type") or {}
        fields = [
            {
                "name": f["name"],
                "type": f["type"].get("name") or (f["type"].get("ofType") or {}).get("name"),
            }
            for f in (t.get("inputFields") or [])
        ]
        if fields:
            self._m.put_constraint(cache_key, "schema", json.dumps(fields))
        return fields

    # --- test: a real entity to operate on ---
    def sandbox_issue_id(self) -> str:
        existing = self._m.get_constraint(SANDBOX_KEY)
        if existing:
            return existing
        team_id = self._m.get_resolution("team", self._default_team_name())
        if not team_id:
            raise SynthesisFailed("no team resolved yet; cannot create a synthesis sandbox")
        data = self._a.execute(
            _CREATE_SANDBOX, {"input": {"teamId": team_id, "title": SANDBOX_TITLE}}
        )
        issue = data["issueCreate"]["issue"]
        self._m.put_constraint(SANDBOX_KEY, "resolution", issue["id"])
        self.log.append(f"created synthesis sandbox {issue['identifier']} (reused hereafter)")
        return issue["id"]

    def _default_team_name(self) -> str:
        row = self._m._db.execute(
            "SELECT name FROM resolutions WHERE kind='team' ORDER BY ts LIMIT 1"
        ).fetchone()
        return row["name"] if row else ""

    # --- the loop ---
    def synthesize(self, step: PlanStep, ctx: dict) -> bool:
        name = step.capability
        purpose = (step.spec.purpose if step.spec else "") or step.description or name
        param_names = sorted(step.params.keys())
        self.log.append(f"gap '{name}': {purpose}")

        # Decide what KIND of capability is missing. Not everything is an API call:
        # grouping and summarising are in-memory work over rows already fetched.
        rows = self._available_rows(ctx)
        # A name that starts with a write verb creates something on the platform,
        # whatever else it says. Without this, 'create_weekly_triage_summary_page'
        # matched 'summar', became an in-memory transform, and the run reported
        # SUCCESS while nothing was ever written to Linear.
        writes = name.lower().startswith(
            ("create", "add", "new", "make", "post", "update", "set", "delete", "assign")
        )
        if rows and not writes and _TRANSFORM_WORDS.search(f"{name} {purpose}"):
            return self._synthesize_transform(name, purpose, param_names, rows)

        candidates = self._candidates(name, purpose)
        if not candidates:
            self.log.append("  no candidate operation in the live schema — cannot synthesise")
            return False
        self.log.append(
            f"  introspected {len(candidates)} candidate operations: "
            + ", ".join(c["field"] for c in candidates)
        )

        feedback = ""
        for attempt in range(1, self._attempts + 1):
            try:
                proposal = self._propose(name, purpose, param_names, ctx, candidates, feedback)
            except Exception as exc:  # noqa: BLE001
                self.log.append(f"  attempt {attempt}: generation failed — {exc}")
                feedback = f"Your previous output was unusable: {exc}"
                continue

            if "error" in proposal:
                self.log.append(f"  attempt {attempt}: model declined — {proposal['error']}")
                return False

            try:
                self._test(proposal, step, ctx)
            except Exception as exc:  # noqa: BLE001 — failed test = not registered
                self.log.append(f"  attempt {attempt}: TEST FAILED — {exc}")
                if self._record_permission_boundary(name, proposal, exc):
                    return False  # a boundary is not something retrying can fix
                feedback = (
                    f"The operation you produced failed when executed against the real "
                    f"Linear API with this error: {exc}. Fix it."
                )
                # A UUID complaint is the one error a model reliably repeats: it
                # keeps sending the human-readable name it was given. Say plainly
                # where the id actually is, or the remaining attempts are wasted.
                if "isUuid" in str(exc) or "must be a UUID" in str(exc):
                    feedback += (
                        " The value you sent is a NAME, not an id. Use the matching "
                        f"context key instead — available now: {sorted(ctx.keys())}. "
                        "For example use {{team_id}} rather than {{team}}."
                    )
                continue

            version = self._r.register_synthesized(
                name=name,
                description=proposal.get("description") or purpose,
                graphql=json.dumps(
                    {
                        "graphql": proposal["graphql"],
                        "variables_template": proposal["variables_template"],
                        "value_maps": proposal.get("value_maps") or {},
                    }
                ),
                params=param_names,
                provenance=(
                    f"synthesized at runtime from live schema introspection; "
                    f"operation={proposal.get('operation_name')}; attempt={attempt}"
                ),
                tests=json.dumps(
                    {"executed_against": "real Linear API", "sandbox_issue": True, "attempt": attempt}
                ),
            )
            self.log.append(
                f"  attempt {attempt}: TEST PASSED on sandbox issue — registered "
                f"'{name}' v{version} (operation: {proposal.get('operation_name')})"
            )
            return True

        self.log.append(f"  gave up after {self._attempts} attempts")
        return False

    _FORBIDDEN = re.compile(
        r"forbidden|access denied|not authori[sz]ed|permission|upgrade to|"
        r"limit of .* allowed|plan",
        re.I,
    )

    def _record_permission_boundary(self, name: str, proposal: dict, exc: Exception) -> bool:
        """Remember a boundary the account cannot cross.

        A validation error is worth retrying with better arguments; a permission or
        plan limit is not — no rewording of the operation will make it succeed. It
        is recorded so later runs can refuse immediately instead of spending
        attempts rediscovering it, and so the limit is visible in memory.
        """
        message = str(exc)
        if not self._FORBIDDEN.search(message):
            return False
        operation = proposal.get("operation_name") or name
        self._m.put_constraint(
            key=f"linear.permission.{operation}",
            kind="permission",
            value=f"denied for this account: {message[:200]}",
        )
        self.log.append(
            f"  recorded a permission boundary for '{operation}' — retrying cannot "
            f"fix an account limit, so synthesis stops here"
        )
        return True

    # --- transform synthesis ---
    @staticmethod
    def _available_rows(ctx: dict) -> list[dict]:
        found = [v for k, v in ctx.items() if k.endswith("::results") and isinstance(v, list)]
        return found[-1] if found else []

    def _synthesize_transform(self, name, purpose, param_names, rows) -> bool:
        """Build an in-memory transform, tested on the real rows it will process.

        The oracle here is execution over actual data: a pipeline referencing a field
        that does not exist, or an unsupported operation, fails immediately — so an
        untested pipeline can never be registered.
        """
        self.log.append(
            f"  classified as a DATA TRANSFORM (not a platform call); "
            f"{len(rows)} row(s) available as input"
        )
        sample = json.dumps(rows[:3], indent=2)[:1500]
        feedback = ""
        for attempt in range(1, self._attempts + 1):
            user = json.dumps(
                {
                    "capability_name": name,
                    "purpose": purpose,
                    "caller_parameters": param_names,
                    "sample_rows": json.loads(sample) if sample.strip().startswith("[") else sample,
                    "feedback_on_previous_attempt": feedback or None,
                },
                indent=2,
            )
            try:
                text = self._llm.complete(_TRANSFORM_SYSTEM, user).strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    text = text[4:].strip() if text.lstrip().startswith("json") else text.strip()
                proposal = json.loads(text)
            except Exception as exc:  # noqa: BLE001
                self.log.append(f"  attempt {attempt}: generation failed — {exc}")
                feedback = f"Your previous output was unusable: {exc}"
                continue

            if "error" in proposal:
                self.log.append(f"  attempt {attempt}: model declined — {proposal['error']}")
                return False
            pipeline = proposal.get("pipeline")
            if not isinstance(pipeline, list) or not pipeline:
                feedback = "The 'pipeline' field must be a non-empty list of operations."
                self.log.append(f"  attempt {attempt}: no usable pipeline")
                continue

            try:
                output = run_pipeline(rows, pipeline)
                if output in (None, "", [], {}):
                    raise TransformError("pipeline produced an empty result")
            except Exception as exc:  # noqa: BLE001 — failed test = not registered
                self.log.append(f"  attempt {attempt}: TEST FAILED — {exc}")
                feedback = (
                    f"Running your pipeline on the real rows raised: {exc}. Fix it, "
                    f"using only fields present in sample_rows."
                )
                continue

            version = self._m.put_capability(
                name=name,
                kind="synthesized",
                description=proposal.get("description") or purpose,
                handler="transform",
                params=param_names,
                graphql=json.dumps({"pipeline": pipeline}),
                provenance=(
                    f"synthesized at runtime as an in-memory transform; "
                    f"ops={[p.get('op') for p in pipeline]}; attempt={attempt}"
                ),
                tests=json.dumps(
                    {"executed_against": f"{len(rows)} real rows", "attempt": attempt}
                ),
            )
            self.log.append(
                f"  attempt {attempt}: TEST PASSED on {len(rows)} real row(s) — registered "
                f"'{name}' v{version} (pipeline: {[p.get('op') for p in pipeline]})"
            )
            return True

        self.log.append(f"  gave up after {self._attempts} attempts")
        return False

    def _propose(self, name, purpose, param_names, ctx, candidates, feedback) -> dict:
        user = json.dumps(
            {
                "capability_name": name,
                "purpose": purpose,
                "caller_parameters": param_names,
                "runtime_context_keys": sorted(ctx.keys()) + ["issue_id"],
                "candidate_operations": candidates,
                "feedback_on_previous_attempt": feedback or None,
            },
            indent=2,
        )
        text = self._llm.complete(_SYSTEM, user).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            text = text[4:].strip() if text.lstrip().startswith("json") else text.strip()
        proposal = json.loads(text)
        if "error" in proposal:
            return proposal
        for key in ("graphql", "variables_template"):
            if key not in proposal:
                raise ValueError(f"proposal missing '{key}'")
        return proposal

    def _test(self, proposal: dict, step: PlanStep, ctx: dict) -> None:
        """Execute the candidate against the REAL API. Linear's own validation is the
        oracle: bad enums, wrong types, unknown fields and permission errors all fail
        here, so only a working artifact can be registered.

        Targets are redirected at a sandbox: mutations that address an existing issue
        get the sandbox issue's id, and anything that creates gets the sandbox title,
        so a test can never mutate real work.
        """
        read_only = proposal["graphql"].strip().lower().startswith("query")
        if read_only:
            # A query changes nothing, so it is tested exactly as it will be used.
            variables = build_variables(
                proposal["variables_template"], proposal.get("value_maps") or {},
                step.params, ctx,
            )
        else:
            sandbox = self.sandbox_issue_id()
            test_ctx = {**ctx, "issue_id": sandbox, "id": sandbox, "issueId": sandbox}
            variables = build_variables(
                proposal["variables_template"], proposal.get("value_maps") or {},
                step.params, test_ctx,
            )
            _redirect_to_sandbox(variables, sandbox)
        data = self._a.execute(proposal["graphql"], variables)
        for value in _walk(data):
            if isinstance(value, dict) and value.get("success") is False:
                raise LinearError(f"operation returned success=false: {json.dumps(data)[:200]}")
        if not data:
            raise LinearError("operation returned no data")


def _redirect_to_sandbox(variables: Any, sandbox_id: str) -> None:
    """Rewrite a test payload so it cannot touch real data: ids point at the sandbox
    issue, and any title/description is marked as a synthesis test."""
    if isinstance(variables, dict):
        for key, value in variables.items():
            if key in ("id", "issueId") and isinstance(value, str):
                variables[key] = sandbox_id
            elif key == "title":
                variables[key] = SANDBOX_TITLE
            elif key == "description":
                variables[key] = "created by capability synthesis test"
            else:
                _redirect_to_sandbox(value, sandbox_id)
    elif isinstance(variables, list):
        for item in variables:
            _redirect_to_sandbox(item, sandbox_id)


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)
