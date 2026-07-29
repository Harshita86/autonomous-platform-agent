"""Persistent memory (SQLite). Single writer, so the invariants can't be broken
by any caller:
  - episodes    : append-only execution history (the learning proof depends on it)
  - resolutions : idempotent, monotonic name->id cache (the semantic world-model)

The procedural/capability store and synthesis land in the 'widen' step; the schema
here is the seam they slot into.
"""
from __future__ import annotations

import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    instruction   TEXT    NOT NULL,
    signature     TEXT    NOT NULL,
    plan_json     TEXT    NOT NULL,
    results_json  TEXT    NOT NULL,
    outcome       TEXT    NOT NULL,
    api_calls     INTEGER NOT NULL,
    latency_ms    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_sig ON episodes(signature);

CREATE TABLE IF NOT EXISTS resolutions (
    kind         TEXT NOT NULL,   -- 'team' | 'label' | ...
    name         TEXT NOT NULL,
    value        TEXT NOT NULL,   -- the resolved id
    episode_id   INTEGER,
    ts           REAL NOT NULL,
    PRIMARY KEY (kind, name)
);

-- Procedural memory: capabilities are DATA, not code. A synthesized capability
-- is stored and invoked identically to a built-in one. Versioned, so a
-- re-synthesis that regresses never destroys a known-good version.
CREATE TABLE IF NOT EXISTS capabilities (
    name         TEXT    NOT NULL,
    version      INTEGER NOT NULL,
    kind         TEXT    NOT NULL,   -- primitive | synthesized | composite
    description  TEXT    NOT NULL,
    handler      TEXT    NOT NULL,   -- 'builtin:<fn>' or 'graphql'
    graphql      TEXT,               -- parametrized operation, for handler='graphql'
    params_json  TEXT    NOT NULL,   -- declared input parameter names
    provenance   TEXT,               -- how it came to exist
    tests_json   TEXT,               -- validation performed before registering
    invocations  INTEGER NOT NULL DEFAULT 0,
    successes    INTEGER NOT NULL DEFAULT 0,
    failures     INTEGER NOT NULL DEFAULT 0,
    inverse_capability_id TEXT,
    superseded   INTEGER NOT NULL DEFAULT 0,
    ts           REAL    NOT NULL,
    PRIMARY KEY (name, version)
);

-- Episodic detail: entities this agent created. Lets a repeated instruction be
-- recognised as already-done instead of silently duplicating real work, and gives
-- rollback something concrete to compensate.
CREATE TABLE IF NOT EXISTS created_entities (
    kind        TEXT NOT NULL,        -- 'issue' | ...
    scope       TEXT NOT NULL,        -- team id, so identical titles in two teams differ
    fingerprint TEXT NOT NULL,        -- normalised title
    entity_id   TEXT NOT NULL,
    identifier  TEXT,
    url         TEXT,
    episode_id  INTEGER,
    ts          REAL NOT NULL,
    PRIMARY KEY (kind, scope, fingerprint)
);

-- Compaction target: the aggregate of episodes that have been folded away. The
-- first run's cost is kept explicitly, because that is the baseline the learning
-- comparison is measured against and it must survive the raw rows being removed.
CREATE TABLE IF NOT EXISTS episode_summaries (
    signature        TEXT PRIMARY KEY,
    runs             INTEGER NOT NULL,
    successes        INTEGER NOT NULL,
    first_ts         REAL,
    last_ts          REAL,
    first_api_calls  INTEGER,
    first_latency_ms INTEGER,
    min_api_calls    INTEGER,
    max_api_calls    INTEGER,
    total_api_calls  INTEGER NOT NULL DEFAULT 0,
    total_latency_ms INTEGER NOT NULL DEFAULT 0,
    total_llm_calls  INTEGER NOT NULL DEFAULT 0
);

-- Semantic memory: constraints discovered at runtime (validation rules, enums,
-- permission boundaries). Monotonic — this is the fuel for pre-validation.
CREATE TABLE IF NOT EXISTS constraints (
    key          TEXT NOT NULL PRIMARY KEY,
    kind         TEXT NOT NULL,   -- validation | rate_limit | permission | resolution
    value        TEXT NOT NULL,
    episode_id   INTEGER,
    ts           REAL NOT NULL
);
"""


class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        """Additive migrations, so an existing memory file keeps its history rather
        than being wiped when the schema grows."""
        caps = {r["name"] for r in self._db.execute("PRAGMA table_info(capabilities)")}
        if "inverse_capability_id" not in caps:
            self._db.execute("ALTER TABLE capabilities ADD COLUMN inverse_capability_id TEXT")
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(episodes)")}
        for name, ddl in (
            ("gen_template", "ALTER TABLE episodes ADD COLUMN gen_template TEXT"),
            ("llm_calls", "ALTER TABLE episodes ADD COLUMN llm_calls INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in cols:
                self._db.execute(ddl)

    # --- semantic store: name -> id resolution cache ---
    def get_resolution(self, kind: str, name: str) -> str | None:
        row = self._db.execute(
            "SELECT value FROM resolutions WHERE kind=? AND name=?",
            (kind, name.lower()),
        ).fetchone()
        return row["value"] if row else None

    def put_resolution(self, kind: str, name: str, value: str, episode_id: int | None = None) -> None:
        # INSERT OR IGNORE keeps it monotonic: the first true resolution wins and
        # is never silently overwritten.
        self._db.execute(
            "INSERT OR IGNORE INTO resolutions(kind, name, value, episode_id, ts) VALUES (?,?,?,?,?)",
            (kind, name.lower(), value, episode_id, time.time()),
        )
        self._db.commit()

    # --- episodic store: append-only ---
    def add_episode(
        self,
        instruction: str,
        signature: str,
        plan_json: str,
        results_json: str,
        outcome: str,
        api_calls: int,
        latency_ms: int,
        llm_calls: int = 0,
        gen_template: str | None = None,
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO episodes(ts, instruction, signature, plan_json, results_json, "
            "outcome, api_calls, latency_ms, llm_calls, gen_template) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(), instruction, signature, plan_json, results_json, outcome,
                api_calls, latency_ms, llm_calls, gen_template,
            ),
        )
        self._db.commit()
        return cur.lastrowid

    def summary_for(self, signature: str) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM episode_summaries WHERE signature=?", (signature,)
        ).fetchone()

    def compact(self, keep_recent: int = 5, apply: bool = False) -> list[dict]:
        """Fold old episodes into per-signature aggregates.

        History is otherwise append-only, so this is a deliberate, explicit and
        opt-in exception rather than something that happens quietly in the
        background. Three things are never folded away:

          * the most recent `keep_recent` runs of each signature, which is what
            anyone actually inspects;
          * the cheapest successful run of each learned sentence pattern, because
            deleting it would destroy the decomposition the agent reuses;
          * the first run's cost, which is preserved in the summary because it is
            the baseline the learning comparison is measured against.

        Aggregates therefore survive compaction, and so does behaviour.
        """
        report: list[dict] = []
        signatures = [
            r["signature"]
            for r in self._db.execute("SELECT DISTINCT signature FROM episodes")
        ]
        for signature in signatures:
            rows = self._db.execute(
                "SELECT * FROM episodes WHERE signature=? ORDER BY id", (signature,)
            ).fetchall()
            keep = {r["id"] for r in rows[-keep_recent:]}

            # Preserve the cheapest successful run per learned pattern.
            best: dict[str, sqlite3.Row] = {}
            for r in rows:
                if r["outcome"] != "success" or not r["gen_template"]:
                    continue
                current = best.get(r["gen_template"])
                if current is None or r["api_calls"] < current["api_calls"]:
                    best[r["gen_template"]] = r
            keep |= {r["id"] for r in best.values()}

            stale = [r for r in rows if r["id"] not in keep]
            if not stale:
                continue
            report.append({
                "signature": signature,
                "folded": len(stale),
                "kept": len(rows) - len(stale),
                "patterns_preserved": len(best),
            })
            if apply:
                self._fold(signature, stale)
                self._db.executemany(
                    "DELETE FROM episodes WHERE id=?", [(r["id"],) for r in stale]
                )
        if apply:
            self._db.commit()
        return report

    def _fold(self, signature: str, rows: list[sqlite3.Row]) -> None:
        existing = self.summary_for(signature)
        runs = len(rows) + (existing["runs"] if existing else 0)
        successes = sum(1 for r in rows if r["outcome"] == "success") + (
            existing["successes"] if existing else 0
        )
        api = [r["api_calls"] for r in rows]
        lat = [r["latency_ms"] for r in rows]
        first = rows[0]
        self._db.execute(
            "INSERT OR REPLACE INTO episode_summaries(signature, runs, successes, "
            "first_ts, last_ts, first_api_calls, first_latency_ms, min_api_calls, "
            "max_api_calls, total_api_calls, total_latency_ms, total_llm_calls) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                signature, runs, successes,
                existing["first_ts"] if existing else first["ts"],
                rows[-1]["ts"],
                # The earliest cost is the baseline; never overwrite it.
                existing["first_api_calls"] if existing else first["api_calls"],
                existing["first_latency_ms"] if existing else first["latency_ms"],
                min(api + ([existing["min_api_calls"]] if existing else [])),
                max(api + ([existing["max_api_calls"]] if existing else [])),
                sum(api) + (existing["total_api_calls"] if existing else 0),
                sum(lat) + (existing["total_latency_ms"] if existing else 0),
                sum(r["llm_calls"] or 0 for r in rows)
                + (existing["total_llm_calls"] if existing else 0),
            ),
        )

    def successful_patterns(self, limit: int = 200) -> list[sqlite3.Row]:
        """Learned sentence patterns from successful runs, one per distinct pattern,
        cheapest first.

        A LIMIT applied before deduplication let repeats of one heavily-used
        pattern fill every slot, so a genuinely new pattern learned minutes ago
        was never even offered to the matcher — cheap-but-irrelevant history
        crowded out relevant history. Distinct patterns are cheap to tell apart
        (a substring of gen_template) and there are far fewer of them than there
        are episodes, so dedup first and cap the result after.
        """
        rows = self._db.execute(
            "SELECT * FROM episodes WHERE outcome='success' AND gen_template IS NOT NULL "
            "ORDER BY api_calls ASC, llm_calls ASC, id DESC"
        ).fetchall()
        seen: set[str] = set()
        out: list[sqlite3.Row] = []
        for row in rows:
            try:
                key = json.loads(row["gen_template"])["pattern"]
            except Exception:  # noqa: BLE001 — an unparseable pattern is simply skipped
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    # --- procedural store: capabilities as data, versioned ---
    def get_capability(self, name: str) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM capabilities WHERE name=? AND superseded=0 "
            "ORDER BY version DESC LIMIT 1",
            (name,),
        ).fetchone()

    def list_capabilities(self) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM capabilities WHERE superseded=0 ORDER BY name"
        ).fetchall()

    def put_capability(
        self,
        name: str,
        kind: str,
        description: str,
        handler: str,
        params: list[str],
        graphql: str | None = None,
        provenance: str | None = None,
        tests: str | None = None,
    ) -> int:
        """Insert as version+1 and supersede the prior version — never overwrite,
        so a regressing re-synthesis can't destroy a known-good capability."""
        prev = self._db.execute(
            "SELECT MAX(version) AS v FROM capabilities WHERE name=?", (name,)
        ).fetchone()
        version = (prev["v"] or 0) + 1
        self._db.execute("UPDATE capabilities SET superseded=1 WHERE name=?", (name,))
        self._db.execute(
            "INSERT INTO capabilities(name, version, kind, description, handler, graphql, "
            "params_json, provenance, tests_json, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                name, version, kind, description, handler, graphql,
                json.dumps(params), provenance, tests, time.time(),
            ),
        )
        self._db.commit()
        return version

    def record_invocation(self, name: str, success: bool) -> None:
        col = "successes" if success else "failures"
        self._db.execute(
            f"UPDATE capabilities SET invocations=invocations+1, {col}={col}+1 "
            "WHERE name=? AND superseded=0",
            (name,),
        )
        self._db.commit()

    # --- created entities: idempotency + rollback targets ---
    @staticmethod
    def fingerprint(text: str) -> str:
        return " ".join(text.lower().split())

    def find_created(self, kind: str, scope: str, title: str) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM created_entities WHERE kind=? AND scope=? AND fingerprint=?",
            (kind, scope, self.fingerprint(title)),
        ).fetchone()

    def record_created(
        self, kind: str, scope: str, title: str, entity_id: str,
        identifier: str | None = None, url: str | None = None,
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO created_entities"
            "(kind, scope, fingerprint, entity_id, identifier, url, ts) VALUES (?,?,?,?,?,?,?)",
            (kind, scope, self.fingerprint(title), entity_id, identifier, url, time.time()),
        )
        self._db.commit()

    def forget_created(self, kind: str, entity_id: str) -> None:
        """Called after a successful rollback: the entity no longer exists, so it
        must not keep suppressing future creates."""
        self._db.execute(
            "DELETE FROM created_entities WHERE kind=? AND entity_id=?", (kind, entity_id)
        )
        self._db.commit()

    # --- semantic store: constraints ---
    def put_constraint(self, key: str, kind: str, value: str, episode_id: int | None = None) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO constraints(key, kind, value, episode_id, ts) VALUES (?,?,?,?,?)",
            (key, kind, value, episode_id, time.time()),
        )
        self._db.commit()

    def get_constraint(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM constraints WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def list_constraints(self) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM constraints ORDER BY key").fetchall()

    def episodes_for_signature(self, signature: str) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM episodes WHERE signature=? ORDER BY id ASC", (signature,)
        ).fetchall()

    def close(self) -> None:
        self._db.close()
