"""Declarative data transforms.

Some capabilities are not API calls at all: 'group these issues by priority',
'sort them', 'render a summary'. The platform cannot do that work, so the agent
must. The obvious route — have the LLM write Python and exec() it — means running
model-authored code, which needs a real sandbox to be safe.

Instead a transform is DATA: a pipeline of named operations with parameters,
interpreted by the deterministic functions below. Synthesis generates the pipeline
spec, it is tested against the real retrieved data before being registered, and it
is inspectable in the database like every other capability.
"""
from __future__ import annotations

from typing import Any

OPS = ("filter", "sort_by", "group_by", "limit", "count", "render_markdown")


class TransformError(RuntimeError):
    pass


def _get(item: dict, path: str) -> Any:
    """Read a possibly nested field: 'assignee.name' -> item['assignee']['name']."""
    node: Any = item
    for part in path.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _op_filter(rows: list[dict], spec: dict) -> list[dict]:
    field = spec.get("field")
    if not field:
        raise TransformError("filter requires 'field'")
    if spec.get("is_null"):
        return [r for r in rows if _get(r, field) in (None, "", [])]
    if spec.get("not_null"):
        return [r for r in rows if _get(r, field) not in (None, "", [])]
    if "equals" in spec:
        want = str(spec["equals"]).lower()
        return [r for r in rows if str(_get(r, field)).lower() == want]
    if "contains" in spec:
        want = str(spec["contains"]).lower()
        return [r for r in rows if want in str(_get(r, field) or "").lower()]
    raise TransformError("filter needs one of: is_null, not_null, equals, contains")


def _op_sort_by(rows: list[dict], spec: dict) -> list[dict]:
    field = spec.get("field")
    if not field:
        raise TransformError("sort_by requires 'field'")
    return sorted(
        rows,
        key=lambda r: (_get(r, field) is None, _get(r, field)),
        reverse=bool(spec.get("desc")),
    )


def _op_group_by(rows: list[dict], spec: dict) -> dict[str, list[dict]]:
    field = spec.get("field")
    if not field:
        raise TransformError("group_by requires 'field'")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = _get(row, field)
        groups.setdefault("(none)" if key in (None, "") else str(key), []).append(row)
    return groups


def _op_render_markdown(data: Any, spec: dict) -> str:
    """Render rows or groups as markdown. The final step of a summary capability."""
    label = spec.get("label_field", "identifier")
    text = spec.get("text_field", "title")
    heading = spec.get("heading")

    def line(row: dict) -> str:
        left = _get(row, label)
        right = _get(row, text)
        return f"- {left}: {right}" if left else f"- {right}"

    out: list[str] = []
    if heading:
        out.append(f"# {heading}\n")
    if isinstance(data, dict):
        total = sum(len(v) for v in data.values())
        out.append(f"{total} item(s) in {len(data)} group(s).\n")
        for key in sorted(data):
            out.append(f"## {key} ({len(data[key])})")
            out.extend(line(r) for r in data[key])
            out.append("")
    elif isinstance(data, list):
        out.append(f"{len(data)} item(s).\n")
        out.extend(line(r) for r in data)
    else:
        out.append(str(data))
    return "\n".join(out).strip()


def run_pipeline(rows: list[dict], pipeline: list[dict]) -> Any:
    """Apply each operation in order. Unknown operations are refused rather than
    skipped — silently ignoring a step would drop part of the instruction."""
    data: Any = rows
    for spec in pipeline:
        op = spec.get("op")
        if op not in OPS:
            raise TransformError(f"unknown transform op '{op}'; supported: {', '.join(OPS)}")
        if op == "filter":
            data = _op_filter(_as_rows(data, op), spec)
        elif op == "sort_by":
            data = _op_sort_by(_as_rows(data, op), spec)
        elif op == "group_by":
            data = _op_group_by(_as_rows(data, op), spec)
        elif op == "limit":
            data = _as_rows(data, op)[: int(spec.get("n", 50))]
        elif op == "count":
            data = len(data) if isinstance(data, (list, dict)) else 0
        elif op == "render_markdown":
            data = _op_render_markdown(data, spec)
    return data


def _as_rows(data: Any, op: str) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):  # flatten groups back to rows
        flat: list[dict] = []
        for value in data.values():
            if isinstance(value, list):
                flat.extend(value)
        return flat
    raise TransformError(f"'{op}' expects rows, got {type(data).__name__}")
